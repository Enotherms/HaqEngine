//! Quality mode in the layer: FSR 3 optical flow at half resolution and HaqEngine's packed blend,
//! with every image in one allocation and the per-frame constants in a ring of slot regions.

use ash::vk;
use haqengine_core::fsr3_flow::{half_size, of_schedule, of_sizes, set_frame_index, OfDispatch, OfInput, OfRes, OF_LEVELS, SCD_HISTOGRAM_WIDTH};
use crate::pipelines::{descriptor_type, FsrPass, QualityPipelines};
use crate::state::DeviceData;
use haqengine_core::quality::{arena_layout, quality_step, ConstantRing, QualityStep};
use haqengine_core::PackedFormat;
use haqengine_shaders::BindingKind;

/// Constant blocks per frame slot: one per flow dispatch, then the luminance pyramid's SPD block.
pub const RING_BLOCKS: usize = 25;
pub const SPD_BLOCK: usize = 24;
pub const BLOCK_SIZE: u64 = 32;

/// The images Quality owns: the raw frames by parity, the blend output, and the flow's images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QRes {
    Raw(usize),
    Out,
    Of(OfRes),
}

/// Every image Quality owns for a swapchain of `extent`: format and size.
pub fn image_specs(extent: vk::Extent2D) -> Vec<(QRes, vk::Format, u32, u32)> {
    let (w, h) = (extent.width, extent.height);
    let (hw, hh) = half_size(w, h);
    let sizes = of_sizes(hw, hh);
    let mut out = vec![
        (QRes::Raw(0), vk::Format::R32_UINT, w, h),
        (QRes::Raw(1), vk::Format::R32_UINT, w, h),
        (QRes::Out, vk::Format::R32_UINT, w, h),
        (QRes::Of(OfRes::Output), vk::Format::R16G16_SINT, sizes.flow[0].0, sizes.flow[0].1),
        (QRes::Of(OfRes::ScdHistogram), vk::Format::R32_UINT, SCD_HISTOGRAM_WIDTH, 1),
        (QRes::Of(OfRes::ScdPreviousHistogram), vk::Format::R32_SFLOAT, SCD_HISTOGRAM_WIDTH, 1),
        (QRes::Of(OfRes::ScdTemp), vk::Format::R32_UINT, 3, 1),
        (QRes::Of(OfRes::ScdOutput), vk::Format::R32_UINT, 3, 1),
    ];
    for slot in 0..2 {
        for level in 0..OF_LEVELS {
            let (lw, lh) = sizes.luma[level];
            let (fw, fh) = sizes.flow[level];
            out.push((QRes::Of(OfRes::Input(slot, level as u32)), vk::Format::R8_UINT, lw, lh));
            out.push((QRes::Of(OfRes::Flow(slot, level as u32)), vk::Format::R16G16_SINT, fw, fh));
        }
    }
    out
}

/// The flow's schedule for even and odd frames, built once per swapchain.
pub fn schedules(extent: vk::Extent2D, format: PackedFormat) -> [Vec<OfDispatch>; 2] {
    let (hw, hh) = half_size(extent.width, extent.height);
    let input = OfInput::PackedHalf { format: format.code(), full_width: extent.width, full_height: extent.height };
    [of_schedule(hw, hh, 0, input), of_schedule(hw, hh, 1, input)]
}

fn of_res(r: OfRes, parity: usize) -> QRes {
    match r {
        OfRes::Raw => QRes::Raw(parity),
        other => QRes::Of(other),
    }
}

/// The blend's images for a frame of `parity`: the previous and current raw frames, the flow,
/// FSR 3's previous and current level-0 luma, the output and the scene-change words.
pub fn blend_binds(parity: usize) -> Vec<(u32, QRes)> {
    let prev = 1 - parity;
    vec![
        (0, QRes::Raw(prev)),
        (1, QRes::Raw(parity)),
        (2, QRes::Of(OfRes::Output)),
        (3, QRes::Of(OfRes::Input(prev, 0))),
        (4, QRes::Of(OfRes::Input(parity, 0))),
        (5, QRes::Out),
        (6, QRes::Of(OfRes::ScdOutput)),
    ]
}

/// Per parity, the images of each flow dispatch's set followed by the blend's set.
pub fn set_plan(schedules: &[Vec<OfDispatch>; 2]) -> [Vec<Vec<(u32, QRes)>>; 2] {
    let plan = |parity: usize| {
        let mut sets: Vec<Vec<(u32, QRes)>> =
            schedules[parity].iter().map(|d| d.bind.iter().map(|(b, r)| (*b, of_res(*r, parity))).collect()).collect();
        sets.push(blend_binds(parity));
        sets
    };
    [plan(0), plan(1)]
}

/// Writes one slot's constant blocks: each flow dispatch's block with the frame index, then SPD.
pub fn fill_constants(schedule: &[OfDispatch], frame_index: u32, ring: &ConstantRing, slot: usize, out: &mut [u8]) {
    for (i, d) in schedule.iter().enumerate().take(SPD_BLOCK) {
        let mut c = d.constants;
        set_frame_index(&mut c, frame_index);
        let o = ring.offset(slot, i) as usize;
        out[o..o + c.len()].copy_from_slice(&c);
        if let Some(spd) = &d.spd_constants {
            let o = ring.offset(slot, SPD_BLOCK) as usize;
            out[o..o + spd.len()].copy_from_slice(spd);
        }
    }
}

struct QImage {
    res: QRes,
    image: vk::Image,
    view: vk::ImageView,
}

/// The constant ring's mapping; written only by the thread holding the swapchain's state lock.
struct Mapped(*mut u8);
unsafe impl Send for Mapped {}

/// Everything Quality needs for one swapchain, created on its first Quality present.
pub struct QualityResources {
    images: Vec<QImage>,
    memory: vk::DeviceMemory,
    pool: vk::DescriptorPool,
    sets: [Vec<vk::DescriptorSet>; 2],
    schedules: [Vec<OfDispatch>; 2],
    ring: ConstantRing,
    ring_buffer: vk::Buffer,
    ring_memory: vk::DeviceMemory,
    mapped: Mapped,
    extent: vk::Extent2D,
    format: PackedFormat,
    parity: usize,
    has_prev: bool,
    frame_index: u32,
    initialized: bool,
    needs_clear: bool,
    pub bytes: u64,
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)
}

fn memory_type(props: &vk::PhysicalDeviceMemoryProperties, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(flags))
}

unsafe fn barrier(d: &ash::Device, cmd: vk::CommandBuffer, src: vk::PipelineStageFlags, dst: vk::PipelineStageFlags) {
    let b = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    d.cmd_pipeline_barrier(cmd, src, dst, vk::DependencyFlags::empty(), &[b], &[], &[]);
}

#[allow(clippy::too_many_arguments)]
unsafe fn layout_barrier(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src: vk::PipelineStageFlags,
    dst: vk::PipelineStageFlags,
) {
    let b = vk::ImageMemoryBarrier::default()
        .image(image)
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(color_range());
    d.cmd_pipeline_barrier(cmd, src, dst, vk::DependencyFlags::empty(), &[], &[], &[b]);
}

fn push_words<const N: usize>(words: [u32; N]) -> [u8; 24] {
    let mut out = [0u8; 24];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

impl QualityResources {
    fn image(&self, r: QRes) -> &QImage {
        self.images.iter().find(|i| i.res == r).expect("every planned image exists")
    }

    /// # Safety
    /// `device` must be live with Quality's features, and `pipelines` created on it.
    pub unsafe fn create(
        device: &DeviceData,
        pipelines: &QualityPipelines,
        extent: vk::Extent2D,
        format: PackedFormat,
        slots: usize,
    ) -> Result<QualityResources, vk::Result> {
        let d = &device.device;
        let inst = &device.instance.instance;
        let props = inst.get_physical_device_memory_properties(device.physical_device);
        let limits = inst.get_physical_device_properties(device.physical_device).limits;
        let schedules = schedules(extent, format);
        let mut me = QualityResources {
            images: Vec::new(),
            memory: vk::DeviceMemory::null(),
            pool: vk::DescriptorPool::null(),
            sets: [Vec::new(), Vec::new()],
            ring: ConstantRing::new(RING_BLOCKS, BLOCK_SIZE, limits.min_uniform_buffer_offset_alignment, slots),
            ring_buffer: vk::Buffer::null(),
            ring_memory: vk::DeviceMemory::null(),
            mapped: Mapped(std::ptr::null_mut()),
            schedules,
            extent,
            format,
            parity: 0,
            has_prev: false,
            frame_index: 0,
            initialized: false,
            needs_clear: true,
            bytes: 0,
        };
        match me.build(d, &props, pipelines) {
            Ok(()) => Ok(me),
            Err(e) => {
                me.destroy(d);
                Err(e)
            }
        }
    }

    unsafe fn build(&mut self, d: &ash::Device, props: &vk::PhysicalDeviceMemoryProperties, p: &QualityPipelines) -> Result<(), vk::Result> {
        let specs = image_specs(self.extent);
        let mut raw: Vec<(QRes, vk::Image, vk::Format)> = Vec::with_capacity(specs.len());
        let result = (|| {
            for (res, format, w, h) in &specs {
                let image = d.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(*format)
                        .extent(vk::Extent3D { width: *w, height: *h, depth: 1 })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )?;
                raw.push((*res, image, *format));
            }
            let reqs: Vec<vk::MemoryRequirements> = raw.iter().map(|(_, i, _)| d.get_image_memory_requirements(*i)).collect();
            let bits = reqs.iter().fold(u32::MAX, |b, r| b & r.memory_type_bits);
            let kind = memory_type(props, bits, vk::MemoryPropertyFlags::DEVICE_LOCAL).ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
            let (offsets, total) = arena_layout(&reqs.iter().map(|r| (r.size, r.alignment)).collect::<Vec<_>>());
            self.memory = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(total).memory_type_index(kind), None)?;
            self.bytes = total;
            for ((_, image, _), offset) in raw.iter().zip(&offsets) {
                d.bind_image_memory(*image, self.memory, *offset)?;
            }
            while let Some((res, image, format)) = raw.pop() {
                let view = d.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(color_range()),
                    None,
                );
                match view {
                    Ok(view) => self.images.push(QImage { res, image, view }),
                    Err(e) => {
                        d.destroy_image(image, None);
                        return Err(e);
                    }
                }
            }
            self.build_ring(d, props)?;
            self.build_sets(d, p)
        })();
        for (_, image, _) in raw {
            d.destroy_image(image, None);
        }
        result
    }

    unsafe fn build_ring(&mut self, d: &ash::Device, props: &vk::PhysicalDeviceMemoryProperties) -> Result<(), vk::Result> {
        let size = self.ring.size();
        self.ring_buffer = d.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::UNIFORM_BUFFER), None)?;
        let req = d.get_buffer_memory_requirements(self.ring_buffer);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let kind = memory_type(props, req.memory_type_bits, flags).ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        self.ring_memory = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(kind), None)?;
        self.bytes += req.size;
        d.bind_buffer_memory(self.ring_buffer, self.ring_memory, 0)?;
        self.mapped = Mapped(d.map_memory(self.ring_memory, 0, size, vk::MemoryMapFlags::empty())? as *mut u8);
        std::ptr::write_bytes(self.mapped.0, 0, size as usize);
        Ok(())
    }

    fn pass_for<'a>(p: &'a QualityPipelines, schedule: &[OfDispatch], i: usize) -> Result<&'a FsrPass, vk::Result> {
        match schedule.get(i) {
            Some(d) => p.flow.get(d.pass).and_then(|x| x.as_ref()).ok_or(vk::Result::ERROR_INITIALIZATION_FAILED),
            None => Ok(&p.blend),
        }
    }

    unsafe fn build_sets(&mut self, d: &ash::Device, p: &QualityPipelines) -> Result<(), vk::Result> {
        let plan = set_plan(&self.schedules);
        let mut counts = [0u32; 3];
        let mut layouts: [Vec<vk::DescriptorSetLayout>; 2] = [Vec::new(), Vec::new()];
        for parity in 0..2 {
            for i in 0..plan[parity].len() {
                let pass = Self::pass_for(p, &self.schedules[parity], i)?;
                layouts[parity].push(pass.set_layout);
                for (_, k) in pass.pass.bindings {
                    match k {
                        BindingKind::StorageImage => counts[0] += 1,
                        BindingKind::SampledImage => counts[1] += 1,
                        BindingKind::UniformBuffer => counts[2] += 1,
                        _ => return Err(vk::Result::ERROR_INITIALIZATION_FAILED),
                    }
                }
            }
        }
        let sizes: Vec<vk::DescriptorPoolSize> = [
            (vk::DescriptorType::STORAGE_IMAGE, counts[0]),
            (vk::DescriptorType::SAMPLED_IMAGE, counts[1]),
            (vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, counts[2]),
        ]
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .map(|(ty, n)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(n))
        .collect();
        let max_sets = (layouts[0].len() + layouts[1].len()) as u32;
        self.pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(max_sets).pool_sizes(&sizes), None)?;
        for parity in 0..2 {
            self.sets[parity] =
                d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.pool).set_layouts(&layouts[parity]))?;
        }
        for parity in 0..2 {
            for (i, binds) in plan[parity].iter().enumerate() {
                let pass = Self::pass_for(p, &self.schedules[parity], i)?;
                let set = self.sets[parity][i];
                let mut image_infos: Vec<(u32, vk::DescriptorType, [vk::DescriptorImageInfo; 1])> = Vec::new();
                let mut buffer_infos: Vec<(u32, [vk::DescriptorBufferInfo; 1])> = Vec::new();
                for (b, k) in pass.pass.bindings {
                    if *k == BindingKind::UniformBuffer {
                        buffer_infos.push((*b, [vk::DescriptorBufferInfo::default().buffer(self.ring_buffer).range(BLOCK_SIZE)]));
                        continue;
                    }
                    let res = binds.iter().find(|(n, _)| n == b).map(|(_, r)| *r).ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    let view = self.image(res).view;
                    image_infos.push((*b, descriptor_type(*k), [vk::DescriptorImageInfo::default().image_view(view).image_layout(vk::ImageLayout::GENERAL)]));
                }
                let mut writes: Vec<vk::WriteDescriptorSet> = image_infos
                    .iter()
                    .map(|(b, ty, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*b).descriptor_type(*ty).image_info(info))
                    .collect();
                writes.extend(buffer_infos.iter().map(|(b, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(*b)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .buffer_info(info)
                }));
                d.update_descriptor_sets(&writes, &[]);
            }
        }
        Ok(())
    }

    /// Records one Quality present into `cmd` (already begun): copies the game image in, runs the
    /// flow, and writes the spare image (the blended frame, or a copy when there is no previous
    /// frame). Leaves both swapchain images in `PRESENT_SRC_KHR`.
    ///
    /// # Safety
    /// `cmd` must be recording on the device these resources belong to, `slot` must be the frame
    /// slot whose fence was waited for, and `game`/`spare` images of this swapchain.
    pub unsafe fn record(&mut self, d: &ash::Device, cmd: vk::CommandBuffer, p: &QualityPipelines, game: vk::Image, spare: vk::Image, slot: usize) -> QualityStep {
        use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
        if !self.initialized {
            let barriers: Vec<vk::ImageMemoryBarrier> = self
                .images
                .iter()
                .map(|img| {
                    vk::ImageMemoryBarrier::default()
                        .image(img.image)
                        .old_layout(L::UNDEFINED)
                        .new_layout(L::GENERAL)
                        .dst_access_mask(A::MEMORY_READ | A::MEMORY_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .subresource_range(color_range())
                })
                .collect();
            d.cmd_pipeline_barrier(cmd, S::TOP_OF_PIPE, S::TRANSFER | S::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &barriers);
            self.initialized = true;
        }
        if self.needs_clear {
            barrier(d, cmd, S::COMPUTE_SHADER | S::TRANSFER, S::TRANSFER);
            for img in self.images.iter().filter(|i| matches!(i.res, QRes::Of(_))) {
                d.cmd_clear_color_image(cmd, img.image, L::GENERAL, &vk::ClearColorValue { uint32: [0; 4] }, &[color_range()]);
            }
            self.frame_index = 0;
            self.needs_clear = false;
        }
        let step = quality_step(self.has_prev, self.frame_index);
        let cur = self.parity;
        let (w, h) = (self.extent.width, self.extent.height);
        let full = vk::ImageCopy::default()
            .src_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .dst_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .extent(vk::Extent3D { width: w, height: h, depth: 1 });
        layout_barrier(d, cmd, game, L::PRESENT_SRC_KHR, L::TRANSFER_SRC_OPTIMAL, A::empty(), A::TRANSFER_READ, S::TRANSFER, S::TRANSFER);
        barrier(d, cmd, S::COMPUTE_SHADER | S::TRANSFER, S::TRANSFER | S::COMPUTE_SHADER);
        d.cmd_copy_image(cmd, game, L::TRANSFER_SRC_OPTIMAL, self.image(QRes::Raw(cur)).image, L::GENERAL, &[full]);
        barrier(d, cmd, S::TRANSFER, S::COMPUTE_SHADER);

        let ring_bytes = std::slice::from_raw_parts_mut(self.mapped.0, self.ring.size() as usize);
        fill_constants(&self.schedules[cur], self.frame_index, &self.ring, slot, ring_bytes);
        for (i, dd) in self.schedules[cur].iter().enumerate() {
            let Some(pass) = p.flow.get(dd.pass).and_then(|x| x.as_ref()) else { continue };
            let mut offsets = [0u32; 2];
            let mut n = 0;
            for (_, k) in pass.pass.bindings {
                if *k == BindingKind::UniformBuffer {
                    let block = if n == 0 { i } else { SPD_BLOCK };
                    offsets[n] = self.ring.offset(slot, block) as u32;
                    n += 1;
                }
            }
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pass.pipeline);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pass.layout, 0, &[self.sets[cur][i]], &offsets[..n]);
            if pass.pass.push_size > 0 {
                d.cmd_push_constants(cmd, pass.layout, vk::ShaderStageFlags::COMPUTE, 0, &dd.push[..pass.pass.push_size as usize]);
            }
            d.cmd_dispatch(cmd, dd.groups[0], dd.groups[1], dd.groups[2]);
            barrier(d, cmd, S::COMPUTE_SHADER, S::COMPUTE_SHADER);
        }

        layout_barrier(d, cmd, spare, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, A::empty(), A::TRANSFER_WRITE, S::TRANSFER, S::TRANSFER);
        if step == QualityStep::Interpolate {
            let (hw, hh) = half_size(w, h);
            let flow = of_sizes(hw, hh).flow[0];
            let push = push_words([w, h, flow.0, flow.1, (self.frame_index == 0) as u32, self.format.code()]);
            let blend = &p.blend;
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, blend.pipeline);
            let blend_set = *self.sets[cur].last().expect("the blend's set is planned last");
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, blend.layout, 0, &[blend_set], &[]);
            d.cmd_push_constants(cmd, blend.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
            d.cmd_dispatch(cmd, w.div_ceil(8), h.div_ceil(8), 1);
            barrier(d, cmd, S::COMPUTE_SHADER, S::TRANSFER);
            d.cmd_copy_image(cmd, self.image(QRes::Out).image, L::GENERAL, spare, L::TRANSFER_DST_OPTIMAL, &[full]);
        } else {
            d.cmd_copy_image(cmd, game, L::TRANSFER_SRC_OPTIMAL, spare, L::TRANSFER_DST_OPTIMAL, &[full]);
        }
        layout_barrier(d, cmd, spare, L::TRANSFER_DST_OPTIMAL, L::PRESENT_SRC_KHR, A::TRANSFER_WRITE, A::empty(), S::TRANSFER, S::BOTTOM_OF_PIPE);
        layout_barrier(d, cmd, game, L::TRANSFER_SRC_OPTIMAL, L::PRESENT_SRC_KHR, A::TRANSFER_READ, A::empty(), S::TRANSFER, S::BOTTOM_OF_PIPE);
        step
    }

    /// Call after the recorded commands were submitted: this frame becomes the previous one.
    pub fn submitted(&mut self) {
        self.parity ^= 1;
        self.has_prev = true;
        self.frame_index = self.frame_index.saturating_add(1);
    }

    /// Call when a present skipped Quality work: the next frame copies and the flow starts over.
    pub fn forget_previous(&mut self) {
        self.has_prev = false;
        self.needs_clear = true;
    }

    /// # Safety
    /// The GPU must be done with these resources.
    pub unsafe fn destroy(&self, d: &ash::Device) {
        if !self.mapped.0.is_null() {
            d.unmap_memory(self.ring_memory);
        }
        if self.ring_buffer != vk::Buffer::null() {
            d.destroy_buffer(self.ring_buffer, None);
        }
        if self.ring_memory != vk::DeviceMemory::null() {
            d.free_memory(self.ring_memory, None);
        }
        if self.pool != vk::DescriptorPool::null() {
            d.destroy_descriptor_pool(self.pool, None);
        }
        for img in &self.images {
            d.destroy_image_view(img.view, None);
            d.destroy_image(img.image, None);
        }
        if self.memory != vk::DeviceMemory::null() {
            d.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXTENT: vk::Extent2D = vk::Extent2D { width: 1280, height: 800 };

    #[test]
    fn quality_sets_per_parity() {
        let s = schedules(EXTENT, PackedFormat::Bgra8);
        assert_eq!(s[0].len(), 24);
        let plan = set_plan(&s);
        assert_eq!(plan[0].len() + plan[1].len(), 50);
        let specs = image_specs(EXTENT);
        let owned: Vec<QRes> = specs.iter().map(|x| x.0).collect();
        for parity in &plan {
            for set in parity {
                for (_, r) in set {
                    assert!(owned.contains(r), "{r:?} has no image");
                }
            }
        }
        assert_eq!(plan[0][0][0], (0, QRes::Raw(0)));
        assert_eq!(plan[1][0][0], (0, QRes::Raw(1)));
        let blend = &plan[0][24];
        assert_eq!(blend[0], (0, QRes::Raw(1)));
        assert_eq!(blend[1], (1, QRes::Raw(0)));
        assert_eq!(blend[3], (3, QRes::Of(OfRes::Input(1, 0))));
        assert_eq!(blend[4], (4, QRes::Of(OfRes::Input(0, 0))));
        assert_eq!(blend[5], (5, QRes::Out));
        let raw = specs.iter().find(|x| x.0 == QRes::Raw(0)).unwrap();
        assert_eq!((raw.1, raw.2, raw.3), (vk::Format::R32_UINT, 1280, 800));
        let luma = specs.iter().find(|x| x.0 == QRes::Of(OfRes::Input(0, 0))).unwrap();
        assert_eq!((luma.1, luma.2, luma.3), (vk::Format::R8_UINT, 640, 400));
    }

    #[test]
    fn constant_writes_patch_only_frame_index() {
        let s = schedules(EXTENT, PackedFormat::Bgra8);
        let ring = ConstantRing::new(RING_BLOCKS, BLOCK_SIZE, 64, 3);
        let mut a = vec![0u8; ring.size() as usize];
        let mut b = a.clone();
        fill_constants(&s[0], 6, &ring, 1, &mut a);
        fill_constants(&s[0], 8, &ring, 1, &mut b);
        let slot = ring.offset(1, 0) as usize;
        assert!(a[..slot].iter().all(|x| *x == 0), "other slots untouched");
        for i in 0..RING_BLOCKS {
            let o = ring.offset(1, i) as usize;
            for k in 0..BLOCK_SIZE as usize {
                if i != SPD_BLOCK && (16..20).contains(&k) {
                    continue;
                }
                assert_eq!(a[o + k], b[o + k], "block {i} byte {k}");
            }
            if i != SPD_BLOCK {
                assert_eq!(u32::from_le_bytes(a[o + 16..o + 20].try_into().unwrap()), 6);
            }
        }
        let spd = ring.offset(1, SPD_BLOCK) as usize;
        assert_eq!(u32::from_le_bytes(a[spd..spd + 4].try_into().unwrap()), 4);
    }
}
