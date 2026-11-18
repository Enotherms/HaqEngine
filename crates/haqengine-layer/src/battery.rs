//! Battery-mode frame generation inside the layer: per-swapchain GPU resources and the commands
//! that turn two real frames into the frame between them.

use crate::pipelines::{Pass, Pipelines};
use crate::state::DeviceData;
use ash::vk;
use haqengine_core::flow::{BLOCK, LEVELS, RADIUS_COARSE, RADIUS_FINE};
use haqengine_core::PackedFormat;

/// Luma pyramid sizes for a frame: half resolution, then halving, rounding up.
pub fn level_sizes(w: u32, h: u32) -> [(u32, u32); LEVELS] {
    let mut out = [(0, 0); LEVELS];
    let (mut lw, mut lh) = (w, h);
    for size in out.iter_mut() {
        (lw, lh) = (lw.div_ceil(2), lh.div_ceil(2));
        *size = (lw, lh);
    }
    out
}

/// Blocks per pyramid level.
pub fn flow_sizes(levels: &[(u32, u32); LEVELS]) -> [(u32, u32); LEVELS] {
    levels.map(|(w, h)| (w.div_ceil(BLOCK), h.div_ceil(BLOCK)))
}

/// Which of the two frame slots holds the current frame; the other holds the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parity(u8);

impl Parity {
    pub fn new() -> Self {
        Parity(0)
    }
    pub fn cur(&self) -> usize {
        self.0 as usize
    }
    pub fn prev(&self) -> usize {
        1 - self.0 as usize
    }
    pub fn advance(&mut self) {
        self.0 = 1 - self.0;
    }
}

impl Default for Parity {
    fn default() -> Self {
        Self::new()
    }
}

/// What a battery present does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatteryStep {
    /// Build the frame between the previous and the current real frame.
    Interpolate,
    /// No usable previous frame: show a copy and keep this frame as the next one's previous.
    CopyAndPrime,
}

pub fn battery_step(has_prev: bool, extent_changed: bool) -> BatteryStep {
    if has_prev && !extent_changed { BatteryStep::Interpolate } else { BatteryStep::CopyAndPrime }
}

const RAW_OUT: usize = 2;
const PYR_BASE: usize = 3;
const FLOW_BASE: usize = PYR_BASE + 2 * LEVELS;
const MEDIAN_BASE: usize = FLOW_BASE + LEVELS;
const DUMMY: usize = MEDIAN_BASE + LEVELS - 1;
const IMAGE_COUNT: usize = DUMMY + 1;

fn pyr(parity: usize, level: usize) -> usize {
    PYR_BASE + parity * LEVELS + level
}

/// One layer-owned storage image, kept in `GENERAL` layout.
pub struct BatImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub width: u32,
    pub height: u32,
}

/// Images and descriptor sets for one swapchain, created on its first battery present.
pub struct BatteryResources {
    images: Vec<BatImage>,
    pool: vk::DescriptorPool,
    luma_sets: [vk::DescriptorSet; 2],
    reduce_sets: [[vk::DescriptorSet; LEVELS - 1]; 2],
    match_sets: [[vk::DescriptorSet; LEVELS]; 2],
    median_sets: [vk::DescriptorSet; LEVELS - 1],
    warp_sets: [vk::DescriptorSet; 2],
    extent: vk::Extent2D,
    format: PackedFormat,
    parity: Parity,
    has_prev: bool,
    initialized: bool,
}

unsafe fn create_image(
    d: &ash::Device,
    memory: &vk::PhysicalDeviceMemoryProperties,
    format: vk::Format,
    width: u32,
    height: u32,
) -> Result<BatImage, vk::Result> {
    let image = d.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED),
        None,
    )?;
    let req = d.get_image_memory_requirements(image);
    let Some(kind) = (0..memory.memory_type_count).find(|&i| {
        req.memory_type_bits & (1 << i) != 0
            && memory.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }) else {
        d.destroy_image(image, None);
        return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
    };
    let mem = match d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(kind), None) {
        Ok(m) => m,
        Err(e) => {
            d.destroy_image(image, None);
            return Err(e);
        }
    };
    if let Err(e) = d.bind_image_memory(image, mem, 0) {
        d.destroy_image(image, None);
        d.free_memory(mem, None);
        return Err(e);
    }
    let view = match d.create_image_view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(color_range()),
        None,
    ) {
        Ok(v) => v,
        Err(e) => {
            d.destroy_image(image, None);
            d.free_memory(mem, None);
            return Err(e);
        }
    };
    Ok(BatImage { image, memory: mem, view, width, height })
}

unsafe fn destroy_image(d: &ash::Device, img: &BatImage) {
    d.destroy_image_view(img.view, None);
    d.destroy_image(img.image, None);
    d.free_memory(img.memory, None);
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

unsafe fn write_set(d: &ash::Device, set: vk::DescriptorSet, views: &[vk::ImageView]) {
    let infos: Vec<[vk::DescriptorImageInfo; 1]> = views
        .iter()
        .map(|v| [vk::DescriptorImageInfo::default().image_view(*v).image_layout(vk::ImageLayout::GENERAL)])
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(info)
        })
        .collect();
    d.update_descriptor_sets(&writes, &[]);
}

unsafe fn allocate(d: &ash::Device, pool: vk::DescriptorPool, pass: &Pass, count: usize) -> Result<Vec<vk::DescriptorSet>, vk::Result> {
    let layouts = vec![pass.set_layout; count];
    d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))
}

fn groups(w: u32, h: u32) -> (u32, u32) {
    (w.div_ceil(8), h.div_ceil(8))
}

/// Push-constant bytes for 2 words, without allocating.
pub fn push2(a: [u8; 4], b: [u8; 4]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&a);
    out[4..].copy_from_slice(&b);
    out
}

/// Push-constant bytes for 3 words, without allocating.
pub fn push3(a: [u8; 4], b: [u8; 4], c: [u8; 4]) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[..4].copy_from_slice(&a);
    out[4..8].copy_from_slice(&b);
    out[8..].copy_from_slice(&c);
    out
}

/// Push-constant bytes for 6 words, without allocating.
pub fn push6(w: [[u8; 4]; 6]) -> [u8; 24] {
    let mut out = [0u8; 24];
    for (i, word) in w.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(word);
    }
    out
}

unsafe fn memory_barrier(d: &ash::Device, cmd: vk::CommandBuffer, src: vk::PipelineStageFlags, dst: vk::PipelineStageFlags) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    d.cmd_pipeline_barrier(cmd, src, dst, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
}

#[allow(clippy::too_many_arguments)]
unsafe fn image_barrier(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .image(image)
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(color_range());
    d.cmd_pipeline_barrier(cmd, src_stage, dst_stage, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
}

unsafe fn dispatch(d: &ash::Device, cmd: vk::CommandBuffer, pass: &Pass, set: vk::DescriptorSet, push: &[u8], g: (u32, u32)) {
    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pass.pipeline);
    d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pass.layout, 0, &[set], &[]);
    d.cmd_push_constants(cmd, pass.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
    d.cmd_dispatch(cmd, g.0, g.1, 1);
}

impl BatteryResources {
    /// # Safety
    /// `device` must be live and `pipelines` created on it.
    pub unsafe fn create(
        device: &DeviceData,
        pipelines: &Pipelines,
        extent: vk::Extent2D,
        format: PackedFormat,
    ) -> Result<BatteryResources, vk::Result> {
        let d = &device.device;
        let memory = device.instance.instance.get_physical_device_memory_properties(device.physical_device);
        let levels = level_sizes(extent.width, extent.height);
        let flows = flow_sizes(&levels);
        let mut specs: Vec<(vk::Format, u32, u32)> = Vec::with_capacity(IMAGE_COUNT);
        for _ in 0..3 {
            specs.push((vk::Format::R32_UINT, extent.width, extent.height));
        }
        for _ in 0..2 {
            for &(w, h) in &levels {
                specs.push((vk::Format::R32_SFLOAT, w, h));
            }
        }
        for &(w, h) in &flows {
            specs.push((vk::Format::R32G32_SFLOAT, w, h));
        }
        for &(w, h) in &flows[1..] {
            specs.push((vk::Format::R32G32_SFLOAT, w, h));
        }
        specs.push((vk::Format::R32G32_SFLOAT, 1, 1));
        let mut images: Vec<BatImage> = Vec::with_capacity(IMAGE_COUNT);
        for (f, w, h) in specs {
            match create_image(d, &memory, f, w, h) {
                Ok(img) => images.push(img),
                Err(e) => {
                    for img in &images {
                        destroy_image(d, img);
                    }
                    return Err(e);
                }
            }
        }
        let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(64)];
        let pool = match d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(24).pool_sizes(&sizes), None) {
            Ok(p) => p,
            Err(e) => {
                for img in &images {
                    destroy_image(d, img);
                }
                return Err(e);
            }
        };
        let sets = (|| -> Result<_, vk::Result> {
            let luma = allocate(d, pool, &pipelines.luma, 2)?;
            let reduce = allocate(d, pool, &pipelines.reduce, 2 * (LEVELS - 1))?;
            let matching = allocate(d, pool, &pipelines.block_match, 2 * LEVELS)?;
            let median = allocate(d, pool, &pipelines.median, LEVELS - 1)?;
            let warp = allocate(d, pool, &pipelines.warp, 2)?;
            Ok((luma, reduce, matching, median, warp))
        })();
        let (luma, reduce, matching, median, warp) = match sets {
            Ok(s) => s,
            Err(e) => {
                d.destroy_descriptor_pool(pool, None);
                for img in &images {
                    destroy_image(d, img);
                }
                return Err(e);
            }
        };
        let mut res = BatteryResources {
            images,
            pool,
            luma_sets: [luma[0], luma[1]],
            reduce_sets: [[vk::DescriptorSet::null(); LEVELS - 1]; 2],
            match_sets: [[vk::DescriptorSet::null(); LEVELS]; 2],
            median_sets: [vk::DescriptorSet::null(); LEVELS - 1],
            warp_sets: [warp[0], warp[1]],
            extent,
            format,
            parity: Parity::new(),
            has_prev: false,
            initialized: false,
        };
        for p in 0..2 {
            for l in 1..LEVELS {
                res.reduce_sets[p][l - 1] = reduce[p * (LEVELS - 1) + l - 1];
            }
            for l in 0..LEVELS {
                res.match_sets[p][l] = matching[p * LEVELS + l];
            }
        }
        res.median_sets.copy_from_slice(&median[..LEVELS - 1]);
        res.write_sets(d);
        Ok(res)
    }

    unsafe fn write_sets(&self, d: &ash::Device) {
        let v = |i: usize| self.images[i].view;
        for p in 0..2 {
            let q = 1 - p;
            write_set(d, self.luma_sets[p], &[v(p), v(pyr(p, 0))]);
            for l in 1..LEVELS {
                write_set(d, self.reduce_sets[p][l - 1], &[v(pyr(p, l - 1)), v(pyr(p, l))]);
            }
            for (l, &set) in self.match_sets[p].iter().enumerate() {
                let coarse = if l + 1 < LEVELS { v(MEDIAN_BASE + l) } else { v(DUMMY) };
                write_set(d, set, &[v(pyr(q, l)), v(pyr(p, l)), coarse, v(FLOW_BASE + l)]);
            }
            write_set(d, self.warp_sets[p], &[v(q), v(p), v(FLOW_BASE), v(RAW_OUT)]);
        }
        for l in 1..LEVELS {
            write_set(d, self.median_sets[l - 1], &[v(FLOW_BASE + l), v(MEDIAN_BASE + l - 1)]);
        }
    }

    /// # Safety
    /// The GPU must be done with these resources.
    pub unsafe fn destroy(&self, d: &ash::Device) {
        d.destroy_descriptor_pool(self.pool, None);
        for img in &self.images {
            destroy_image(d, img);
        }
    }

    /// Records the battery pass into `cmd` (already begun): copies the game image in, builds its
    /// pyramid, and writes the spare image (interpolated, or a copy when there is no previous
    /// frame). Leaves both swapchain images in `PRESENT_SRC_KHR`.
    ///
    /// # Safety
    /// `cmd` must be recording; `game` and `spare` must be images of this swapchain.
    pub unsafe fn record(&mut self, d: &ash::Device, cmd: vk::CommandBuffer, p: &Pipelines, game: vk::Image, spare: vk::Image) -> BatteryStep {
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
        let step = battery_step(self.has_prev, false);
        let cur = self.parity.cur();
        let (w, h) = (self.extent.width, self.extent.height);
        let full = vk::ImageCopy::default()
            .src_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .dst_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .extent(vk::Extent3D { width: w, height: h, depth: 1 });

        image_barrier(d, cmd, game, L::PRESENT_SRC_KHR, L::TRANSFER_SRC_OPTIMAL, A::empty(), A::TRANSFER_READ, S::TRANSFER, S::TRANSFER);
        memory_barrier(d, cmd, S::COMPUTE_SHADER | S::TRANSFER, S::TRANSFER | S::COMPUTE_SHADER);
        d.cmd_copy_image(cmd, game, L::TRANSFER_SRC_OPTIMAL, self.images[cur].image, L::GENERAL, &[full]);
        memory_barrier(d, cmd, S::TRANSFER, S::COMPUTE_SHADER);

        let levels = level_sizes(w, h);
        let flows = flow_sizes(&levels);
        let fmt = self.format.code();
        dispatch(d, cmd, &p.luma, self.luma_sets[cur], &push3(w.to_le_bytes(), h.to_le_bytes(), fmt.to_le_bytes()), groups(levels[0].0, levels[0].1));
        for l in 1..LEVELS {
            memory_barrier(d, cmd, S::COMPUTE_SHADER, S::COMPUTE_SHADER);
            let (iw, ih) = levels[l - 1];
            dispatch(d, cmd, &p.reduce, self.reduce_sets[cur][l - 1], &push2(iw.to_le_bytes(), ih.to_le_bytes()), groups(levels[l].0, levels[l].1));
        }

        image_barrier(d, cmd, spare, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, A::empty(), A::TRANSFER_WRITE, S::TRANSFER, S::TRANSFER);
        if step == BatteryStep::Interpolate {
            for l in (0..LEVELS).rev() {
                memory_barrier(d, cmd, S::COMPUTE_SHADER, S::COMPUTE_SHADER);
                let (lw, lh) = levels[l];
                let (bw, bh) = flows[l];
                let has_coarse = l + 1 < LEVELS;
                let (cw, ch) = if has_coarse { flows[l + 1] } else { (1, 1) };
                let radius = if has_coarse { RADIUS_FINE } else { RADIUS_COARSE };
                let push = push6([
                    lw.to_le_bytes(),
                    lh.to_le_bytes(),
                    cw.to_le_bytes(),
                    ch.to_le_bytes(),
                    radius.to_le_bytes(),
                    (has_coarse as u32).to_le_bytes(),
                ]);
                dispatch(d, cmd, &p.block_match, self.match_sets[cur][l], &push, (bw, bh));
                if l > 0 {
                    memory_barrier(d, cmd, S::COMPUTE_SHADER, S::COMPUTE_SHADER);
                    dispatch(d, cmd, &p.median, self.median_sets[l - 1], &push2(bw.to_le_bytes(), bh.to_le_bytes()), groups(bw, bh));
                }
            }
            memory_barrier(d, cmd, S::COMPUTE_SHADER, S::COMPUTE_SHADER);
            let push = push6([
                w.to_le_bytes(),
                h.to_le_bytes(),
                flows[0].0.to_le_bytes(),
                flows[0].1.to_le_bytes(),
                0.5f32.to_le_bytes(),
                fmt.to_le_bytes(),
            ]);
            dispatch(d, cmd, &p.warp, self.warp_sets[cur], &push, groups(w, h));
            memory_barrier(d, cmd, S::COMPUTE_SHADER, S::TRANSFER);
            d.cmd_copy_image(cmd, self.images[RAW_OUT].image, L::GENERAL, spare, L::TRANSFER_DST_OPTIMAL, &[full]);
        } else {
            d.cmd_copy_image(cmd, game, L::TRANSFER_SRC_OPTIMAL, spare, L::TRANSFER_DST_OPTIMAL, &[full]);
        }
        image_barrier(d, cmd, spare, L::TRANSFER_DST_OPTIMAL, L::PRESENT_SRC_KHR, A::TRANSFER_WRITE, A::empty(), S::TRANSFER, S::BOTTOM_OF_PIPE);
        image_barrier(d, cmd, game, L::TRANSFER_SRC_OPTIMAL, L::PRESENT_SRC_KHR, A::TRANSFER_READ, A::empty(), S::TRANSFER, S::BOTTOM_OF_PIPE);
        step
    }

    /// Debug snapshot: after this frame's submit (same queue), copies the previous and current real
    /// frames, the generated frame and the level-0 flow to files in `dir`. Waits for the GPU.
    ///
    /// # Safety
    /// Call after this frame's commands were submitted to `queue` and before `submitted()`.
    pub unsafe fn dump(&self, device: &DeviceData, pool: vk::CommandPool, queue: vk::Queue, dir: &std::path::Path) -> Result<(), vk::Result> {
        let d = &device.device;
        let memory = device.instance.instance.get_physical_device_memory_properties(device.physical_device);
        let flow0 = FLOW_BASE;
        let picks = [(self.parity.prev(), "prev", 4u64), (self.parity.cur(), "cur", 4), (RAW_OUT, "out", 4), (flow0, "flow", 8)];
        let mut buffers: Vec<(vk::Buffer, vk::DeviceMemory, u64, &str)> = Vec::new();
        let result = (|| -> Result<(), vk::Result> {
            for &(index, name, bpp) in &picks {
                let img = &self.images[index];
                let size = img.width as u64 * img.height as u64 * bpp;
                let buffer = d.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::TRANSFER_DST), None)?;
                let req = d.get_buffer_memory_requirements(buffer);
                let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                let Some(kind) = (0..memory.memory_type_count)
                    .find(|&i| req.memory_type_bits & (1 << i) != 0 && memory.memory_types[i as usize].property_flags.contains(flags))
                else {
                    d.destroy_buffer(buffer, None);
                    return Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
                };
                let mem = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(kind), None)?;
                d.bind_buffer_memory(buffer, mem, 0)?;
                buffers.push((buffer, mem, size, name));
            }
            let cmd = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
            )?[0];
            if let Some(set) = device.set_loader_data {
                let _ = set(d.handle(), vk::Handle::as_raw(cmd) as *mut std::ffi::c_void);
            }
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            memory_barrier(d, cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER);
            for (i, &(index, _, _)) in picks.iter().enumerate() {
                let img = &self.images[index];
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                    .image_extent(vk::Extent3D { width: img.width, height: img.height, depth: 1 });
                d.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::GENERAL, buffers[i].0, &[region]);
            }
            let to_host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[to_host], &[], &[]);
            d.end_command_buffer(cmd)?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            let submitted = d.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence);
            let waited = submitted.and_then(|_| d.wait_for_fences(&[fence], true, 2_000_000_000));
            d.destroy_fence(fence, None);
            d.free_command_buffers(pool, &cmds);
            waited?;
            let _ = std::fs::create_dir_all(dir);
            for &(_, mem, size, name) in &buffers {
                let ptr = d.map_memory(mem, 0, size, vk::MemoryMapFlags::empty())?;
                let bytes = std::slice::from_raw_parts(ptr as *const u8, size as usize);
                let _ = std::fs::write(dir.join(format!("{name}.raw")), bytes);
                d.unmap_memory(mem);
            }
            let f0 = &self.images[flow0];
            let meta = format!(
                "{} {} {} {} {}\n",
                self.extent.width,
                self.extent.height,
                self.format.code(),
                f0.width,
                f0.height
            );
            let _ = std::fs::write(dir.join("meta.txt"), meta);
            Ok(())
        })();
        for (buffer, mem, _, _) in buffers {
            d.destroy_buffer(buffer, None);
            d.free_memory(mem, None);
        }
        result
    }

    /// Call after the recorded commands were submitted: the current frame becomes the previous one.
    pub fn submitted(&mut self) {
        self.parity.advance();
        self.has_prev = true;
    }

    /// Call when a present skipped battery work: the next frame starts over with a copy.
    pub fn forget_previous(&mut self) {
        self.has_prev = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_sizes_1280x800() {
        assert_eq!(level_sizes(1280, 800), [(640, 400), (320, 200), (160, 100), (80, 50)]);
        assert_eq!(level_sizes(1281, 801), [(641, 401), (321, 201), (161, 101), (81, 51)]);
    }

    #[test]
    fn flow_sizes_match_blocks() {
        assert_eq!(flow_sizes(&level_sizes(1280, 800)), [(80, 50), (40, 25), (20, 13), (10, 7)]);
        assert_eq!(flow_sizes(&level_sizes(1281, 801)), [(81, 51), (41, 26), (21, 13), (11, 7)]);
    }

    #[test]
    fn parity_alternates_and_skip_keeps_prev() {
        let mut p = Parity::new();
        assert_ne!(p.cur(), p.prev());
        let first = p.cur();
        p.advance();
        assert_eq!(p.prev(), first);
        assert_ne!(p.cur(), first);
        p.advance();
        assert_eq!(p.cur(), first);
    }

    #[test]
    fn push_bytes_are_little_endian_in_order() {
        assert_eq!(push2(1u32.to_le_bytes(), 2u32.to_le_bytes()), [1, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(push3(1u32.to_le_bytes(), 2u32.to_le_bytes(), 0.5f32.to_le_bytes())[8..], 0.5f32.to_le_bytes());
        let six = push6([1u32, 2, 3, 4, 5, 6].map(u32::to_le_bytes));
        assert_eq!(six[20..], 6u32.to_le_bytes());
        assert_eq!(six[..4], 1u32.to_le_bytes());
    }

    #[test]
    fn battery_state_resets_on_new_extent() {
        assert_eq!(battery_step(true, false), BatteryStep::Interpolate);
        assert_eq!(battery_step(false, false), BatteryStep::CopyAndPrime);
        assert_eq!(battery_step(true, true), BatteryStep::CopyAndPrime);
    }
}
