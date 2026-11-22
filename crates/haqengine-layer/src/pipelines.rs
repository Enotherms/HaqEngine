//! Compute pipelines for battery-mode frame generation, created once per device.

use ash::vk;
use haqengine_core::PackedFormat;
use haqengine_shaders::{
    BindingKind, Fsr3Pass, Shader, BATTERY_BLEND_PACKED, BLOCK_MATCH, CUT_FIXUP, FI_BLEND_PACKED, FLOW_MEDIAN, LUMA_DOWNSAMPLE_PACKED,
    LUMA_REDUCE, OF_PASSES,
};

/// The packed layout of a swapchain format, or `None` when battery mode cannot read it.
pub fn packed_format(f: vk::Format) -> Option<PackedFormat> {
    match f {
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => Some(PackedFormat::Bgra8),
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => Some(PackedFormat::Rgba8),
        vk::Format::A2B10G10R10_UNORM_PACK32 => Some(PackedFormat::A2b10g10r10),
        _ => None,
    }
}

/// One compute pass: its descriptor set layout (all bindings are storage images), pipeline layout
/// and pipeline.
pub struct Pass {
    pub set_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub bindings: u32,
}

/// Every pass battery mode records.
pub struct Pipelines {
    pub luma: Pass,
    pub reduce: Pass,
    pub block_match: Pass,
    pub median: Pass,
    pub warp: Pass,
    pub fixup: Pass,
}

fn spirv_words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

unsafe fn create_pass(d: &ash::Device, shader: &Shader, bindings: u32, push_size: u32) -> Result<Pass, vk::Result> {
    let words = spirv_words(shader.spirv);
    let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
    let binding_list: Vec<_> = (0..bindings)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let set_layout = match d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding_list), None) {
        Ok(l) => l,
        Err(e) => {
            d.destroy_shader_module(module, None);
            return Err(e);
        }
    };
    let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push_size)];
    let set_layouts = [set_layout];
    let layout = match d
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&ranges), None)
    {
        Ok(l) => l,
        Err(e) => {
            d.destroy_descriptor_set_layout(set_layout, None);
            d.destroy_shader_module(module, None);
            return Err(e);
        }
    };
    let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
    let created = d.create_compute_pipelines(
        vk::PipelineCache::null(),
        &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
        None,
    );
    d.destroy_shader_module(module, None);
    match created {
        Ok(p) => Ok(Pass { set_layout, layout, pipeline: p[0], bindings }),
        Err((_, e)) => {
            d.destroy_pipeline_layout(layout, None);
            d.destroy_descriptor_set_layout(set_layout, None);
            Err(e)
        }
    }
}

unsafe fn destroy_pass(d: &ash::Device, p: &Pass) {
    d.destroy_pipeline(p.pipeline, None);
    d.destroy_pipeline_layout(p.layout, None);
    d.destroy_descriptor_set_layout(p.set_layout, None);
}

impl Pipelines {
    /// # Safety
    /// `d` must be a live device.
    pub unsafe fn create(d: &ash::Device) -> Result<Pipelines, vk::Result> {
        let mut made: Vec<Pass> = Vec::with_capacity(6);
        let specs: [(&Shader, u32, u32); 6] = [
            (&LUMA_DOWNSAMPLE_PACKED, 2, 12),
            (&LUMA_REDUCE, 2, 8),
            (&BLOCK_MATCH, 4, 24),
            (&FLOW_MEDIAN, 2, 8),
            (&BATTERY_BLEND_PACKED, 8, 28),
            (&CUT_FIXUP, 5, 24),
        ];
        for (shader, bindings, push) in specs {
            match create_pass(d, shader, bindings, push) {
                Ok(p) => made.push(p),
                Err(e) => {
                    for p in &made {
                        destroy_pass(d, p);
                    }
                    return Err(e);
                }
            }
        }
        let mut it = made.into_iter();
        let mut next = || it.next().expect("six passes were created");
        Ok(Pipelines { luma: next(), reduce: next(), block_match: next(), median: next(), warp: next(), fixup: next() })
    }

    /// # Safety
    /// `d` must be the device the pipelines were created on, idle for these pipelines.
    pub unsafe fn destroy(&self, d: &ash::Device) {
        for p in [&self.luma, &self.reduce, &self.block_match, &self.median, &self.warp, &self.fixup] {
            destroy_pass(d, p);
        }
    }
}

/// One Quality pass, built from its binding table: uniform buffers are dynamic, so one descriptor
/// set serves every frame slot.
pub struct FsrPass {
    pub set_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub pass: &'static Fsr3Pass,
}

/// The descriptor type HaqEngine binds for a binding kind.
pub fn descriptor_type(k: BindingKind) -> vk::DescriptorType {
    match k {
        BindingKind::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        BindingKind::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
        BindingKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        BindingKind::Sampler => vk::DescriptorType::SAMPLER,
        BindingKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
    }
}

unsafe fn create_fsr_pass(d: &ash::Device, pass: &'static Fsr3Pass) -> Result<FsrPass, vk::Result> {
    let words = spirv_words(pass.spirv);
    let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
    let bindings: Vec<_> = pass
        .bindings
        .iter()
        .map(|(b, k)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(*b)
                .descriptor_type(descriptor_type(*k))
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let set_layout = match d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) {
        Ok(l) => l,
        Err(e) => {
            d.destroy_shader_module(module, None);
            return Err(e);
        }
    };
    let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(pass.push_size)];
    let set_layouts = [set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    let info = if pass.push_size > 0 { info.push_constant_ranges(&ranges) } else { info };
    let layout = match d.create_pipeline_layout(&info, None) {
        Ok(l) => l,
        Err(e) => {
            d.destroy_descriptor_set_layout(set_layout, None);
            d.destroy_shader_module(module, None);
            return Err(e);
        }
    };
    let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
    let created = d.create_compute_pipelines(
        vk::PipelineCache::null(),
        &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
        None,
    );
    d.destroy_shader_module(module, None);
    match created {
        Ok(p) => Ok(FsrPass { set_layout, layout, pipeline: p[0], pass }),
        Err((_, e)) => {
            d.destroy_pipeline_layout(layout, None);
            d.destroy_descriptor_set_layout(set_layout, None);
            Err(e)
        }
    }
}

unsafe fn destroy_fsr_pass(d: &ash::Device, p: &FsrPass) {
    d.destroy_pipeline(p.pipeline, None);
    d.destroy_pipeline_layout(p.layout, None);
    d.destroy_descriptor_set_layout(p.set_layout, None);
}

/// Every pass Quality mode records: the flow passes it schedules (by `OF_PASSES` index; AMD's
/// prepare-luma pass, index 0, is never built) and the packed blend.
pub struct QualityPipelines {
    pub flow: Vec<Option<FsrPass>>,
    pub blend: FsrPass,
}

impl QualityPipelines {
    /// # Safety
    /// `d` must be a live device with Quality's features enabled.
    pub unsafe fn create(d: &ash::Device) -> Result<QualityPipelines, vk::Result> {
        let mut flow: Vec<Option<FsrPass>> = Vec::with_capacity(OF_PASSES.len());
        let free = |flow: &[Option<FsrPass>]| {
            for p in flow.iter().flatten() {
                destroy_fsr_pass(d, p);
            }
        };
        for (i, pass) in OF_PASSES.iter().enumerate() {
            if i == haqengine_core::fsr3_flow::PASS_PREPARE_LUMA {
                flow.push(None);
                continue;
            }
            match create_fsr_pass(d, pass) {
                Ok(p) => flow.push(Some(p)),
                Err(e) => {
                    free(&flow);
                    return Err(e);
                }
            }
        }
        match create_fsr_pass(d, &FI_BLEND_PACKED) {
            Ok(blend) => Ok(QualityPipelines { flow, blend }),
            Err(e) => {
                free(&flow);
                Err(e)
            }
        }
    }

    /// # Safety
    /// `d` must be the device the pipelines were created on, idle for them.
    pub unsafe fn destroy(&self, d: &ash::Device) {
        for p in self.flow.iter().flatten() {
            destroy_fsr_pass(d, p);
        }
        destroy_fsr_pass(d, &self.blend);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_format_mapping() {
        assert_eq!(packed_format(vk::Format::B8G8R8A8_UNORM), Some(PackedFormat::Bgra8));
        assert_eq!(packed_format(vk::Format::B8G8R8A8_SRGB), Some(PackedFormat::Bgra8));
        assert_eq!(packed_format(vk::Format::R8G8B8A8_UNORM), Some(PackedFormat::Rgba8));
        assert_eq!(packed_format(vk::Format::R8G8B8A8_SRGB), Some(PackedFormat::Rgba8));
        assert_eq!(packed_format(vk::Format::A2B10G10R10_UNORM_PACK32), Some(PackedFormat::A2b10g10r10));
        for f in [vk::Format::R16G16B16A16_SFLOAT, vk::Format::R5G6B5_UNORM_PACK16, vk::Format::B10G11R11_UFLOAT_PACK32] {
            assert_eq!(packed_format(f), None, "{f:?}");
        }
    }
}
