//! A headless Vulkan device for running HaqEngine's shaders on test images.

use ash::vk;
use haqengine_core::{Image, Rgba8};
use haqengine_shaders::Shader;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicU32, Ordering};

const FENCE_TIMEOUT_NS: u64 = 5_000_000_000;
const VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";

static VALIDATION_ERRORS: AtomicU32 = AtomicU32::new(0);

unsafe extern "system" fn on_debug_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _: *mut c_void,
) -> vk::Bool32 {
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        let msg = if data.is_null() || (*data).p_message.is_null() {
            "(no message)".into()
        } else {
            CStr::from_ptr((*data).p_message).to_string_lossy()
        };
        eprintln!("validation error: {msg}");
    }
    vk::FALSE
}

/// A device-local storage image kept in `GENERAL` layout between commands.
pub struct StorageImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub width: u32,
    pub height: u32,
}

pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    query_pool: vk::QueryPool,
    timestamps: bool,
    timestamp_period_ns: f32,
    memory: vk::PhysicalDeviceMemoryProperties,
}

fn err(what: &str, r: vk::Result) -> String {
    format!("{what}: {r:?}")
}

impl Gpu {
    pub fn new(validation: bool) -> Result<Gpu, String> {
        std::env::remove_var("HAQENGINE");
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("cannot load Vulkan: {e}"))?;
            let mut layers = Vec::new();
            let mut extensions = Vec::new();
            if validation {
                let available = entry.enumerate_instance_layer_properties().map_err(|e| err("listing layers", e))?;
                if !available.iter().any(|l| l.layer_name_as_c_str() == Ok(VALIDATION)) {
                    return Err("validation requested but VK_LAYER_KHRONOS_validation is not installed".into());
                }
                layers.push(VALIDATION.as_ptr());
                extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            }
            let app = vk::ApplicationInfo::default()
                .application_name(c"haqengine-gpu-tests")
                .api_version(vk::API_VERSION_1_1);
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default()
                        .application_info(&app)
                        .enabled_layer_names(&layers)
                        .enabled_extension_names(&extensions),
                    None,
                )
                .map_err(|e| err("creating the instance", e))?;
            let debug = if validation {
                let utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let messenger = utils
                    .create_debug_utils_messenger(
                        &vk::DebugUtilsMessengerCreateInfoEXT::default()
                            .message_severity(
                                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                            )
                            .message_type(
                                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                            )
                            .pfn_user_callback(Some(on_debug_message)),
                        None,
                    )
                    .map_err(|e| err("creating the debug messenger", e))?;
                Some((utils, messenger))
            } else {
                None
            };
            let physicals = instance.enumerate_physical_devices().map_err(|e| err("listing GPUs", e))?;
            let (physical, family, valid_bits) = physicals
                .iter()
                .find_map(|&p| {
                    instance
                        .get_physical_device_queue_family_properties(p)
                        .iter()
                        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                        .map(|i| {
                            let f = instance.get_physical_device_queue_family_properties(p)[i];
                            (p, i as u32, f.timestamp_valid_bits)
                        })
                })
                .ok_or("no GPU with a compute queue")?;
            let props = instance.get_physical_device_properties(physical);
            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
            let device = instance
                .create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info), None)
                .map_err(|e| err("creating the device", e))?;
            let queue = device.get_device_queue(family, 0);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| err("creating the command pool", e))?;
            let query_pool = device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2),
                    None,
                )
                .map_err(|e| err("creating the query pool", e))?;
            let memory = instance.get_physical_device_memory_properties(physical);
            Ok(Gpu {
                _entry: entry,
                instance,
                debug,
                device,
                queue,
                pool,
                query_pool,
                timestamps: valid_bits > 0,
                timestamp_period_ns: props.limits.timestamp_period,
                memory,
            })
        }
    }

    pub fn validation_errors(&self) -> u32 {
        VALIDATION_ERRORS.load(Ordering::Relaxed)
    }

    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32, String> {
        (0..self.memory.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.memory.memory_types[i as usize].property_flags.contains(flags))
            .ok_or_else(|| format!("no memory type with {flags:?}"))
    }

    unsafe fn host_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        let d = &self.device;
        let buffer = d
            .create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None)
            .map_err(|e| err("creating a buffer", e))?;
        let req = d.get_buffer_memory_requirements(buffer);
        let kind = self.memory_type(
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let memory = d
            .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(kind), None)
            .map_err(|e| err("allocating buffer memory", e))?;
        d.bind_buffer_memory(buffer, memory, 0).map_err(|e| err("binding buffer memory", e))?;
        Ok((buffer, memory))
    }

    unsafe fn submit(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<(), String> {
        let d = &self.device;
        let cmd = d
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| err("allocating a command buffer", e))?[0];
        let result = (|| {
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(|e| err("beginning commands", e))?;
            let all = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[all],
                &[],
                &[],
            );
            record(cmd);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
            d.end_command_buffer(cmd).map_err(|e| err("ending commands", e))?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| err("creating a fence", e))?;
            let cmds = [cmd];
            let r = d
                .queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence)
                .map_err(|e| err("submitting", e))
                .and_then(|_| match d.wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS) {
                    Ok(()) => Ok(()),
                    Err(vk::Result::TIMEOUT) => Err("GPU timeout".to_string()),
                    Err(e) => Err(err("waiting for the GPU", e)),
                });
            d.destroy_fence(fence, None);
            r
        })();
        d.free_command_buffers(self.pool, &[cmd]);
        result
    }

    unsafe fn transition(&self, cmd: vk::CommandBuffer, image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
        let barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .old_layout(old)
            .new_layout(new)
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(color_range());
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    pub fn create_storage_image(&self, format: vk::Format, width: u32, height: u32) -> Result<StorageImage, String> {
        unsafe {
            let d = &self.device;
            let image = d
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D { width, height, depth: 1 })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(|e| err("creating an image", e))?;
            let req = d.get_image_memory_requirements(image);
            let kind = self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let memory = d
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(kind), None)
                .map_err(|e| err("allocating image memory", e))?;
            d.bind_image_memory(image, memory, 0).map_err(|e| err("binding image memory", e))?;
            let view = d
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(color_range()),
                    None,
                )
                .map_err(|e| err("creating an image view", e))?;
            self.submit(|cmd| self.transition(cmd, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL))?;
            Ok(StorageImage { image, memory, view, width, height })
        }
    }

    pub fn destroy_image(&self, img: StorageImage) {
        unsafe {
            self.device.destroy_image_view(img.view, None);
            self.device.destroy_image(img.image, None);
            self.device.free_memory(img.memory, None);
        }
    }

    pub fn clear(&self, img: &StorageImage, value: [f32; 4]) -> Result<(), String> {
        unsafe {
            self.submit(|cmd| {
                self.device.cmd_clear_color_image(
                    cmd,
                    img.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue { float32: value },
                    &[color_range()],
                )
            })
        }
    }

    unsafe fn copy_in(&self, img: &StorageImage, bytes: &[u8]) -> Result<(), String> {
        let d = &self.device;
        let (buffer, memory) = self.host_buffer(bytes.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
        let result = (|| {
            let ptr = d
                .map_memory(memory, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| err("mapping memory", e))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
            d.unmap_memory(memory);
            self.submit(|cmd| {
                self.transition(cmd, img.image, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                d.cmd_copy_buffer_to_image(cmd, buffer, img.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy_region(img)]);
                self.transition(cmd, img.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL);
            })
        })();
        d.destroy_buffer(buffer, None);
        d.free_memory(memory, None);
        result
    }

    unsafe fn copy_out(&self, img: &StorageImage, len: usize) -> Result<Vec<u8>, String> {
        let d = &self.device;
        let (buffer, memory) = self.host_buffer(len as u64, vk::BufferUsageFlags::TRANSFER_DST)?;
        let result = (|| {
            self.submit(|cmd| {
                self.transition(cmd, img.image, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                d.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer, &[copy_region(img)]);
                self.transition(cmd, img.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::GENERAL);
            })?;
            let ptr = d
                .map_memory(memory, 0, len as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| err("mapping memory", e))?;
            let mut out = vec![0u8; len];
            std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), len);
            d.unmap_memory(memory);
            Ok(out)
        })();
        d.destroy_buffer(buffer, None);
        d.free_memory(memory, None);
        result
    }

    pub fn upload_rgba8(&self, src: &Image<Rgba8>) -> Result<StorageImage, String> {
        let img = self.create_storage_image(vk::Format::R8G8B8A8_UNORM, src.width, src.height)?;
        let bytes: Vec<u8> = src.data.iter().flatten().copied().collect();
        unsafe { self.copy_in(&img, &bytes)? };
        Ok(img)
    }

    pub fn read_r32f(&self, img: &StorageImage) -> Result<Image<f32>, String> {
        let n = (img.width * img.height) as usize;
        let bytes = unsafe { self.copy_out(img, n * 4)? };
        let data = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(Image { width: img.width, height: img.height, data })
    }

    pub fn read_rgba8(&self, img: &StorageImage) -> Result<Image<Rgba8>, String> {
        let n = (img.width * img.height) as usize;
        let bytes = unsafe { self.copy_out(img, n * 4)? };
        let data = bytes.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
        Ok(Image { width: img.width, height: img.height, data })
    }

    /// Runs one dispatch of `shader` over `images` (bound in order) and returns its GPU time in µs.
    pub fn run(&self, shader: &Shader, images: &[&StorageImage], push: &[u8], groups: [u32; 2]) -> Result<f64, String> {
        unsafe {
            let d = &self.device;
            let code = ash::util::read_spv(&mut std::io::Cursor::new(shader.spirv)).map_err(|e| e.to_string())?;
            let module = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .map_err(|e| err("creating the shader module", e))?;
            let bindings: Vec<_> = (0..images.len() as u32)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = d
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
                .map_err(|e| err("creating the set layout", e))?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push.len() as u32)];
            let set_layouts = [set_layout];
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&ranges),
                    None,
                )
                .map_err(|e| err("creating the pipeline layout", e))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline = d
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
                    None,
                )
                .map_err(|(_, e)| err("creating the pipeline", e))?[0];
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(images.len() as u32)];
            let desc_pool = d
                .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)
                .map_err(|e| err("creating the descriptor pool", e))?;
            let set = d
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&set_layouts))
                .map_err(|e| err("allocating the descriptor set", e))?[0];
            let infos: Vec<[vk::DescriptorImageInfo; 1]> = images
                .iter()
                .map(|img| [vk::DescriptorImageInfo::default().image_view(img.view).image_layout(vk::ImageLayout::GENERAL)])
                .collect();
            let writes: Vec<_> = infos
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
            let result = self.submit(|cmd| {
                d.cmd_reset_query_pool(cmd, self.query_pool, 0, 2);
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.query_pool, 0);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
                d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, push);
                d.cmd_dispatch(cmd, groups[0], groups[1], 1);
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.query_pool, 1);
            });
            let time = result.and_then(|_| {
                if !self.timestamps {
                    return Ok(f64::NAN);
                }
                let mut stamps = [0u64; 2];
                d.get_query_pool_results(self.query_pool, 0, &mut stamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                    .map_err(|e| err("reading timestamps", e))?;
                Ok(stamps[1].wrapping_sub(stamps[0]) as f64 * self.timestamp_period_ns as f64 / 1000.0)
            });
            d.destroy_descriptor_pool(desc_pool, None);
            d.destroy_pipeline(pipeline, None);
            d.destroy_pipeline_layout(layout, None);
            d.destroy_descriptor_set_layout(set_layout, None);
            d.destroy_shader_module(module, None);
            time
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            if let Some((utils, messenger)) = self.debug.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

fn copy_region(img: &StorageImage) -> vk::BufferImageCopy {
    vk::BufferImageCopy::default()
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1),
        )
        .image_extent(vk::Extent3D { width: img.width, height: img.height, depth: 1 })
}

/// Push-constant bytes from 32-bit values in order.
pub fn push_bytes(words: &[[u8; 4]]) -> Vec<u8> {
    words.iter().flatten().copied().collect()
}
