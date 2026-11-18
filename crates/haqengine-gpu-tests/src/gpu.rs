//! A headless Vulkan device for running HaqEngine's shaders on test images.

use ash::vk;
use haqengine_core::{Image, Rgba8};
use haqengine_shaders::{BindingKind, Fsr3Pass, Shader};
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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
    pub bytes: u64,
    pub mip_views: Vec<vk::ImageView>,
}

/// A host-visible storage buffer.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

/// What an FSR 3 binding points at: a whole image, one mip level of it, or a storage buffer.
/// Sampler bindings are filled with the device's linear-clamp sampler.
#[derive(Clone, Copy)]
pub enum Bind<'a> {
    Image(&'a StorageImage),
    Mip(&'a StorageImage, u32),
    Buffer(&'a GpuBuffer),
}

pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    query_pool: vk::QueryPool,
    sampler: vk::Sampler,
    timestamps: bool,
    timestamp_period_ns: f32,
    memory: vk::PhysicalDeviceMemoryProperties,
    lost: AtomicBool,
    pub device_name: String,
    pub subgroup_size: u32,
    pub subgroup_ok: bool,
    pub extended_formats: bool,
    pub robust_images: bool,
    /// `shaderStorageImageWriteWithoutFormat` and `shaderStorageImageReadWithoutFormat`.
    pub write_without_format: bool,
}

/// Index of the device to test on: a discrete or integrated GPU first, never a CPU renderer
/// when a real GPU exists.
pub fn pick_device(types: &[vk::PhysicalDeviceType]) -> Option<usize> {
    let rank = |t: vk::PhysicalDeviceType| match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 4,
        _ => 3,
    };
    (0..types.len()).min_by_key(|&i| (rank(types[i]), i))
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
            let candidates: Vec<(vk::PhysicalDevice, u32, u32)> = physicals
                .iter()
                .filter_map(|&p| {
                    instance
                        .get_physical_device_queue_family_properties(p)
                        .iter()
                        .enumerate()
                        .find(|(_, f)| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                        .map(|(i, f)| (p, i as u32, f.timestamp_valid_bits))
                })
                .collect();
            let types: Vec<vk::PhysicalDeviceType> = candidates
                .iter()
                .map(|(p, _, _)| instance.get_physical_device_properties(*p).device_type)
                .collect();
            let (physical, family, valid_bits) =
                pick_device(&types).map(|i| candidates[i]).ok_or("no GPU with a compute queue")?;
            let props = instance.get_physical_device_properties(physical);
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
            instance.get_physical_device_properties2(physical, &mut props2);
            let needed = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC | vk::SubgroupFeatureFlags::QUAD;
            let subgroup_ok = subgroup.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
                && subgroup.supported_operations.contains(needed);
            let subgroup_size = subgroup.subgroup_size;
            let supported = instance.get_physical_device_features(physical);
            let extended_formats = supported.shader_storage_image_extended_formats == vk::TRUE;
            let write_without_format = supported.shader_storage_image_write_without_format == vk::TRUE
                && supported.shader_storage_image_read_without_format == vk::TRUE;
            let has_robustness = instance
                .enumerate_device_extension_properties(physical)
                .map(|exts| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::ext::image_robustness::NAME)))
                .unwrap_or(false);
            let mut robust_query = vk::PhysicalDeviceImageRobustnessFeatures::default();
            if has_robustness {
                let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut robust_query);
                instance.get_physical_device_features2(physical, &mut features2);
            }
            let robust_images = has_robustness && robust_query.robust_image_access == vk::TRUE;
            let mut robust_enable = vk::PhysicalDeviceImageRobustnessFeatures::default().robust_image_access(true);
            let device_extensions = [ash::ext::image_robustness::NAME.as_ptr()];
            let features = vk::PhysicalDeviceFeatures::default()
                .shader_storage_image_extended_formats(extended_formats)
                .shader_storage_image_write_without_format(write_without_format)
                .shader_storage_image_read_without_format(write_without_format);
            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
            let mut device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info).enabled_features(&features);
            if robust_images {
                device_info = device_info.enabled_extension_names(&device_extensions).push_next(&mut robust_enable);
            }
            let device = instance.create_device(physical, &device_info, None).map_err(|e| err("creating the device", e))?;
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
            let sampler = device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .max_lod(vk::LOD_CLAMP_NONE),
                    None,
                )
                .map_err(|e| err("creating the sampler", e))?;
            Ok(Gpu {
                _entry: entry,
                instance,
                debug,
                device,
                queue,
                pool,
                query_pool,
                sampler,
                timestamps: valid_bits > 0,
                timestamp_period_ns: props.limits.timestamp_period,
                memory,
                lost: AtomicBool::new(false),
                device_name: props.device_name_as_c_str().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                subgroup_size,
                subgroup_ok,
                extended_formats,
                robust_images,
                write_without_format,
            })
        }
    }

    /// True after a hang or device loss; nothing the GPU may still use is freed after that.
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Relaxed)
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
        let memory = self
            .memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            .and_then(|kind| {
                d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(kind), None)
                    .map_err(|e| err("allocating buffer memory", e))
            });
        let memory = match memory {
            Ok(m) => m,
            Err(e) => {
                d.destroy_buffer(buffer, None);
                return Err(e);
            }
        };
        if let Err(e) = d.bind_buffer_memory(buffer, memory, 0) {
            d.destroy_buffer(buffer, None);
            d.free_memory(memory, None);
            return Err(err("binding buffer memory", e));
        }
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
            if r.as_ref().is_err_and(|e| e.contains("GPU timeout") || e.contains("ERROR_DEVICE_LOST")) {
                self.lost.store(true, Ordering::Relaxed);
            } else {
                d.destroy_fence(fence, None);
            }
            r
        })();
        if !self.is_lost() {
            d.free_command_buffers(self.pool, &[cmd]);
        }
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
        self.create_image(format, width, height, 1)
    }

    /// A device-local image with `mips` levels in `GENERAL` layout; with more than one level it
    /// also gets one view per level for storage writes.
    pub fn create_image(&self, format: vk::Format, width: u32, height: u32, mips: u32) -> Result<StorageImage, String> {
        unsafe {
            let d = &self.device;
            let image = d
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D { width, height, depth: 1 })
                        .mip_levels(mips)
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
                )
                .map_err(|e| err("creating an image", e))?;
            let req = d.get_image_memory_requirements(image);
            let memory = match self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL).and_then(|kind| {
                d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(kind), None)
                    .map_err(|e| err("allocating image memory", e))
            }) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_image(image, None);
                    return Err(e);
                }
            };
            let free = |views: &[vk::ImageView]| {
                for v in views {
                    d.destroy_image_view(*v, None);
                }
                d.destroy_image(image, None);
                d.free_memory(memory, None);
            };
            if let Err(e) = d.bind_image_memory(image, memory, 0) {
                free(&[]);
                return Err(err("binding image memory", e));
            }
            let make_view = |range: vk::ImageSubresourceRange| {
                d.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(range),
                    None,
                )
            };
            let mut views = Vec::with_capacity(mips as usize + 1);
            let levels = std::iter::once(color_range()).chain(
                (0..if mips > 1 { mips } else { 0 }).map(|m| color_range().base_mip_level(m).level_count(1)),
            );
            for range in levels {
                match make_view(range) {
                    Ok(v) => views.push(v),
                    Err(e) => {
                        free(&views);
                        return Err(err("creating an image view", e));
                    }
                }
            }
            if let Err(e) = self.submit(|cmd| self.transition(cmd, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL)) {
                if !self.is_lost() {
                    free(&views);
                }
                return Err(e);
            }
            let view = views.remove(0);
            Ok(StorageImage { image, memory, view, width, height, bytes: req.size, mip_views: views })
        }
    }

    pub fn destroy_image(&self, img: StorageImage) {
        if self.is_lost() {
            return;
        }
        unsafe {
            for v in &img.mip_views {
                self.device.destroy_image_view(*v, None);
            }
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

    /// Replaces the contents of an existing `R8G8B8A8_UNORM` image.
    pub fn write_rgba8(&self, img: &StorageImage, src: &Image<Rgba8>) -> Result<(), String> {
        let bytes: Vec<u8> = src.data.iter().flatten().copied().collect();
        unsafe { self.copy_in(img, &bytes) }
    }

    pub fn create_storage_buffer(&self, size: u64) -> Result<GpuBuffer, String> {
        let (buffer, memory) = unsafe { self.host_buffer(size, vk::BufferUsageFlags::STORAGE_BUFFER)? };
        let buf = GpuBuffer { buffer, memory, size };
        if let Err(e) = self.write_buffer(&buf, &vec![0u8; size as usize]) {
            self.destroy_buffer(buf);
            return Err(e);
        }
        Ok(buf)
    }

    pub fn write_buffer(&self, buf: &GpuBuffer, bytes: &[u8]) -> Result<(), String> {
        unsafe {
            let len = bytes.len().min(buf.size as usize);
            let ptr = self
                .device
                .map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty())
                .map_err(|e| err("mapping memory", e))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, len);
            self.device.unmap_memory(buf.memory);
        }
        Ok(())
    }

    pub fn destroy_buffer(&self, buf: GpuBuffer) {
        if self.is_lost() {
            return;
        }
        unsafe {
            self.device.destroy_buffer(buf.buffer, None);
            self.device.free_memory(buf.memory, None);
        }
    }

    /// Mip 0 of an image as raw bytes, `texel_bytes` per texel.
    pub fn read_raw(&self, img: &StorageImage, texel_bytes: usize) -> Result<Vec<u8>, String> {
        unsafe { self.copy_out(img, (img.width * img.height) as usize * texel_bytes) }
    }

    pub fn write_u32(&self, img: &StorageImage, src: &Image<u32>) -> Result<(), String> {
        let bytes: Vec<u8> = src.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        unsafe { self.copy_in(img, &bytes) }
    }

    pub fn read_r8(&self, img: &StorageImage) -> Result<Image<u8>, String> {
        let n = (img.width * img.height) as usize;
        let data = unsafe { self.copy_out(img, n)? };
        Ok(Image { width: img.width, height: img.height, data })
    }

    pub fn read_rg16i(&self, img: &StorageImage) -> Result<Image<[i16; 2]>, String> {
        let n = (img.width * img.height) as usize;
        let bytes = unsafe { self.copy_out(img, n * 4)? };
        let data = bytes
            .chunks_exact(4)
            .map(|c| [i16::from_le_bytes([c[0], c[1]]), i16::from_le_bytes([c[2], c[3]])])
            .collect();
        Ok(Image { width: img.width, height: img.height, data })
    }

    pub fn upload_u32(&self, src: &Image<u32>) -> Result<StorageImage, String> {
        let img = self.create_storage_image(vk::Format::R32_UINT, src.width, src.height)?;
        let bytes: Vec<u8> = src.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        unsafe { self.copy_in(&img, &bytes)? };
        Ok(img)
    }

    pub fn read_u32(&self, img: &StorageImage) -> Result<Image<u32>, String> {
        let n = (img.width * img.height) as usize;
        let bytes = unsafe { self.copy_out(img, n * 4)? };
        let data = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(Image { width: img.width, height: img.height, data })
    }

    pub fn clear_u32(&self, img: &StorageImage, value: u32) -> Result<(), String> {
        unsafe {
            self.submit(|cmd| {
                self.device.cmd_clear_color_image(
                    cmd,
                    img.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue { uint32: [value, 0, 0, 0] },
                    &[color_range()],
                )
            })
        }
    }

    pub fn upload_r32f(&self, src: &Image<f32>) -> Result<StorageImage, String> {
        let img = self.create_storage_image(vk::Format::R32_SFLOAT, src.width, src.height)?;
        let bytes: Vec<u8> = src.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        unsafe { self.copy_in(&img, &bytes)? };
        Ok(img)
    }

    pub fn upload_rg32f(&self, src: &Image<[f32; 2]>) -> Result<StorageImage, String> {
        let img = self.create_storage_image(vk::Format::R32G32_SFLOAT, src.width, src.height)?;
        let bytes: Vec<u8> = src.data.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        unsafe { self.copy_in(&img, &bytes)? };
        Ok(img)
    }

    pub fn read_rg32f(&self, img: &StorageImage) -> Result<Image<[f32; 2]>, String> {
        let n = (img.width * img.height) as usize;
        let bytes = unsafe { self.copy_out(img, n * 8)? };
        let f = |c: &[u8]| f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let data = bytes.chunks_exact(8).map(|c| [f(&c[..4]), f(&c[4..])]).collect();
        Ok(Image { width: img.width, height: img.height, data })
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
            if self.is_lost() {
                return time;
            }
            d.destroy_descriptor_pool(desc_pool, None);
            d.destroy_pipeline(pipeline, None);
            d.destroy_pipeline_layout(layout, None);
            d.destroy_descriptor_set_layout(set_layout, None);
            d.destroy_shader_module(module, None);
            time
        }
    }

    /// Runs one dispatch of an FSR 3 pass and returns its GPU time in µs. `images` maps binding
    /// numbers to images; `constants` fills the pass's uniform buffers in binding order; `push`
    /// holds the pass's push constants (`pass.push_size` bytes). Everything it creates is freed
    /// on every path, unless the device is lost.
    pub fn run_fsr3(
        &self,
        pass: &Fsr3Pass,
        binds: &[(u32, Bind)],
        constants: &[&[u8]],
        push: &[u8],
        groups: [u32; 3],
    ) -> Result<f64, String> {
        let kind_of = |k: BindingKind| match k {
            BindingKind::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            BindingKind::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            BindingKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            BindingKind::Sampler => vk::DescriptorType::SAMPLER,
            BindingKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        };
        let mut cb_bindings: Vec<u32> =
            pass.bindings.iter().filter(|b| b.1 == BindingKind::UniformBuffer).map(|b| b.0).collect();
        cb_bindings.sort_unstable();
        if cb_bindings.len() != constants.len() {
            return Err(format!("{} needs {} constant blocks, got {}", pass.name, cb_bindings.len(), constants.len()));
        }
        if push.len() != pass.push_size as usize {
            return Err(format!("{} needs {} push bytes, got {}", pass.name, pass.push_size, push.len()));
        }
        for (b, kind) in pass.bindings {
            let bound = binds.iter().find(|(n, _)| n == b).map(|(_, x)| *x);
            let fits = match (kind, bound) {
                (BindingKind::UniformBuffer | BindingKind::Sampler, None) => true,
                (BindingKind::StorageBuffer, Some(Bind::Buffer(_))) => true,
                (BindingKind::StorageImage | BindingKind::SampledImage, Some(Bind::Image(_))) => true,
                (BindingKind::StorageImage, Some(Bind::Mip(img, m))) => (m as usize) < img.mip_views.len(),
                _ => false,
            };
            if !fits {
                return Err(format!("{} binding {b} ({kind:?}) is not bound to a matching resource", pass.name));
            }
        }
        let mut owned = Owned::default();
        let result = unsafe { self.dispatch_fsr3(pass, binds, constants, push, groups, &cb_bindings, &kind_of, &mut owned) };
        if !self.is_lost() {
            unsafe { owned.free(&self.device) };
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn dispatch_fsr3(
        &self,
        pass: &Fsr3Pass,
        binds: &[(u32, Bind)],
        constants: &[&[u8]],
        push: &[u8],
        groups: [u32; 3],
        cb_bindings: &[u32],
        kind_of: &dyn Fn(BindingKind) -> vk::DescriptorType,
        owned: &mut Owned,
    ) -> Result<f64, String> {
        let d = &self.device;
        for bytes in constants {
            let (buffer, memory) = self.host_buffer(bytes.len() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER)?;
            owned.buffers.push((buffer, memory));
            let ptr = d
                .map_memory(memory, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| err("mapping memory", e))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
            d.unmap_memory(memory);
        }
        let code = ash::util::read_spv(&mut std::io::Cursor::new(pass.spirv)).map_err(|e| e.to_string())?;
        let module = d
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .map_err(|e| err("creating the shader module", e))?;
        owned.module = module;
        let layout_bindings: Vec<_> = pass
            .bindings
            .iter()
            .map(|(b, k)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(*b)
                    .descriptor_type(kind_of(*k))
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = d
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings), None)
            .map_err(|e| err("creating the set layout", e))?;
        owned.set_layout = set_layout;
        let set_layouts = [set_layout];
        let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(pass.push_size)];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let layout_info = if pass.push_size > 0 { layout_info.push_constant_ranges(&ranges) } else { layout_info };
        let layout = d.create_pipeline_layout(&layout_info, None).map_err(|e| err("creating the pipeline layout", e))?;
        owned.layout = layout;
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
        owned.pipeline = pipeline;
        let sizes: Vec<_> = [
            vk::DescriptorType::STORAGE_IMAGE,
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::DescriptorType::SAMPLER,
            vk::DescriptorType::STORAGE_BUFFER,
        ]
            .into_iter()
            .map(|ty| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(pass.bindings.len() as u32))
            .collect();
        let desc_pool = d
            .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)
            .map_err(|e| err("creating the descriptor pool", e))?;
        owned.desc_pool = desc_pool;
        let set = d
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&set_layouts))
            .map_err(|e| err("allocating the descriptor set", e))?[0];
        let image_infos: Vec<(u32, vk::DescriptorType, [vk::DescriptorImageInfo; 1])> = pass
            .bindings
            .iter()
            .filter_map(|(b, k)| {
                let bound = binds.iter().find(|(n, _)| n == b).map(|(_, x)| *x);
                let view = match (k, bound) {
                    (BindingKind::Sampler, _) => return Some((*b, kind_of(*k), [vk::DescriptorImageInfo::default().sampler(self.sampler)])),
                    (_, Some(Bind::Image(img))) => img.view,
                    (_, Some(Bind::Mip(img, m))) => img.mip_views[m as usize],
                    _ => return None,
                };
                Some((*b, kind_of(*k), [vk::DescriptorImageInfo::default().image_view(view).image_layout(vk::ImageLayout::GENERAL)]))
            })
            .collect();
        let storage_infos: Vec<(u32, [vk::DescriptorBufferInfo; 1])> = binds
            .iter()
            .filter_map(|(b, x)| match x {
                Bind::Buffer(buf) => Some((*b, [vk::DescriptorBufferInfo::default().buffer(buf.buffer).range(buf.size)])),
                _ => None,
            })
            .collect();
        let buffer_infos: Vec<(u32, [vk::DescriptorBufferInfo; 1])> = cb_bindings
            .iter()
            .zip(&owned.buffers)
            .zip(constants)
            .map(|((b, (buffer, _)), bytes)| (*b, [vk::DescriptorBufferInfo::default().buffer(*buffer).range(bytes.len() as u64)]))
            .collect();
        let mut writes: Vec<vk::WriteDescriptorSet> = image_infos
            .iter()
            .map(|(b, ty, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*b).descriptor_type(*ty).image_info(info))
            .collect();
        writes.extend(buffer_infos.iter().map(|(b, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(*b)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(info)
        }));
        writes.extend(storage_infos.iter().map(|(b, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(*b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        }));
        d.update_descriptor_sets(&writes, &[]);
        self.submit(|cmd| {
            d.cmd_reset_query_pool(cmd, self.query_pool, 0, 2);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.query_pool, 0);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            if pass.push_size > 0 {
                d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            d.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.query_pool, 1);
        })?;
        if !self.timestamps {
            return Ok(f64::NAN);
        }
        let mut stamps = [0u64; 2];
        d.get_query_pool_results(self.query_pool, 0, &mut stamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
            .map_err(|e| err("reading timestamps", e))?;
        Ok(stamps[1].wrapping_sub(stamps[0]) as f64 * self.timestamp_period_ns as f64 / 1000.0)
    }
}

/// Vulkan objects made for one dispatch; null handles are skipped when freeing.
#[derive(Default)]
struct Owned {
    buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    desc_pool: vk::DescriptorPool,
}

impl Owned {
    unsafe fn free(self, d: &ash::Device) {
        if self.desc_pool != vk::DescriptorPool::null() {
            d.destroy_descriptor_pool(self.desc_pool, None);
        }
        if self.pipeline != vk::Pipeline::null() {
            d.destroy_pipeline(self.pipeline, None);
        }
        if self.layout != vk::PipelineLayout::null() {
            d.destroy_pipeline_layout(self.layout, None);
        }
        if self.set_layout != vk::DescriptorSetLayout::null() {
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
        if self.module != vk::ShaderModule::null() {
            d.destroy_shader_module(self.module, None);
        }
        for (buffer, memory) in self.buffers {
            d.destroy_buffer(buffer, None);
            d.free_memory(memory, None);
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        if self.is_lost() {
            return;
        }
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_sampler(self.sampler, None);
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
        .level_count(vk::REMAINING_MIP_LEVELS)
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

#[cfg(test)]
mod tests {
    use super::*;
    use vk::PhysicalDeviceType as T;

    #[test]
    fn pick_device_prefers_real_gpus() {
        assert_eq!(pick_device(&[T::CPU, T::INTEGRATED_GPU]), Some(1));
        assert_eq!(pick_device(&[T::INTEGRATED_GPU, T::DISCRETE_GPU]), Some(1));
        assert_eq!(pick_device(&[T::CPU]), Some(0));
        assert_eq!(pick_device(&[]), None);
    }
}
