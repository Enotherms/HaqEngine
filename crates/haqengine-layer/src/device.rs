//! Device-level hooks: create, destroy and the device-level function table.

use crate::guard::guard;
use crate::loader_abi::{find_device_link, find_device_loader_data};
use crate::state::{dispatch_key, DeviceData, InstanceData, DEVICES, INSTANCES, LOG};
use haqengine_core::quality::{quality_gate, QualityCaps};
use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;

pub unsafe extern "system" fn create_device(
    physical_device: vk::PhysicalDevice,
    create_info: *const vk::DeviceCreateInfo,
    allocator: *const vk::AllocationCallbacks,
    out: *mut vk::Device,
) -> vk::Result {
    guard("vkCreateDevice", vk::Result::ERROR_INITIALIZATION_FAILED, || unsafe {
        create_device_inner(physical_device, create_info, allocator, out)
    })
}

unsafe fn create_device_inner(
    physical_device: vk::PhysicalDevice,
    create_info: *const vk::DeviceCreateInfo,
    allocator: *const vk::AllocationCallbacks,
    out: *mut vk::Device,
) -> vk::Result {
    let Some(instance) = INSTANCES.get(dispatch_key(physical_device)) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let set_loader_data = find_device_loader_data((*create_info).p_next);
    let Some(info) = find_device_link((*create_info).p_next) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let link = (*info).u.p_layer_info;
    if link.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let next_gipa = (*link).pfn_next_get_instance_proc_addr;
    let next_gdpa = (*link).pfn_next_get_device_proc_addr;
    (*info).u.p_layer_info = (*link).p_next;
    let layer_link = (*info).u.p_layer_info;

    let Some(next_create) = next_gipa(instance.instance.handle(), c"vkCreateDevice".as_ptr()) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let next_create: vk::PFN_vkCreateDevice = std::mem::transmute(next_create);
    let wants_quality = crate::state::SETTINGS.mode == haqengine_core::Mode::Quality;
    let mut quality: Result<(), String> = Err("quality mode was not requested".into());
    let patch = if wants_quality {
        match quality_patch(&instance, physical_device, &*create_info) {
            Ok(p) => {
                quality = Ok(());
                p
            }
            Err(e) => {
                quality = Err(e);
                None
            }
        }
    } else {
        None
    };
    let base: &vk::DeviceCreateInfo = patch.as_ref().map_or(&*create_info, |p| &p.info);
    let wait_state = crate::features::present_wait_state(base);
    let present = if wait_state == crate::features::PresentWaitState::Absent && present_wait_supported(&instance, physical_device) {
        crate::features::present_patch(base)
    } else {
        None
    };
    let mut paced = wait_state == crate::features::PresentWaitState::Enabled;
    let mut created = None;
    if let Some(p) = &present {
        let r = next_create(physical_device, &p.info, allocator, out);
        if r == vk::Result::SUCCESS {
            paced = true;
            created = Some((r, patch.is_some()));
        } else {
            LOG.log(&format!("pacing off: the device would not start with present waits ({r:?})"));
            if let Some(l) = find_device_link(base.p_next) {
                (*l).u.p_layer_info = layer_link;
            }
        }
    }
    let (result, patched) = match created {
        Some(c) => c,
        None => crate::features::create_with_fallback(
            next_create,
            physical_device,
            create_info,
            patch.as_ref(),
            &mut || (*info).u.p_layer_info = layer_link,
            allocator,
            out,
        ),
    };
    if result != vk::Result::SUCCESS {
        return result;
    }
    if !paced && present.is_none() {
        LOG.log(match wait_state {
            crate::features::PresentWaitState::Absent => "pacing off: the GPU lacks present waits",
            _ => "pacing off: no swapchain extension, or the game set present-wait features itself",
        });
    }
    if wants_quality {
        if patch.is_some() && !patched {
            quality = Err("the device would not start with quality's features".into());
        }
        if quality.is_ok() {
            let used = if patched { &patch.as_ref().expect("patched").info } else { &*create_info };
            quality = quality_verdict(&instance, physical_device, used);
        }
        match &quality {
            Ok(()) => LOG.log(&format!("quality ready{}", if patched { " (features enabled by HaqEngine)" } else { "" })),
            Err(e) => LOG.log(&format!("quality unavailable: {e}; using battery")),
        }
    }

    let handle = *out;
    let load = |name: &CStr| std::mem::transmute(next_gdpa(handle, name.as_ptr()));
    let device = ash::Device::load_with(load, handle);
    let swapchain = ash::khr::swapchain::DeviceFn::load(load);
    let queue_families = instance.instance.get_physical_device_queue_family_properties(physical_device);
    let wait_for_present = if paced {
        let f: Option<vk::PFN_vkWaitForPresentKHR> =
            next_gdpa(handle, c"vkWaitForPresentKHR".as_ptr()).map(|p| std::mem::transmute(p));
        if f.is_none() {
            LOG.log("pacing off: vkWaitForPresentKHR missing");
        }
        f
    } else {
        None
    };
    DEVICES.insert(
        dispatch_key(handle),
        Arc::new(DeviceData {
            next_gdpa,
            device,
            swapchain,
            physical_device,
            instance,
            queue_families,
            set_loader_data,
            pipelines: std::sync::OnceLock::new(),
            quality,
            quality_pipelines: std::sync::OnceLock::new(),
            wait_for_present,
        }),
    );
    vk::Result::SUCCESS
}

pub unsafe extern "system" fn destroy_device(device: vk::Device, allocator: *const vk::AllocationCallbacks) {
    guard("vkDestroyDevice", (), || unsafe {
        let key = dispatch_key(device);
        if let Some(data) = DEVICES.remove(key) {
            crate::queue::forget_device(key);
            if let Some(Ok(p)) = data.pipelines.get() {
                p.destroy(&data.device);
            }
            if let Some(Ok(p)) = data.quality_pipelines.get() {
                p.destroy(&data.device);
            }
            data.device.destroy_device(allocator.as_ref());
        }
    })
}

fn at_least_1_1(api_version: u32) -> bool {
    (vk::api_version_major(api_version), vk::api_version_minor(api_version)) >= (1, 1)
}

/// The create-info patch that turns on Quality's features, if any is needed, or why Quality cannot
/// run on this device.
unsafe fn quality_patch(
    instance: &InstanceData,
    physical: vk::PhysicalDevice,
    ci: &vk::DeviceCreateInfo,
) -> Result<Option<crate::features::FeaturePatch>, String> {
    if !at_least_1_1(instance.api_version) {
        return Err(format!(
            "the app asked for Vulkan {}.{}, quality needs 1.1",
            vk::api_version_major(instance.api_version),
            vk::api_version_minor(instance.api_version)
        ));
    }
    let i = &instance.instance;
    let props = i.get_physical_device_properties(physical);
    let version = |v: u32| (vk::api_version_major(v), vk::api_version_minor(v));
    let device_api_13 = version(props.api_version) >= (1, 3) && version(instance.api_version) >= (1, 3);
    let image_robustness_ext = i
        .enumerate_device_extension_properties(physical)
        .map(|exts| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::ext::image_robustness::NAME)))
        .unwrap_or(false);
    let mut robust = vk::PhysicalDeviceImageRobustnessFeatures::default();
    if image_robustness_ext || device_api_13 {
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut robust);
        i.get_physical_device_features2(physical, &mut f2);
    }
    let supported = crate::features::Supported {
        extended_formats: i.get_physical_device_features(physical).shader_storage_image_extended_formats == vk::TRUE,
        robust_image_access: robust.robust_image_access == vk::TRUE,
        image_robustness_ext,
        device_api_13,
    };
    crate::features::patch_create_info(ci, &supported)
}

/// Whether the GPU offers present ids and present waits (both extensions and features).
/// Below Vulkan 1.1 the features cannot be read, so the extensions decide (a device that then
/// refuses them is created again without).
unsafe fn present_wait_supported(instance: &InstanceData, physical: vk::PhysicalDevice) -> bool {
    let i = &instance.instance;
    let has = |name: &CStr| {
        i.enumerate_device_extension_properties(physical)
            .map(|exts| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(name)))
            .unwrap_or(false)
    };
    if !has(ash::khr::present_id::NAME) || !has(ash::khr::present_wait::NAME) {
        return false;
    }
    if !at_least_1_1(instance.api_version) {
        return true;
    }
    let mut id = vk::PhysicalDevicePresentIdFeaturesKHR::default();
    let mut wait = vk::PhysicalDevicePresentWaitFeaturesKHR::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut id).push_next(&mut wait);
    i.get_physical_device_features2(physical, &mut f2);
    id.present_id == vk::TRUE && wait.present_wait == vk::TRUE
}

/// Quality's gate over what the created device really enabled.
unsafe fn quality_verdict(instance: &InstanceData, physical: vk::PhysicalDevice, used: &vk::DeviceCreateInfo) -> Result<(), String> {
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    instance.instance.get_physical_device_properties2(physical, &mut props2);
    let needed = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC | vk::SubgroupFeatureFlags::QUAD;
    let (extended_formats, robust_images) = crate::features::enabled_caps(used);
    quality_gate(&QualityCaps {
        api_version: instance.api_version,
        subgroup_ops_ok: subgroup.supported_stages.contains(vk::ShaderStageFlags::COMPUTE) && subgroup.supported_operations.contains(needed),
        subgroup_size: subgroup.subgroup_size,
        extended_formats,
        robust_images,
    })
}

/// Our replacement for a core device-level function, if HaqEngine hooks it.
pub fn device_hook(name: &CStr) -> vk::PFN_vkVoidFunction {
    match name.to_bytes() {
        b"vkGetDeviceProcAddr" => crate::erase(crate::vkGetDeviceProcAddr as *const ()),
        b"vkDestroyDevice" => crate::erase(destroy_device as *const ()),
        b"vkGetDeviceQueue" => crate::erase(crate::queue::get_device_queue as *const ()),
        _ => None,
    }
}

/// Our replacement for an extension or Vulkan 1.1+ function. Only handed out when the next layer
/// provides the real one, so a disabled extension stays invisible.
pub fn extension_hook(name: &CStr) -> vk::PFN_vkVoidFunction {
    match name.to_bytes() {
        b"vkGetDeviceQueue2" => crate::erase(crate::queue::get_device_queue2 as *const ()),
        b"vkCreateSwapchainKHR" => crate::erase(crate::swapchain::create_swapchain as *const ()),
        b"vkDestroySwapchainKHR" => crate::erase(crate::swapchain::destroy_swapchain as *const ()),
        b"vkQueuePresentKHR" => crate::erase(crate::swapchain::queue_present as *const ()),
        b"vkWaitForPresentKHR" => crate::erase(crate::swapchain::wait_for_present as *const ()),
        _ => None,
    }
}
