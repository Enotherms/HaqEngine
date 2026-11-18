//! Device-level hooks: create, destroy and the device-level function table.

use crate::guard::guard;
use crate::loader_abi::{find_device_link, find_device_loader_data};
use crate::state::{dispatch_key, DeviceData, DEVICES, INSTANCES};
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

    let Some(next_create) = next_gipa(instance.instance.handle(), c"vkCreateDevice".as_ptr()) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let next_create: vk::PFN_vkCreateDevice = std::mem::transmute(next_create);
    let result = next_create(physical_device, create_info, allocator, out);
    if result != vk::Result::SUCCESS {
        return result;
    }

    let handle = *out;
    let load = |name: &CStr| std::mem::transmute(next_gdpa(handle, name.as_ptr()));
    let device = ash::Device::load_with(load, handle);
    let swapchain = ash::khr::swapchain::DeviceFn::load(load);
    let queue_families = instance.instance.get_physical_device_queue_family_properties(physical_device);
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
        }),
    );
    vk::Result::SUCCESS
}

pub unsafe extern "system" fn destroy_device(device: vk::Device, allocator: *const vk::AllocationCallbacks) {
    guard("vkDestroyDevice", (), || unsafe {
        let key = dispatch_key(device);
        if let Some(data) = DEVICES.remove(key) {
            crate::queue::forget_device(key);
            data.device.destroy_device(allocator.as_ref());
        }
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
        _ => None,
    }
}
