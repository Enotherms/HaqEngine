//! Device-level hooks: create, destroy and the device-level function table.

use crate::loader_abi::find_device_link;
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
    let Some(instance) = INSTANCES.get(dispatch_key(physical_device)) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
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
    DEVICES.insert(dispatch_key(handle), Arc::new(DeviceData { next_gdpa, device, swapchain }));
    vk::Result::SUCCESS
}

pub unsafe extern "system" fn destroy_device(device: vk::Device, allocator: *const vk::AllocationCallbacks) {
    if let Some(data) = DEVICES.remove(dispatch_key(device)) {
        data.device.destroy_device(allocator.as_ref());
    }
}

/// Our replacement for a device-level function, if HaqEngine hooks it.
pub fn device_hook(name: &CStr) -> vk::PFN_vkVoidFunction {
    match name.to_bytes() {
        b"vkGetDeviceProcAddr" => crate::erase(crate::vkGetDeviceProcAddr as *const ()),
        b"vkDestroyDevice" => crate::erase(destroy_device as *const ()),
        _ => None,
    }
}
