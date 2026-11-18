//! The HaqEngine Vulkan layer.
#![allow(non_snake_case, clippy::missing_safety_doc)]

pub mod device;
pub mod guard;
pub mod instance;
pub mod loader_abi;
pub mod state;

use ash::vk;
use loader_abi::NegotiateLayerInterface;
use state::{dispatch_key, DEVICES, INSTANCES};
use std::ffi::{c_char, CStr};

/// Layer interface version HaqEngine implements.
const LAYER_INTERFACE_VERSION: u32 = 2;

#[no_mangle]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    p_version: *mut NegotiateLayerInterface,
) -> vk::Result {
    guard::guard("vkNegotiateLoaderLayerInterfaceVersion", vk::Result::ERROR_INITIALIZATION_FAILED, || unsafe {
        negotiate(p_version)
    })
}

unsafe fn negotiate(p_version: *mut NegotiateLayerInterface) -> vk::Result {
    let Some(n) = p_version.as_mut() else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    if n.loader_layer_interface_version < LAYER_INTERFACE_VERSION {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    n.loader_layer_interface_version = LAYER_INTERFACE_VERSION;
    n.pfn_get_instance_proc_addr = Some(vkGetInstanceProcAddr);
    n.pfn_get_device_proc_addr = Some(vkGetDeviceProcAddr);
    n.pfn_get_physical_device_proc_addr = std::ptr::null();
    vk::Result::SUCCESS
}

fn erase(f: *const ()) -> vk::PFN_vkVoidFunction {
    Some(unsafe { std::mem::transmute::<*const (), unsafe extern "system" fn()>(f) })
}

fn instance_hook(name: &CStr) -> vk::PFN_vkVoidFunction {
    match name.to_bytes() {
        b"vkGetInstanceProcAddr" => erase(vkGetInstanceProcAddr as *const ()),
        b"vkCreateInstance" => erase(instance::create_instance as *const ()),
        b"vkDestroyInstance" => erase(instance::destroy_instance as *const ()),
        b"vkCreateDevice" => erase(device::create_device as *const ()),
        _ => None,
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetInstanceProcAddr(
    instance: vk::Instance,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    guard::guard("vkGetInstanceProcAddr", None, || unsafe {
        if p_name.is_null() {
            return None;
        }
        let name = CStr::from_ptr(p_name);
        if let Some(f) = instance_hook(name).or_else(|| device::device_hook(name)) {
            return Some(f);
        }
        let data = INSTANCES.get(dispatch_key(instance))?;
        (data.next_gipa)(instance, p_name)
    })
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceProcAddr(device: vk::Device, p_name: *const c_char) -> vk::PFN_vkVoidFunction {
    guard::guard("vkGetDeviceProcAddr", None, || unsafe {
        if p_name.is_null() {
            return None;
        }
        let name = CStr::from_ptr(p_name);
        if let Some(f) = device::device_hook(name) {
            return Some(f);
        }
        let data = DEVICES.get(dispatch_key(device))?;
        (data.next_gdpa)(device, p_name)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    fn negotiate(version: u32) -> (vk::Result, NegotiateLayerInterface) {
        let mut n = NegotiateLayerInterface {
            s_type: loader_abi::LAYER_NEGOTIATE_INTERFACE_STRUCT,
            p_next: ptr::null_mut(),
            loader_layer_interface_version: version,
            pfn_get_instance_proc_addr: None,
            pfn_get_device_proc_addr: None,
            pfn_get_physical_device_proc_addr: ptr::null(),
        };
        let r = unsafe { vkNegotiateLoaderLayerInterfaceVersion(&mut n) };
        (r, n)
    }

    #[test]
    fn negotiate_caps_version_at_2() {
        let (r, n) = negotiate(5);
        assert_eq!(r, vk::Result::SUCCESS);
        assert_eq!(n.loader_layer_interface_version, 2);
        assert!(n.pfn_get_instance_proc_addr.is_some());
        assert!(n.pfn_get_device_proc_addr.is_some());
    }

    #[test]
    fn negotiate_rejects_version_below_2() {
        assert_eq!(negotiate(1).0, vk::Result::ERROR_INITIALIZATION_FAILED);
    }

    #[test]
    fn gipa_unknown_name_without_instance_is_null() {
        let f = unsafe { vkGetInstanceProcAddr(vk::Instance::null(), c"vkSomethingMadeUp".as_ptr()) };
        assert!(f.is_none());
    }

    #[test]
    fn gipa_returns_own_create_instance() {
        let f = unsafe { vkGetInstanceProcAddr(vk::Instance::null(), c"vkCreateInstance".as_ptr()) };
        assert_eq!(f.map(|f| f as usize), Some(instance::create_instance as *const () as usize));
    }
}
