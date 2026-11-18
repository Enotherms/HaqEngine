//! The HaqEngine Vulkan layer.
#![allow(non_snake_case, clippy::missing_safety_doc)]

pub mod chain;
pub mod device;
pub mod extensions;
pub mod framegen;
pub mod guard;
pub mod instance;
pub mod loader_abi;
pub mod plan;
pub mod queue;
pub mod state;
pub mod swapchain;

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
        b"vkEnumerateDeviceExtensionProperties" => {
            erase(extensions::enumerate_device_extension_properties as *const ())
        }
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
        let next = (data.next_gdpa)(device, p_name);
        match (next, device::extension_hook(name)) {
            (Some(_), Some(ours)) => Some(ours),
            _ => next,
        }
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
    fn gdpa_hides_swapchain_hooks_when_next_layer_lacks_them() {
        let (handle, _) = state::test_support::fake_device(|_| ptr::null(), vec![]);
        for name in [c"vkCreateSwapchainKHR", c"vkDestroySwapchainKHR", c"vkQueuePresentKHR"] {
            assert!(unsafe { vkGetDeviceProcAddr(handle, name.as_ptr()) }.is_none(), "{name:?}");
        }
        assert!(unsafe { vkGetDeviceProcAddr(handle, c"vkDestroyDevice".as_ptr()) }.is_some());
        DEVICES.remove(dispatch_key(handle));
    }

    #[test]
    fn gdpa_returns_own_get_device_queue() {
        unsafe extern "system" fn dummy() {}
        unsafe extern "system" fn next_gdpa(_: vk::Device, name: *const c_char) -> vk::PFN_vkVoidFunction {
            let name = unsafe { CStr::from_ptr(name) };
            (name == c"vkGetDeviceQueue" || name == c"vkGetDeviceQueue2").then_some(dummy as unsafe extern "system" fn())
        }
        let (handle, _) = state::test_support::fake_device_with_gdpa(|_| ptr::null(), vec![], next_gdpa);
        let f = unsafe { vkGetDeviceProcAddr(handle, c"vkGetDeviceQueue".as_ptr()) };
        assert_eq!(f.map(|f| f as usize), Some(queue::get_device_queue as *const () as usize));
        let f = unsafe { vkGetDeviceProcAddr(handle, c"vkGetDeviceQueue2".as_ptr()) };
        assert_eq!(f.map(|f| f as usize), Some(queue::get_device_queue2 as *const () as usize));
        DEVICES.remove(dispatch_key(handle));
    }

    #[test]
    fn gipa_returns_own_enumerate_device_extensions() {
        let f = unsafe { vkGetInstanceProcAddr(vk::Instance::null(), c"vkEnumerateDeviceExtensionProperties".as_ptr()) };
        assert_eq!(f.map(|f| f as usize), Some(extensions::enumerate_device_extension_properties as *const () as usize));
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
