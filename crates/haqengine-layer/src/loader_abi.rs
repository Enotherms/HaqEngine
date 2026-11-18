//! `#[repr(C)]` mirrors of the Vulkan loader/layer interface from `vk_layer.h`, which ash does not include.

use ash::vk;
use std::ffi::c_void;

/// `LAYER_NEGOTIATE_INTERFACE_STRUCT`.
pub const LAYER_NEGOTIATE_INTERFACE_STRUCT: u32 = 1;
/// `VK_LAYER_LINK_INFO` in `VkLayerFunction`.
pub const VK_LAYER_LINK_INFO: u32 = 0;
/// `VK_LOADER_DATA_CALLBACK` in `VkLayerFunction`.
pub const VK_LOADER_DATA_CALLBACK: u32 = 1;

/// `PFN_vkSetDeviceLoaderData`: gives a layer-created dispatchable object (e.g. a command buffer)
/// the loader's dispatch table, so layers below can find its device.
pub type PfnSetDeviceLoaderData = unsafe extern "system" fn(vk::Device, *mut c_void) -> vk::Result;

#[repr(C)]
pub struct NegotiateLayerInterface {
    pub s_type: u32,
    pub p_next: *mut c_void,
    pub loader_layer_interface_version: u32,
    pub pfn_get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub pfn_get_device_proc_addr: Option<vk::PFN_vkGetDeviceProcAddr>,
    pub pfn_get_physical_device_proc_addr: *const c_void,
}

#[repr(C)]
pub struct LayerInstanceLink {
    pub p_next: *mut LayerInstanceLink,
    pub pfn_next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub pfn_next_get_physical_device_proc_addr: *const c_void,
}

#[repr(C)]
pub union LayerInstanceCreateInfoU {
    pub p_layer_info: *mut LayerInstanceLink,
    pub other: *const c_void,
}

#[repr(C)]
pub struct LayerInstanceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub function: u32,
    pub u: LayerInstanceCreateInfoU,
}

#[repr(C)]
pub struct LayerDeviceLink {
    pub p_next: *mut LayerDeviceLink,
    pub pfn_next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub pfn_next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

#[repr(C)]
pub union LayerDeviceCreateInfoU {
    pub p_layer_info: *mut LayerDeviceLink,
    pub pfn_set_device_loader_data: Option<PfnSetDeviceLoaderData>,
    pub other: *const c_void,
}

#[repr(C)]
pub struct LayerDeviceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub function: u32,
    pub u: LayerDeviceCreateInfoU,
}

/// Finds the loader's instance link-info struct in a `pNext` chain.
///
/// # Safety
/// `p_next` must be null or the head of a valid Vulkan `pNext` chain.
pub unsafe fn find_instance_link(p_next: *const c_void) -> Option<*mut LayerInstanceCreateInfo> {
    let mut cur = p_next as *const vk::BaseInStructure;
    while !cur.is_null() {
        if (*cur).s_type == vk::StructureType::LOADER_INSTANCE_CREATE_INFO {
            let info = cur as *mut LayerInstanceCreateInfo;
            if (*info).function == VK_LAYER_LINK_INFO {
                return Some(info);
            }
        }
        cur = (*cur).p_next;
    }
    None
}

/// Finds the loader's device link-info struct in a `pNext` chain.
///
/// # Safety
/// `p_next` must be null or the head of a valid Vulkan `pNext` chain.
pub unsafe fn find_device_link(p_next: *const c_void) -> Option<*mut LayerDeviceCreateInfo> {
    let mut cur = p_next as *const vk::BaseInStructure;
    while !cur.is_null() {
        if (*cur).s_type == vk::StructureType::LOADER_DEVICE_CREATE_INFO {
            let info = cur as *mut LayerDeviceCreateInfo;
            if (*info).function == VK_LAYER_LINK_INFO {
                return Some(info);
            }
        }
        cur = (*cur).p_next;
    }
    None
}

/// Finds the loader's `pfnSetDeviceLoaderData` callback in a device `pNext` chain.
///
/// # Safety
/// `p_next` must be null or the head of a valid Vulkan `pNext` chain.
pub unsafe fn find_device_loader_data(p_next: *const c_void) -> Option<PfnSetDeviceLoaderData> {
    let mut cur = p_next as *const vk::BaseInStructure;
    while !cur.is_null() {
        if (*cur).s_type == vk::StructureType::LOADER_DEVICE_CREATE_INFO {
            let info = cur as *const LayerDeviceCreateInfo;
            if (*info).function == VK_LOADER_DATA_CALLBACK {
                return (*info).u.pfn_set_device_loader_data;
            }
        }
        cur = (*cur).p_next;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    unsafe extern "system" fn fake_gipa(_: vk::Instance, _: *const std::ffi::c_char) -> vk::PFN_vkVoidFunction {
        None
    }

    fn link() -> LayerInstanceLink {
        LayerInstanceLink {
            p_next: ptr::null_mut(),
            pfn_next_get_instance_proc_addr: fake_gipa,
            pfn_next_get_physical_device_proc_addr: ptr::null(),
        }
    }

    #[test]
    fn find_instance_link_skips_other_structs() {
        let mut l = link();
        let wanted = LayerInstanceCreateInfo {
            s_type: vk::StructureType::LOADER_INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            function: VK_LAYER_LINK_INFO,
            u: LayerInstanceCreateInfoU { p_layer_info: &mut l },
        };
        let data_callback = LayerInstanceCreateInfo {
            s_type: vk::StructureType::LOADER_INSTANCE_CREATE_INFO,
            p_next: &wanted as *const _ as *const c_void,
            function: 1,
            u: LayerInstanceCreateInfoU { other: ptr::null() },
        };
        let head = vk::BaseInStructure {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: &data_callback as *const _ as *const vk::BaseInStructure,
            _marker: std::marker::PhantomData,
        };
        let found = unsafe { find_instance_link(&head as *const _ as *const c_void) };
        assert_eq!(found, Some(&wanted as *const _ as *mut LayerInstanceCreateInfo));
    }

    unsafe extern "system" fn fake_set_data(_: vk::Device, _: *mut c_void) -> vk::Result {
        vk::Result::SUCCESS
    }

    #[test]
    fn find_device_loader_data_finds_callback_after_link() {
        let callback = LayerDeviceCreateInfo {
            s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            function: VK_LOADER_DATA_CALLBACK,
            u: LayerDeviceCreateInfoU { pfn_set_device_loader_data: Some(fake_set_data) },
        };
        let link_info = LayerDeviceCreateInfo {
            s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
            p_next: &callback as *const _ as *const c_void,
            function: VK_LAYER_LINK_INFO,
            u: LayerDeviceCreateInfoU { other: ptr::null() },
        };
        let found = unsafe { find_device_loader_data(&link_info as *const _ as *const c_void) };
        assert_eq!(found.map(|f| f as usize), Some(fake_set_data as *const () as usize));
        assert!(unsafe { find_device_loader_data(ptr::null()) }.is_none());
        assert!(unsafe { find_device_loader_data(&callback as *const _ as *const c_void) }.is_some());
    }

    #[test]
    fn find_instance_link_none_when_absent() {
        let head = vk::BaseInStructure {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            _marker: std::marker::PhantomData,
        };
        assert_eq!(unsafe { find_instance_link(&head as *const _ as *const c_void) }, None);
        assert_eq!(unsafe { find_instance_link(ptr::null()) }, None);
        assert_eq!(unsafe { find_device_link(ptr::null()) }, None);
    }
}
