//! Hides device extensions that conflict with frame generation.

use crate::guard::guard;
use crate::state::{dispatch_key, multiplier, INSTANCES};
use ash::vk;
use std::ffi::{c_char, CStr};
use std::sync::Once;

static HID_REPORTED: Once = Once::new();

/// Extensions hidden from games while HaqEngine generates frames. `VK_EXT_swapchain_maintenance1`
/// lets a game switch present modes on every present, which breaks forced FIFO pacing; without it,
/// DXVK and VKD3D-Proton keep one present mode and recreate the swapchain to change it.
pub const HIDDEN_WHEN_GENERATING: &[&CStr] = &[c"VK_EXT_swapchain_maintenance1"];

/// `list` without the extensions named in `hide`.
pub fn filter_extensions(list: &[vk::ExtensionProperties], hide: &[&CStr]) -> Vec<vk::ExtensionProperties> {
    list.iter()
        .filter(|p| p.extension_name_as_c_str().map_or(true, |name| !hide.contains(&name)))
        .copied()
        .collect()
}

/// Writes `list` back with Vulkan's two-call idiom (`props` null: report the count).
///
/// # Safety
/// `count` must be valid; `props`, if not null, must hold `*count` elements.
pub unsafe fn write_out(list: &[vk::ExtensionProperties], count: *mut u32, props: *mut vk::ExtensionProperties) -> vk::Result {
    if props.is_null() {
        *count = list.len() as u32;
        return vk::Result::SUCCESS;
    }
    let n = (*count as usize).min(list.len());
    std::ptr::copy_nonoverlapping(list.as_ptr(), props, n);
    *count = n as u32;
    if n < list.len() { vk::Result::INCOMPLETE } else { vk::Result::SUCCESS }
}

pub unsafe extern "system" fn enumerate_device_extension_properties(
    physical_device: vk::PhysicalDevice,
    layer_name: *const c_char,
    count: *mut u32,
    props: *mut vk::ExtensionProperties,
) -> vk::Result {
    guard("vkEnumerateDeviceExtensionProperties", vk::Result::ERROR_INITIALIZATION_FAILED, || unsafe {
        let Some(instance) = INSTANCES.get(dispatch_key(physical_device)) else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let next = instance.instance.fp_v1_0().enumerate_device_extension_properties;
        if !layer_name.is_null() || multiplier() < 2 {
            return next(physical_device, layer_name, count, props);
        }
        let mut n = 0u32;
        let r = next(physical_device, layer_name, &mut n, std::ptr::null_mut());
        if r != vk::Result::SUCCESS {
            return r;
        }
        let mut all = vec![vk::ExtensionProperties::default(); n as usize];
        let r = next(physical_device, layer_name, &mut n, all.as_mut_ptr());
        if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
            return r;
        }
        all.truncate(n as usize);
        let shown = filter_extensions(&all, HIDDEN_WHEN_GENERATING);
        if shown.len() < all.len() {
            HID_REPORTED.call_once(|| crate::state::LOG.log("hid VK_EXT_swapchain_maintenance1 from the game"));
        }
        write_out(&shown, count, props)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ext(name: &str) -> vk::ExtensionProperties {
        let mut p = vk::ExtensionProperties::default();
        for (i, b) in name.bytes().enumerate() {
            p.extension_name[i] = b as c_char;
        }
        p
    }

    fn names(list: &[vk::ExtensionProperties]) -> Vec<String> {
        list.iter().map(|p| p.extension_name_as_c_str().unwrap().to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn filter_hides_swapchain_maintenance1_only() {
        let list = [ext("VK_KHR_swapchain"), ext("VK_EXT_swapchain_maintenance1"), ext("VK_KHR_present_id")];
        assert_eq!(names(&filter_extensions(&list, HIDDEN_WHEN_GENERATING)), ["VK_KHR_swapchain", "VK_KHR_present_id"]);
    }

    #[test]
    fn write_out_follows_two_call_idiom() {
        let list = [ext("A"), ext("B"), ext("C")];
        let mut count = 0u32;
        assert_eq!(unsafe { write_out(&list, &mut count, std::ptr::null_mut()) }, vk::Result::SUCCESS);
        assert_eq!(count, 3);
        let mut out = [vk::ExtensionProperties::default(); 2];
        let mut count = 2u32;
        assert_eq!(unsafe { write_out(&list, &mut count, out.as_mut_ptr()) }, vk::Result::INCOMPLETE);
        assert_eq!(count, 2);
        assert_eq!(names(&out), ["A", "B"]);
        let mut out = [vk::ExtensionProperties::default(); 4];
        let mut count = 4u32;
        assert_eq!(unsafe { write_out(&list, &mut count, out.as_mut_ptr()) }, vk::Result::SUCCESS);
        assert_eq!(count, 3);
    }
}
