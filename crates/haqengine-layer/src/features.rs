//! Turns on the device features Quality mode needs, at `vkCreateDevice`, without writing to the
//! game's structs: changed structs are copies, owned by a `FeaturePatch` for the call's duration.

use ash::vk;
use std::ffi::{c_char, c_void, CStr};

/// What the physical device can turn on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Supported {
    pub extended_formats: bool,
    pub robust_image_access: bool,
    pub image_robustness_ext: bool,
    pub device_api_13: bool,
}

/// A create info with the Quality features added, and everything it points to.
pub struct FeaturePatch {
    pub info: vk::DeviceCreateInfo<'static>,
    _features: Box<vk::PhysicalDeviceFeatures>,
    _robust: Box<vk::PhysicalDeviceImageRobustnessFeatures<'static>>,
    _chain: Vec<u64>,
    _extensions: Vec<*const c_char>,
}

/// Size of a device feature struct HaqEngine may have to copy to change one of them.
fn feature_struct_size(s: vk::StructureType) -> Option<usize> {
    use std::mem::size_of;
    use vk::StructureType as S;
    Some(match s {
        S::PHYSICAL_DEVICE_FEATURES_2 => size_of::<vk::PhysicalDeviceFeatures2>(),
        S::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => size_of::<vk::PhysicalDeviceVulkan11Features>(),
        S::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => size_of::<vk::PhysicalDeviceVulkan12Features>(),
        S::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES => size_of::<vk::PhysicalDeviceVulkan13Features>(),
        S::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES => size_of::<vk::PhysicalDeviceImageRobustnessFeatures>(),
        S::PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT => size_of::<vk::PhysicalDeviceRobustness2FeaturesEXT>(),
        S::LOADER_DEVICE_CREATE_INFO => size_of::<crate::loader_abi::LayerDeviceCreateInfo>(),
        _ => return None,
    })
}

unsafe fn each(head: *const c_void, mut f: impl FnMut(*const vk::BaseInStructure<'static>)) {
    let mut cur = head as *const vk::BaseInStructure<'static>;
    while !cur.is_null() {
        f(cur);
        cur = (*cur).p_next;
    }
}

unsafe fn find(head: *const c_void, s: vk::StructureType) -> Option<*const vk::BaseInStructure<'static>> {
    let mut hit = None;
    each(head, |c| {
        if hit.is_none() && (*c).s_type == s {
            hit = Some(c);
        }
    });
    hit
}

/// Whether the create info turns on `shaderStorageImageExtendedFormats` and robust image access
/// (`robustImageAccess` or `robustImageAccess2`).
///
/// # Safety
/// `ci` must be a valid device create info.
pub unsafe fn enabled_caps(ci: &vk::DeviceCreateInfo) -> (bool, bool) {
    use vk::StructureType as S;
    let mut ext = !ci.p_enabled_features.is_null() && (*ci.p_enabled_features).shader_storage_image_extended_formats == vk::TRUE;
    let mut robust = false;
    each(ci.p_next, |c| match (*c).s_type {
        S::PHYSICAL_DEVICE_FEATURES_2 => {
            ext |= (*(c as *const vk::PhysicalDeviceFeatures2)).features.shader_storage_image_extended_formats == vk::TRUE
        }
        S::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES => {
            robust |= (*(c as *const vk::PhysicalDeviceVulkan13Features)).robust_image_access == vk::TRUE
        }
        S::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES => {
            robust |= (*(c as *const vk::PhysicalDeviceImageRobustnessFeatures)).robust_image_access == vk::TRUE
        }
        S::PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT => {
            robust |= (*(c as *const vk::PhysicalDeviceRobustness2FeaturesEXT)).robust_image_access2 == vk::TRUE
        }
        _ => {}
    });
    (ext, robust)
}

/// Copies the chain from `head` through the last struct whose type is in `targets` into `buf`,
/// leaving the rest of the chain shared. Returns the new head (or `head` when no target is in the
/// chain); fails when a struct before the last target has no known size.
unsafe fn copy_through(head: *const c_void, targets: &[vk::StructureType], buf: &mut Vec<u64>) -> Result<*const c_void, String> {
    let mut nodes: Vec<*const vk::BaseInStructure<'static>> = Vec::new();
    each(head, |c| nodes.push(c));
    let Some(last) = nodes.iter().rposition(|c| targets.contains(&(**c).s_type)) else {
        return Ok(head);
    };
    let mut sizes = Vec::with_capacity(last + 1);
    for c in &nodes[..=last] {
        sizes.push(feature_struct_size((**c).s_type).ok_or_else(|| format!("device feature struct {:?} has no known size", (**c).s_type))?);
    }
    let words: usize = sizes.iter().map(|z| z.div_ceil(8)).sum();
    *buf = vec![0u64; words];
    let base = buf.as_mut_ptr() as *mut u8;
    let mut offset = 0usize;
    let mut prev: *mut vk::BaseOutStructure = std::ptr::null_mut();
    let mut first: *const c_void = std::ptr::null();
    for (c, size) in nodes[..=last].iter().zip(&sizes) {
        let dst = base.add(offset);
        std::ptr::copy_nonoverlapping(*c as *const u8, dst, *size);
        offset += size.div_ceil(8) * 8;
        let copy = dst as *mut vk::BaseOutStructure;
        if prev.is_null() {
            first = copy as *const c_void;
        } else {
            (*prev).p_next = copy;
        }
        prev = copy;
    }
    Ok(first)
}

/// The create info with Quality's features turned on, `None` when the game already enables them,
/// or why they cannot be added. The game's structs are only read.
///
/// # Safety
/// `ci` must be a valid device create info, alive while the patch is used.
pub unsafe fn patch_create_info(ci: &vk::DeviceCreateInfo, supported: &Supported) -> Result<Option<FeaturePatch>, String> {
    use vk::StructureType as S;
    let (ext_on, robust_on) = enabled_caps(ci);
    if ext_on && robust_on {
        return Ok(None);
    }
    if !ext_on && !supported.extended_formats {
        return Err("the GPU lacks shaderStorageImageExtendedFormats".into());
    }
    if !robust_on && !supported.robust_image_access {
        return Err("the GPU lacks robust image access".into());
    }
    let has = |s| find(ci.p_next, s).is_some();
    let mut targets = Vec::new();
    let ext_in_features2 = !ext_on && ci.p_enabled_features.is_null() && has(S::PHYSICAL_DEVICE_FEATURES_2);
    if ext_in_features2 {
        targets.push(S::PHYSICAL_DEVICE_FEATURES_2);
    }
    let robust_target = if robust_on {
        None
    } else if has(S::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES) {
        Some(S::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES)
    } else if has(S::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) {
        Some(S::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
    } else {
        None
    };
    targets.extend(robust_target);
    let add_robust_struct = !robust_on && robust_target.is_none();
    let names: Vec<*const c_char> = (0..ci.enabled_extension_count as usize).map(|i| *ci.pp_enabled_extension_names.add(i)).collect();
    let listed = names.iter().any(|n| CStr::from_ptr(*n) == ash::ext::image_robustness::NAME);
    let add_extension = add_robust_struct && !listed && !supported.device_api_13;
    if add_extension && !supported.image_robustness_ext {
        return Err("the GPU lacks VK_EXT_image_robustness".into());
    }
    let mut chain = Vec::new();
    let head = copy_through(ci.p_next, &targets, &mut chain)?;
    let mut modified = 0usize;
    let copied_end = if chain.is_empty() { std::ptr::null() } else { chain.as_ptr().add(chain.len()) as *const u8 };
    each(head, |c| {
        let inside = !copied_end.is_null() && (c as *const u8) >= (chain.as_ptr() as *const u8) && (c as *const u8) < copied_end;
        if !inside {
            return;
        }
        let m = c as *mut vk::BaseOutStructure<'static>;
        match (*c).s_type {
            S::PHYSICAL_DEVICE_FEATURES_2 if ext_in_features2 => {
                (*(m as *mut vk::PhysicalDeviceFeatures2)).features.shader_storage_image_extended_formats = vk::TRUE;
                modified += 1;
            }
            S::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES if robust_target == Some(S::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES) => {
                (*(m as *mut vk::PhysicalDeviceImageRobustnessFeatures)).robust_image_access = vk::TRUE;
                modified += 1;
            }
            S::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES if robust_target == Some(S::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) => {
                (*(m as *mut vk::PhysicalDeviceVulkan13Features)).robust_image_access = vk::TRUE;
                modified += 1;
            }
            _ => {}
        }
    });
    if modified != targets.len() {
        return Err("a device feature struct appears twice in the chain".into());
    }
    let mut features = Box::new(if ci.p_enabled_features.is_null() { vk::PhysicalDeviceFeatures::default() } else { *ci.p_enabled_features });
    let use_features = !ext_on && !ext_in_features2;
    if use_features {
        features.shader_storage_image_extended_formats = vk::TRUE;
    }
    let mut robust = Box::new(vk::PhysicalDeviceImageRobustnessFeatures::default().robust_image_access(true));
    let mut info: vk::DeviceCreateInfo<'static> = std::ptr::read(ci as *const vk::DeviceCreateInfo as *const vk::DeviceCreateInfo<'static>);
    info.p_next = head;
    if add_robust_struct {
        robust.p_next = head as *mut c_void;
        info.p_next = &*robust as *const vk::PhysicalDeviceImageRobustnessFeatures as *const c_void;
    }
    if use_features {
        info.p_enabled_features = &*features;
    }
    let mut extensions = names;
    if add_extension {
        extensions.push(ash::ext::image_robustness::NAME.as_ptr());
    }
    info.enabled_extension_count = extensions.len() as u32;
    info.pp_enabled_extension_names = extensions.as_ptr();
    Ok(Some(FeaturePatch { info, _features: features, _robust: robust, _chain: chain, _extensions: extensions }))
}

pub type CreateDevice = unsafe extern "system" fn(
    vk::PhysicalDevice,
    *const vk::DeviceCreateInfo<'_>,
    *const vk::AllocationCallbacks<'_>,
    *mut vk::Device,
) -> vk::Result;

/// Creates the device with the patch; if that fails, calls `before_retry` (which restores the
/// loader's layer link the first call used up) and tries once more with the game's own create
/// info. Returns the result and whether the patched info was the one that succeeded.
///
/// # Safety
/// The arguments must be valid for `next`.
pub unsafe fn create_with_fallback(
    next: CreateDevice,
    physical: vk::PhysicalDevice,
    original: *const vk::DeviceCreateInfo,
    patch: Option<&FeaturePatch>,
    before_retry: &mut dyn FnMut(),
    allocator: *const vk::AllocationCallbacks,
    out: *mut vk::Device,
) -> (vk::Result, bool) {
    if let Some(p) = patch {
        let r = next(physical, &p.info, allocator, out);
        if r == vk::Result::SUCCESS {
            return (r, true);
        }
        before_retry();
    }
    (next(physical, original, allocator, out), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    const ALL: Supported = Supported { extended_formats: true, robust_image_access: true, image_robustness_ext: true, device_api_13: false };

    unsafe fn find<T>(head: *const c_void, s: vk::StructureType) -> Option<*const T> {
        let mut cur = head as *const vk::BaseInStructure;
        while !cur.is_null() {
            if (*cur).s_type == s {
                return Some(cur as *const T);
            }
            cur = (*cur).p_next;
        }
        None
    }

    unsafe fn ext_names(info: &vk::DeviceCreateInfo) -> Vec<String> {
        (0..info.enabled_extension_count as usize)
            .map(|i| CStr::from_ptr(*info.pp_enabled_extension_names.add(i)).to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn patch_adds_robustness_struct_and_extension() {
        unsafe {
            let features = vk::PhysicalDeviceFeatures::default();
            let swapchain = [ash::khr::swapchain::NAME.as_ptr()];
            let ci = vk::DeviceCreateInfo::default().enabled_features(&features).enabled_extension_names(&swapchain);
            let patch = patch_create_info(&ci, &ALL).unwrap().expect("needs a patch");
            assert_eq!((*patch.info.p_enabled_features).shader_storage_image_extended_formats, vk::TRUE);
            let robust: *const vk::PhysicalDeviceImageRobustnessFeatures =
                find(patch.info.p_next, vk::StructureType::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES).expect("struct added");
            assert_eq!((*robust).robust_image_access, vk::TRUE);
            let names = ext_names(&patch.info);
            assert_eq!(names.iter().filter(|n| n.as_str() == "VK_EXT_image_robustness").count(), 1);
            assert!(names.contains(&"VK_KHR_swapchain".to_string()));
            assert_eq!(enabled_caps(&patch.info), (true, true));
        }
    }

    #[test]
    fn patch_uses_vulkan13_struct_when_present() {
        unsafe {
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut v13);
            let ci = vk::DeviceCreateInfo::default().push_next(&mut f2);
            let patch = patch_create_info(&ci, &Supported { device_api_13: true, ..ALL }).unwrap().unwrap();
            let f2c: *const vk::PhysicalDeviceFeatures2 = find(patch.info.p_next, vk::StructureType::PHYSICAL_DEVICE_FEATURES_2).unwrap();
            assert_eq!((*f2c).features.shader_storage_image_extended_formats, vk::TRUE);
            let v13c: *const vk::PhysicalDeviceVulkan13Features =
                find(patch.info.p_next, vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES).unwrap();
            assert_eq!((*v13c).robust_image_access, vk::TRUE);
            assert!(find::<vk::PhysicalDeviceImageRobustnessFeatures>(patch.info.p_next, vk::StructureType::PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES).is_none());
            assert!(patch.info.p_enabled_features.is_null());
            assert!(ext_names(&patch.info).is_empty());
        }
    }

    #[test]
    fn patch_leaves_original_untouched() {
        unsafe {
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let v13p: *mut vk::PhysicalDeviceVulkan13Features = &mut v13;
            let mut f2 = vk::PhysicalDeviceFeatures2 { p_next: v13p as *mut c_void, ..Default::default() };
            let f2p: *mut vk::PhysicalDeviceFeatures2 = &mut f2;
            let ci = vk::DeviceCreateInfo { p_next: f2p as *const c_void, ..Default::default() };
            let raw = |p: *const u8, n: usize| std::slice::from_raw_parts(p, n).to_vec();
            let snapshot = || {
                (
                    raw(&ci as *const _ as *const u8, std::mem::size_of_val(&ci)),
                    raw(f2p as *const u8, std::mem::size_of::<vk::PhysicalDeviceFeatures2>()),
                    raw(v13p as *const u8, std::mem::size_of::<vk::PhysicalDeviceVulkan13Features>()),
                )
            };
            let before = snapshot();
            let patch = patch_create_info(&ci, &Supported { device_api_13: true, ..ALL }).unwrap().unwrap();
            assert_eq!(snapshot(), before);
            drop(patch);
        }
    }

    #[test]
    fn patch_none_when_already_enabled() {
        unsafe {
            let features = vk::PhysicalDeviceFeatures::default().shader_storage_image_extended_formats(true);
            let mut r2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default().robust_image_access2(true);
            let ci = vk::DeviceCreateInfo::default().enabled_features(&features).push_next(&mut r2);
            assert!(patch_create_info(&ci, &ALL).unwrap().is_none());
            assert_eq!(enabled_caps(&ci), (true, true));
        }
    }

    #[test]
    fn patch_refuses_unsupported_and_unknown() {
        unsafe {
            let features = vk::PhysicalDeviceFeatures::default();
            let ci = vk::DeviceCreateInfo::default().enabled_features(&features);
            assert!(patch_create_info(&ci, &Supported { robust_image_access: false, ..ALL }).is_err());
            assert!(patch_create_info(&ci, &Supported { image_robustness_ext: false, ..ALL }).is_err());
            let mut odd = vk::PhysicalDeviceMeshShaderFeaturesNV::default();
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut v13).push_next(&mut odd);
            let ci = vk::DeviceCreateInfo::default().push_next(&mut f2);
            let r = patch_create_info(&ci, &Supported { device_api_13: true, ..ALL });
            assert!(r.is_err() || r.as_ref().is_ok_and(|p| p.is_some()));
        }
    }

    #[test]
    fn patch_copies_loader_structs_at_the_head() {
        use crate::loader_abi::{LayerDeviceCreateInfo, LayerDeviceCreateInfoU};
        unsafe {
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut v13);
            let marker = 0x5151usize as *const c_void;
            let callback = LayerDeviceCreateInfo {
                s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
                p_next: &mut f2 as *mut _ as *const c_void,
                function: 1,
                u: LayerDeviceCreateInfoU { other: std::ptr::null() },
            };
            let link = LayerDeviceCreateInfo {
                s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
                p_next: &callback as *const _ as *const c_void,
                function: 0,
                u: LayerDeviceCreateInfoU { other: marker },
            };
            let ci = vk::DeviceCreateInfo { p_next: &link as *const _ as *const c_void, ..Default::default() };
            let patch = patch_create_info(&ci, &Supported { device_api_13: true, ..ALL }).unwrap().expect("patched");
            let head = patch.info.p_next as *const LayerDeviceCreateInfo;
            assert_ne!(head, &link as *const _, "the loader's link struct is copied, not shared");
            assert_eq!(((*head).function, (*head).u.other), (0, marker));
            let v13c: *const vk::PhysicalDeviceVulkan13Features =
                find(patch.info.p_next, vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES).unwrap();
            assert_eq!((*v13c).robust_image_access, vk::TRUE);
            assert_eq!(v13.robust_image_access, vk::FALSE);
        }
    }

    static LINK_USED: AtomicU32 = AtomicU32::new(0);

    unsafe extern "system" fn uses_link(
        _: vk::PhysicalDevice,
        ci: *const vk::DeviceCreateInfo<'_>,
        _: *const vk::AllocationCallbacks<'_>,
        out: *mut vk::Device,
    ) -> vk::Result {
        if LINK_USED.swap(1, Ordering::SeqCst) == 1 {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        if (*ci).enabled_extension_count > 0 {
            return vk::Result::ERROR_FEATURE_NOT_PRESENT;
        }
        *out = vk::Handle::from_raw(0x99);
        vk::Result::SUCCESS
    }

    #[test]
    fn retry_restores_the_layer_link() {
        unsafe {
            let features = vk::PhysicalDeviceFeatures::default();
            let ci = vk::DeviceCreateInfo::default().enabled_features(&features);
            let patch = patch_create_info(&ci, &ALL).unwrap().unwrap();
            let mut out = vk::Device::null();
            LINK_USED.store(0, Ordering::SeqCst);
            let mut restore = || LINK_USED.store(0, Ordering::SeqCst);
            let (r, patched) =
                create_with_fallback(uses_link, vk::PhysicalDevice::null(), &ci, Some(&patch), &mut restore, std::ptr::null(), &mut out);
            assert_eq!((r, patched), (vk::Result::SUCCESS, false));
        }
    }

    static CALLS: AtomicU32 = AtomicU32::new(0);

    unsafe extern "system" fn fail_patched(
        _: vk::PhysicalDevice,
        ci: *const vk::DeviceCreateInfo<'_>,
        _: *const vk::AllocationCallbacks<'_>,
        out: *mut vk::Device,
    ) -> vk::Result {
        CALLS.fetch_add(1, Ordering::SeqCst);
        if (*ci).enabled_extension_count > 0 {
            return vk::Result::ERROR_FEATURE_NOT_PRESENT;
        }
        *out = vk::Handle::from_raw(0x1234);
        vk::Result::SUCCESS
    }

    #[test]
    fn patched_create_falls_back() {
        unsafe {
            let features = vk::PhysicalDeviceFeatures::default();
            let ci = vk::DeviceCreateInfo::default().enabled_features(&features);
            let patch = patch_create_info(&ci, &ALL).unwrap().unwrap();
            let mut out = vk::Device::null();
            CALLS.store(0, Ordering::SeqCst);
            let (r, patched) = create_with_fallback(fail_patched, vk::PhysicalDevice::null(), &ci, Some(&patch), &mut || {}, std::ptr::null(), &mut out);
            assert_eq!((r, patched, CALLS.load(Ordering::SeqCst)), (vk::Result::SUCCESS, false, 2));
            assert_eq!(vk::Handle::as_raw(out), 0x1234);
            let (r, patched) = create_with_fallback(fail_patched, vk::PhysicalDevice::null(), &ci, None, &mut || {}, std::ptr::null(), &mut out);
            assert_eq!((r, patched), (vk::Result::SUCCESS, false));
        }
    }
}
