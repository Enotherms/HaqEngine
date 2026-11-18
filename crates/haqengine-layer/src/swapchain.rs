//! Swapchain and present pass-through. M0 only counts presents and forwards everything unchanged.

use crate::guard::guard;
use crate::chain::{present_ext_size, swapchain_ext_size, ChainCopy, PrefixCopy};
use crate::framegen::FrameGen;
use crate::pipelines::packed_format;
use haqengine_core::Mode;
use crate::plan::plan_swapchain_with;
use crate::state::{device_can_transfer, dispatch_key, multiplier, DeviceData, DEVICES, LOG};
use ash::vk;
use ash::vk::Handle;
use haqengine_core::HandleMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Once};

pub struct SwapchainInfo {
    pub id: u64,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub images: u32,
    pub presents: AtomicU64,
    /// The device that created it, used when a present arrives on a queue we can't map.
    pub device: Option<Arc<DeviceData>>,
    /// Generation resources, when this swapchain generates frames.
    pub gen: Option<crate::framegen::FrameGen>,
    /// Created with the game's present-mode list rewritten to `[FIFO]`; presents must stay FIFO.
    pub fifo_only_modes: bool,
}

/// Live swapchains, keyed by raw handle.
pub struct SwapchainRegistry {
    map: HandleMap<Arc<SwapchainInfo>>,
}

impl SwapchainRegistry {
    pub fn new() -> Self {
        SwapchainRegistry { map: HandleMap::new() }
    }
    pub fn created(&self, handle: u64, info: SwapchainInfo) {
        self.map.insert(handle as usize, Arc::new(info));
    }
    /// Counts one present. No allocation: an `Arc` clone and an atomic add.
    pub fn presented(&self, handle: u64) {
        if let Some(info) = self.map.get(handle as usize) {
            info.presents.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn destroyed(&self, handle: u64) -> Option<Arc<SwapchainInfo>> {
        self.map.remove(handle as usize)
    }
    /// A live swapchain's record.
    pub fn get(&self, handle: u64) -> Option<Arc<SwapchainInfo>> {
        self.map.get(handle as usize)
    }
    /// The device that created a live swapchain.
    pub fn device_of(&self, handle: u64) -> Option<Arc<DeviceData>> {
        self.map.get(handle as usize).and_then(|info| info.device.clone())
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub unsafe extern "system" fn create_swapchain(
    device: vk::Device,
    create_info: *const vk::SwapchainCreateInfoKHR,
    allocator: *const vk::AllocationCallbacks,
    out: *mut vk::SwapchainKHR,
) -> vk::Result {
    guard("vkCreateSwapchainKHR", vk::Result::ERROR_INITIALIZATION_FAILED, || unsafe {
        create_swapchain_inner(device, create_info, allocator, out)
    })
}

pub unsafe extern "system" fn destroy_swapchain(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    allocator: *const vk::AllocationCallbacks,
) {
    guard("vkDestroySwapchainKHR", (), || unsafe { destroy_swapchain_inner(device, swapchain, allocator) })
}

pub unsafe extern "system" fn queue_present(queue: vk::Queue, present_info: *const vk::PresentInfoKHR) -> vk::Result {
    guard("vkQueuePresentKHR", vk::Result::ERROR_DEVICE_LOST, || unsafe { queue_present_inner(queue, present_info) })
}
static NO_DEVICE_REPORTED: Once = Once::new();
static MODE_INFO_UNREWRITTEN: Once = Once::new();

/// The first non-FIFO mode a present's `VkSwapchainPresentModeInfoEXT` asks for on a swapchain
/// created with a FIFO-only list, read in place without copying the chain.
///
/// # Safety
/// `pi` must be a valid present.
pub unsafe fn non_fifo_mode_on_fifo_only(pi: &vk::PresentInfoKHR) -> Option<vk::PresentModeKHR> {
    let mut cur = pi.p_next as *const vk::BaseInStructure;
    while !cur.is_null() && (*cur).s_type != vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT {
        cur = (*cur).p_next;
    }
    if cur.is_null() || pi.p_swapchains.is_null() {
        return None;
    }
    let info = &*(cur as *const vk::SwapchainPresentModeInfoEXT);
    if info.p_present_modes.is_null() {
        return None;
    }
    let n = (info.swapchain_count as usize).min(pi.swapchain_count as usize);
    let modes = std::slice::from_raw_parts(info.p_present_modes, n);
    let swapchains = std::slice::from_raw_parts(pi.p_swapchains, n);
    swapchains.iter().zip(modes).find_map(|(sc, mode)| {
        let fifo_only = SWAPCHAINS.get(sc.as_raw()).is_some_and(|s| s.fifo_only_modes);
        (fifo_only && *mode != vk::PresentModeKHR::FIFO).then_some(*mode)
    })
}

/// The structure types in a `pNext` chain, for diagnostics (at most 16).
unsafe fn chain_types(p_next: *const std::ffi::c_void) -> Vec<i32> {
    let mut out = Vec::new();
    let mut cur = p_next as *const vk::BaseInStructure;
    while !cur.is_null() && out.len() < 16 {
        out.push((*cur).s_type.as_raw());
        cur = (*cur).p_next;
    }
    out
}

/// FIFO repeated, for rewritten present-mode lists (create: 1 entry; present: one per swapchain).
pub static FIFO_MODES: [vk::PresentModeKHR; 16] = [vk::PresentModeKHR::FIFO; 16];

/// The present modes listed in a chained `VkSwapchainPresentModesCreateInfoEXT`, if any.
unsafe fn game_present_modes<'a>(p_next: *const std::ffi::c_void) -> Option<&'a [vk::PresentModeKHR]> {
    let mut cur = p_next as *const vk::BaseInStructure;
    while !cur.is_null() {
        if (*cur).s_type == vk::StructureType::SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT {
            let info = &*(cur as *const vk::SwapchainPresentModesCreateInfoEXT);
            if info.p_present_modes.is_null() {
                return Some(&[]);
            }
            return Some(std::slice::from_raw_parts(info.p_present_modes, info.present_mode_count as usize));
        }
        cur = (*cur).p_next;
    }
    None
}

unsafe fn create_swapchain_inner(
    device: vk::Device,
    create_info: *const vk::SwapchainCreateInfoKHR,
    allocator: *const vk::AllocationCallbacks,
    out: *mut vk::SwapchainKHR,
) -> vk::Result {
    let Some(data) = DEVICES.get(dispatch_key(device)) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let ci = &*create_info;
    let mut chain = ChainCopy::new();
    let mut rewritten_head: Option<*const std::ffi::c_void> = None;
    let mut rewrite_unsupported = false;
    if multiplier() >= 2 && game_present_modes(ci.p_next).is_some() {
        match chain.copy_prefix(ci.p_next, vk::StructureType::SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT, swapchain_ext_size) {
            PrefixCopy::Copied { head, target } => {
                let list = &mut *(target as *mut vk::SwapchainPresentModesCreateInfoEXT);
                list.present_mode_count = 1;
                list.p_present_modes = FIFO_MODES.as_ptr();
                rewritten_head = Some(head);
            }
            PrefixCopy::Unsupported => rewrite_unsupported = true,
            PrefixCopy::NotFound => {}
        }
    }
    let plan = guard("swapchain plan", None, || unsafe {
        let mut caps = vk::SurfaceCapabilitiesKHR::default();
        let r = (data.instance.surface.get_physical_device_surface_capabilities_khr)(
            data.physical_device,
            ci.surface,
            &mut caps,
        );
        let game_modes = if rewritten_head.is_some() { Some(&FIFO_MODES[..1]) } else { game_present_modes(ci.p_next) };
        (r == vk::Result::SUCCESS)
            .then(|| plan_swapchain_with(ci, &caps, multiplier(), device_can_transfer(&data), game_modes))
    });
    let mut planned = *ci;
    let generating = plan.is_some_and(|p| p.generate);
    if let Some(p) = plan.filter(|p| p.generate) {
        planned.min_image_count = p.min_image_count;
        planned.image_usage = p.image_usage;
        planned.present_mode = p.present_mode;
        if let Some(head) = rewritten_head {
            planned.p_next = head;
        }
    }
    let fifo_only_modes = generating && rewritten_head.is_some();
    let result = (data.swapchain.create_swapchain_khr)(device, &planned, allocator, out);
    if result == vk::Result::SUCCESS {
        guard("swapchain bookkeeping", (), || unsafe {
            let mut count = 0u32;
            let _ = (data.swapchain.get_swapchain_images_khr)(device, *out, &mut count, std::ptr::null_mut());
            let mut images = vec![vk::Image::null(); count as usize];
            let r = (data.swapchain.get_swapchain_images_khr)(device, *out, &mut count, images.as_mut_ptr());
            images.truncate(count as usize);
            let reason = match plan {
                None => "surface capabilities unavailable",
                Some(p) => p.reason,
            };
            let generate = plan.is_some_and(|p| p.generate) && r == vk::Result::SUCCESS && !images.is_empty();
            let (battery_format, engine) = match (crate::state::SETTINGS.mode, packed_format(ci.image_format)) {
                (Mode::Copy, _) => (None, " copy (mode copy)".to_string()),
                (Mode::Battery | Mode::Quality, Some(f)) => (Some(f), " battery".to_string()),
                (Mode::Battery | Mode::Quality, None) => (None, format!(" copy (format {:?} not supported)", ci.image_format)),
            };
            let engine = if generate { engine } else { String::new() };
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            LOG.log(&format!(
                "swapchain {id}: {}x{} format {:?} colour {:?} images {count} mode {:?} gen {reason}{engine}{}",
                ci.image_extent.width,
                ci.image_extent.height,
                ci.image_format,
                ci.image_color_space,
                planned.present_mode,
                game_present_modes(ci.p_next)
                    .map(|m| match (fifo_only_modes, rewrite_unsupported) {
                        (true, _) => format!(" (game modes {m:?} -> [FIFO])"),
                        (false, true) => format!(" (game modes {m:?}; cannot rewrite present modes: unknown struct)"),
                        (false, false) => format!(" (game modes {m:?})"),
                    })
                    .unwrap_or_default()
            ));
            SWAPCHAINS.created(
                (*out).as_raw(),
                SwapchainInfo {
                    id,
                    extent: ci.image_extent,
                    format: ci.image_format,
                    images: count,
                    presents: AtomicU64::new(0),
                    device: Some(Arc::clone(&data)),
                    gen: generate.then(|| FrameGen::new(images, ci.image_extent, battery_format)),
                    fifo_only_modes,
                },
            );
        });
    }
    result
}

unsafe fn destroy_swapchain_inner(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    allocator: *const vk::AllocationCallbacks,
) {
    guard("swapchain destroy log", (), || unsafe {
        if let Some(info) = SWAPCHAINS.destroyed(swapchain.as_raw()) {
            if let (Some(gen), Some(dev)) = (info.gen.as_ref(), info.device.as_ref()) {
                gen.destroy(dev);
            }
            match info.gen.as_ref() {
                Some(gen) => {
                    let (real, generated, skipped) = gen.counts();
                    LOG.log(&format!(
                        "swapchain {} destroyed after {real} real + {generated} generated presents, {skipped} skipped",
                        info.id
                    ));
                }
                None => LOG.log(&format!(
                    "swapchain {} destroyed after {} presents",
                    info.id,
                    info.presents.load(Ordering::Relaxed)
                )),
            }
        }
    });
    if let Some(data) = DEVICES.get(dispatch_key(device)) {
        (data.swapchain.destroy_swapchain_khr)(device, swapchain, allocator);
    }
}

unsafe fn queue_present_inner(queue: vk::Queue, present_info: *const vk::PresentInfoKHR) -> vk::Result {
    let start = std::time::Instant::now();
    guard("present counting", (), || unsafe {
        let pi = &*present_info;
        if !pi.p_swapchains.is_null() {
            for sc in std::slice::from_raw_parts(pi.p_swapchains, pi.swapchain_count as usize) {
                SWAPCHAINS.presented(sc.as_raw());
            }
        }
    });
    let device = DEVICES.get(dispatch_key(queue)).or_else(|| {
        let pi = &*present_info;
        if pi.swapchain_count == 0 || pi.p_swapchains.is_null() {
            return None;
        }
        SWAPCHAINS.device_of((*pi.p_swapchains).as_raw())
    });
    match device {
        Some(data) => {
            let mut chain = ChainCopy::new();
            let fifo_only = any_fifo_only(&*present_info);
            let rewritten = if fifo_only { fifo_present_info(&*present_info, &mut chain) } else { None };
            if fifo_only && rewritten.is_none() {
                if let Some(mode) = non_fifo_mode_on_fifo_only(&*present_info) {
                    MODE_INFO_UNREWRITTEN.call_once(|| {
                        LOG.log(&format!(
                            "warning: present asks for {mode:?} on a FIFO-only swapchain and could not be rewritten; chain {:?}",
                            chain_types((*present_info).p_next)
                        ))
                    });
                }
            }
            let present_info: *const vk::PresentInfoKHR = match rewritten.as_ref() {
                Some(pi) => pi,
                None => present_info,
            };
            let pi = &*present_info;
            if crate::framegen::present_shape_ok(pi) {
                if let Some(info) = SWAPCHAINS.get((*pi.p_swapchains).as_raw()) {
                    if let Some(gen) = info.gen.as_ref().filter(|g| !g.disabled.load(Ordering::Relaxed)) {
                        return gen.present(info.id, &data, queue, pi, start);
                    }
                }
            }
            (data.swapchain.queue_present_khr)(queue, present_info)
        }
        None => {
            NO_DEVICE_REPORTED.call_once(|| LOG.log("error: present on an unknown device"));
            vk::Result::ERROR_DEVICE_LOST
        }
    }
}

/// True when any swapchain in the present was created with a rewritten present-mode list.
unsafe fn any_fifo_only(pi: &vk::PresentInfoKHR) -> bool {
    if pi.p_swapchains.is_null() {
        return false;
    }
    std::slice::from_raw_parts(pi.p_swapchains, pi.swapchain_count as usize)
        .iter()
        .any(|sc| SWAPCHAINS.get(sc.as_raw()).is_some_and(|info| info.fifo_only_modes))
}

/// `pi` with its `VkSwapchainPresentModeInfoEXT` copied and set to FIFO, or `None` when it has none
/// or the chain cannot be copied.
///
/// # Safety
/// `pi` must be a valid present; the result borrows `chain`.
pub unsafe fn fifo_present_info<'a>(pi: &vk::PresentInfoKHR<'a>, chain: &'a mut ChainCopy) -> Option<vk::PresentInfoKHR<'a>> {
    match chain.copy_prefix(pi.p_next, vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT, present_ext_size) {
        PrefixCopy::Copied { head, target } => {
            let info = &mut *(target as *mut vk::SwapchainPresentModeInfoEXT);
            let n = info.swapchain_count as usize;
            if n > chain.modes.len() || n != pi.swapchain_count as usize || info.p_present_modes.is_null() {
                return None;
            }
            let game_modes = std::slice::from_raw_parts(info.p_present_modes, n);
            let swapchains = std::slice::from_raw_parts(pi.p_swapchains, n);
            for (i, (sc, mode)) in swapchains.iter().zip(game_modes).enumerate() {
                let fifo_only = SWAPCHAINS.get(sc.as_raw()).is_some_and(|s| s.fifo_only_modes);
                chain.modes[i] = if fifo_only { vk::PresentModeKHR::FIFO } else { *mode };
            }
            info.p_present_modes = chain.modes.as_ptr();
            let mut out = *pi;
            out.p_next = head;
            Some(out)
        }
        PrefixCopy::NotFound | PrefixCopy::Unsupported => None,
    }
}

impl Default for SwapchainRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub static SWAPCHAINS: LazyLock<SwapchainRegistry> = LazyLock::new(SwapchainRegistry::new);

#[cfg(test)]
mod tests {
    use super::*;

    fn info(id: u64) -> SwapchainInfo {
        SwapchainInfo {
            id,
            extent: vk::Extent2D { width: 1280, height: 800 },
            format: vk::Format::B8G8R8A8_UNORM,
            images: 3,
            presents: AtomicU64::new(0),
            device: None,
            gen: None,
            fifo_only_modes: false,
        }
    }

    static CAPTURED: std::sync::Mutex<Option<(u32, vk::ImageUsageFlags, vk::PresentModeKHR)>> =
        std::sync::Mutex::new(None);
    /// Serialises tests that set the global multiplier override.
    static MULTIPLIER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static CAPTURED_MODES: std::sync::Mutex<Option<(vk::PresentModeKHR, Vec<vk::PresentModeKHR>)>> =
        std::sync::Mutex::new(None);

    unsafe extern "system" fn capture_create_modes(
        _: vk::Device,
        ci: *const vk::SwapchainCreateInfoKHR,
        _: *const vk::AllocationCallbacks,
        out: *mut vk::SwapchainKHR,
    ) -> vk::Result {
        let ci = unsafe { &*ci };
        let modes = unsafe { game_present_modes(ci.p_next) }.map(|m| m.to_vec()).unwrap_or_default();
        *CAPTURED_MODES.lock().unwrap() = Some((ci.present_mode, modes));
        unsafe { *out = vk::SwapchainKHR::from_raw(0x5c_5555) };
        vk::Result::SUCCESS
    }

    unsafe extern "system" fn capture_create(
        _: vk::Device,
        ci: *const vk::SwapchainCreateInfoKHR,
        _: *const vk::AllocationCallbacks,
        out: *mut vk::SwapchainKHR,
    ) -> vk::Result {
        let ci = unsafe { &*ci };
        *CAPTURED.lock().unwrap() = Some((ci.min_image_count, ci.image_usage, ci.present_mode));
        unsafe { *out = vk::SwapchainKHR::from_raw(0x5c_4444) };
        vk::Result::SUCCESS
    }

    unsafe extern "system" fn two_images(
        _: vk::Device,
        _: vk::SwapchainKHR,
        count: *mut u32,
        images: *mut vk::Image,
    ) -> vk::Result {
        unsafe {
            if !images.is_null() {
                *images = vk::Image::from_raw(0x1);
                *images.add(1) = vk::Image::from_raw(0x2);
            }
            *count = 2;
        }
        vk::Result::SUCCESS
    }

    unsafe extern "system" fn all_caps(
        _: vk::PhysicalDevice,
        _: vk::SurfaceKHR,
        caps: *mut vk::SurfaceCapabilitiesKHR,
    ) -> vk::Result {
        unsafe {
            *caps = vk::SurfaceCapabilitiesKHR {
                min_image_count: 2,
                max_image_count: 8,
                supported_usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
                ..Default::default()
            };
        }
        vk::Result::SUCCESS
    }

    #[test]
    fn create_swapchain_applies_plan() {
        use std::ffi::{c_void, CStr};
        let load = |name: &CStr| -> *const c_void {
            match name.to_bytes() {
                b"vkCreateSwapchainKHR" => capture_create as *const c_void,
                b"vkGetSwapchainImagesKHR" => two_images as *const c_void,
                _ => std::ptr::null(),
            }
        };
        let instance_load = |name: &CStr| -> *const c_void {
            if name == c"vkGetPhysicalDeviceSurfaceCapabilitiesKHR" { all_caps as *const c_void } else { std::ptr::null() }
        };
        let families = vec![vk::QueueFamilyProperties { queue_flags: vk::QueueFlags::GRAPHICS, ..Default::default() }];
        let (device, _) = crate::state::test_support::fake_device_full(
            load,
            families,
            crate::state::test_support::gdpa_nothing,
            instance_load,
        );
        let _lock = MULTIPLIER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::state::set_test_multiplier(2);
        let ci = vk::SwapchainCreateInfoKHR {
            min_image_count: 3,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            present_mode: vk::PresentModeKHR::MAILBOX,
            image_extent: vk::Extent2D { width: 1280, height: 800 },
            ..Default::default()
        };
        let mut out = vk::SwapchainKHR::null();
        let result = unsafe { create_swapchain(device, &ci, std::ptr::null(), &mut out) };
        crate::state::set_test_multiplier(0);
        let info = SWAPCHAINS.destroyed(out.as_raw());
        DEVICES.remove(dispatch_key(device));
        assert_eq!(result, vk::Result::SUCCESS);
        let (min, usage, mode) = CAPTURED.lock().unwrap().take().unwrap();
        assert_eq!(mode, vk::PresentModeKHR::FIFO);
        assert_eq!(min, 4);
        assert!(usage.contains(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST));
        let gen = info.unwrap().gen.as_ref().map(|g| g.images.len());
        assert_eq!(gen, Some(2));
    }

    static FAKE_PRESENTS: AtomicU64 = AtomicU64::new(0);

    unsafe extern "system" fn fake_present(_: vk::Queue, _: *const vk::PresentInfoKHR) -> vk::Result {
        FAKE_PRESENTS.fetch_add(1, Ordering::SeqCst);
        vk::Result::SUCCESS
    }

    /// A device whose next layer only provides `vkQueuePresentKHR`, and a queue handle whose
    /// dispatch key is not in `DEVICES`.
    fn fake_device_and_stray_queue() -> (Arc<DeviceData>, vk::Queue) {
        let load = |name: &std::ffi::CStr| {
            if name == c"vkQueuePresentKHR" { fake_present as *const std::ffi::c_void } else { std::ptr::null() }
        };
        let (handle, data) = crate::state::test_support::fake_device(load, vec![]);
        DEVICES.remove(dispatch_key(handle));
        (data, vk::Queue::from_raw(crate::state::test_support::fake_handle()))
    }

    #[test]
    fn present_on_unmapped_queue_uses_the_swapchains_device() {
        let (device, queue) = fake_device_and_stray_queue();
        let handle = 0x5c_0001u64;
        SWAPCHAINS.created(handle, SwapchainInfo { device: Some(device), ..info(100) });
        let swapchain = vk::SwapchainKHR::from_raw(handle);
        let index = 0u32;
        let present = vk::PresentInfoKHR {
            swapchain_count: 1,
            p_swapchains: &swapchain,
            p_image_indices: &index,
            ..Default::default()
        };
        let before = FAKE_PRESENTS.load(Ordering::SeqCst);
        let result = unsafe { queue_present(queue, &present) };
        SWAPCHAINS.destroyed(handle);
        assert_eq!(result, vk::Result::SUCCESS);
        assert_eq!(FAKE_PRESENTS.load(Ordering::SeqCst), before + 1);
    }

    #[test]
    fn create_swapchain_rewrites_present_modes() {
        use std::ffi::{c_void, CStr};
        let load = |name: &CStr| -> *const c_void {
            match name.to_bytes() {
                b"vkCreateSwapchainKHR" => capture_create_modes as *const c_void,
                b"vkGetSwapchainImagesKHR" => two_images as *const c_void,
                _ => std::ptr::null(),
            }
        };
        let instance_load = |name: &CStr| -> *const c_void {
            if name == c"vkGetPhysicalDeviceSurfaceCapabilitiesKHR" { all_caps as *const c_void } else { std::ptr::null() }
        };
        let families = vec![vk::QueueFamilyProperties { queue_flags: vk::QueueFlags::GRAPHICS, ..Default::default() }];
        let (device, _) = crate::state::test_support::fake_device_full(
            load,
            families,
            crate::state::test_support::gdpa_nothing,
            instance_load,
        );
        let game_modes = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::IMMEDIATE];
        let list = vk::SwapchainPresentModesCreateInfoEXT {
            present_mode_count: 2,
            p_present_modes: game_modes.as_ptr(),
            ..Default::default()
        };
        let ci = vk::SwapchainCreateInfoKHR {
            p_next: &list as *const _ as *const c_void,
            min_image_count: 3,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            present_mode: vk::PresentModeKHR::FIFO,
            image_extent: vk::Extent2D { width: 1280, height: 800 },
            ..Default::default()
        };
        let _lock = MULTIPLIER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::state::set_test_multiplier(2);
        let mut out = vk::SwapchainKHR::null();
        let result = unsafe { create_swapchain(device, &ci, std::ptr::null(), &mut out) };
        crate::state::set_test_multiplier(0);
        let info = SWAPCHAINS.destroyed(out.as_raw());
        DEVICES.remove(dispatch_key(device));
        assert_eq!(result, vk::Result::SUCCESS);
        let (mode, modes) = CAPTURED_MODES.lock().unwrap().take().unwrap();
        assert_eq!(mode, vk::PresentModeKHR::FIFO);
        assert_eq!(modes, [vk::PresentModeKHR::FIFO]);
        assert_eq!(list.present_mode_count, 2, "the game's struct must not change");
        assert_eq!(game_modes, [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::IMMEDIATE]);
        let info = info.expect("registered");
        assert!(info.gen.is_some());
        assert!(info.fifo_only_modes);
    }

    static PRESENTED_MODES: std::sync::Mutex<Vec<(u64, Vec<vk::PresentModeKHR>)>> = std::sync::Mutex::new(Vec::new());

    unsafe extern "system" fn capture_present_modes(_: vk::Queue, pi: *const vk::PresentInfoKHR) -> vk::Result {
        unsafe {
            let pi = &*pi;
            let sc = (*pi.p_swapchains).as_raw();
            let mut cur = pi.p_next as *const vk::BaseInStructure;
            let mut modes = Vec::new();
            while !cur.is_null() {
                if (*cur).s_type == vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT {
                    let m = &*(cur as *const vk::SwapchainPresentModeInfoEXT);
                    modes = std::slice::from_raw_parts(m.p_present_modes, m.swapchain_count as usize).to_vec();
                }
                cur = (*cur).p_next;
            }
            PRESENTED_MODES.lock().unwrap().push((sc, modes));
        }
        vk::Result::SUCCESS
    }

    /// Presents `[IMMEDIATE]` per-present mode info on a swapchain registered with `fifo_only_modes`.
    fn present_with_mode_info(handle: u64, fifo_only_modes: bool) -> Vec<vk::PresentModeKHR> {
        use std::ffi::{c_void, CStr};
        let load = |name: &CStr| -> *const c_void {
            if name == c"vkQueuePresentKHR" { capture_present_modes as *const c_void } else { std::ptr::null() }
        };
        let (device, data) = crate::state::test_support::fake_device(load, vec![]);
        SWAPCHAINS.created(handle, SwapchainInfo { device: Some(data), fifo_only_modes, ..info(handle) });
        let queue = vk::Queue::from_raw(device.as_raw());
        let game_modes = [vk::PresentModeKHR::IMMEDIATE];
        let mode_info = vk::SwapchainPresentModeInfoEXT {
            swapchain_count: 1,
            p_present_modes: game_modes.as_ptr(),
            ..Default::default()
        };
        let swapchain = vk::SwapchainKHR::from_raw(handle);
        let index = 0u32;
        let present = vk::PresentInfoKHR {
            p_next: &mode_info as *const _ as *const c_void,
            swapchain_count: 1,
            p_swapchains: &swapchain,
            p_image_indices: &index,
            ..Default::default()
        };
        let r = unsafe { queue_present(queue, &present) };
        SWAPCHAINS.destroyed(handle);
        DEVICES.remove(dispatch_key(device));
        assert_eq!(r, vk::Result::SUCCESS);
        assert_eq!(game_modes, [vk::PresentModeKHR::IMMEDIATE], "the game's struct must not change");
        let seen = PRESENTED_MODES.lock().unwrap();
        seen.iter().rev().find(|(sc, _)| *sc == handle).map(|(_, m)| m.clone()).expect("presented")
    }

    #[test]
    fn non_fifo_modes_found_only_for_fifo_only_swapchains() {
        use std::ffi::c_void;
        let (a, b) = (0x5c_8001u64, 0x5c_8002u64);
        SWAPCHAINS.created(a, SwapchainInfo { fifo_only_modes: true, ..info(a) });
        SWAPCHAINS.created(b, SwapchainInfo { fifo_only_modes: false, ..info(b) });
        let swapchains = [vk::SwapchainKHR::from_raw(a), vk::SwapchainKHR::from_raw(b)];
        let check = |modes: [vk::PresentModeKHR; 2]| {
            let mode_info = vk::SwapchainPresentModeInfoEXT {
                swapchain_count: 2,
                p_present_modes: modes.as_ptr(),
                ..Default::default()
            };
            let unknown = vk::ApplicationInfo { p_next: &mode_info as *const _ as *const c_void, ..Default::default() };
            let pi = vk::PresentInfoKHR {
                p_next: &unknown as *const _ as *const c_void,
                swapchain_count: 2,
                p_swapchains: swapchains.as_ptr(),
                ..Default::default()
            };
            unsafe { non_fifo_mode_on_fifo_only(&pi) }
        };
        assert_eq!(check([vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX]), None);
        assert_eq!(check([vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::FIFO]), Some(vk::PresentModeKHR::MAILBOX));
        SWAPCHAINS.destroyed(a);
        SWAPCHAINS.destroyed(b);
    }

    #[test]
    fn present_mode_info_rewrites_only_fifo_only_swapchains() {
        use std::ffi::{c_void, CStr};
        let load = |name: &CStr| -> *const c_void {
            if name == c"vkQueuePresentKHR" { capture_present_modes as *const c_void } else { std::ptr::null() }
        };
        let (device, data) = crate::state::test_support::fake_device(load, vec![]);
        let (a, b) = (0x5c_7001u64, 0x5c_7002u64);
        SWAPCHAINS.created(a, SwapchainInfo { device: Some(Arc::clone(&data)), fifo_only_modes: true, ..info(a) });
        SWAPCHAINS.created(b, SwapchainInfo { device: Some(data), fifo_only_modes: false, ..info(b) });
        let queue = vk::Queue::from_raw(device.as_raw());
        let game_modes = [vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::MAILBOX];
        let mode_info = vk::SwapchainPresentModeInfoEXT {
            swapchain_count: 2,
            p_present_modes: game_modes.as_ptr(),
            ..Default::default()
        };
        let swapchains = [vk::SwapchainKHR::from_raw(a), vk::SwapchainKHR::from_raw(b)];
        let indices = [0u32, 0];
        let present = vk::PresentInfoKHR {
            p_next: &mode_info as *const _ as *const c_void,
            swapchain_count: 2,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: indices.as_ptr(),
            ..Default::default()
        };
        let r = unsafe { queue_present(queue, &present) };
        SWAPCHAINS.destroyed(a);
        SWAPCHAINS.destroyed(b);
        DEVICES.remove(dispatch_key(device));
        assert_eq!(r, vk::Result::SUCCESS);
        let seen = PRESENTED_MODES.lock().unwrap();
        let modes = seen.iter().rev().find(|(sc, _)| *sc == a).map(|(_, m)| m.clone()).expect("presented");
        assert_eq!(modes, [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX]);
        assert_eq!(game_modes, [vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::MAILBOX]);
    }

    #[test]
    fn present_mode_info_rewritten_on_pass_through() {
        assert_eq!(present_with_mode_info(0x5c_6001, true), [vk::PresentModeKHR::FIFO]);
    }

    #[test]
    fn present_mode_info_untouched_without_rewrite() {
        assert_eq!(present_with_mode_info(0x5c_6002, false), [vk::PresentModeKHR::IMMEDIATE]);
    }

    #[test]
    fn swapchain_registry_counts_presents() {
        let r = SwapchainRegistry::new();
        r.created(1, info(1));
        for _ in 0..3 {
            r.presented(1);
        }
        assert_eq!(r.destroyed(1).unwrap().presents.load(Ordering::Relaxed), 3);
        assert!(r.destroyed(1).is_none());
    }

    #[test]
    fn swapchain_registry_tracks_recreate() {
        let r = SwapchainRegistry::new();
        r.created(1, info(1));
        r.presented(1);
        r.created(2, info(2));
        r.presented(2);
        r.presented(2);
        assert_eq!(r.destroyed(1).unwrap().presents.load(Ordering::Relaxed), 1);
        assert_eq!(r.destroyed(2).unwrap().presents.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn swapchain_registry_ignores_unknown_present() {
        let r = SwapchainRegistry::new();
        r.presented(99);
        assert!(r.destroyed(99).is_none());
    }
}
