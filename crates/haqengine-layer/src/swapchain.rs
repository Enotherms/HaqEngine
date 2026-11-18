//! Swapchain and present pass-through. M0 only counts presents and forwards everything unchanged.

use crate::guard::guard;
use crate::framegen::FrameGen;
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

/// True when a `pNext` chain contains a struct of type `s_type`.
unsafe fn chain_has(p_next: *const std::ffi::c_void, s_type: vk::StructureType) -> bool {
    let mut cur = p_next as *const vk::BaseInStructure;
    while !cur.is_null() {
        if (*cur).s_type == s_type {
            return true;
        }
        cur = (*cur).p_next;
    }
    false
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
    let plan = guard("swapchain plan", None, || unsafe {
        let mut caps = vk::SurfaceCapabilitiesKHR::default();
        let r = (data.instance.surface.get_physical_device_surface_capabilities_khr)(
            data.physical_device,
            ci.surface,
            &mut caps,
        );
        let manages_modes = chain_has(ci.p_next, vk::StructureType::SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT);
        (r == vk::Result::SUCCESS)
            .then(|| plan_swapchain_with(ci, &caps, multiplier(), device_can_transfer(&data), manages_modes))
    });
    let mut planned = *ci;
    if let Some(p) = plan.filter(|p| p.generate) {
        planned.min_image_count = p.min_image_count;
        planned.image_usage = p.image_usage;
        planned.present_mode = p.present_mode;
    }
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
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            LOG.log(&format!(
                "swapchain {id}: {}x{} format {:?} colour {:?} images {count} mode {:?} gen {reason}",
                ci.image_extent.width, ci.image_extent.height, ci.image_format, ci.image_color_space, planned.present_mode
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
                    gen: generate.then(|| FrameGen::new(images, ci.image_extent)),
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
        }
    }

    static CAPTURED: std::sync::Mutex<Option<(u32, vk::ImageUsageFlags, vk::PresentModeKHR)>> =
        std::sync::Mutex::new(None);

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
