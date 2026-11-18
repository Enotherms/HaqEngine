//! Swapchain and present pass-through. M0 only counts presents and forwards everything unchanged.

use crate::guard::guard;
use crate::state::{dispatch_key, DeviceData, DEVICES, LOG};
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

unsafe fn create_swapchain_inner(
    device: vk::Device,
    create_info: *const vk::SwapchainCreateInfoKHR,
    allocator: *const vk::AllocationCallbacks,
    out: *mut vk::SwapchainKHR,
) -> vk::Result {
    let Some(data) = DEVICES.get(dispatch_key(device)) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let result = (data.swapchain.create_swapchain_khr)(device, create_info, allocator, out);
    if result == vk::Result::SUCCESS {
        guard("swapchain bookkeeping", (), || unsafe {
            let ci = &*create_info;
            let mut images = 0u32;
            let _ = (data.swapchain.get_swapchain_images_khr)(device, *out, &mut images, std::ptr::null_mut());
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            LOG.log(&format!(
                "swapchain {id}: {}x{} format {:?} images {images}",
                ci.image_extent.width, ci.image_extent.height, ci.image_format
            ));
            SWAPCHAINS.created(
                (*out).as_raw(),
                SwapchainInfo {
                    id,
                    extent: ci.image_extent,
                    format: ci.image_format,
                    images,
                    presents: AtomicU64::new(0),
                    device: Some(Arc::clone(&data)),
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
    guard("swapchain destroy log", (), || {
        if let Some(info) = SWAPCHAINS.destroyed(swapchain.as_raw()) {
            LOG.log(&format!(
                "swapchain {} destroyed after {} presents",
                info.id,
                info.presents.load(Ordering::Relaxed)
            ));
        }
    });
    if let Some(data) = DEVICES.get(dispatch_key(device)) {
        (data.swapchain.destroy_swapchain_khr)(device, swapchain, allocator);
    }
}

unsafe fn queue_present_inner(queue: vk::Queue, present_info: *const vk::PresentInfoKHR) -> vk::Result {
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
        Some(data) => (data.swapchain.queue_present_khr)(queue, present_info),
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
        }
    }

    static FAKE_PRESENTS: AtomicU64 = AtomicU64::new(0);

    unsafe extern "system" fn fake_present(_: vk::Queue, _: *const vk::PresentInfoKHR) -> vk::Result {
        FAKE_PRESENTS.fetch_add(1, Ordering::SeqCst);
        vk::Result::SUCCESS
    }

    unsafe extern "system" fn fake_gdpa(_: vk::Device, _: *const std::ffi::c_char) -> vk::PFN_vkVoidFunction {
        None
    }

    /// A device whose next layer only provides `vkQueuePresentKHR`, and a queue handle whose
    /// dispatch key is not in `DEVICES`.
    fn fake_device_and_stray_queue() -> (Arc<DeviceData>, vk::Queue) {
        let device_word: &'static usize = Box::leak(Box::new(0x0dd0_0001));
        let queue_word: &'static usize = Box::leak(Box::new(0x0dd0_0002));
        let handle = vk::Device::from_raw(device_word as *const usize as u64);
        let load = |name: &std::ffi::CStr| {
            if name == c"vkQueuePresentKHR" { fake_present as *const std::ffi::c_void } else { std::ptr::null() }
        };
        let data = DeviceData {
            next_gdpa: fake_gdpa,
            device: unsafe { ash::Device::load_with(load, handle) },
            swapchain: ash::khr::swapchain::DeviceFn::load(load),
        };
        (Arc::new(data), vk::Queue::from_raw(queue_word as *const usize as u64))
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
