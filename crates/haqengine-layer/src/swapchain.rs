//! Swapchain and present pass-through. M0 only counts presents and forwards everything unchanged.

use crate::guard::guard;
use crate::state::{dispatch_key, DEVICES, LOG};
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
    match DEVICES.get(dispatch_key(queue)) {
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
        }
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
