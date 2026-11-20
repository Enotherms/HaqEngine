//! Copies the front of a Vulkan `pNext` chain so one struct can be changed without writing to the
//! game's memory. Fixed-size buffer, no heap allocation.

use ash::vk;
use std::ffi::c_void;
use std::mem::size_of;

/// Buffer size in 8-byte words (512 bytes).
pub const CHAIN_BUF_WORDS: usize = 64;

/// Storage for copied structs. Copies live as long as this value.
pub struct ChainCopy {
    buf: [u64; CHAIN_BUF_WORDS],
    used: usize,
    /// Per-swapchain present modes for a rewritten `VkSwapchainPresentModeInfoEXT`.
    pub modes: [vk::PresentModeKHR; 16],
    /// Per-swapchain ids for a rewritten `VkPresentIdKHR`.
    pub present_ids: [u64; 16],
}

impl Default for ChainCopy {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of `ChainCopy::copy_prefix`.
#[derive(Debug, PartialEq, Eq)]
pub enum PrefixCopy {
    /// The target is not in the chain; nothing was copied.
    NotFound,
    /// The prefix was copied; `head` starts the new chain and `target` is the writable copy.
    Copied { head: *const c_void, target: *mut vk::BaseOutStructure<'static> },
    /// A struct before (or at) the target has no known size, or the buffer is too small.
    Unsupported,
}

/// Size of an extension struct allowed in `VkSwapchainCreateInfoKHR::pNext`.
pub fn swapchain_ext_size(s: vk::StructureType) -> Option<usize> {
    use vk::StructureType as S;
    Some(match s {
        S::IMAGE_FORMAT_LIST_CREATE_INFO => size_of::<vk::ImageFormatListCreateInfo>(),
        S::SWAPCHAIN_COUNTER_CREATE_INFO_EXT => size_of::<vk::SwapchainCounterCreateInfoEXT>(),
        S::DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR => size_of::<vk::DeviceGroupSwapchainCreateInfoKHR>(),
        S::SWAPCHAIN_DISPLAY_NATIVE_HDR_CREATE_INFO_AMD => size_of::<vk::SwapchainDisplayNativeHdrCreateInfoAMD>(),
        S::SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT => size_of::<vk::SurfaceFullScreenExclusiveInfoEXT>(),
        S::SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT => size_of::<vk::SurfaceFullScreenExclusiveWin32InfoEXT>(),
        S::SWAPCHAIN_PRESENT_BARRIER_CREATE_INFO_NV => size_of::<vk::SwapchainPresentBarrierCreateInfoNV>(),
        S::SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT => size_of::<vk::SwapchainPresentModesCreateInfoEXT>(),
        S::SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT => size_of::<vk::SwapchainPresentScalingCreateInfoEXT>(),
        S::IMAGE_COMPRESSION_CONTROL_EXT => size_of::<vk::ImageCompressionControlEXT>(),
        S::SWAPCHAIN_LATENCY_CREATE_INFO_NV => size_of::<vk::SwapchainLatencyCreateInfoNV>(),
        _ => return None,
    })
}

/// Size of an extension struct allowed in `VkPresentInfoKHR::pNext`.
pub fn present_ext_size(s: vk::StructureType) -> Option<usize> {
    use vk::StructureType as S;
    Some(match s {
        S::PRESENT_REGIONS_KHR => size_of::<vk::PresentRegionsKHR>(),
        S::DISPLAY_PRESENT_INFO_KHR => size_of::<vk::DisplayPresentInfoKHR>(),
        S::PRESENT_ID_KHR => size_of::<vk::PresentIdKHR>(),
        S::PRESENT_TIMES_INFO_GOOGLE => size_of::<vk::PresentTimesInfoGOOGLE>(),
        S::DEVICE_GROUP_PRESENT_INFO_KHR => size_of::<vk::DeviceGroupPresentInfoKHR>(),
        S::SWAPCHAIN_PRESENT_FENCE_INFO_EXT => size_of::<vk::SwapchainPresentFenceInfoEXT>(),
        S::SWAPCHAIN_PRESENT_MODE_INFO_EXT => size_of::<vk::SwapchainPresentModeInfoEXT>(),
        S::FRAME_BOUNDARY_EXT => size_of::<vk::FrameBoundaryEXT>(),
        _ => return None,
    })
}

impl ChainCopy {
    pub fn new() -> Self {
        ChainCopy { buf: [0; CHAIN_BUF_WORDS], used: 0, modes: [vk::PresentModeKHR::FIFO; 16], present_ids: [0; 16] }
    }

    /// Copies the chain from `head` up to and including the first struct of type `target`.
    ///
    /// # Safety
    /// `head` must be null or the head of a valid Vulkan `pNext` chain.
    pub unsafe fn copy_prefix(
        &mut self,
        head: *const c_void,
        target: vk::StructureType,
        size_fn: fn(vk::StructureType) -> Option<usize>,
    ) -> PrefixCopy {
        let mut needed_words = 0usize;
        let mut cur = head as *const vk::BaseInStructure;
        loop {
            if cur.is_null() {
                return PrefixCopy::NotFound;
            }
            let Some(size) = size_fn((*cur).s_type) else {
                if Self::contains(cur, target) {
                    return PrefixCopy::Unsupported;
                }
                return PrefixCopy::NotFound;
            };
            needed_words += size.div_ceil(8);
            if (*cur).s_type == target {
                break;
            }
            cur = (*cur).p_next;
        }
        if self.used + needed_words > CHAIN_BUF_WORDS {
            return PrefixCopy::Unsupported;
        }
        let base = self.buf.as_mut_ptr();
        let mut first: *const c_void = std::ptr::null();
        let mut prev: *mut vk::BaseOutStructure = std::ptr::null_mut();
        let mut cur = head as *const vk::BaseInStructure;
        loop {
            let s_type = (*cur).s_type;
            let size = size_fn(s_type).expect("checked above");
            let dst = base.add(self.used) as *mut u8;
            std::ptr::copy_nonoverlapping(cur as *const u8, dst, size);
            self.used += size.div_ceil(8);
            let copy = dst as *mut vk::BaseOutStructure;
            if prev.is_null() {
                first = copy as *const c_void;
            } else {
                (*prev).p_next = copy;
            }
            if s_type == target {
                return PrefixCopy::Copied { head: first, target: copy as *mut vk::BaseOutStructure<'static> };
            }
            prev = copy;
            cur = (*cur).p_next;
        }
    }

    unsafe fn contains(mut cur: *const vk::BaseInStructure, target: vk::StructureType) -> bool {
        while !cur.is_null() {
            if (*cur).s_type == target {
                return true;
            }
            cur = (*cur).p_next;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes<T>(v: &T) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(v as *const T as *const u8, size_of::<T>()).to_vec() }
    }

    #[test]
    fn copy_prefix_relinks_and_keeps_tail() {
        let tail = vk::PresentIdKHR::default();
        let modes = [vk::PresentModeKHR::MAILBOX];
        let target = vk::SwapchainPresentModesCreateInfoEXT {
            p_next: &tail as *const _ as *const c_void,
            present_mode_count: 1,
            p_present_modes: modes.as_ptr(),
            ..Default::default()
        };
        let formats = [vk::Format::B8G8R8A8_UNORM];
        let first = vk::ImageFormatListCreateInfo {
            p_next: &target as *const _ as *const c_void,
            view_format_count: 1,
            p_view_formats: formats.as_ptr(),
            ..Default::default()
        };
        let (before_first, before_target) = (bytes(&first), bytes(&target));
        let mut copy = ChainCopy::new();
        let PrefixCopy::Copied { head, target: t } = (unsafe {
            copy.copy_prefix(&first as *const _ as *const c_void, vk::StructureType::SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT, swapchain_ext_size)
        }) else {
            panic!("expected a copy");
        };
        unsafe {
            let fifo = [vk::PresentModeKHR::FIFO];
            let t = &mut *(t as *mut vk::SwapchainPresentModesCreateInfoEXT);
            t.p_present_modes = fifo.as_ptr();
            let h = &*(head as *const vk::ImageFormatListCreateInfo);
            assert_eq!(h.s_type, vk::StructureType::IMAGE_FORMAT_LIST_CREATE_INFO);
            assert_eq!(h.p_next, t as *const _ as *const c_void);
            assert_ne!(head, &first as *const _ as *const c_void);
            assert_eq!(t.p_next, &tail as *const _ as *const c_void);
            assert_eq!(*t.p_present_modes, vk::PresentModeKHR::FIFO);
        }
        assert_eq!(bytes(&first), before_first);
        assert_eq!(bytes(&target), before_target);
    }

    #[test]
    fn copy_prefix_target_first() {
        let target = vk::SwapchainPresentModeInfoEXT::default();
        let mut copy = ChainCopy::new();
        let r = unsafe {
            copy.copy_prefix(&target as *const _ as *const c_void, vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT, present_ext_size)
        };
        let PrefixCopy::Copied { head, target: t } = r else { panic!("expected a copy") };
        assert_eq!(head, t as *const c_void);
    }

    #[test]
    fn copy_prefix_not_found() {
        let other = vk::PresentIdKHR::default();
        let mut copy = ChainCopy::new();
        let t = vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT;
        assert_eq!(unsafe { copy.copy_prefix(&other as *const _ as *const c_void, t, present_ext_size) }, PrefixCopy::NotFound);
        assert_eq!(unsafe { copy.copy_prefix(std::ptr::null(), t, present_ext_size) }, PrefixCopy::NotFound);
    }

    #[test]
    fn copy_prefix_unknown_struct_is_none() {
        let target = vk::SwapchainPresentModeInfoEXT::default();
        let unknown = vk::ApplicationInfo { p_next: &target as *const _ as *const c_void, ..Default::default() };
        let mut copy = ChainCopy::new();
        let r = unsafe {
            copy.copy_prefix(&unknown as *const _ as *const c_void, vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT, present_ext_size)
        };
        assert_eq!(r, PrefixCopy::Unsupported);
    }

    #[test]
    fn copy_prefix_overflow_is_none() {
        fn huge(_: vk::StructureType) -> Option<usize> {
            Some(400)
        }
        let target = vk::SwapchainPresentModeInfoEXT::default();
        let first = vk::PresentIdKHR { p_next: &target as *const _ as *const c_void, ..Default::default() };
        let mut copy = ChainCopy::new();
        let r = unsafe { copy.copy_prefix(&first as *const _ as *const c_void, vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT, huge) };
        assert_eq!(r, PrefixCopy::Unsupported);
    }

    #[test]
    fn size_tables_cover_present_mode_structs() {
        assert!(swapchain_ext_size(vk::StructureType::SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT).is_some());
        assert!(present_ext_size(vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT).is_some());
        assert!(present_ext_size(vk::StructureType::APPLICATION_INFO).is_none());
    }
}
