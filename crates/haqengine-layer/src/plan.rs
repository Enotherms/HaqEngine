//! Decides, per swapchain, whether HaqEngine generates frames and how the swapchain must change.

use ash::vk;

/// What HaqEngine does with one swapchain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapchainPlan {
    pub generate: bool,
    pub reason: &'static str,
    pub min_image_count: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub present_mode: vk::PresentModeKHR,
}

/// Plans a swapchain from the game's create info and the surface capabilities.
pub fn plan_swapchain(
    ci: &vk::SwapchainCreateInfoKHR,
    caps: &vk::SurfaceCapabilitiesKHR,
    multiplier: u32,
    queue_can_transfer: bool,
) -> SwapchainPlan {
    let copy = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    let off = |reason| SwapchainPlan {
        generate: false,
        reason,
        min_image_count: ci.min_image_count,
        image_usage: ci.image_usage,
        present_mode: ci.present_mode,
    };
    if multiplier < 2 {
        return off("multiplier 1");
    }
    if ci.image_color_space != vk::ColorSpaceKHR::SRGB_NONLINEAR {
        return off("HDR colour space: pass-through until M8");
    }
    if !caps.supported_usage_flags.contains(copy) {
        return off("surface cannot copy images");
    }
    if !queue_can_transfer {
        return off("present queue cannot transfer");
    }
    if caps.max_image_count != 0 && ci.min_image_count + 1 > caps.max_image_count {
        return off("no room for an extra image");
    }
    SwapchainPlan {
        generate: true,
        reason: "generating",
        min_image_count: ci.min_image_count + 1,
        image_usage: ci.image_usage | copy,
        present_mode: vk::PresentModeKHR::FIFO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ci(min: u32, mode: vk::PresentModeKHR) -> vk::SwapchainCreateInfoKHR<'static> {
        vk::SwapchainCreateInfoKHR {
            min_image_count: min,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            present_mode: mode,
            ..Default::default()
        }
    }

    fn caps(max: u32, usage: vk::ImageUsageFlags) -> vk::SurfaceCapabilitiesKHR {
        vk::SurfaceCapabilitiesKHR {
            min_image_count: 2,
            max_image_count: max,
            supported_usage_flags: usage,
            ..Default::default()
        }
    }

    fn all_usage() -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
    }

    fn assert_unchanged(p: &SwapchainPlan, c: &vk::SwapchainCreateInfoKHR) {
        assert!(!p.generate);
        assert_eq!(p.min_image_count, c.min_image_count);
        assert_eq!(p.image_usage, c.image_usage);
        assert_eq!(p.present_mode, c.present_mode);
    }

    #[test]
    fn plan_off_at_multiplier_1() {
        let c = ci(3, vk::PresentModeKHR::FIFO);
        let p = plan_swapchain(&c, &caps(8, all_usage()), 1, true);
        assert_unchanged(&p, &c);
        assert_eq!(p.reason, "multiplier 1");
    }

    #[test]
    fn plan_off_for_hdr_colour_space() {
        let mut c = ci(3, vk::PresentModeKHR::FIFO);
        c.image_color_space = vk::ColorSpaceKHR::HDR10_ST2084_EXT;
        let p = plan_swapchain(&c, &caps(8, all_usage()), 2, true);
        assert_unchanged(&p, &c);
        assert!(p.reason.contains("HDR"));
    }

    #[test]
    fn plan_off_without_transfer_usage() {
        let c = ci(3, vk::PresentModeKHR::FIFO);
        let p = plan_swapchain(&c, &caps(8, vk::ImageUsageFlags::COLOR_ATTACHMENT), 2, true);
        assert_unchanged(&p, &c);
        assert_eq!(p.reason, "surface cannot copy images");
    }

    #[test]
    fn plan_off_without_transfer_queue() {
        let c = ci(3, vk::PresentModeKHR::FIFO);
        let p = plan_swapchain(&c, &caps(8, all_usage()), 2, false);
        assert_unchanged(&p, &c);
        assert_eq!(p.reason, "present queue cannot transfer");
    }

    #[test]
    fn plan_off_at_max_image_count() {
        let c = ci(3, vk::PresentModeKHR::FIFO);
        let p = plan_swapchain(&c, &caps(3, all_usage()), 2, true);
        assert_unchanged(&p, &c);
        assert_eq!(p.reason, "no room for an extra image");
    }

    #[test]
    fn plan_unlimited_max_image_count_is_ok() {
        let c = ci(3, vk::PresentModeKHR::FIFO);
        assert!(plan_swapchain(&c, &caps(0, all_usage()), 2, true).generate);
    }

    #[test]
    fn plan_forces_fifo_when_generating() {
        let c = ci(3, vk::PresentModeKHR::MAILBOX);
        let p = plan_swapchain(&c, &caps(8, all_usage()), 2, true);
        assert!(p.generate);
        assert_eq!(p.reason, "generating");
        assert_eq!(p.present_mode, vk::PresentModeKHR::FIFO);
        assert_eq!(p.min_image_count, 4);
        assert!(p.image_usage.contains(all_usage()));
    }
}
