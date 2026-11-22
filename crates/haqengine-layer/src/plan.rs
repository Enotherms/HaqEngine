//! Decides, per swapchain, whether HaqEngine generates frames and how the swapchain must change.

use ash::vk;
use haqengine_core::MAX_MULTIPLIER;

/// What HaqEngine does with one swapchain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapchainPlan {
    pub generate: bool,
    pub reason: &'static str,
    /// Frames shown per real frame this swapchain can hold (1 when not generating).
    pub multiplier: u32,
    pub min_image_count: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub present_mode: vk::PresentModeKHR,
}

/// Like `plan_swapchain`, also given the modes the game lists in `VkSwapchainPresentModesCreateInfoEXT`
/// (`None` when it chains none). Such a game may switch present modes itself on every present.
pub fn plan_swapchain_with(
    ci: &vk::SwapchainCreateInfoKHR,
    caps: &vk::SurfaceCapabilitiesKHR,
    multiplier: u32,
    queue_can_transfer: bool,
    game_present_modes: Option<&[vk::PresentModeKHR]>,
) -> SwapchainPlan {
    let fifo_only = |modes: &[vk::PresentModeKHR]| {
        modes.contains(&vk::PresentModeKHR::FIFO)
            && modes.iter().all(|m| matches!(*m, vk::PresentModeKHR::FIFO | vk::PresentModeKHR::FIFO_RELAXED))
    };
    if multiplier >= 2 && game_present_modes.is_some_and(|m| !fifo_only(m)) {
        return SwapchainPlan {
            generate: false,
            reason: "game present modes include non-FIFO (swapchain_maintenance1)",
            multiplier: 1,
            min_image_count: ci.min_image_count,
            image_usage: ci.image_usage,
            present_mode: ci.present_mode,
        };
    }
    plan_swapchain(ci, caps, multiplier, queue_can_transfer)
}

/// The multiplier a swapchain asks for: the setting for Battery (also when Quality is unavailable
/// and Battery runs instead), at most 2 for Quality and copy mode, which present one generated frame.
pub fn engine_multiplier(mode: haqengine_core::Mode, battery_format: bool, quality_ok: bool, multiplier: u32) -> u32 {
    use haqengine_core::Mode;
    match mode {
        Mode::Battery if battery_format => multiplier,
        Mode::Quality if battery_format && !quality_ok => multiplier,
        _ => multiplier.min(2),
    }
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
        multiplier: 1,
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
    let room = if caps.max_image_count == 0 { u32::MAX } else { caps.max_image_count.saturating_sub(ci.min_image_count) };
    let fits = multiplier.min(MAX_MULTIPLIER).min(room.saturating_add(1));
    if fits < 2 {
        return off("no room for an extra image");
    }
    let reason = match (fits < multiplier.min(MAX_MULTIPLIER), fits) {
        (false, _) => "generating",
        (true, 2) => "generating at 2x: no room for more images",
        (true, _) => "generating at 3x: no room for more images",
    };
    SwapchainPlan {
        generate: true,
        reason,
        multiplier: fits,
        min_image_count: ci.min_image_count + fits - 1,
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
        vk::SurfaceCapabilitiesKHR { min_image_count: 2, max_image_count: max, supported_usage_flags: usage, ..Default::default() }
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
    fn only_battery_asks_for_more_than_two() {
        use haqengine_core::Mode;
        assert_eq!(engine_multiplier(Mode::Battery, true, true, 3), 3);
        assert_eq!(engine_multiplier(Mode::Quality, true, true, 3), 2);
        assert_eq!(engine_multiplier(Mode::Quality, true, false, 4), 4, "quality unavailable runs battery");
        assert_eq!(engine_multiplier(Mode::Copy, true, true, 3), 2);
        assert_eq!(engine_multiplier(Mode::Battery, false, true, 3), 2, "unsupported format copies");
        assert_eq!(engine_multiplier(Mode::Battery, true, true, 1), 1);
    }

    #[test]
    fn plan_adds_m_minus_1_images() {
        let c = ci(3, vk::PresentModeKHR::FIFO);
        for (m, images) in [(2, 4), (3, 5), (4, 6)] {
            let p = plan_swapchain(&c, &caps(8, all_usage()), m, true);
            assert!(p.generate);
            assert_eq!((p.multiplier, p.min_image_count), (m, images), "{m}");
        }
    }

    #[test]
    fn plan_falls_back_when_no_room_for_three() {
        let c = ci(3, vk::PresentModeKHR::FIFO);
        let p = plan_swapchain(&c, &caps(4, all_usage()), 3, true);
        assert!(p.generate);
        assert_eq!((p.multiplier, p.min_image_count), (2, 4));
        assert_eq!(p.reason, "generating at 2x: no room for more images");
        let p = plan_swapchain(&c, &caps(5, all_usage()), 4, true);
        assert_eq!((p.multiplier, p.min_image_count, p.reason), (3, 5, "generating at 3x: no room for more images"));
    }

    #[test]
    fn plan_off_when_no_room_for_two() {
        let c = ci(3, vk::PresentModeKHR::FIFO);
        let p = plan_swapchain(&c, &caps(3, all_usage()), 3, true);
        assert_unchanged(&p, &c);
        assert_eq!(p.multiplier, 1);
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
    fn plan_off_when_game_present_modes_include_non_fifo() {
        let c = ci(3, vk::PresentModeKHR::FIFO);
        for list in [
            &[vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX][..],
            &[vk::PresentModeKHR::IMMEDIATE][..],
            &[vk::PresentModeKHR::FIFO_RELAXED][..],
        ] {
            let p = plan_swapchain_with(&c, &caps(8, all_usage()), 2, true, Some(list));
            assert_unchanged(&p, &c);
            assert!(p.reason.contains("present modes"), "{list:?}");
        }
    }

    #[test]
    fn plan_generates_when_game_present_modes_are_fifo_only() {
        let c = ci(3, vk::PresentModeKHR::FIFO);
        for list in [&[vk::PresentModeKHR::FIFO][..], &[vk::PresentModeKHR::FIFO, vk::PresentModeKHR::FIFO_RELAXED][..]] {
            let p = plan_swapchain_with(&c, &caps(8, all_usage()), 2, true, Some(list));
            assert!(p.generate, "{list:?}");
            assert_eq!(p.present_mode, vk::PresentModeKHR::FIFO);
        }
        assert!(plan_swapchain_with(&c, &caps(8, all_usage()), 2, true, None).generate);
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
