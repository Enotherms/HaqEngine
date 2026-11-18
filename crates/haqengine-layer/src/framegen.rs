//! Per-swapchain resources for generated frames. Created once per swapchain, never per frame.

use crate::queue::{queue_can_transfer, QUEUES};
use crate::state::{DeviceData, LOG};
use ash::vk;
use ash::vk::Handle;
use haqengine_core::{should_log, FrameStats};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

/// Most wait semaphores a game present may carry for HaqEngine to duplicate it.
const MAX_WAITS: usize = 16;
/// How long to wait for a slot's previous copy before skipping a frame.
const SLOT_WAIT_NS: u64 = 50_000_000;
/// How often pacing statistics are logged.
const LOG_EVERY_MS: u64 = 10_000;

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// One frame's worth of reusable copy resources.
pub struct Slot {
    pub cmd: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub acquired: vk::Semaphore,
    pub copy_ready: vk::Semaphore,
    pub real_ready: vk::Semaphore,
}

/// Command pool and slots, created on the first generating present (the queue family is only known then).
pub struct Ready {
    pub family: u32,
    pub pool: vk::CommandPool,
    pub slots: Vec<Slot>,
}

pub struct GenStats {
    pub real: u64,
    pub generated: u64,
    pub skipped: u64,
    pub intervals: FrameStats,
    pub present_cost: FrameStats,
    pub last_present: Option<Instant>,
    pub last_log_ms: u64,
}

pub struct GenState {
    pub ready: Option<Ready>,
    pub next: usize,
    pub stats: GenStats,
}

pub struct FrameGen {
    pub images: Vec<vk::Image>,
    pub extent: vk::Extent2D,
    pub disabled: AtomicBool,
    pub state: Mutex<GenState>,
}

impl FrameGen {
    pub fn new(images: Vec<vk::Image>, extent: vk::Extent2D) -> FrameGen {
        FrameGen {
            images,
            extent,
            disabled: AtomicBool::new(false),
            state: Mutex::new(GenState {
                ready: None,
                next: 0,
                stats: GenStats {
                    real: 0,
                    generated: 0,
                    skipped: 0,
                    intervals: FrameStats::new(),
                    present_cost: FrameStats::new(),
                    last_present: None,
                    last_log_ms: 0,
                },
            }),
        }
    }

    /// Creates the command pool and slots for `family` on first use.
    ///
    /// # Safety
    /// `device` must be the live device that owns this swapchain.
    pub unsafe fn ensure_ready(&self, state: &mut GenState, device: &DeviceData, family: u32) -> Result<(), vk::Result> {
        if let Some(ready) = &state.ready {
            return if ready.family == family { Ok(()) } else { Err(vk::Result::ERROR_FEATURE_NOT_PRESENT) };
        }
        let d = &device.device;
        let pool = d.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER | vk::CommandPoolCreateFlags::TRANSIENT),
            None,
        )?;
        let mut ready = Ready { family, pool, slots: Vec::with_capacity(self.images.len() + 2) };
        let result = (|| -> Result<(), vk::Result> {
            let cmds = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(self.images.len() as u32 + 2),
            )?;
            let Some(set_loader_data) = device.set_loader_data else {
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            };
            for cmd in &cmds {
                let r = set_loader_data(d.handle(), cmd.as_raw() as *mut std::ffi::c_void);
                if r != vk::Result::SUCCESS {
                    d.free_command_buffers(pool, &cmds);
                    return Err(r);
                }
            }
            for cmd in cmds {
                let fence = d.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?;
                let sem = || d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None);
                let slot = Slot { cmd, fence, acquired: vk::Semaphore::null(), copy_ready: vk::Semaphore::null(), real_ready: vk::Semaphore::null() };
                ready.slots.push(slot);
                let last = ready.slots.last_mut().expect("just pushed");
                last.acquired = sem()?;
                last.copy_ready = sem()?;
                last.real_ready = sem()?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                state.ready = Some(ready);
                Ok(())
            }
            Err(e) => {
                destroy_ready(device, &ready);
                Err(e)
            }
        }
    }

    /// Waits for in-flight copies (up to 1 s) and destroys the pool and slots.
    ///
    /// # Safety
    /// `device` must be the live device that owns this swapchain.
    pub unsafe fn destroy(&self, device: &DeviceData) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ready) = state.ready.take() {
            destroy_ready(device, &ready);
        }
    }
}

/// Waits (up to 1 s) for the slots' copies, then destroys everything in `ready`.
unsafe fn destroy_ready(device: &DeviceData, ready: &Ready) {
    let d = &device.device;
    let fences: Vec<vk::Fence> = ready.slots.iter().map(|s| s.fence).filter(|f| *f != vk::Fence::null()).collect();
    if !fences.is_empty() {
        let _ = d.wait_for_fences(&fences, true, 1_000_000_000);
    }
    for s in &ready.slots {
        for sem in [s.acquired, s.copy_ready, s.real_ready] {
            if sem != vk::Semaphore::null() {
                d.destroy_semaphore(sem, None);
            }
        }
        if s.fence != vk::Fence::null() {
            d.destroy_fence(s.fence, None);
        }
    }
    d.destroy_command_pool(ready.pool, None);
}

impl FrameGen {
    /// Turns generation off for this swapchain and logs it once.
    fn disable(&self, id: u64, step: &str, result: vk::Result) {
        if !self.disabled.swap(true, Ordering::Relaxed) {
            LOG.log(&format!("generation off for swapchain {id}: {step} {result:?}"));
        }
    }

    /// Presents a GPU copy of the game's frame, then the frame itself. Falls back to the game's
    /// own present on any problem.
    ///
    /// # Safety
    /// `pi` must be a valid present for `queue`, and `device` the live device owning the swapchain.
    pub unsafe fn present(
        &self,
        id: u64,
        device: &DeviceData,
        queue: vk::Queue,
        pi: &vk::PresentInfoKHR,
        start: Instant,
    ) -> vk::Result {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let next_present = device.swapchain.queue_present_khr;
        let pass = |st: &mut GenState, skipped: bool| {
            let r = next_present(queue, pi);
            record(st, id, start, false, skipped);
            r
        };
        if pi.wait_semaphore_count as usize > MAX_WAITS {
            return pass(&mut st, true);
        }
        let family = match QUEUES.get(queue.as_raw() as usize) {
            Some(q) if queue_can_transfer(queue) => q.family,
            _ => {
                self.disable(id, "present queue cannot transfer", vk::Result::ERROR_FEATURE_NOT_PRESENT);
                return pass(&mut st, false);
            }
        };
        if let Err(e) = self.ensure_ready(&mut st, device, family) {
            self.disable(id, "creating copy resources", e);
            return pass(&mut st, false);
        }
        let slot = {
            let ready = st.ready.as_ref().expect("ensure_ready succeeded");
            let s = &ready.slots[st.next % ready.slots.len()];
            Slot { cmd: s.cmd, fence: s.fence, acquired: s.acquired, copy_ready: s.copy_ready, real_ready: s.real_ready }
        };
        st.next = st.next.wrapping_add(1);
        let d = &device.device;
        match d.wait_for_fences(&[slot.fence], true, SLOT_WAIT_NS) {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => return pass(&mut st, true),
            Err(e) => {
                self.disable(id, "waiting for a copy", e);
                return pass(&mut st, false);
            }
        }
        let swapchain = *pi.p_swapchains;
        let game_index = *pi.p_image_indices;
        let mut spare = 0u32;
        let r = (device.swapchain.acquire_next_image_khr)(
            d.handle(),
            swapchain,
            ACQUIRE_TIMEOUT_NS,
            slot.acquired,
            vk::Fence::null(),
            &mut spare,
        );
        let spare = match acquire_outcome(r, spare) {
            AcquireAction::Generate(i) => i,
            AcquireAction::SkipThisFrame => return pass(&mut st, true),
            AcquireAction::Disable => {
                self.disable(id, "acquiring a spare image", r);
                return pass(&mut st, false);
            }
        };
        let (Some(&game_image), Some(&spare_image)) =
            (self.images.get(game_index as usize), self.images.get(spare as usize))
        else {
            self.disable(id, "image index out of range", vk::Result::ERROR_UNKNOWN);
            return self.give_back(&mut st, device, queue, pi, swapchain, spare, slot.acquired, id, start);
        };
        if let Err(e) = self.record_copy(d, slot.cmd, game_image, spare_image) {
            self.disable(id, "recording the copy", e);
            return self.give_back(&mut st, device, queue, pi, swapchain, spare, slot.acquired, id, start);
        }
        let mut waits = [vk::Semaphore::null(); MAX_WAITS + 1];
        let n = pi.wait_semaphore_count as usize;
        if n > 0 {
            waits[..n].copy_from_slice(std::slice::from_raw_parts(pi.p_wait_semaphores, n));
        }
        waits[n] = slot.acquired;
        let stages = [vk::PipelineStageFlags::TRANSFER; MAX_WAITS + 1];
        let signals = [slot.copy_ready, slot.real_ready];
        let cmds = [slot.cmd];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&waits[..=n])
            .wait_dst_stage_mask(&stages[..=n])
            .command_buffers(&cmds)
            .signal_semaphores(&signals);
        if let Err(e) = d.reset_fences(&[slot.fence]) {
            self.disable(id, "resetting a fence", e);
            return self.give_back(&mut st, device, queue, pi, swapchain, spare, slot.acquired, id, start);
        }
        if let Err(e) = d.queue_submit(queue, &[submit], slot.fence) {
            let _ = d.queue_submit(queue, &[], slot.fence);
            self.disable(id, "submitting the copy", e);
            return self.give_back(&mut st, device, queue, pi, swapchain, spare, slot.acquired, id, start);
        }
        let copy_wait = [slot.copy_ready];
        let copy_indices = [spare];
        let swapchains = [swapchain];
        let copy_present = vk::PresentInfoKHR::default()
            .wait_semaphores(&copy_wait)
            .swapchains(&swapchains)
            .image_indices(&copy_indices);
        let r_copy = next_present(queue, &copy_present);
        let real_wait = [slot.real_ready];
        let mut real_present = *pi;
        real_present.wait_semaphore_count = 1;
        real_present.p_wait_semaphores = real_wait.as_ptr();
        let r_real = next_present(queue, &real_present);
        record(&mut st, id, start, true, false);
        combine_present_results(r_copy, r_real)
    }

    /// Records copying the game's image into the spare image, leaving both ready to present.
    unsafe fn record_copy(&self, d: &ash::Device, cmd: vk::CommandBuffer, src: vk::Image, dst: vk::Image) -> Result<(), vk::Result> {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let barrier = |image, old, new, src_access, dst_access| {
            vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(range)
        };
        d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let to_copy = [
            barrier(src, vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_READ),
            barrier(dst, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE),
        ];
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_copy);
        let layers = vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1);
        let region = vk::ImageCopy::default()
            .src_subresource(layers)
            .dst_subresource(layers)
            .extent(vk::Extent3D { width: self.extent.width, height: self.extent.height, depth: 1 });
        d.cmd_copy_image(cmd, src, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
        let to_present = [
            barrier(src, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::empty()),
            barrier(dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty()),
        ];
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[], &[], &to_present);
        d.end_command_buffer(cmd)
    }

    /// Error path after a spare image was acquired: present it as-is (waiting on its acquire
    /// semaphore) so the swapchain does not lose an image, then present the game's frame unchanged.
    #[allow(clippy::too_many_arguments)]
    unsafe fn give_back(
        &self,
        st: &mut GenState,
        device: &DeviceData,
        queue: vk::Queue,
        pi: &vk::PresentInfoKHR,
        swapchain: vk::SwapchainKHR,
        spare: u32,
        acquired: vk::Semaphore,
        id: u64,
        start: Instant,
    ) -> vk::Result {
        let waits = [acquired];
        let swapchains = [swapchain];
        let indices = [spare];
        let spare_present = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&swapchains).image_indices(&indices);
        let r_spare = (device.swapchain.queue_present_khr)(queue, &spare_present);
        let r_real = (device.swapchain.queue_present_khr)(queue, pi);
        record(st, id, start, false, false);
        combine_present_results(r_spare, r_real)
    }

    /// The destroy line's counts.
    pub fn counts(&self) -> (u64, u64, u64) {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        (st.stats.real, st.stats.generated, st.stats.skipped)
    }
}

/// Updates counters and timing windows after one real present, logging every 10 seconds.
fn record(st: &mut GenState, id: u64, start: Instant, generated: bool, skipped: bool) {
    let now = Instant::now();
    let s = &mut st.stats;
    s.real += 1;
    if generated {
        s.generated += 1;
    }
    if skipped {
        s.skipped += 1;
    }
    if let Some(last) = s.last_present {
        s.intervals.push(now.duration_since(last).as_micros().min(u32::MAX as u128) as u32);
    }
    s.last_present = Some(now);
    s.present_cost.push(now.duration_since(start).as_micros().min(u32::MAX as u128) as u32);
    let now_ms = now.duration_since(*START).as_millis() as u64;
    if s.last_log_ms == 0 {
        s.last_log_ms = now_ms.max(1);
    } else if should_log(now_ms, s.last_log_ms, LOG_EVERY_MS) {
        let secs = (now_ms - s.last_log_ms) as f32 / 1000.0;
        if let (Some(iv), Some(cost)) = (s.intervals.summary(), s.present_cost.summary()) {
            let real_fps = iv.count as f32 / secs;
            let out_per_real = if s.real > 0 { (s.real + s.generated) as f32 / s.real as f32 } else { 1.0 };
            LOG.log(&format!(
                "swapchain {id}: real {real_fps:.1} fps, out {:.1} fps, interval mean {:.2} ms sd {:.2} p99 {:.2}, added {:.2} ms p99 {:.2}, skipped {}",
                real_fps * out_per_real,
                iv.mean_ms,
                iv.stddev_ms,
                iv.p99_ms,
                cost.mean_ms,
                cost.p99_ms,
                s.skipped
            ));
        }
        s.intervals.clear();
        s.present_cost.clear();
        s.last_log_ms = now_ms;
    }
}

/// How long a present may wait for a spare image before showing the real frame alone.
pub const ACQUIRE_TIMEOUT_NS: u64 = 50_000_000;

/// The worst of the two present results, so the game always learns about a stale or lost swapchain.
pub fn combine_present_results(copy: vk::Result, real: vk::Result) -> vk::Result {
    fn severity(r: vk::Result) -> u8 {
        match r {
            vk::Result::ERROR_DEVICE_LOST => 4,
            vk::Result::ERROR_OUT_OF_DATE_KHR => 3,
            r if r.as_raw() < 0 => 2,
            vk::Result::SUBOPTIMAL_KHR => 1,
            _ => 0,
        }
    }
    if severity(copy) > severity(real) { copy } else { real }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquireAction {
    Generate(u32),
    SkipThisFrame,
    Disable,
}

/// What to do after acquiring the spare image.
pub fn acquire_outcome(result: vk::Result, index: u32) -> AcquireAction {
    match result {
        vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR => AcquireAction::Generate(index),
        vk::Result::TIMEOUT | vk::Result::NOT_READY | vk::Result::ERROR_OUT_OF_DATE_KHR => AcquireAction::SkipThisFrame,
        _ => AcquireAction::Disable,
    }
}

/// True for presents HaqEngine can duplicate: one swapchain and no extension structs.
pub fn present_shape_ok(pi: &vk::PresentInfoKHR) -> bool {
    pi.swapchain_count == 1 && !pi.p_swapchains.is_null() && !pi.p_image_indices.is_null() && pi.p_next.is_null()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::Result as R;

    #[test]
    fn combine_present_results_prefers_worst() {
        assert_eq!(combine_present_results(R::SUCCESS, R::SUBOPTIMAL_KHR), R::SUBOPTIMAL_KHR);
        assert_eq!(combine_present_results(R::ERROR_OUT_OF_DATE_KHR, R::SUCCESS), R::ERROR_OUT_OF_DATE_KHR);
        assert_eq!(combine_present_results(R::ERROR_OUT_OF_DATE_KHR, R::ERROR_DEVICE_LOST), R::ERROR_DEVICE_LOST);
        assert_eq!(combine_present_results(R::SUCCESS, R::SUCCESS), R::SUCCESS);
        assert_eq!(combine_present_results(R::ERROR_SURFACE_LOST_KHR, R::SUBOPTIMAL_KHR), R::ERROR_SURFACE_LOST_KHR);
    }

    #[test]
    fn acquire_outcome_maps_to_action() {
        assert_eq!(acquire_outcome(R::SUCCESS, 2), AcquireAction::Generate(2));
        assert_eq!(acquire_outcome(R::SUBOPTIMAL_KHR, 1), AcquireAction::Generate(1));
        assert_eq!(acquire_outcome(R::TIMEOUT, 0), AcquireAction::SkipThisFrame);
        assert_eq!(acquire_outcome(R::NOT_READY, 0), AcquireAction::SkipThisFrame);
        assert_eq!(acquire_outcome(R::ERROR_OUT_OF_DATE_KHR, 0), AcquireAction::SkipThisFrame);
        assert_eq!(acquire_outcome(R::ERROR_DEVICE_LOST, 0), AcquireAction::Disable);
        assert_eq!(acquire_outcome(R::ERROR_SURFACE_LOST_KHR, 0), AcquireAction::Disable);
    }

    #[test]
    fn present_shape_requires_single_swapchain() {
        let sc = [vk::SwapchainKHR::null(); 2];
        let idx = [0u32; 2];
        let one = vk::PresentInfoKHR {
            swapchain_count: 1,
            p_swapchains: sc.as_ptr(),
            p_image_indices: idx.as_ptr(),
            ..Default::default()
        };
        assert!(present_shape_ok(&one));
        let two = vk::PresentInfoKHR { swapchain_count: 2, ..one };
        assert!(!present_shape_ok(&two));
        let ext = 0u64;
        let chained = vk::PresentInfoKHR { p_next: &ext as *const u64 as *const std::ffi::c_void, ..one };
        assert!(!present_shape_ok(&chained));
        let none = vk::PresentInfoKHR { swapchain_count: 1, p_swapchains: std::ptr::null(), ..Default::default() };
        assert!(!present_shape_ok(&none));
    }
}
