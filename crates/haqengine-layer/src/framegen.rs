//! Per-swapchain resources for generated frames. Created once per swapchain, never per frame.

use crate::battery::BatteryResources;
use crate::pipelines::QualityPipelines;
use crate::quality::QualityResources;
use haqengine_core::frame_change::{is_cut, ChangeCounts, StillBuckets, StillDetector};
use crate::pipelines::Pipelines;
use crate::queue::{queue_can_transfer, QUEUES};
use crate::state::{DeviceData, LOG};
use ash::vk;
use ash::vk::Handle;
use crate::present_watch::PresentWatch;
use haqengine_core::pacing::pace_target;
use haqengine_core::{should_log, FrameStats, Mode, PackedFormat, MAX_GENERATED};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

/// Most wait semaphores a game present may carry for HaqEngine to duplicate it.
const MAX_WAITS: usize = 16;
/// How long to wait for a slot's previous copy before skipping a frame.
const SLOT_WAIT_NS: u64 = 50_000_000;
/// Longest wait for the screen before queueing a real frame's presents.
const PACE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(40);
/// How often pacing statistics are logged.
const LOG_EVERY_MS: u64 = 10_000;

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// How often (in generated frames) the snapshot trigger file is checked.
const DUMP_CHECK_EVERY: u64 = 30;

/// `HAQENGINE_DUMP`: `1` for `~/haqengine-dump`, or a directory; unset for no snapshot. A snapshot
/// is taken when the file `now` appears in that directory.
static DUMP_DIR: LazyLock<Option<std::path::PathBuf>> = LazyLock::new(|| {
    let v = std::env::var("HAQENGINE_DUMP").ok().filter(|v| !v.is_empty() && v != "0")?;
    if v == "1" {
        std::env::var("HOME").ok().map(|h| std::path::PathBuf::from(h).join("haqengine-dump"))
    } else {
        Some(std::path::PathBuf::from(v))
    }
});

/// One frame's worth of reusable copy resources.
pub struct Slot {
    pub cmd: vk::CommandBuffer,
    pub fence: vk::Fence,
    /// One acquire semaphore per spare image a present may use (null beyond the swapchain's
    /// multiplier).
    pub acquired: [vk::Semaphore; MAX_GENERATED],
}

/// Command pool and slots, created on the first generating present (the queue family is only known then).
pub struct Ready {
    pub family: u32,
    pub pool: vk::CommandPool,
    pub slots: Vec<Slot>,
    /// Two timestamps per slot, or null when the queue cannot time.
    pub queries: vk::QueryPool,
    pub timestamp_period_ns: f32,
    /// Whether a slot's timestamps were written and can be read.
    pub stamped: Vec<bool>,
    /// One "ready to present" semaphore per swapchain image. A present may still hold its
    /// semaphore until that image is acquired again, so it is reused only for the same image.
    pub image_ready: Vec<vk::Semaphore>,
}

pub struct GenStats {
    pub real: u64,
    pub generated: u64,
    pub skipped: u64,
    pub intervals: FrameStats,
    pub present_cost: FrameStats,
    pub gpu_cost: FrameStats,
    pub last_present: Option<Instant>,
    pub last_log_ms: u64,
    /// Real and generated presents since the last log line.
    pub window_real: u64,
    pub window_generated: u64,
    /// Time spent waiting for the screen before queueing generated presents.
    pub pace_wait: FrameStats,
}

pub struct GenState {
    pub ready: Option<Ready>,
    pub next: usize,
    pub stats: GenStats,
    pub battery: Option<BatteryResources>,
    pub battery_failed: bool,
    pub quality: Option<QualityResources>,
    pub quality_failed: bool,
    /// Decides when Battery skips generation on a still screen.
    pub still: StillDetector,
    /// The frame number last recorded in each slot (0 = never).
    pub slot_frames: Vec<u64>,
    pub frames: u64,
    /// Cuts seen and still presents since the last log line.
    pub window_cuts: u64,
    pub window_still: u64,
    /// How still each frame was since the last log line.
    pub window_buckets: StillBuckets,
    /// The engine this swapchain runs now (changes with a live config edit).
    pub mode: Mode,
    /// Frames per real frame now: the live setting, at most the swapchain's `multiplier`.
    pub live_multiplier: u32,
    /// The live-settings change counter last applied.
    pub seen_generation: u64,
    /// Watches tagged presents reach the screen while pacing is on.
    pub watch: Option<PresentWatch>,
    /// Pacing is off for this swapchain (the game tags presents itself, or watching failed).
    pub watch_off: bool,
    /// The last present id handed out, and the newest one queued for the screen.
    pub next_present_id: u64,
    pub last_present_id: u64,
}

/// Which engine recorded a present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recorded {
    Copy,
    Battery,
    Quality,
}

pub struct FrameGen {
    pub images: Vec<vk::Image>,
    pub extent: vk::Extent2D,
    /// The packed layout Battery and Quality use for this swapchain, or `None` when its format
    /// only allows copy mode.
    pub battery_format: Option<PackedFormat>,
    /// Whether the device can run Quality mode.
    pub quality_ok: bool,
    /// Frames shown per real frame this swapchain was created for (its extra images).
    pub multiplier: u32,
    pub disabled: AtomicBool,
    pub state: Mutex<GenState>,
}

impl FrameGen {
    pub fn new(
        images: Vec<vk::Image>,
        extent: vk::Extent2D,
        battery_format: Option<PackedFormat>,
        quality_ok: bool,
        mode: Mode,
        multiplier: u32,
    ) -> FrameGen {
        FrameGen {
            images,
            extent,
            battery_format,
            quality_ok,
            multiplier,
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
                    gpu_cost: FrameStats::new(),
                    last_present: None,
                    last_log_ms: 0,
                    window_real: 0,
                    window_generated: 0,
                    pace_wait: FrameStats::new(),
                },
                battery: None,
                battery_failed: false,
                quality: None,
                quality_failed: false,
                still: StillDetector::new(),
                slot_frames: Vec::new(),
                frames: 0,
                window_cuts: 0,
                window_still: 0,
                window_buckets: StillBuckets::default(),
                mode: engine_mode(mode, battery_format.is_some(), quality_ok),
                live_multiplier: multiplier,
                seen_generation: crate::state::live().0,
                watch: None,
                watch_off: false,
                next_present_id: 0,
                last_present_id: 0,
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
        let slot_count = self.images.len() + 2;
        let family_props = device.queue_families.get(family as usize).copied().unwrap_or_default();
        let queries = if family_props.timestamp_valid_bits > 0 {
            d.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2 * slot_count as u32),
                None,
            )
            .unwrap_or(vk::QueryPool::null())
        } else {
            vk::QueryPool::null()
        };
        let timestamp_period_ns =
            device.instance.instance.get_physical_device_properties(device.physical_device).limits.timestamp_period;
        let mut ready = Ready {
            family,
            pool,
            slots: Vec::with_capacity(slot_count),
            queries,
            timestamp_period_ns,
            stamped: vec![false; slot_count],
            image_ready: Vec::with_capacity(self.images.len()),
        };
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
                ready.slots.push(Slot { cmd, fence, acquired: [vk::Semaphore::null(); MAX_GENERATED] });
                let last = ready.slots.last_mut().expect("just pushed");
                for a in last.acquired.iter_mut().take(self.max_generated()) {
                    *a = sem()?;
                }
            }
            for _ in 0..self.images.len() {
                let s = d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                ready.image_ready.push(s);
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
        state.watch = None;
        if let Some(ready) = state.ready.take() {
            destroy_ready(device, &ready);
        }
        if let Some(battery) = state.battery.take() {
            battery.destroy(&device.device);
        }
        if let Some(quality) = state.quality.take() {
            quality.destroy(&device.device);
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
        for &a in s.acquired.iter().filter(|a| **a != vk::Semaphore::null()) {
            d.destroy_semaphore(a, None);
        }
        if s.fence != vk::Fence::null() {
            d.destroy_fence(s.fence, None);
        }
    }
    for sem in &ready.image_ready {
        d.destroy_semaphore(*sem, None);
    }
    d.destroy_command_pool(ready.pool, None);
    if ready.queries != vk::QueryPool::null() {
        d.destroy_query_pool(ready.queries, None);
    }
}

impl FrameGen {
    /// Generated frames per real frame the swapchain has images for.
    fn max_generated(&self) -> usize {
        crate::battery::outputs_for_multiplier(self.multiplier)
    }

    /// Generated frames this present makes: the swapchain's count for Battery, one for Quality and
    /// copy mode.
    fn generated_now(&self, st: &GenState) -> usize {
        let quality = st.mode == Mode::Quality && !st.quality_failed;
        let battery = st.mode != Mode::Copy && self.battery_format.is_some() && !st.battery_failed && !quality;
        if battery { crate::battery::outputs_for_multiplier(st.live_multiplier).min(self.max_generated()) } else { 1 }
    }

    /// Applies a live config change to this swapchain: the multiplier at once (within what the
    /// swapchain was created for), a mode change after the slots' work is done, freeing the
    /// resources the new mode does not use.
    unsafe fn apply_live(&self, st: &mut GenState, id: u64, device: &DeviceData, generation: u64, live: haqengine_core::config::Live) {
        st.seen_generation = generation;
        let m = effective_multiplier(live.multiplier, self.multiplier);
        if m < live.multiplier {
            LOG.log(&format!("swapchain {id}: multiplier {} applies after the game recreates its swapchain; using {m}", live.multiplier));
        }
        let mode = engine_mode(live.mode, self.battery_format.is_some(), self.quality_ok);
        if live.mode == Mode::Quality && mode != Mode::Quality {
            LOG.log(&format!("swapchain {id}: quality unavailable on this device or format; using {mode:?}"));
        }
        if mode != st.mode {
            let (drop_battery, drop_quality) = mode_change_drops(st.mode, mode);
            let d = &device.device;
            if let Some(ready) = st.ready.as_ref() {
                let fences: Vec<vk::Fence> = ready.slots.iter().map(|s| s.fence).collect();
                if d.wait_for_fences(&fences, true, SLOT_WAIT_NS).is_err() {
                    LOG.log(&format!("swapchain {id}: mode change waits for in-flight frames"));
                    st.seen_generation = generation.wrapping_sub(1);
                    return;
                }
            }
            if drop_battery {
                if let Some(b) = st.battery.take() {
                    b.destroy(d);
                }
                st.battery_failed = false;
            }
            if drop_quality {
                if let Some(q) = st.quality.take() {
                    q.destroy(d);
                }
                st.quality_failed = false;
            }
            st.still = StillDetector::new();
            st.mode = mode;
        }
        st.live_multiplier = m;
        LOG.log(&format!("swapchain {id}: now {:?} {}x", st.mode, shown_multiplier(st.mode, m)));
    }

    /// Whether this present's frames are tagged and watched (and paced, unless `HAQENGINE_PACING`
    /// is off): the device has present waits, the game does not tag its own presents, and the
    /// watch thread runs (started here on first use).
    unsafe fn pacing_ready(&self, st: &mut GenState, id: u64, device: &DeviceData, swapchain: vk::SwapchainKHR, pi: &vk::PresentInfoKHR) -> bool {
        let Some(wait) = device.wait_for_present else { return false };
        if st.watch_off {
            return false;
        }
        let off = if has_present_id(pi.p_next) {
            Some("the game sets present ids".to_string())
        } else {
            st.watch.as_ref().and_then(|w| w.failed()).map(|e| format!("waiting for presents failed {e:?}"))
        };
        if let Some(why) = off {
            st.watch_off = true;
            st.watch = None;
            LOG.log(&format!("swapchain {id}: pacing off: {why}"));
            return false;
        }
        if st.watch.is_none() {
            st.watch = PresentWatch::start(device.device.handle(), swapchain, wait);
            if st.watch.is_none() {
                st.watch_off = true;
                LOG.log(&format!("swapchain {id}: pacing off: the watch thread did not start"));
                return false;
            }
            LOG.log(&format!("swapchain {id}: watching presents, pacing {}", if crate::state::SETTINGS.pacing { "on" } else { "off" }));
        }
        true
    }

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
            if let Some(b) = st.battery.as_mut() {
                b.forget_previous();
            }
            if let Some(q) = st.quality.as_mut() {
                q.forget_previous();
            }
            let r = next_present(queue, pi);
            record(st, id, start, 0, skipped);
            r
        };
        crate::state::poll_config(start);
        let (generation, live) = crate::state::live();
        if generation != st.seen_generation {
            self.apply_live(&mut st, id, device, generation, live);
        }
        if st.live_multiplier < 2 {
            return pass(&mut st, false);
        }
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
        let slot_index = st.next % st.ready.as_ref().expect("ensure_ready succeeded").slots.len();
        let slot = {
            let ready = st.ready.as_ref().expect("ensure_ready succeeded");
            let s = &ready.slots[slot_index];
            Slot { cmd: s.cmd, fence: s.fence, acquired: s.acquired }
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
        read_gpu_time(&mut st, d, slot_index);
        let swapchain = *pi.p_swapchains;
        let game_index = *pi.p_image_indices;
        if self.still_screen(&mut st, d, slot_index) {
            return self.present_still(&mut st, id, device, queue, pi, slot, slot_index, game_index, start);
        }
        let count = self.generated_now(&st);
        let tag = self.pacing_ready(&mut st, id, device, swapchain, pi);
        if tag && crate::state::SETTINGS.pacing {
            if let (Some(target), Some(w)) = (pace_target(st.last_present_id), st.watch.as_ref()) {
                let waited = w.wait_shown(target, PACE_TIMEOUT);
                st.stats.pace_wait.push(waited.as_micros().min(u32::MAX as u128) as u32);
            }
        }
        let mut spares = [0u32; MAX_GENERATED];
        for k in 0..count {
            let r = (device.swapchain.acquire_next_image_khr)(
                d.handle(),
                swapchain,
                ACQUIRE_TIMEOUT_NS,
                slot.acquired[k],
                vk::Fence::null(),
                &mut spares[k],
            );
            match acquire_outcome(r, spares[k]) {
                AcquireAction::Generate(i) => spares[k] = i,
                AcquireAction::SkipThisFrame if k == 0 => return pass(&mut st, true),
                AcquireAction::Disable if k == 0 => {
                    self.disable(id, "acquiring a spare image", r);
                    return pass(&mut st, false);
                }
                AcquireAction::SkipThisFrame => {
                    return self.give_back(&mut st, device, queue, pi, swapchain, &spares[..k], &slot.acquired[..k], id, start, true);
                }
                AcquireAction::Disable => {
                    self.disable(id, "acquiring a spare image", r);
                    return self.give_back(&mut st, device, queue, pi, swapchain, &spares[..k], &slot.acquired[..k], id, start, false);
                }
            }
        }
        let spares = &spares[..count];
        let acquired = &slot.acquired[..count];
        let mut spare_images = [vk::Image::null(); MAX_GENERATED];
        let mut in_range = true;
        for (k, &i) in spares.iter().enumerate() {
            match self.images.get(i as usize) {
                Some(&img) => spare_images[k] = img,
                None => in_range = false,
            }
        }
        let Some(&game_image) = self.images.get(game_index as usize).filter(|_| in_range) else {
            self.disable(id, "image index out of range", vk::Result::ERROR_UNKNOWN);
            return self.give_back(&mut st, device, queue, pi, swapchain, spares, acquired, id, start, false);
        };
        let recorded = match self.record_frame(&mut st, id, device, slot.cmd, slot_index, game_image, &spare_images[..count]) {
            Ok(b) => b,
            Err(e) => {
                self.disable(id, "recording the frame", e);
                return self.give_back(&mut st, device, queue, pi, swapchain, spares, acquired, id, start, false);
            }
        };
        let mut waits = [vk::Semaphore::null(); MAX_WAITS + MAX_GENERATED];
        let n = pi.wait_semaphore_count as usize;
        if n > 0 {
            waits[..n].copy_from_slice(std::slice::from_raw_parts(pi.p_wait_semaphores, n));
        }
        waits[n..n + count].copy_from_slice(acquired);
        let stages = [vk::PipelineStageFlags::TRANSFER; MAX_WAITS + MAX_GENERATED];
        let ready_sems = st.ready.as_ref().map(|r| r.image_ready.as_slice());
        let Some((indices, sems)) =
            ready_sems.and_then(|sems| present_semaphore_indices(sems.len(), spares, game_index).map(|i| (i, sems)))
        else {
            self.disable(id, "present semaphores for the images", vk::Result::ERROR_UNKNOWN);
            return self.give_back(&mut st, device, queue, pi, swapchain, spares, acquired, id, start, false);
        };
        let mut signals = [vk::Semaphore::null(); MAX_GENERATED + 1];
        for k in 0..=count {
            signals[k] = sems[indices[k]];
        }
        let cmds = [slot.cmd];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&waits[..n + count])
            .wait_dst_stage_mask(&stages[..n + count])
            .command_buffers(&cmds)
            .signal_semaphores(&signals[..=count]);
        if let Err(e) = d.reset_fences(&[slot.fence]) {
            self.disable(id, "resetting a fence", e);
            return self.give_back(&mut st, device, queue, pi, swapchain, spares, acquired, id, start, false);
        }
        if let Err(e) = d.queue_submit(queue, &[submit], slot.fence) {
            let _ = d.queue_submit(queue, &[], slot.fence);
            self.disable(id, "submitting the copy", e);
            return self.give_back(&mut st, device, queue, pi, swapchain, spares, acquired, id, start, false);
        }
        if let Some(ready) = st.ready.as_mut() {
            ready.stamped[slot_index] = ready.queries != vk::QueryPool::null();
        }
        note_slot_frame(&mut st, slot_index);
        if recorded == Recorded::Quality {
            if let Some(q) = st.quality.as_mut() {
                q.submitted();
            }
        }
        if recorded == Recorded::Battery {
            if let Some(dir) = DUMP_DIR.as_ref().filter(|_| st.stats.generated % DUMP_CHECK_EVERY == 0) {
                let trigger = dir.join("now");
                if trigger.exists() {
                    let _ = std::fs::remove_file(&trigger);
                    if let (Some(b), Some(ready)) = (st.battery.as_ref(), st.ready.as_ref()) {
                        let r = b.dump(device, ready.pool, queue, dir);
                        LOG.log(&format!("debug snapshot to {}: {r:?}", dir.display()));
                    }
                }
            }
            if let Some(b) = st.battery.as_mut() {
                b.submitted();
            }
        }
        let swapchains = [swapchain];
        let mut ids = [0u64; MAX_GENERATED + 1];
        if tag {
            for slot_id in ids.iter_mut().take(count + 1) {
                st.next_present_id += 1;
                *slot_id = st.next_present_id;
            }
        }
        let mut result = vk::Result::SUCCESS;
        for k in 0..count {
            let wait = [signals[k]];
            let index = [spares[k]];
            let mut present_id = vk::PresentIdKHR::default().present_ids(&ids[k..=k]);
            let mut present = vk::PresentInfoKHR::default().wait_semaphores(&wait).swapchains(&swapchains).image_indices(&index);
            if tag {
                present = present.push_next(&mut present_id);
            }
            let r = next_present(queue, &present);
            if tag {
                note_queued(&mut st, r, ids[k], None);
            }
            result = combine_present_results(result, r);
        }
        let real_wait = [signals[count]];
        let mut real_id = vk::PresentIdKHR::default().present_ids(&ids[count..=count]);
        real_id.p_next = pi.p_next;
        let mut real_present = *pi;
        real_present.wait_semaphore_count = 1;
        real_present.p_wait_semaphores = real_wait.as_ptr();
        if tag {
            real_present.p_next = &real_id as *const vk::PresentIdKHR as *const std::ffi::c_void;
        }
        let r_real = next_present(queue, &real_present);
        if tag {
            let resumed = st.stats.last_present;
            note_queued(&mut st, r_real, ids[count], resumed);
        }
        record(&mut st, id, start, count as u64, false);
        combine_present_results(result, r_real)
    }

    /// Whether this present should skip generation because the screen has been still. Also counts
    /// the reused slot's frame as a cut when it was one. Battery swapchains only.
    unsafe fn still_screen(&self, st: &mut GenState, d: &ash::Device, slot_index: usize) -> bool {
        if st.mode != Mode::Battery {
            return false;
        }
        let Some(battery) = st.battery.as_ref() else { return false };
        if let Some(c) = battery.counts(slot_index) {
            if is_cut(&c) {
                st.window_cuts += 1;
            }
            st.window_buckets.add(&c);
        }
        let Some(ready) = st.ready.as_ref() else { return false };
        let slots = ready.slots.iter().enumerate().map(|(i, s)| {
            let done = i == slot_index || d.get_fence_status(s.fence).unwrap_or(false);
            (done, st.slot_frames.get(i).copied().unwrap_or(0), battery.counts(i))
        });
        let counts = newest_ready_counts(slots);
        st.still.observe(counts)
    }

    /// A present on a still screen: Battery still runs (so the counts stay current and the next
    /// moving frame has its history), but only the game's frame is shown, without waiting a
    /// refresh behind a generated one.
    #[allow(clippy::too_many_arguments)]
    unsafe fn present_still(
        &self,
        st: &mut GenState,
        id: u64,
        device: &DeviceData,
        queue: vk::Queue,
        pi: &vk::PresentInfoKHR,
        slot: Slot,
        slot_index: usize,
        game_index: u32,
        start: Instant,
    ) -> vk::Result {
        let d = &device.device;
        let next_present = device.swapchain.queue_present_khr;
        let plain = |st: &mut GenState| {
            let r = next_present(queue, pi);
            record(st, id, start, 0, false);
            r
        };
        let Some(&game_image) = self.images.get(game_index as usize) else {
            return plain(st);
        };
        let real_ready = match st.ready.as_ref().and_then(|r| r.image_ready.get(game_index as usize).copied()) {
            Some(s) => s,
            None => return plain(st),
        };
        if let Err(e) = self.record_frame(st, id, device, slot.cmd, slot_index, game_image, &[]) {
            self.disable(id, "recording a still frame", e);
            return plain(st);
        }
        let n = pi.wait_semaphore_count as usize;
        let mut waits = [vk::Semaphore::null(); MAX_WAITS + 1];
        if n > 0 {
            waits[..n].copy_from_slice(std::slice::from_raw_parts(pi.p_wait_semaphores, n));
        }
        let stages = [vk::PipelineStageFlags::TRANSFER; MAX_WAITS + 1];
        let signals = [real_ready];
        let cmds = [slot.cmd];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&waits[..n])
            .wait_dst_stage_mask(&stages[..n])
            .command_buffers(&cmds)
            .signal_semaphores(&signals);
        if let Err(e) = d.reset_fences(&[slot.fence]) {
            self.disable(id, "resetting a fence", e);
            return plain(st);
        }
        if let Err(e) = d.queue_submit(queue, &[submit], slot.fence) {
            let _ = d.queue_submit(queue, &[], slot.fence);
            self.disable(id, "submitting a still frame", e);
            return plain(st);
        }
        if let Some(ready) = st.ready.as_mut() {
            ready.stamped[slot_index] = ready.queries != vk::QueryPool::null();
        }
        note_slot_frame(st, slot_index);
        if let Some(b) = st.battery.as_mut() {
            b.submitted();
        }
        st.window_still += 1;
        let real_wait = [real_ready];
        let mut real_present = *pi;
        real_present.wait_semaphore_count = 1;
        real_present.p_wait_semaphores = real_wait.as_ptr();
        let r = next_present(queue, &real_present);
        record(st, id, start, 0, false);
        r
    }

    /// Records this present's commands: Quality or Battery interpolation when available, otherwise
    /// a copy, bracketed by timestamps. Returns which engine recorded the frame.
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_frame(
        &self,
        st: &mut GenState,
        id: u64,
        device: &DeviceData,
        cmd: vk::CommandBuffer,
        slot_index: usize,
        game: vk::Image,
        spares: &[vk::Image],
    ) -> Result<Recorded, vk::Result> {
        let spare = spares.first().copied();
        let d = &device.device;
        d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let queries = st.ready.as_ref().map_or(vk::QueryPool::null(), |r| r.queries);
        let first = 2 * slot_index as u32;
        if queries != vk::QueryPool::null() {
            d.cmd_reset_query_pool(cmd, queries, first, 2);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, first);
        }
        let quality = self.quality_pipelines(st, id, device);
        let recorded = if let (Some(p), Some(q), Some(spare)) = (quality, st.quality.as_mut(), spare) {
            q.record(d, cmd, &p, game, spare, slot_index);
            Recorded::Quality
        } else {
            match (self.battery_pipelines(st, id, device), st.battery.as_mut()) {
                (Some(p), Some(b)) => {
                    b.record(d, cmd, &p, game, spares, slot_index);
                    Recorded::Battery
                }
                _ => {
                    if let Some(spare) = spare {
                        self.record_copy_body(d, cmd, game, spare);
                    }
                    Recorded::Copy
                }
            }
        };
        if queries != vk::QueryPool::null() {
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, first + 1);
        }
        d.end_command_buffer(cmd)?;
        Ok(recorded)
    }

    /// The device's Quality pipelines with this swapchain's resources created, or `None` when the
    /// swapchain does not run Quality (or after a creation failure, logged once; Battery takes over).
    unsafe fn quality_pipelines(&self, st: &mut GenState, id: u64, device: &DeviceData) -> Option<Arc<QualityPipelines>> {
        if st.mode != Mode::Quality || st.quality_failed {
            return None;
        }
        let format = self.battery_format?;
        let pipelines = match device.quality_pipelines.get_or_init(|| QualityPipelines::create(&device.device).map(Arc::new)) {
            Ok(p) => Arc::clone(p),
            Err(e) => {
                st.quality_failed = true;
                LOG.log(&format!("quality mode off for swapchain {id}: creating pipelines {e:?}; using battery"));
                return None;
            }
        };
        if st.quality.is_none() {
            let slots = st.ready.as_ref().map_or(0, |r| r.slots.len());
            match QualityResources::create(device, &pipelines, self.extent, format, slots) {
                Ok(q) => {
                    LOG.log(&format!("swapchain {id}: quality resources {} bytes in 2 allocations", q.bytes));
                    st.quality = Some(q);
                }
                Err(e) => {
                    st.quality_failed = true;
                    LOG.log(&format!("quality mode off for swapchain {id}: creating resources {e:?}; using battery"));
                    return None;
                }
            }
        }
        Some(pipelines)
    }

    /// The device's battery pipelines with this swapchain's resources created, or `None` for copy
    /// mode (or after a creation failure, logged once).
    unsafe fn battery_pipelines(&self, st: &mut GenState, id: u64, device: &DeviceData) -> Option<Arc<Pipelines>> {
        let format = self.battery_format?;
        if st.battery_failed || st.mode == Mode::Copy {
            return None;
        }
        let pipelines = match device.pipelines.get_or_init(|| Pipelines::create(&device.device).map(Arc::new)) {
            Ok(p) => Arc::clone(p),
            Err(e) => {
                st.battery_failed = true;
                LOG.log(&format!("battery mode off for swapchain {id}: creating pipelines {e:?}"));
                return None;
            }
        };
        if st.battery.is_none() {
            let slots = st.ready.as_ref().map_or(0, |r| r.slots.len());
            match BatteryResources::create(device, &pipelines, self.extent, format, slots, self.max_generated()) {
                Ok(b) => st.battery = Some(b),
                Err(e) => {
                    st.battery_failed = true;
                    LOG.log(&format!("battery mode off for swapchain {id}: creating resources {e:?}"));
                    return None;
                }
            }
        }
        Some(pipelines)
    }

    /// Records copying the game's image into the spare image, leaving both ready to present.
    unsafe fn record_copy_body(&self, d: &ash::Device, cmd: vk::CommandBuffer, src: vk::Image, dst: vk::Image) {
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
    }

    /// Error path after spare images were acquired: present each as-is (waiting on its acquire
    /// semaphore) so the swapchain does not lose images, then present the game's frame unchanged.
    #[allow(clippy::too_many_arguments)]
    unsafe fn give_back(
        &self,
        st: &mut GenState,
        device: &DeviceData,
        queue: vk::Queue,
        pi: &vk::PresentInfoKHR,
        swapchain: vk::SwapchainKHR,
        spares: &[u32],
        acquired: &[vk::Semaphore],
        id: u64,
        start: Instant,
        skipped: bool,
    ) -> vk::Result {
        let swapchains = [swapchain];
        let mut result = vk::Result::SUCCESS;
        for (&spare, &sem) in spares.iter().zip(acquired) {
            let waits = [sem];
            let indices = [spare];
            let spare_present = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&swapchains).image_indices(&indices);
            result = combine_present_results(result, (device.swapchain.queue_present_khr)(queue, &spare_present));
        }
        let r_real = (device.swapchain.queue_present_khr)(queue, pi);
        record(st, id, start, 0, skipped);
        combine_present_results(result, r_real)
    }

    /// The destroy line's counts.
    pub fn counts(&self) -> (u64, u64, u64) {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        (st.stats.real, st.stats.generated, st.stats.skipped)
    }
}

/// Reads a slot's GPU time from its previous use (the slot's fence has signalled).
unsafe fn read_gpu_time(st: &mut GenState, d: &ash::Device, slot_index: usize) {
    let Some(ready) = st.ready.as_mut() else { return };
    if ready.queries == vk::QueryPool::null() || !ready.stamped[slot_index] {
        return;
    }
    let mut stamps = [0u64; 2];
    let r = d.get_query_pool_results(ready.queries, 2 * slot_index as u32, &mut stamps, vk::QueryResultFlags::TYPE_64);
    ready.stamped[slot_index] = false;
    if r.is_ok() && stamps[1] >= stamps[0] {
        let us = (stamps[1] - stamps[0]) as f64 * ready.timestamp_period_ns as f64 / 1000.0;
        st.stats.gpu_cost.push(us.min(u32::MAX as f64) as u32);
    }
}

/// Hands a tagged present that reached the queue to the watch thread.
fn note_queued(st: &mut GenState, r: vk::Result, present_id: u64, real_start: Option<Instant>) {
    if r != vk::Result::SUCCESS && r != vk::Result::SUBOPTIMAL_KHR {
        return;
    }
    if let Some(w) = st.watch.as_ref() {
        w.queued(present_id, real_start);
        st.last_present_id = present_id;
    }
}

/// Updates counters and timing windows after one real present, logging every 10 seconds.
fn record(st: &mut GenState, id: u64, start: Instant, generated: u64, skipped: bool) {
    let now = Instant::now();
    let GenState { stats: s, quality: q, battery: bat, window_cuts, window_still, window_buckets, watch, .. } = st;
    s.real += 1;
    s.generated += generated;
    s.window_real += 1;
    s.window_generated += generated;
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
            let quality = q
                .as_mut()
                .map(|q| q.take_counts())
                .map(|c| format!(", quality cuts {} copies {} restarts {}", c.cuts, c.copies, c.restarts))
                .unwrap_or_default();
            let quality = if bat.is_some() {
                let c = change_suffix(*window_cuts, *window_still, window_buckets);
                *window_cuts = 0;
                *window_still = 0;
                window_buckets.clear();
                quality + &c
            } else {
                quality
            };
            let gpu = s
                .gpu_cost
                .summary()
                .map(|g| format!(", gpu {:.2} ms p99 {:.2}", g.mean_ms, g.p99_ms))
                .unwrap_or_default();
            let shown = watch
                .as_ref()
                .and_then(|w| w.take_shown())
                .map(|v| format!(", to screen {:.2} ms p99 {:.2}", v.mean_ms, v.p99_ms))
                .unwrap_or_default();
            let paced = s.pace_wait.summary().map(|p| format!(", paced {:.2} ms p99 {:.2}", p.mean_ms, p.p99_ms)).unwrap_or_default();
            LOG.log(&format!(
                "swapchain {id}: real {real_fps:.1} fps, out {:.1} fps, interval mean {:.2} ms sd {:.2} p99 {:.2}, added {:.2} ms p99 {:.2}, skipped {}{shown}{paced}{gpu}{quality}",
                out_fps(real_fps, s.window_real, s.window_generated),
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
        s.gpu_cost.clear();
        s.pace_wait.clear();
        s.window_real = 0;
        s.window_generated = 0;
        s.last_log_ms = now_ms;
    }
}

/// How long a present may wait for a spare image before showing the real frame alone.
pub const ACQUIRE_TIMEOUT_NS: u64 = 50_000_000;

/// The worst of the two present results, so the game always learns about a stale or lost swapchain.
/// Remembers which frame number the slot now holds.
fn note_slot_frame(st: &mut GenState, slot_index: usize) {
    st.frames += 1;
    if st.slot_frames.len() <= slot_index {
        st.slot_frames.resize(slot_index + 1, 0);
    }
    st.slot_frames[slot_index] = st.frames;
}

/// The counts of the newest frame whose slot has finished: `(finished, frame number, counts)` per
/// slot. `None` when no slot has finished or the newest finished one has no counts. Takes an
/// iterator so a present allocates nothing.
pub fn newest_ready_counts(slots: impl Iterator<Item = (bool, u64, Option<ChangeCounts>)>) -> Option<ChangeCounts> {
    slots.filter(|s| s.0 && s.1 > 0).max_by_key(|s| s.1).and_then(|s| s.2)
}

/// Whether a present's `pNext` chain carries the game's own `VkPresentIdKHR`.
///
/// # Safety
/// `p_next` must be null or a valid structure chain.
pub unsafe fn has_present_id(p_next: *const std::ffi::c_void) -> bool {
    let mut cur = p_next as *const vk::BaseInStructure;
    while !cur.is_null() {
        if (*cur).s_type == vk::StructureType::PRESENT_ID_KHR {
            return true;
        }
        cur = (*cur).p_next;
    }
    false
}

/// Frames shown per second in a log window with `real` real and `generated` generated presents.
pub fn out_fps(real_fps: f32, real: u64, generated: u64) -> f32 {
    if real == 0 { real_fps } else { real_fps * (real + generated) as f32 / real as f32 }
}

/// The log line's change fields.
pub fn change_suffix(cuts: u64, still: u64, buckets: &StillBuckets) -> String {
    format!(", cuts {cuts}, still {still}, {buckets}")
}

/// Frames per real frame a swapchain created for `created` runs when the live setting is `live`.
pub fn effective_multiplier(live: u32, created: u32) -> u32 {
    live.min(created)
}

/// Frames per real frame the engine `mode` shows at multiplier `m`: Quality and copy mode make
/// one generated frame.
pub fn shown_multiplier(mode: Mode, m: u32) -> u32 {
    if mode == Mode::Battery { m } else { m.min(2) }
}

/// Which resources a switch from `from` to `to` frees: `(battery, quality)`. Copy mode keeps
/// neither; Battery drops Quality's and Quality drops Battery's.
pub fn mode_change_drops(from: Mode, to: Mode) -> (bool, bool) {
    if from == to {
        return (false, false);
    }
    match to {
        Mode::Copy => (true, true),
        Mode::Battery => (false, true),
        Mode::Quality => (true, false),
    }
}

/// The engine a swapchain can run for the requested mode: Quality needs the device's support
/// (else Battery), and both need a packed layout for the swapchain's format (else copy).
pub fn engine_mode(requested: Mode, battery_format: bool, quality_ok: bool) -> Mode {
    match requested {
        _ if !battery_format => Mode::Copy,
        Mode::Quality if !quality_ok => Mode::Battery,
        m => m,
    }
}

/// The image semaphores a generated present signals, in present order: each spare image's, then
/// the game image's (the entries after those are unused). `None` when there are no spares or more
/// than `MAX_GENERATED`, or when an index is out of range or two are the same image.
pub fn present_semaphore_indices(images: usize, spares: &[u32], game: u32) -> Option<[usize; MAX_GENERATED + 1]> {
    if spares.is_empty() || spares.len() > MAX_GENERATED {
        return None;
    }
    let mut out = [0usize; MAX_GENERATED + 1];
    for (k, &i) in spares.iter().chain(std::iter::once(&game)).enumerate() {
        let i = i as usize;
        if i >= images || out[..k].contains(&i) {
            return None;
        }
        out[k] = i;
    }
    Some(out)
}

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

/// True for presents HaqEngine can duplicate: exactly one swapchain. Extension structs in `pNext`
/// (e.g. `VkPresentIdKHR`) are kept on the real present; the copy is presented without them.
pub fn present_shape_ok(pi: &vk::PresentInfoKHR) -> bool {
    pi.swapchain_count == 1 && !pi.p_swapchains.is_null() && !pi.p_image_indices.is_null()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::Result as R;

    #[test]
    fn game_present_ids_are_found_in_the_chain() {
        let ids = [7u64];
        let mut id = vk::PresentIdKHR::default().present_ids(&ids);
        let mut regions = vk::PresentRegionsKHR::default();
        let with = vk::PresentInfoKHR::default().push_next(&mut regions).push_next(&mut id);
        assert!(unsafe { has_present_id(with.p_next) });
        let mut regions = vk::PresentRegionsKHR::default();
        let without = vk::PresentInfoKHR::default().push_next(&mut regions);
        assert!(!unsafe { has_present_id(without.p_next) });
        assert!(!unsafe { has_present_id(std::ptr::null()) });
    }

    #[test]
    fn out_fps_counts_this_window_only() {
        assert_eq!(out_fps(30.0, 300, 600), 90.0);
        assert_eq!(out_fps(30.0, 300, 0), 30.0);
        assert_eq!(out_fps(30.0, 0, 0), 30.0);
    }

    #[test]
    fn combine_present_results_prefers_worst() {
        assert_eq!(combine_present_results(R::SUCCESS, R::SUBOPTIMAL_KHR), R::SUBOPTIMAL_KHR);
        assert_eq!(combine_present_results(R::ERROR_OUT_OF_DATE_KHR, R::SUCCESS), R::ERROR_OUT_OF_DATE_KHR);
        assert_eq!(combine_present_results(R::ERROR_OUT_OF_DATE_KHR, R::ERROR_DEVICE_LOST), R::ERROR_DEVICE_LOST);
        assert_eq!(combine_present_results(R::SUCCESS, R::SUCCESS), R::SUCCESS);
        assert_eq!(combine_present_results(R::ERROR_SURFACE_LOST_KHR, R::SUBOPTIMAL_KHR), R::ERROR_SURFACE_LOST_KHR);
    }

    #[test]
    fn newest_ready_slot_counts() {
        use haqengine_core::frame_change::ChangeCounts;
        let c = |n: u32| Some(ChangeCounts { disagree: n, still: 0, pixels: 100 });
        let n = |slots: &[(bool, u64, Option<ChangeCounts>)]| newest_ready_counts(slots.iter().copied());
        assert_eq!(n(&[(true, 5, c(1)), (true, 7, c(2)), (false, 9, c(3))]), c(2));
        assert_eq!(n(&[(false, 5, c(1)), (false, 7, c(2))]), None);
        assert_eq!(n(&[(true, 8, None), (true, 7, c(4))]), None, "the newest ready frame has no counts");
        assert_eq!(n(&[]), None);
    }

    #[test]
    fn log_line_shows_cuts_and_still() {
        let mut b = StillBuckets::default();
        b.add(&ChangeCounts { disagree: 0, still: 90, pixels: 100 });
        assert_eq!(change_suffix(3, 41, &b), ", cuts 3, still 41, still% <50 0, <75 0, <85 0, <99.5 1, rest 0");
        assert_eq!(change_suffix(0, 0, &StillBuckets::default()), ", cuts 0, still 0, still% <50 0, <75 0, <85 0, <99.5 0, rest 0");
    }

    #[test]
    fn present_semaphore_indices_orders_spares_then_game() {
        assert_eq!(present_semaphore_indices(4, &[2], 3), Some([2, 3, 0, 0]));
        assert_eq!(present_semaphore_indices(6, &[4, 1], 0), Some([4, 1, 0, 0]));
        assert_eq!(present_semaphore_indices(6, &[4, 1, 5], 2), Some([4, 1, 5, 2]));
    }

    #[test]
    fn present_semaphore_indices_rejects_duplicates_and_out_of_range() {
        assert_eq!(present_semaphore_indices(4, &[0], 0), None);
        assert_eq!(present_semaphore_indices(4, &[4], 1), None);
        assert_eq!(present_semaphore_indices(4, &[1], 7), None);
        assert_eq!(present_semaphore_indices(6, &[2, 2], 1), None);
        assert_eq!(present_semaphore_indices(6, &[], 1), None);
        assert_eq!(present_semaphore_indices(9, &[1, 2, 3, 4], 0), None);
    }

    #[test]
    fn live_multiplier_change_applies_within_cap() {
        assert_eq!(effective_multiplier(2, 3), 2);
        assert_eq!(effective_multiplier(3, 3), 3);
        assert_eq!(effective_multiplier(4, 3), 3);
        assert_eq!(effective_multiplier(1, 3), 1);
    }

    #[test]
    fn mode_change_marks_resources_for_rebuild() {
        use haqengine_core::Mode;
        assert_eq!(mode_change_drops(Mode::Battery, Mode::Battery), (false, false));
        assert_eq!(mode_change_drops(Mode::Battery, Mode::Copy), (true, true));
        assert_eq!(mode_change_drops(Mode::Quality, Mode::Battery), (false, true));
        assert_eq!(mode_change_drops(Mode::Battery, Mode::Quality), (true, false));
    }

    #[test]
    fn shown_multiplier_follows_the_engine() {
        use haqengine_core::Mode;
        assert_eq!(shown_multiplier(Mode::Battery, 3), 3);
        assert_eq!(shown_multiplier(Mode::Copy, 3), 2);
        assert_eq!(shown_multiplier(Mode::Quality, 4), 2);
        assert_eq!(shown_multiplier(Mode::Copy, 1), 1);
    }

    #[test]
    fn engine_mode_falls_back() {
        use haqengine_core::Mode;
        assert_eq!(engine_mode(Mode::Quality, true, true), Mode::Quality);
        assert_eq!(engine_mode(Mode::Quality, true, false), Mode::Battery);
        assert_eq!(engine_mode(Mode::Battery, false, true), Mode::Copy);
        assert_eq!(engine_mode(Mode::Quality, false, true), Mode::Copy);
        assert_eq!(engine_mode(Mode::Copy, true, true), Mode::Copy);
    }

    #[test]
    fn battery_outputs_for_multiplier() {
        assert_eq!(crate::battery::outputs_for_multiplier(1), 1);
        assert_eq!(crate::battery::outputs_for_multiplier(2), 1);
        assert_eq!(crate::battery::outputs_for_multiplier(3), 2);
        assert_eq!(crate::battery::outputs_for_multiplier(4), 3);
        assert_eq!(crate::battery::outputs_for_multiplier(9), 3);
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
        assert!(present_shape_ok(&chained));
        let none = vk::PresentInfoKHR { swapchain_count: 1, p_swapchains: std::ptr::null(), ..Default::default() };
        assert!(!present_shape_ok(&none));
    }
}
