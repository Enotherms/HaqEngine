//! Per-swapchain resources for generated frames. Created once per swapchain, never per frame.

use crate::battery::BatteryResources;
use crate::pipelines::QualityPipelines;
use crate::quality::QualityResources;
use haqengine_core::frame_change::{is_cut, ChangeCounts, StillDetector};
use crate::pipelines::Pipelines;
use crate::queue::{queue_can_transfer, QUEUES};
use crate::state::{DeviceData, LOG};
use ash::vk;
use ash::vk::Handle;
use haqengine_core::{should_log, FrameStats, PackedFormat};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

/// Most wait semaphores a game present may carry for HaqEngine to duplicate it.
const MAX_WAITS: usize = 16;
/// How long to wait for a slot's previous copy before skipping a frame.
const SLOT_WAIT_NS: u64 = 50_000_000;
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
    pub acquired: vk::Semaphore,
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
    /// Battery mode's packed layout for this swapchain, or `None` for copy mode.
    pub battery_format: Option<PackedFormat>,
    /// Whether this swapchain runs Quality mode (falling back to Battery if it cannot start).
    pub quality: bool,
    pub disabled: AtomicBool,
    pub state: Mutex<GenState>,
}

impl FrameGen {
    pub fn new(images: Vec<vk::Image>, extent: vk::Extent2D, battery_format: Option<PackedFormat>, quality: bool) -> FrameGen {
        FrameGen {
            images,
            extent,
            battery_format,
            quality: quality && battery_format.is_some(),
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
                ready.slots.push(Slot { cmd, fence, acquired: vk::Semaphore::null() });
                let last = ready.slots.last_mut().expect("just pushed");
                last.acquired = sem()?;
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
        if s.acquired != vk::Semaphore::null() {
            d.destroy_semaphore(s.acquired, None);
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
        let recorded = match self.record_frame(&mut st, id, device, slot.cmd, slot_index, game_image, Some(spare_image)) {
            Ok(b) => b,
            Err(e) => {
                self.disable(id, "recording the frame", e);
                return self.give_back(&mut st, device, queue, pi, swapchain, spare, slot.acquired, id, start);
            }
        };
        let mut waits = [vk::Semaphore::null(); MAX_WAITS + 1];
        let n = pi.wait_semaphore_count as usize;
        if n > 0 {
            waits[..n].copy_from_slice(std::slice::from_raw_parts(pi.p_wait_semaphores, n));
        }
        waits[n] = slot.acquired;
        let stages = [vk::PipelineStageFlags::TRANSFER; MAX_WAITS + 1];
        let ready_sems = st.ready.as_ref().map(|r| (r.image_ready.len(), r.image_ready.as_slice()));
        let Some((copy_ready, real_ready)) = ready_sems
            .and_then(|(n, sems)| present_semaphore_indices(n, spare, game_index).map(|(c, r)| (sems[c], sems[r])))
        else {
            self.disable(id, "present semaphores for the images", vk::Result::ERROR_UNKNOWN);
            return self.give_back(&mut st, device, queue, pi, swapchain, spare, slot.acquired, id, start);
        };
        let signals = [copy_ready, real_ready];
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
        let copy_wait = [copy_ready];
        let copy_indices = [spare];
        let swapchains = [swapchain];
        let copy_present = vk::PresentInfoKHR::default()
            .wait_semaphores(&copy_wait)
            .swapchains(&swapchains)
            .image_indices(&copy_indices);
        let r_copy = next_present(queue, &copy_present);
        let real_wait = [real_ready];
        let mut real_present = *pi;
        real_present.wait_semaphore_count = 1;
        real_present.p_wait_semaphores = real_wait.as_ptr();
        let r_real = next_present(queue, &real_present);
        record(&mut st, id, start, true, false);
        combine_present_results(r_copy, r_real)
    }

    /// Whether this present should skip generation because the screen has been still. Also counts
    /// the reused slot's frame as a cut when it was one. Battery swapchains only.
    unsafe fn still_screen(&self, st: &mut GenState, d: &ash::Device, slot_index: usize) -> bool {
        if self.quality {
            return false;
        }
        let Some(battery) = st.battery.as_ref() else { return false };
        if battery.counts(slot_index).is_some_and(|c| is_cut(&c)) {
            st.window_cuts += 1;
        }
        let Some(ready) = st.ready.as_ref() else { return false };
        let slots: Vec<(bool, u64, Option<ChangeCounts>)> = ready
            .slots
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let done = i == slot_index || d.get_fence_status(s.fence).unwrap_or(false);
                (done, st.slot_frames.get(i).copied().unwrap_or(0), battery.counts(i))
            })
            .collect();
        let counts = newest_ready_counts(&slots);
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
            record(st, id, start, false, false);
            r
        };
        let Some(&game_image) = self.images.get(game_index as usize) else {
            return plain(st);
        };
        let real_ready = match st.ready.as_ref().and_then(|r| r.image_ready.get(game_index as usize).copied()) {
            Some(s) => s,
            None => return plain(st),
        };
        if let Err(e) = self.record_frame(st, id, device, slot.cmd, slot_index, game_image, None) {
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
        record(st, id, start, false, false);
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
        spare: Option<vk::Image>,
    ) -> Result<Recorded, vk::Result> {
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
                    b.record(d, cmd, &p, game, spare, slot_index);
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
        if !self.quality || st.quality_failed {
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
        if st.battery_failed {
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
            match BatteryResources::create(device, &pipelines, self.extent, format, slots) {
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

/// Updates counters and timing windows after one real present, logging every 10 seconds.
fn record(st: &mut GenState, id: u64, start: Instant, generated: bool, skipped: bool) {
    let now = Instant::now();
    let GenState { stats: s, quality: q, battery: bat, window_cuts, window_still, .. } = st;
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
            let quality = q
                .as_mut()
                .map(|q| q.take_counts())
                .map(|c| format!(", quality cuts {} copies {} restarts {}", c.cuts, c.copies, c.restarts))
                .unwrap_or_default();
            let quality = if bat.is_some() {
                let c = change_suffix(*window_cuts, *window_still);
                *window_cuts = 0;
                *window_still = 0;
                quality + &c
            } else {
                quality
            };
            let gpu = s
                .gpu_cost
                .summary()
                .map(|g| format!(", gpu {:.2} ms p99 {:.2}", g.mean_ms, g.p99_ms))
                .unwrap_or_default();
            LOG.log(&format!(
                "swapchain {id}: real {real_fps:.1} fps, out {:.1} fps, interval mean {:.2} ms sd {:.2} p99 {:.2}, added {:.2} ms p99 {:.2}, skipped {}{gpu}{quality}",
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
        s.gpu_cost.clear();
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
/// slot. `None` when no slot has finished or the newest finished one has no counts.
pub fn newest_ready_counts(slots: &[(bool, u64, Option<ChangeCounts>)]) -> Option<ChangeCounts> {
    slots.iter().filter(|s| s.0 && s.1 > 0).max_by_key(|s| s.1).and_then(|s| s.2)
}

/// The log line's change fields.
pub fn change_suffix(cuts: u64, still: u64) -> String {
    format!(", cuts {cuts}, still {still}")
}

/// The image semaphores a generated present signals: the spare image's (shown first) and the game
/// image's. `None` when an index is out of range or both name the same image.
pub fn present_semaphore_indices(images: usize, spare: u32, game: u32) -> Option<(usize, usize)> {
    let (s, g) = (spare as usize, game as usize);
    (s < images && g < images && s != g).then_some((s, g))
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
        assert_eq!(newest_ready_counts(&[(true, 5, c(1)), (true, 7, c(2)), (false, 9, c(3))]), c(2));
        assert_eq!(newest_ready_counts(&[(false, 5, c(1)), (false, 7, c(2))]), None);
        assert_eq!(newest_ready_counts(&[(true, 8, None), (true, 7, c(4))]), None, "the newest ready frame has no counts");
        assert_eq!(newest_ready_counts(&[]), None);
    }

    #[test]
    fn log_line_shows_cuts_and_still() {
        assert_eq!(change_suffix(3, 41), ", cuts 3, still 41");
        assert_eq!(change_suffix(0, 0), ", cuts 0, still 0");
    }

    #[test]
    fn present_semaphores_follow_images() {
        assert_eq!(present_semaphore_indices(4, 2, 3), Some((2, 3)));
        assert_eq!(present_semaphore_indices(4, 0, 0), None);
        assert_eq!(present_semaphore_indices(4, 4, 1), None);
        assert_eq!(present_semaphore_indices(4, 1, 7), None);
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
