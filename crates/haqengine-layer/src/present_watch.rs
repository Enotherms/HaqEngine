//! Watches when a swapchain's tagged presents reach the screen, on a thread of its own, so
//! generation can keep at most one present queued ahead and log how long real frames take to show.

use ash::vk;
use haqengine_core::pacing::ShownQueue;
use haqengine_core::{FrameStats, Summary};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// One `vkWaitForPresentKHR` slice, so a stop request is seen quickly.
const SLICE_NS: u64 = 20_000_000;
/// A present not on screen after this long is forgotten (it was dropped or never shown).
const LOST_AFTER: Duration = Duration::from_millis(200);
/// RADV can report `ERROR_DEVICE_LOST` for a wait that starts before the driver submitted the
/// present while the GPU is busy; the wait is retried until the error lasts this long.
const DEVICE_LOST_FOR: Duration = Duration::from_secs(1);
/// Pause before retrying a wait that reported `ERROR_DEVICE_LOST`.
const RETRY_PAUSE: Duration = Duration::from_millis(5);

struct Watched {
    queue: ShownQueue,
    shown_id: u64,
    stop: bool,
    failed: Option<vk::Result>,
    shown_us: FrameStats,
}

struct Shared {
    state: Mutex<Watched>,
    queued: Condvar,
    shown: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Watched> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub struct PresentWatch {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl PresentWatch {
    /// Starts watching `swapchain`, or `None` when the thread cannot start.
    pub fn start(device: vk::Device, swapchain: vk::SwapchainKHR, wait: vk::PFN_vkWaitForPresentKHR) -> Option<PresentWatch> {
        let shared = Arc::new(Shared {
            state: Mutex::new(Watched { queue: ShownQueue::default(), shown_id: 0, stop: false, failed: None, shown_us: FrameStats::new() }),
            queued: Condvar::new(),
            shown: Condvar::new(),
        });
        let theirs = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("haqengine-pacing".into())
            .spawn(move || watch(&theirs, device, swapchain, wait))
            .ok()?;
        Some(PresentWatch { shared, thread: Some(thread) })
    }

    /// Notes a present queued with `id`; a real frame carries when the game got control back
    /// before making it, so its time on screen is measured from there.
    pub fn queued(&self, id: u64, real_start: Option<Instant>) {
        self.shared.lock().queue.push(id, real_start);
        self.shared.queued.notify_one();
    }

    /// Waits until present `id` is on screen, at most `timeout`. Returns how long it waited.
    pub fn wait_shown(&self, id: u64, timeout: Duration) -> Duration {
        let began = Instant::now();
        let mut w = self.shared.lock();
        while w.shown_id < id && !w.stop && w.failed.is_none() {
            let Some(left) = timeout.checked_sub(began.elapsed()) else { break };
            w = self.shared.shown.wait_timeout(w, left).unwrap_or_else(|e| e.into_inner()).0;
        }
        began.elapsed()
    }

    /// Why watching stopped, if it did.
    pub fn failed(&self) -> Option<vk::Result> {
        self.shared.lock().failed
    }

    /// Times from the game getting control back to its real frame on screen, since the last call.
    pub fn take_shown(&self) -> Option<Summary> {
        let mut w = self.shared.lock();
        let s = w.shown_us.summary();
        w.shown_us.clear();
        s
    }
}

impl Drop for PresentWatch {
    fn drop(&mut self) {
        self.shared.lock().stop = true;
        self.shared.queued.notify_all();
        self.shared.shown.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn watch(shared: &Shared, device: vk::Device, swapchain: vk::SwapchainKHR, wait: vk::PFN_vkWaitForPresentKHR) {
    loop {
        let id = {
            let mut w = shared.lock();
            while w.queue.is_empty() && !w.stop {
                w = shared.queued.wait(w).unwrap_or_else(|e| e.into_inner());
            }
            if w.stop {
                return;
            }
            w.queue.oldest().expect("not empty")
        };
        let began = Instant::now();
        let mut lost_since: Option<Instant> = None;
        loop {
            let r = unsafe { wait(device, swapchain, id, SLICE_NS) };
            let mut w = shared.lock();
            match r {
                vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR => {
                    let Watched { queue, shown_us, .. } = &mut *w;
                    queue.shown(id, Instant::now(), |us| shown_us.push(us));
                    w.shown_id = w.shown_id.max(id);
                }
                vk::Result::TIMEOUT if w.stop => return,
                vk::Result::TIMEOUT if began.elapsed() < LOST_AFTER => continue,
                vk::Result::TIMEOUT => {
                    w.queue.drop_oldest();
                    w.shown_id = w.shown_id.max(id);
                }
                vk::Result::ERROR_DEVICE_LOST if !w.stop && lost_since.get_or_insert_with(Instant::now).elapsed() < DEVICE_LOST_FOR => {
                    drop(w);
                    std::thread::sleep(RETRY_PAUSE);
                    continue;
                }
                e => {
                    w.failed = Some(e);
                    drop(w);
                    shared.shown.notify_all();
                    return;
                }
            }
            drop(w);
            shared.shown.notify_all();
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    unsafe extern "system" fn shows_at_once(_: vk::Device, _: vk::SwapchainKHR, _: u64, _: u64) -> vk::Result {
        vk::Result::SUCCESS
    }

    unsafe extern "system" fn out_of_date(_: vk::Device, _: vk::SwapchainKHR, _: u64, _: u64) -> vk::Result {
        vk::Result::ERROR_OUT_OF_DATE_KHR
    }

    unsafe extern "system" fn never_shows(_: vk::Device, _: vk::SwapchainKHR, _: u64, timeout: u64) -> vk::Result {
        std::thread::sleep(Duration::from_nanos(timeout.min(SLICE_NS)));
        vk::Result::TIMEOUT
    }

    fn start(f: vk::PFN_vkWaitForPresentKHR) -> PresentWatch {
        PresentWatch::start(vk::Device::from_raw(1), vk::SwapchainKHR::from_raw(2), f).expect("thread")
    }

    #[test]
    fn shown_present_ends_the_wait_and_reports_real_frames() {
        let w = start(shows_at_once);
        w.queued(1, None);
        w.queued(2, Some(Instant::now()));
        assert!(w.wait_shown(2, Duration::from_secs(2)) < Duration::from_secs(1));
        assert_eq!(w.take_shown().map(|s| s.count), Some(1));
        assert!(w.take_shown().is_none());
        assert!(w.failed().is_none());
    }

    static FLAKY_CALLS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    unsafe extern "system" fn lost_three_times(_: vk::Device, _: vk::SwapchainKHR, _: u64, _: u64) -> vk::Result {
        if FLAKY_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 3 {
            vk::Result::ERROR_DEVICE_LOST
        } else {
            vk::Result::SUCCESS
        }
    }

    unsafe extern "system" fn always_lost(_: vk::Device, _: vk::SwapchainKHR, _: u64, _: u64) -> vk::Result {
        vk::Result::ERROR_DEVICE_LOST
    }

    #[test]
    fn a_passing_device_lost_is_retried() {
        let w = start(lost_three_times);
        w.queued(1, Some(Instant::now()));
        assert!(w.wait_shown(1, Duration::from_secs(2)) < Duration::from_secs(1));
        assert!(w.failed().is_none());
        assert_eq!(w.take_shown().map(|s| s.count), Some(1));
    }

    #[test]
    fn a_lasting_device_lost_stops_watching() {
        let w = start(always_lost);
        w.queued(1, None);
        w.wait_shown(1, Duration::from_secs(5));
        assert_eq!(w.failed(), Some(vk::Result::ERROR_DEVICE_LOST));
    }

    #[test]
    fn failure_stops_waiting_at_once() {
        let w = start(out_of_date);
        w.queued(1, None);
        assert!(w.wait_shown(1, Duration::from_secs(2)) < Duration::from_secs(1));
        assert_eq!(w.failed(), Some(vk::Result::ERROR_OUT_OF_DATE_KHR));
    }

    #[test]
    fn wait_gives_up_after_its_timeout_and_a_lost_present_is_forgotten() {
        let w = start(never_shows);
        w.queued(1, Some(Instant::now()));
        let waited = w.wait_shown(1, Duration::from_millis(30));
        assert!(waited >= Duration::from_millis(30) && waited < LOST_AFTER, "{waited:?}");
        assert!(w.wait_shown(1, Duration::from_secs(2)) < Duration::from_secs(1), "forgotten after LOST_AFTER");
        assert!(w.take_shown().is_none());
    }

    #[test]
    fn drop_stops_the_thread() {
        let w = start(never_shows);
        w.queued(1, None);
        let began = Instant::now();
        drop(w);
        assert!(began.elapsed() < Duration::from_millis(500));
    }
}
