//! How many frames are shown per real frame, when each generated frame sits in time, and how far
//! presents may queue ahead of the screen.

use std::collections::VecDeque;
use std::time::Instant;

/// The most frames shown per real frame.
pub const MAX_MULTIPLIER: u32 = 4;
/// The most generated frames per real frame.
pub const MAX_GENERATED: usize = (MAX_MULTIPLIER - 1) as usize;

/// When each generated frame sits between the previous real frame (0) and the current one (1), at
/// multiplier `m`: `k / m` for `k = 1..m`, in presentation order.
pub fn blend_times(m: u32) -> impl Iterator<Item = f32> {
    (1..m).map(move |k| k as f32 / m as f32)
}

/// The present id to wait for before queueing the next real frame's presents, given the id of the
/// newest present already queued: every older present must be on screen, so at most one present
/// waits ahead of the new ones. `None` when nothing older exists.
pub fn pace_target(last_id: u64) -> Option<u64> {
    last_id.checked_sub(1).filter(|&t| t > 0)
}

/// Most presents [`ShownQueue`] remembers; older ones are forgotten.
pub const SHOWN_QUEUE_MAX: usize = 64;

/// Presents queued for the screen, oldest first, each real one with the time its measurement
/// starts from.
#[derive(Default)]
pub struct ShownQueue {
    ids: VecDeque<(u64, Option<Instant>)>,
}

impl ShownQueue {
    /// Adds a queued present (ids rise).
    pub fn push(&mut self, id: u64, real_start: Option<Instant>) {
        if self.ids.len() == SHOWN_QUEUE_MAX {
            self.ids.pop_front();
        }
        self.ids.push_back((id, real_start));
    }
    pub fn oldest(&self) -> Option<u64> {
        self.ids.front().map(|&(id, _)| id)
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    /// Forgets the oldest present (it never reached the screen in time).
    pub fn drop_oldest(&mut self) {
        self.ids.pop_front();
    }
    /// Every present up to `id` is on screen at `now`: forgets them, reporting each real one's
    /// microseconds from its start time to screen.
    pub fn shown(&mut self, id: u64, now: Instant, mut real_us: impl FnMut(u32)) {
        while let Some((_, start)) = self.ids.front().filter(|(f, _)| *f <= id).copied() {
            if let Some(s) = start {
                real_us(now.saturating_duration_since(s).as_micros().min(u32::MAX as u128) as u32);
            }
            self.ids.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn pace_waits_for_all_but_the_newest_present() {
        assert_eq!(pace_target(0), None);
        assert_eq!(pace_target(1), None);
        assert_eq!(pace_target(2), Some(1));
        assert_eq!(pace_target(9), Some(8));
    }

    #[test]
    fn shown_queue_reports_real_frames_up_to_the_shown_id() {
        let t0 = Instant::now();
        let mut q = ShownQueue::default();
        q.push(1, None);
        q.push(2, Some(t0));
        q.push(3, None);
        q.push(4, Some(t0 + Duration::from_millis(5)));
        assert_eq!(q.oldest(), Some(1));
        let mut seen = Vec::new();
        q.shown(2, t0 + Duration::from_millis(30), |us| seen.push(us));
        assert_eq!(seen, [30_000]);
        assert_eq!(q.oldest(), Some(3));
        q.shown(4, t0 + Duration::from_millis(45), |us| seen.push(us));
        assert_eq!(seen, [30_000, 40_000]);
        assert_eq!(q.oldest(), None);
    }

    #[test]
    fn shown_queue_drops_a_lost_present_without_a_time() {
        let t0 = Instant::now();
        let mut q = ShownQueue::default();
        q.push(5, Some(t0));
        q.push(6, Some(t0));
        q.drop_oldest();
        let mut seen = Vec::new();
        q.shown(6, t0 + Duration::from_millis(10), |us| seen.push(us));
        assert_eq!(seen, [10_000]);
        assert!(q.is_empty());
    }

    #[test]
    fn shown_queue_stays_bounded() {
        let mut q = ShownQueue::default();
        for id in 1..=(SHOWN_QUEUE_MAX as u64 + 10) {
            q.push(id, None);
        }
        assert_eq!(q.len(), SHOWN_QUEUE_MAX);
        assert_eq!(q.oldest(), Some(11));
    }

    #[test]
    fn blend_times_split_evenly() {
        assert_eq!(blend_times(2).collect::<Vec<_>>(), [0.5]);
        assert_eq!(blend_times(3).collect::<Vec<_>>(), [1.0 / 3.0, 2.0 / 3.0]);
        assert_eq!(blend_times(4).collect::<Vec<_>>(), [0.25, 0.5, 0.75]);
        assert_eq!(MAX_GENERATED as u32, MAX_MULTIPLIER - 1);
    }
}
