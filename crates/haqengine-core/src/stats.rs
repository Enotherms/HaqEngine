//! Frame-time statistics over a fixed-size window. No heap allocation.

/// Samples kept per window.
pub const STATS_CAPACITY: usize = 1024;

/// Summary of the samples in a window, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub count: u32,
    pub mean_ms: f32,
    pub stddev_ms: f32,
    pub p99_ms: f32,
}

/// A ring of frame intervals in microseconds.
pub struct FrameStats {
    samples: [u32; STATS_CAPACITY],
    len: usize,
    next: usize,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameStats {
    pub fn new() -> Self {
        FrameStats { samples: [0; STATS_CAPACITY], len: 0, next: 0 }
    }
    /// Adds a sample, overwriting the oldest when full.
    pub fn push(&mut self, interval_us: u32) {
        self.samples[self.next] = interval_us;
        self.next = (self.next + 1) % STATS_CAPACITY;
        self.len = (self.len + 1).min(STATS_CAPACITY);
    }
    pub fn summary(&self) -> Option<Summary> {
        let n = self.len;
        if n == 0 {
            return None;
        }
        let mut sorted = [0u32; STATS_CAPACITY];
        sorted[..n].copy_from_slice(&self.samples[..n]);
        let sorted = &mut sorted[..n];
        sorted.sort_unstable();
        let mean = sorted.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = sorted.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let p99_index = ((0.99 * n as f64).ceil() as usize).clamp(1, n) - 1;
        Some(Summary {
            count: n as u32,
            mean_ms: (mean / 1000.0) as f32,
            stddev_ms: (var.sqrt() / 1000.0) as f32,
            p99_ms: (sorted[p99_index] as f64 / 1000.0) as f32,
        })
    }
    pub fn clear(&mut self) {
        self.len = 0;
        self.next = 0;
    }
}

/// True when at least `every_ms` has passed since `last_ms`. A clock that went backwards never logs.
pub fn should_log(now_ms: u64, last_ms: u64, every_ms: u64) -> bool {
    now_ms.checked_sub(last_ms).is_some_and(|d| d >= every_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_empty_is_none() {
        assert!(FrameStats::new().summary().is_none());
    }

    #[test]
    fn stats_even_intervals() {
        let mut s = FrameStats::new();
        for _ in 0..100 {
            s.push(16_667);
        }
        let m = s.summary().unwrap();
        assert_eq!(m.count, 100);
        assert!((m.mean_ms - 16.667).abs() < 0.01 && m.stddev_ms < 0.01 && (m.p99_ms - 16.667).abs() < 0.01, "{m:?}");
    }

    #[test]
    fn stats_p99_catches_spike() {
        let mut s = FrameStats::new();
        for i in 0..200 {
            s.push(if i % 70 == 0 { 50_000 } else { 16_667 });
        }
        assert!(s.summary().unwrap().p99_ms >= 49.0);
    }

    #[test]
    fn stats_wraps_at_capacity() {
        let mut s = FrameStats::new();
        for _ in 0..STATS_CAPACITY {
            s.push(1_000_000);
        }
        for _ in 0..STATS_CAPACITY {
            s.push(10_000);
        }
        let m = s.summary().unwrap();
        assert_eq!(m.count as usize, STATS_CAPACITY);
        assert!((m.mean_ms - 10.0).abs() < 0.01);
    }

    #[test]
    fn should_log_every_interval() {
        assert!(!should_log(9_999, 0, 10_000));
        assert!(should_log(10_000, 0, 10_000));
        assert!(!should_log(5, 10, 10_000));
    }
}
