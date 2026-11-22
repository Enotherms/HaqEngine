//! GPU time per Battery phase, from the timestamps written between phases.

use crate::FrameStats;

/// Battery's phases, in command-buffer order.
pub const PHASES: [&str; 5] = ["copy+pyramid", "flow", "blend", "fixup", "out-copy"];
/// Timestamps per slot when timing: one before each phase and one after the last.
pub const STAMPS: usize = PHASES.len() + 1;

/// Milliseconds per phase from a slot's timestamps (`period_ns` per tick); `None` where either
/// boundary was not written or the stamps run backwards.
pub fn phase_durations(stamps: &[u64; STAMPS], written: [bool; STAMPS], period_ns: f32) -> [Option<f32>; 5] {
    let mut out = [None; 5];
    for (i, d) in out.iter_mut().enumerate() {
        if written[i] && written[i + 1] && stamps[i + 1] >= stamps[i] {
            *d = Some((stamps[i + 1] - stamps[i]) as f32 * period_ns / 1_000_000.0);
        }
    }
    out
}

/// Phase times collected over a log window.
pub struct PhaseStats {
    stats: [FrameStats; 5],
}

impl Default for PhaseStats {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseStats {
    pub fn new() -> PhaseStats {
        PhaseStats { stats: std::array::from_fn(|_| FrameStats::new()) }
    }

    pub fn push(&mut self, phases: &[Option<f32>; 5]) {
        for (s, p) in self.stats.iter_mut().zip(phases) {
            if let Some(ms) = p {
                s.push((ms * 1000.0).round().min(u32::MAX as f32) as u32);
            }
        }
    }

    /// `phase mean/p99` for every phase that ran in the window, in milliseconds.
    pub fn report(&self) -> Option<String> {
        let parts: Vec<String> = PHASES
            .iter()
            .zip(&self.stats)
            .filter_map(|(name, s)| s.summary().map(|v| format!("{name} {:.2}/{:.2}", v.mean_ms, v.p99_ms)))
            .collect();
        (!parts.is_empty()).then(|| format!("gpu phases {} ms (mean/p99)", parts.join(" ")))
    }

    pub fn clear(&mut self) {
        for s in &mut self.stats {
            s.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_durations_from_stamps() {
        let d = phase_durations(&[0, 100, 300, 600, 700, 800], [true; STAMPS], 1000.0);
        let got: Vec<f32> = d.iter().map(|p| (p.expect("written") * 1000.0).round() / 1000.0).collect();
        assert_eq!(got, [0.1, 0.2, 0.3, 0.1, 0.1]);
    }

    #[test]
    fn phase_report_skips_unwritten_phases() {
        let written = [true, true, false, false, false, true];
        let d = phase_durations(&[0, 100, 0, 0, 0, 800], written, 1000.0);
        assert_eq!(d[0], Some(0.1));
        assert!(d[1..].iter().all(Option::is_none), "{d:?}");
        let mut s = PhaseStats::new();
        s.push(&d);
        let line = s.report().expect("one phase");
        assert!(line.contains("copy+pyramid 0.10/0.10") && !line.contains("flow"), "{line}");
        s.clear();
        assert!(s.report().is_none());
    }

    #[test]
    fn phase_totals_sum_outputs() {
        let d = phase_durations(&[0, 100, 300, 1500, 1700, 1800], [true; STAMPS], 1000.0);
        assert_eq!(d[2].map(|ms| (ms * 1000.0).round()), Some(1200.0), "three blends inside one phase");
    }

    #[test]
    fn backwards_stamps_are_dropped() {
        let d = phase_durations(&[0, 500, 300, 600, 700, 800], [true; STAMPS], 1000.0);
        assert_eq!(d[1], None);
    }
}
