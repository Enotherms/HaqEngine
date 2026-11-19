//! How much changed between two real frames, and the decisions taken from it: a scene cut (show
//! the new frame) and a still screen (skip generation).

pub const CUT_FRACTION: f32 = 0.45;
pub const STILL_FRACTION: f32 = 0.995;
pub const STILL_FRAMES: u32 = 2;

/// Per-frame pixel counts from the blend: pixels whose best candidate still disagrees, pixels that
/// are exactly still, and the frame's pixel count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChangeCounts {
    pub disagree: u32,
    pub still: u32,
    pub pixels: u32,
}

/// More than `CUT_FRACTION` of the pixels match nowhere: a new scene.
pub fn is_cut(c: &ChangeCounts) -> bool {
    c.pixels > 0 && c.disagree as f32 > CUT_FRACTION * c.pixels as f32
}

/// At least `STILL_FRACTION` of the pixels did not change at all.
pub fn is_still(c: &ChangeCounts) -> bool {
    c.pixels > 0 && c.still as f32 >= STILL_FRACTION * c.pixels as f32
}

/// Decides, from the newest available counts, whether the next present skips generation: after
/// `STILL_FRAMES` still frames in a row, until the first frame that is not still (or has no counts).
#[derive(Debug, Default)]
pub struct StillDetector {
    run: u32,
}

impl StillDetector {
    pub fn new() -> StillDetector {
        StillDetector { run: 0 }
    }

    pub fn observe(&mut self, counts: Option<ChangeCounts>) -> bool {
        match counts {
            Some(c) if is_still(&c) => self.run = self.run.saturating_add(1),
            _ => self.run = 0,
        }
        self.run >= STILL_FRAMES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(disagree: u32, still: u32) -> ChangeCounts {
        ChangeCounts { disagree, still, pixels: 1000 }
    }

    #[test]
    fn cut_and_still_thresholds() {
        assert!(!is_cut(&counts(450, 0)));
        assert!(is_cut(&counts(451, 0)));
        assert!(!is_still(&counts(0, 994)));
        assert!(is_still(&counts(0, 995)));
        assert!(!is_cut(&ChangeCounts::default()) && !is_still(&ChangeCounts::default()));
    }

    #[test]
    fn still_detector_enters_after_two_and_leaves_immediately() {
        let mut d = StillDetector::new();
        assert!(!d.observe(Some(counts(0, 1000))));
        assert!(d.observe(Some(counts(0, 1000))));
        assert!(d.observe(Some(counts(0, 999))));
        assert!(!d.observe(Some(counts(3, 900))), "one changed frame resumes generation");
        assert!(!d.observe(Some(counts(0, 1000))));
        assert!(d.observe(Some(counts(0, 1000))));
    }

    #[test]
    fn still_detector_needs_counts() {
        let mut d = StillDetector::new();
        assert!(!d.observe(None));
        assert!(!d.observe(Some(counts(0, 1000))));
        assert!(!d.observe(None), "a missing count resets the run");
        assert!(!d.observe(Some(counts(0, 1000))));
        assert!(d.observe(Some(counts(0, 1000))));
    }
}
