//! How much changed between two real frames, and the decisions taken from it: a scene cut (show
//! the new frame) and a mostly still screen (skip generation).

use std::fmt;

pub const CUT_FRACTION: f32 = 0.45;
/// Still pixels needed to enter still mode: a menu with a small animated part still qualifies.
pub const STILL_FRACTION: f32 = 0.85;
/// Still pixels needed to stay in still mode once in it, so a cursor or a small effect does not
/// flip generation on and off.
pub const STILL_KEEP_FRACTION: f32 = 0.75;
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

/// The disagreeing-pixel count above which a frame of `pixels` is a cut, as an integer for the GPU:
/// `disagree > cut_threshold_pixels(pixels)` exactly when `is_cut`.
pub fn cut_threshold_pixels(pixels: u32) -> u32 {
    (CUT_FRACTION * pixels as f32) as u32
}

/// The share of pixels that did not change at all (0 for an empty frame).
pub fn still_fraction(c: &ChangeCounts) -> f32 {
    if c.pixels == 0 {
        0.0
    } else {
        c.still as f32 / c.pixels as f32
    }
}

/// At least `STILL_FRACTION` of the pixels did not change at all.
pub fn is_still(c: &ChangeCounts) -> bool {
    c.pixels > 0 && c.still as f32 >= STILL_FRACTION * c.pixels as f32
}

/// At least `STILL_KEEP_FRACTION` of the pixels did not change at all.
pub fn stays_still(c: &ChangeCounts) -> bool {
    c.pixels > 0 && c.still as f32 >= STILL_KEEP_FRACTION * c.pixels as f32
}

/// Decides, from the newest available counts, whether the next present skips generation: after
/// `STILL_FRAMES` still frames in a row, until the first frame below `STILL_KEEP_FRACTION` (or
/// without counts).
#[derive(Debug, Default)]
pub struct StillDetector {
    run: u32,
    active: bool,
}

impl StillDetector {
    pub fn new() -> StillDetector {
        StillDetector::default()
    }

    pub fn observe(&mut self, counts: Option<ChangeCounts>) -> bool {
        match counts {
            Some(c) if self.active && stays_still(&c) => {}
            Some(c) if is_still(&c) => {
                self.run = self.run.saturating_add(1);
                self.active = self.run >= STILL_FRAMES;
            }
            _ => {
                self.run = 0;
                self.active = false;
            }
        }
        self.active
    }
}

/// How many frames fell in each still-fraction band, for the log: below 50%, 75%, 85%, 99.5%, and
/// the rest. Empty frames are not counted.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StillBuckets {
    pub counts: [u64; 5],
}

const BUCKET_EDGES: [f32; 4] = [0.5, 0.75, 0.85, 0.995];

impl StillBuckets {
    pub fn add(&mut self, c: &ChangeCounts) {
        if c.pixels == 0 {
            return;
        }
        let f = still_fraction(c);
        let i = BUCKET_EDGES.iter().position(|&e| f < e).unwrap_or(BUCKET_EDGES.len());
        self.counts[i] += 1;
    }

    pub fn clear(&mut self) {
        self.counts = [0; 5];
    }
}

impl fmt::Display for StillBuckets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.counts;
        write!(f, "still% <50 {}, <75 {}, <85 {}, <99.5 {}, rest {}", c[0], c[1], c[2], c[3], c[4])
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
        assert!(!is_still(&counts(0, 849)));
        assert!(is_still(&counts(0, 850)));
        assert!(!stays_still(&counts(0, 749)));
        assert!(stays_still(&counts(0, 750)));
        assert!(!is_cut(&ChangeCounts::default()) && !is_still(&ChangeCounts::default()));
    }

    #[test]
    fn integer_cut_threshold_agrees() {
        for pixels in [1000u32, 1280 * 800, 1283 * 797, 7] {
            let t = cut_threshold_pixels(pixels);
            for d in [t.saturating_sub(1), t, t + 1] {
                assert_eq!(d > t, is_cut(&ChangeCounts { disagree: d, still: 0, pixels }), "{pixels} {d}");
            }
        }
    }

    #[test]
    fn still_detector_enters_after_two_and_leaves_immediately() {
        let mut d = StillDetector::new();
        assert!(!d.observe(Some(counts(0, 1000))));
        assert!(d.observe(Some(counts(0, 1000))));
        assert!(d.observe(Some(counts(0, 999))));
        assert!(!d.observe(Some(counts(3, 600))), "one moving frame resumes generation");
        assert!(!d.observe(Some(counts(0, 1000))));
        assert!(d.observe(Some(counts(0, 1000))));
    }

    #[test]
    fn mostly_still_enters_and_keeps_with_hysteresis() {
        let mut d = StillDetector::new();
        assert!(!d.observe(Some(counts(0, 849))));
        assert!(!d.observe(Some(counts(0, 849))), "below the enter line never enters");
        assert!(!d.observe(Some(counts(0, 850))));
        assert!(d.observe(Some(counts(0, 850))), "two mostly-still frames enter");
        assert!(d.observe(Some(counts(0, 750))), "a small moving part keeps still mode");
        assert!(!d.observe(Some(counts(0, 749))), "more motion leaves at once");
        assert!(!d.observe(Some(counts(0, 800))), "re-entering needs the enter line again");
    }

    #[test]
    fn still_buckets_split_the_fraction() {
        let mut b = StillBuckets::default();
        for still in [0, 499, 500, 749, 750, 849, 850, 994, 995, 1000] {
            b.add(&counts(0, still));
        }
        b.add(&ChangeCounts::default());
        assert_eq!(b.counts, [2, 2, 2, 2, 2]);
        assert_eq!(b.to_string(), "still% <50 2, <75 2, <85 2, <99.5 2, rest 2");
        b.clear();
        assert_eq!(b.counts, [0; 5]);
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
