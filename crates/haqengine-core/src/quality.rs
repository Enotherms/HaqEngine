//! Quality mode: which devices can run it, and the layout of its memory and constants.

/// What a device offers Quality mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualityCaps {
    pub api_version: u32,
    pub subgroup_ops_ok: bool,
    pub subgroup_size: u32,
    pub extended_formats: bool,
    pub robust_images: bool,
}

/// `Ok` when the device can run Quality mode, else the first requirement it misses.
pub fn quality_gate(c: &QualityCaps) -> Result<(), String> {
    let (major, minor) = (c.api_version >> 22 & 0x7f, c.api_version >> 12 & 0x3ff);
    if (major, minor) < (1, 1) {
        return Err(format!("the app asked for Vulkan {major}.{minor}, quality needs 1.1"));
    }
    if !c.subgroup_ops_ok {
        return Err("no subgroup basic, arithmetic and quad operations in compute".into());
    }
    if c.subgroup_size != 32 && c.subgroup_size != 64 {
        return Err(format!("subgroup size {} is not 32 or 64", c.subgroup_size));
    }
    if !c.extended_formats {
        return Err("shaderStorageImageExtendedFormats is not enabled".into());
    }
    if !c.robust_images {
        return Err("robust image access is not enabled".into());
    }
    Ok(())
}

/// Offsets for placing every `(size, alignment)` in one allocation, and its total size.
pub fn arena_layout(reqs: &[(u64, u64)]) -> (Vec<u64>, u64) {
    let mut offsets = Vec::with_capacity(reqs.len());
    let mut end = 0u64;
    for &(size, align) in reqs {
        let start = end.div_ceil(align.max(1)) * align.max(1);
        offsets.push(start);
        end = start + size;
    }
    (offsets, end)
}

/// Per-slot regions of one constant buffer: `blocks` blocks per slot, each aligned for a dynamic
/// uniform offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantRing {
    pub stride: u64,
    pub block_stride: u64,
    pub blocks: usize,
    pub slots: usize,
}

impl ConstantRing {
    pub fn new(blocks: usize, block_size: u64, min_alignment: u64, slots: usize) -> ConstantRing {
        let align = min_alignment.max(1);
        let block_stride = block_size.div_ceil(align) * align;
        ConstantRing { stride: block_stride * blocks as u64, block_stride, blocks, slots }
    }

    pub fn offset(&self, slot: usize, block: usize) -> u64 {
        self.stride * slot as u64 + self.block_stride * block as u64
    }

    pub fn size(&self) -> u64 {
        self.stride * self.slots as u64
    }
}

/// What a Quality present does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityStep {
    Interpolate,
    CopyAndPrime,
}

/// A copy when there is no previous frame; otherwise interpolation, also during FSR 3's warm-up,
/// when the flow is zero and the blend still produces a valid frame.
pub fn quality_step(has_prev: bool, _frames_since_reset: u32) -> QualityStep {
    if has_prev {
        QualityStep::Interpolate
    } else {
        QualityStep::CopyAndPrime
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: QualityCaps = QualityCaps {
        api_version: (1 << 22) | (3 << 12),
        subgroup_ops_ok: true,
        subgroup_size: 64,
        extended_formats: true,
        robust_images: true,
    };

    #[test]
    fn quality_gate_reports_first_failure() {
        assert_eq!(quality_gate(&GOOD), Ok(()));
        assert_eq!(quality_gate(&QualityCaps { subgroup_size: 32, ..GOOD }), Ok(()));
        let cases = [
            (QualityCaps { api_version: 1 << 22, ..GOOD }, "Vulkan 1.0"),
            (QualityCaps { subgroup_ops_ok: false, ..GOOD }, "subgroup"),
            (QualityCaps { subgroup_size: 16, ..GOOD }, "subgroup size 16"),
            (QualityCaps { extended_formats: false, ..GOOD }, "shaderStorageImageExtendedFormats"),
            (QualityCaps { robust_images: false, ..GOOD }, "robust image access"),
        ];
        for (caps, word) in cases {
            let e = quality_gate(&caps).unwrap_err();
            assert!(e.contains(word), "{e:?} should mention {word}");
        }
    }

    #[test]
    fn arena_layout_aligns() {
        assert_eq!(arena_layout(&[(100, 256), (10, 4096), (8, 16)]), (vec![0, 4096, 4112], 4120));
        assert_eq!(arena_layout(&[]), (vec![], 0));
    }

    #[test]
    fn constant_ring_offsets() {
        let r = ConstantRing::new(25, 32, 256, 4);
        assert_eq!(r.stride, 25 * 256);
        assert_eq!(r.offset(2, 3), 2 * 6400 + 3 * 256);
        assert_eq!(r.size(), 4 * 6400);
        let small = ConstantRing::new(2, 32, 16, 1);
        assert_eq!((small.block_stride, small.offset(0, 1)), (32, 32));
    }

    #[test]
    fn quality_step_restarts_after_skip() {
        assert_eq!(quality_step(false, 9), QualityStep::CopyAndPrime);
        assert_eq!(quality_step(true, 0), QualityStep::Interpolate);
        assert_eq!(quality_step(true, 7), QualityStep::Interpolate);
    }
}
