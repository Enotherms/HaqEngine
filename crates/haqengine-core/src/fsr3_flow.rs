//! The FSR 3 optical flow schedule: what FidelityFX SDK v1.1.4's host code dispatches per frame.

pub const OF_LEVELS: usize = 7;
pub const OF_BLOCK: u32 = 8;
pub const OF_SEARCH_RADIUS: u32 = 8;

/// Pass indices into `haqengine_shaders::OF_PASSES`.
pub const PASS_PREPARE_LUMA: usize = 0;
pub const PASS_LUMINANCE_PYRAMID: usize = 1;
pub const PASS_SCD_HISTOGRAM: usize = 2;
pub const PASS_SCD_DIVERGENCE: usize = 3;
pub const PASS_SEARCH: usize = 4;
pub const PASS_FILTER: usize = 5;
pub const PASS_SCALE: usize = 6;

/// Sizes of the luma pyramid and of the flow texture for every level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfSizes {
    pub luma: [(u32, u32); OF_LEVELS],
    pub flow: [(u32, u32); OF_LEVELS],
}

/// The images the optical flow reads and writes. `Input` and `Flow` come in two ping-pong slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OfRes {
    Input(usize, u32),
    Flow(usize, u32),
    ScdHistogram,
    ScdPreviousHistogram,
    ScdTemp,
    ScdOutput,
    Color,
    Output,
}

/// One compute dispatch of the schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfDispatch {
    pub pass: usize,
    pub groups: [u32; 3],
    pub bind: Vec<(u32, OfRes)>,
    pub constants: [u8; 32],
    pub spd_constants: Option<[u8; 32]>,
}

pub fn of_sizes(width: u32, height: u32) -> OfSizes {
    let mut luma = [(1, 1); OF_LEVELS];
    let mut flow = [(1, 1); OF_LEVELS];
    for (l, size) in luma.iter_mut().enumerate() {
        *size = ((width >> l).max(1), (height >> l).max(1));
    }
    flow[0] = (width.div_ceil(OF_BLOCK).max(1), height.div_ceil(OF_BLOCK).max(1));
    for l in 1..OF_LEVELS {
        flow[l] = (flow[l - 1].0.div_ceil(2), flow[l - 1].1.div_ceil(2));
    }
    OfSizes { luma, flow }
}

const HISTOGRAMS_PER_DIM: u32 = 3;
const HISTOGRAM_SHIFTS: u32 = 3;
const SPD_MIPS: u32 = 4;

fn constants(width: u32, height: u32, level: u32, frame_index: u32) -> [u8; 32] {
    let words = [width, height, level, OF_LEVELS as u32, frame_index, 0, 0f32.to_bits(), 1f32.to_bits()];
    let mut out = [0u8; 32];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

fn spd_constants(width: u32, height: u32) -> ([u8; 32], [u32; 2]) {
    let groups = [(width - 1) / 64 + 1, (height - 1) / 64 + 1];
    let count = groups[0] * groups[1];
    let words = [SPD_MIPS, count, 0, 0, count, 0, 0, 0];
    let mut out = [0u8; 32];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    (out, groups)
}

/// Every dispatch of one frame, in order. `frame_index` 0 is the first frame after creation.
pub fn of_schedule(width: u32, height: u32, frame_index: u32) -> Vec<OfDispatch> {
    let sizes = of_sizes(width, height);
    let odd_frame = frame_index & 1 == 1;
    let (cur, prev) = if odd_frame { (1, 0) } else { (0, 1) };
    let base = constants(width, height, 0, frame_index);
    let (spd, spd_groups) = spd_constants(width, height);
    let mut out = Vec::with_capacity(24);
    let plain = |pass: usize, groups: [u32; 3], bind: Vec<(u32, OfRes)>, constants: [u8; 32]| OfDispatch {
        pass,
        groups,
        bind,
        constants,
        spd_constants: None,
    };
    out.push(plain(
        PASS_PREPARE_LUMA,
        [(width.div_ceil(2)).div_ceil(16), (height.div_ceil(2)).div_ceil(16), 1],
        vec![(0, OfRes::Color), (1, OfRes::Input(cur, 0))],
        base,
    ));
    out.push(OfDispatch {
        pass: PASS_LUMINANCE_PYRAMID,
        groups: [spd_groups[0], spd_groups[1], 1],
        bind: (0..OF_LEVELS as u32).map(|l| (l, OfRes::Input(cur, l))).collect(),
        constants: base,
        spd_constants: Some(spd),
    });
    let strata_width = (width / 4) / HISTOGRAMS_PER_DIM;
    out.push(plain(
        PASS_SCD_HISTOGRAM,
        [strata_width.div_ceil(32), 16, HISTOGRAMS_PER_DIM * HISTOGRAMS_PER_DIM],
        vec![(0, OfRes::Input(cur, 0)), (1, OfRes::ScdHistogram)],
        base,
    ));
    out.push(plain(
        PASS_SCD_DIVERGENCE,
        [HISTOGRAMS_PER_DIM * HISTOGRAMS_PER_DIM, HISTOGRAM_SHIFTS, 1],
        vec![(0, OfRes::ScdHistogram), (1, OfRes::ScdPreviousHistogram), (2, OfRes::ScdTemp), (3, OfRes::ScdOutput)],
        base,
    ));
    for level in (0..OF_LEVELS).rev() {
        let l = level as u32;
        let odd_level = level & 1 == 1;
        let (a, b) = if odd_frame != odd_level { (1, 0) } else { (0, 1) };
        let c = constants(width, height, l, frame_index);
        let (lw, lh) = sizes.luma[level];
        out.push(plain(
            PASS_SEARCH,
            [(lw.div_ceil(4) * 16).div_ceil(64), lh.div_ceil(16), 1],
            vec![(0, OfRes::Input(cur, l)), (1, OfRes::Input(prev, l)), (2, OfRes::Flow(a, l)), (3, OfRes::ScdOutput)],
            c,
        ));
        let filtered = if level == 0 { OfRes::Output } else { OfRes::Flow(b, l) };
        let (fw, fh) = sizes.flow[level];
        out.push(plain(PASS_FILTER, [fw.div_ceil(16), fh.div_ceil(4), 1], vec![(0, OfRes::Flow(a, l)), (1, filtered)], c));
        if level > 0 {
            let (nw, nh) = sizes.flow[level - 1];
            out.push(plain(
                PASS_SCALE,
                [nw.div_ceil(4), nh.div_ceil(4), 1],
                vec![
                    (0, OfRes::Input(cur, l)),
                    (1, OfRes::Input(prev, l)),
                    (2, OfRes::Flow(b, l)),
                    (3, OfRes::Flow(b, l - 1)),
                    (4, OfRes::ScdOutput),
                ],
                c,
            ));
        }
    }
    out
}

/// An FSR 3 level-0 flow texel as a HaqEngine vector: `cur(p) ≈ prev(p − v)`, in pixels.
pub fn ffx_to_haq(v: [i16; 2]) -> [f32; 2] {
    [v[0] as f32, v[1] as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(b: &[u8; 32], offset: usize) -> u32 {
        u32::from_le_bytes(b[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn of_sizes_1280x800() {
        let s = of_sizes(1280, 800);
        assert_eq!(s.flow, [(160, 100), (80, 50), (40, 25), (20, 13), (10, 7), (5, 4), (3, 2)]);
        assert_eq!(s.luma[0], (1280, 800));
        assert_eq!(s.luma[6], (20, 12));
    }

    #[test]
    fn of_sizes_1281x801() {
        let s = of_sizes(1281, 801);
        assert_eq!(s.flow[0], (161, 101));
        assert!(s.flow.iter().chain(&s.luma).all(|&(w, h)| w >= 1 && h >= 1));
        assert_eq!(of_sizes(3, 2).luma[6], (1, 1));
    }

    #[test]
    fn of_schedule_order() {
        let d = of_schedule(1280, 800, 0);
        let mut expected = vec![PASS_PREPARE_LUMA, PASS_LUMINANCE_PYRAMID, PASS_SCD_HISTOGRAM, PASS_SCD_DIVERGENCE];
        for level in (0..OF_LEVELS).rev() {
            expected.extend([PASS_SEARCH, PASS_FILTER]);
            if level > 0 {
                expected.push(PASS_SCALE);
            }
        }
        assert_eq!(expected.len(), 24);
        assert_eq!(d.iter().map(|x| x.pass).collect::<Vec<_>>(), expected);
    }

    #[test]
    fn of_dispatch_sizes_1280x800() {
        let d = of_schedule(1280, 800, 0);
        assert_eq!(d[0].groups, [40, 25, 1]);
        assert_eq!(d[1].groups, [20, 13, 1]);
        assert_eq!(d[2].groups, [4, 16, 9]);
        assert_eq!(d[3].groups, [9, 3, 1]);
        let level0_search = d.iter().rposition(|x| x.pass == PASS_SEARCH).unwrap();
        assert_eq!(d[level0_search].groups, [80, 50, 1]);
        assert_eq!(d[level0_search + 1].groups, [10, 25, 1]);
        let first_scale = d.iter().position(|x| x.pass == PASS_SCALE).unwrap();
        assert_eq!(d[first_scale].groups, [2, 1, 1]);
        let last_scale = d.iter().rposition(|x| x.pass == PASS_SCALE).unwrap();
        assert_eq!(d[last_scale].groups, [40, 25, 1]);
    }

    #[test]
    fn of_ping_pong() {
        let even = of_schedule(1280, 800, 0);
        let odd = of_schedule(1280, 800, 1);
        assert_eq!(even[0].bind, vec![(0, OfRes::Color), (1, OfRes::Input(0, 0))]);
        assert_eq!(odd[0].bind, vec![(0, OfRes::Color), (1, OfRes::Input(1, 0))]);
        assert_eq!(
            even[4].bind,
            vec![(0, OfRes::Input(0, 6)), (1, OfRes::Input(1, 6)), (2, OfRes::Flow(0, 6)), (3, OfRes::ScdOutput)]
        );
        assert_eq!(
            odd[4].bind,
            vec![(0, OfRes::Input(1, 6)), (1, OfRes::Input(0, 6)), (2, OfRes::Flow(1, 6)), (3, OfRes::ScdOutput)]
        );
        assert_eq!(even[5].bind, vec![(0, OfRes::Flow(0, 6)), (1, OfRes::Flow(1, 6))]);
        assert_eq!(
            even[6].bind,
            vec![
                (0, OfRes::Input(0, 6)),
                (1, OfRes::Input(1, 6)),
                (2, OfRes::Flow(1, 6)),
                (3, OfRes::Flow(1, 5)),
                (4, OfRes::ScdOutput)
            ]
        );
        assert_eq!(even[7].bind[2], (2, OfRes::Flow(1, 5)));
        let last = even.last().unwrap();
        assert_eq!(last.pass, PASS_FILTER);
        assert_eq!(last.bind[1], (1, OfRes::Output));
    }

    #[test]
    fn of_constants_bytes() {
        let d = of_schedule(1280, 800, 5);
        assert_eq!(word(&d[0].constants, 0), 1280);
        assert_eq!(word(&d[0].constants, 4), 800);
        assert_eq!(word(&d[0].constants, 16), 5);
        assert_eq!(word(&d[0].constants, 20), 0);
        assert_eq!(f32::from_bits(word(&d[0].constants, 28)), 1.0);
        let search = &d[4];
        assert_eq!(word(&search.constants, 8), 6);
        assert_eq!(word(&search.constants, 12), 7);
        let spd = d[1].spd_constants.expect("pyramid has SPD constants");
        assert_eq!([word(&spd, 0), word(&spd, 4), word(&spd, 8), word(&spd, 12), word(&spd, 16)], [4, 260, 0, 0, 260]);
        assert!(d.iter().filter(|x| x.pass != PASS_LUMINANCE_PYRAMID).all(|x| x.spd_constants.is_none()));
    }
}
