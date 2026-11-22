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
pub const PASS_LUMA_HALF: usize = 7;

/// Where level-0 luma comes from: AMD's prepare pass on a colour image, or HaqEngine's pass that
/// averages a packed `full_width × full_height` frame down to the flow input size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfInput {
    Color,
    PackedHalf { format: u32, full_width: u32, full_height: u32 },
}

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
    Raw,
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
    pub push: [u8; 12],
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

/// The flow input size for a frame of `width × height` at half resolution.
pub fn half_size(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(2), height.div_ceil(2))
}

/// Texel bytes of every image the flow owns for an input of `width × height`. The packed input
/// frame is borrowed and not counted; drivers round each allocation up, so the real cost is
/// measured by the harness (`fsr3 flow memory within budget`).
pub fn of_memory_bytes(width: u32, height: u32) -> u64 {
    let s = of_sizes(width, height);
    let area = |(w, h): (u32, u32)| w as u64 * h as u64;
    let luma: u64 = s.luma.iter().map(|&l| area(l)).sum();
    let flow: u64 = s.flow.iter().map(|&f| area(f) * 4).sum();
    2 * luma + 2 * flow + area(s.flow[0]) * 4 + 2 * SCD_HISTOGRAM_WIDTH as u64 * 4 + 2 * 3 * 4
}

/// Width of the scene-change histograms: 256 bins for each of the 3×3 regions.
pub const SCD_HISTOGRAM_WIDTH: u32 = 256 * HISTOGRAMS_PER_DIM * HISTOGRAMS_PER_DIM;

/// Rewrites the `frameIndex` word of a dispatch's constants, so schedules can be built once.
pub fn set_frame_index(constants: &mut [u8; 32], frame_index: u32) {
    constants[16..20].copy_from_slice(&frame_index.to_le_bytes());
}

/// CPU reference of the half-size luma pass: the 2×2 average of BT.709 luma, rounded to 8 bits,
/// computed in integers exactly as the shader does.
pub fn luma_half_reference(f: crate::packed::PackedFormat, input: &crate::Image<u32>) -> crate::Image<u8> {
    use crate::packed::PackedFormat;
    let (w, h) = half_size(input.width, input.height);
    let mut out = crate::Image::new(w, h, 0u8);
    for oy in 0..h {
        for ox in 0..w {
            let (mut sum, mut count) = (0u32, 0u32);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (x, y) = (ox * 2 + dx, oy * 2 + dy);
                if x < input.width && y < input.height {
                    let word = input.get(x, y);
                    let (r, g, b) = match f {
                        PackedFormat::Rgba8 => (word & 255, (word >> 8) & 255, (word >> 16) & 255),
                        PackedFormat::Bgra8 => ((word >> 16) & 255, (word >> 8) & 255, word & 255),
                        PackedFormat::A2b10g10r10 => (word & 1023, (word >> 10) & 1023, (word >> 20) & 1023),
                    };
                    sum += 2126 * r + 7152 * g + 722 * b;
                    count += 1;
                }
            }
            let value = match f {
                PackedFormat::A2b10g10r10 => {
                    let den = 10000 * count * 341;
                    (sum * 85 + den / 2) / den
                }
                _ => {
                    let den = 10000 * count;
                    (sum + den / 2) / den
                }
            };
            out.set(ox, oy, value.min(255) as u8);
        }
    }
    out
}

/// Every dispatch of one frame, in order. `frame_index` 0 is the first frame after creation.
pub fn of_schedule(width: u32, height: u32, frame_index: u32, input: OfInput) -> Vec<OfDispatch> {
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
        push: [0; 12],
    };
    match input {
        OfInput::Color => out.push(plain(
            PASS_PREPARE_LUMA,
            [(width.div_ceil(2)).div_ceil(16), (height.div_ceil(2)).div_ceil(16), 1],
            vec![(0, OfRes::Color), (1, OfRes::Input(cur, 0))],
            base,
        )),
        OfInput::PackedHalf { format, full_width, full_height } => {
            let mut d =
                plain(PASS_LUMA_HALF, [width.div_ceil(8), height.div_ceil(8), 1], vec![(0, OfRes::Raw), (1, OfRes::Input(cur, 0))], base);
            d.push = push_words([full_width, full_height, format]);
            out.push(d);
        }
    }
    out.push(OfDispatch {
        pass: PASS_LUMINANCE_PYRAMID,
        groups: [spd_groups[0], spd_groups[1], 1],
        bind: (0..OF_LEVELS as u32).map(|l| (l, OfRes::Input(cur, l))).collect(),
        constants: base,
        spd_constants: Some(spd),
        push: [0; 12],
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

fn push_words(words: [u32; 3]) -> [u8; 12] {
    let mut out = [0u8; 12];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// An FSR 3 level-0 flow texel as a HaqEngine vector: `cur(p) ≈ prev(p − v)`, in full-resolution
/// pixels. FSR 3 stores the opposite vector, from the current frame back to the previous one, in
/// pixels of its input; `scale` is the full size over the input size (2.0 at half resolution).
pub fn ffx_to_haq(v: [i16; 2], scale: f32) -> [f32; 2] {
    [-(v[0] as f32) * scale, -(v[1] as f32) * scale]
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
        let d = of_schedule(1280, 800, 0, OfInput::Color);
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
        let d = of_schedule(1280, 800, 0, OfInput::Color);
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
        let even = of_schedule(1280, 800, 0, OfInput::Color);
        let odd = of_schedule(1280, 800, 1, OfInput::Color);
        assert_eq!(even[0].bind, vec![(0, OfRes::Color), (1, OfRes::Input(0, 0))]);
        assert_eq!(odd[0].bind, vec![(0, OfRes::Color), (1, OfRes::Input(1, 0))]);
        assert_eq!(even[4].bind, vec![(0, OfRes::Input(0, 6)), (1, OfRes::Input(1, 6)), (2, OfRes::Flow(0, 6)), (3, OfRes::ScdOutput)]);
        assert_eq!(odd[4].bind, vec![(0, OfRes::Input(1, 6)), (1, OfRes::Input(0, 6)), (2, OfRes::Flow(1, 6)), (3, OfRes::ScdOutput)]);
        assert_eq!(even[5].bind, vec![(0, OfRes::Flow(0, 6)), (1, OfRes::Flow(1, 6))]);
        assert_eq!(
            even[6].bind,
            vec![(0, OfRes::Input(0, 6)), (1, OfRes::Input(1, 6)), (2, OfRes::Flow(1, 6)), (3, OfRes::Flow(1, 5)), (4, OfRes::ScdOutput)]
        );
        assert_eq!(even[7].bind[2], (2, OfRes::Flow(1, 5)));
        let last = even.last().unwrap();
        assert_eq!(last.pass, PASS_FILTER);
        assert_eq!(last.bind[1], (1, OfRes::Output));
    }

    #[test]
    fn ffx_vectors_point_back_to_the_previous_frame() {
        assert_eq!(ffx_to_haq([-12, 8], 1.0), [12.0, -8.0]);
        assert_eq!(ffx_to_haq([96, 0], 1.0), [-96.0, 0.0]);
        assert_eq!(ffx_to_haq([0, 0], 1.0), [0.0, 0.0]);
        assert_eq!(ffx_to_haq([-6, 4], 2.0), [12.0, -8.0]);
    }

    #[test]
    fn packed_half_input_replaces_prepare_luma() {
        let d = of_schedule(640, 400, 0, OfInput::PackedHalf { format: 1, full_width: 1280, full_height: 800 });
        assert_eq!(d.len(), 24);
        assert_eq!(d[0].pass, PASS_LUMA_HALF);
        assert_eq!(d[0].groups, [80, 50, 1]);
        assert_eq!(d[0].bind, vec![(0, OfRes::Raw), (1, OfRes::Input(0, 0))]);
        let push: Vec<u32> = d[0].push.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(push, [1280, 800, 1]);
        let odd = of_schedule(641, 401, 1, OfInput::PackedHalf { format: 0, full_width: 1281, full_height: 801 });
        assert_eq!(odd[0].bind[1], (1, OfRes::Input(1, 0)));
        let push: Vec<u32> = odd[0].push.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(push, [1281, 801, 0]);
        assert_eq!(of_schedule(640, 400, 0, OfInput::Color)[0].pass, PASS_PREPARE_LUMA);
    }

    #[test]
    fn half_size_rounds_up() {
        assert_eq!(half_size(1280, 800), (640, 400));
        assert_eq!(half_size(1281, 801), (641, 401));
    }

    #[test]
    fn half_resolution_memory_is_bounded() {
        let half = of_memory_bytes(640, 400);
        let full = of_memory_bytes(1280, 800);
        assert!(half <= 1 << 20, "{half} bytes");
        assert!(half * 3 < full, "{half} vs {full}");
        assert!(of_memory_bytes(1, 1) > 0);
    }

    #[test]
    fn schedule_reuse() {
        let input = OfInput::PackedHalf { format: 1, full_width: 1280, full_height: 800 };
        for f in [6u32, 7] {
            let a = of_schedule(640, 400, f, input);
            let b = of_schedule(640, 400, f + 2, input);
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert_eq!((x.pass, x.groups, &x.bind, x.spd_constants, x.push), (y.pass, y.groups, &y.bind, y.spd_constants, y.push));
                let mut patched = x.constants;
                set_frame_index(&mut patched, f + 2);
                assert_eq!(patched, y.constants);
            }
        }
    }

    #[test]
    fn luma_half_reference_rounds_exactly() {
        use crate::packed::{to_packed, PackedFormat};
        let mut img = crate::Image::new(3, 1, [0u8, 0, 0, 255]);
        img.set(0, 0, [255, 255, 255, 255]);
        img.set(1, 0, [255, 255, 255, 255]);
        img.set(2, 0, [10, 20, 30, 255]);
        let out = luma_half_reference(PackedFormat::Bgra8, &to_packed(PackedFormat::Bgra8, &img));
        assert_eq!((out.width, out.height), (2, 1));
        assert_eq!(out.get(0, 0), 255);
        assert_eq!(out.get(1, 0), ((2126 * 10 + 7152 * 20 + 722 * 30 + 5000) / 10000) as u8);
        let ten = luma_half_reference(PackedFormat::A2b10g10r10, &to_packed(PackedFormat::A2b10g10r10, &img));
        assert_eq!(ten.get(0, 0), 255);
    }

    #[test]
    fn of_constants_bytes() {
        let d = of_schedule(1280, 800, 5, OfInput::Color);
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
