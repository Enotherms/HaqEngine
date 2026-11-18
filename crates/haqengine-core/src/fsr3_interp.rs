//! The FSR 3 frame interpolation schedule: what FidelityFX SDK v1.1.4's host code dispatches for
//! one interpolated frame, fed with zero game motion, a constant depth and half-resolution flow.

use crate::fsr3_flow::{half_size, of_sizes};
use crate::{Image, Rgba8};

pub const PASS_FI_SETUP: usize = 0;
pub const PASS_FI_RECONSTRUCT_AND_DILATE: usize = 1;
pub const PASS_FI_RECONSTRUCT_PREVIOUS_DEPTH: usize = 2;
pub const PASS_FI_GAME_MOTION_VECTOR_FIELD: usize = 3;
pub const PASS_FI_GAME_VECTOR_FIELD_INPAINTING_PYRAMID: usize = 4;
pub const PASS_FI_OPTICAL_FLOW_VECTOR_FIELD: usize = 5;
pub const PASS_FI_DISOCCLUSION_MASK: usize = 6;
pub const PASS_FI_INTERPOLATION: usize = 7;
pub const PASS_FI_INPAINTING_PYRAMID: usize = 8;
pub const PASS_FI_INPAINTING: usize = 9;
pub const PASS_FI_BLEND: usize = 10;

/// Above this summed 8-bit luma difference over the 3×3 window the best candidate's two samples
/// show different things, and the blend takes the current frame instead of a ghost of both.
pub const DISAGREE8: u32 = 288;

pub const FI_CONSTANTS_SIZE: usize = 176;
pub const FI_PYRAMID_BINDINGS: u32 = 13;
pub const CAMERA_NEAR: f32 = 0.1;
pub const CAMERA_FAR: f32 = 100.0;
pub const CAMERA_FOV_VERTICAL: f32 = std::f32::consts::FRAC_PI_3;

/// Image sizes for one display size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiSizes {
    pub display: (u32, u32),
    pub render: (u32, u32),
    pub flow: (u32, u32),
    pub pyramid: (u32, u32),
    pub pyramid_mips: u32,
}

/// The images interpolation reads and writes. `Flow` and `Scd` are the optical flow's output
/// and scene-change words; `PrevColor` and `CurColor` are the two real frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FiRes {
    PrevColor,
    CurColor,
    Output,
    InputDepth,
    InputMotion,
    DilatedDepth,
    DilatedMotion,
    ReconstructedPrevDepth,
    ReconstructedInterpDepth,
    GameFieldX,
    GameFieldY,
    FlowFieldX,
    FlowFieldY,
    Disocclusion,
    Pyramid,
    PyramidMip(u32),
    Counters,
    Distortion,
    Confidence,
    Flow,
    Scd,
    PrevLuma,
    CurLuma,
}

/// Per-frame inputs to the constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiParams {
    pub frame_since_reset: u32,
    pub delta_time_ms: f32,
}

/// One compute dispatch. `clears` run before it, as `(image or buffer, 32-bit fill)`; `push` is
/// used by `PASS_FI_BLEND` only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiDispatch {
    pub pass: usize,
    pub groups: [u32; 3],
    pub bind: Vec<(u32, FiRes)>,
    pub pyramid_constants: Option<[u8; 16]>,
    pub clears: Vec<(FiRes, u32)>,
    pub push: [u8; 20],
}

pub fn fi_sizes(display_w: u32, display_h: u32) -> FiSizes {
    let render = half_size(display_w, display_h);
    let pyramid = ((display_w / 2).max(1), (display_h / 2).max(1));
    FiSizes {
        display: (display_w, display_h),
        render,
        flow: of_sizes(render.0, render.1).flow[0],
        pyramid,
        pyramid_mips: pyramid.0.max(pyramid.1).ilog2() + 1,
    }
}

fn put(out: &mut [u8], offset: usize, word: u32) {
    out[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
}

/// `cbFI` in std140 order. Depth is non-inverted and finite, as `setupDeviceDepthToViewSpaceDepthParams`
/// computes for the fixed camera.
pub fn fi_constants(sizes: &FiSizes, p: &FiParams) -> [u8; FI_CONSTANTS_SIZE] {
    let mut c = [0u8; FI_CONSTANTS_SIZE];
    let (rw, rh) = sizes.render;
    let (dw, dh) = sizes.display;
    let f = |v: f32| v.to_bits();
    let q = CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR);
    let aspect = rw as f32 / rh as f32;
    let cot_half = (0.5 * CAMERA_FOV_VERTICAL).cos() / (0.5 * CAMERA_FOV_VERTICAL).sin();
    let horizontal = ((CAMERA_FOV_VERTICAL / 2.0).tan() * aspect).atan() * 2.0;
    let words: [(usize, u32); 30] = [
        (0, rw),
        (4, rh),
        (8, dw),
        (12, dh),
        (16, f(1.0 / dw as f32)),
        (20, f(1.0 / dh as f32)),
        (24, f(CAMERA_NEAR)),
        (28, f(CAMERA_FAR)),
        (32, dw),
        (36, dh),
        (40, 0),
        (44, (p.frame_since_reset == 0) as u32),
        (48, f(-q)),
        (52, f(q * CAMERA_NEAR)),
        (56, f(1.0 / (cot_half / aspect))),
        (60, f(1.0 / cot_half)),
        (64, f(p.delta_time_ms)),
        (68, 0),
        (72, 1),
        (76, 1),
        (80, f(1.0 / sizes.render.0 as f32)),
        (84, f(1.0 / sizes.render.1 as f32)),
        (88, 8),
        (96, rw),
        (100, rh),
        (108, 1),
        (120, dw),
        (124, dh),
        (148, f(1.0)),
        (152, f((horizontal * 0.5).tan())),
    ];
    for (offset, w) in words {
        put(&mut c, offset, w);
    }
    c
}

/// Patches the per-frame words: `reset` in the constants and in the blend's push constants.
pub fn set_fi_frame(constants: &mut [u8; FI_CONSTANTS_SIZE], schedule: &mut [FiDispatch], frame_since_reset: u32) {
    let reset = (frame_since_reset == 0) as u32;
    put(constants, 44, reset);
    for d in schedule.iter_mut().filter(|d| d.pass == PASS_FI_BLEND) {
        put(&mut d.push, 16, reset);
    }
}

fn spd(width: u32, height: u32) -> ([u8; 16], [u32; 3]) {
    let groups = [(width - 1) / 64 + 1, (height - 1) / 64 + 1, 1];
    let mips = width.max(height).ilog2().min(12);
    let mut out = [0u8; 16];
    put(&mut out, 0, mips);
    put(&mut out, 4, groups[0] * groups[1]);
    (out, groups)
}

fn groups8((w, h): (u32, u32)) -> [u32; 3] {
    [w.div_ceil(8), h.div_ceil(8), 1]
}

/// One interpolated frame: our blend on FSR 3's flow, then AMD's inpainting pyramid and
/// inpainting, which fill holes and fall back to the current frame on a scene cut.
pub fn fi_schedule(sizes: &FiSizes, p: &FiParams) -> Vec<FiDispatch> {
    use FiRes::*;
    let d = |pass, groups, bind: Vec<(u32, FiRes)>| FiDispatch {
        pass,
        groups,
        bind,
        pyramid_constants: None,
        clears: Vec::new(),
        push: [0; 20],
    };
    let display = groups8(sizes.display);
    let mut blend = d(
        PASS_FI_BLEND,
        display,
        vec![(0, PrevColor), (1, CurColor), (2, Flow), (3, PrevLuma), (4, CurLuma), (5, Output)],
    );
    blend.clears.push((Counters, 0));
    let words = [sizes.display.0, sizes.display.1, sizes.flow.0, sizes.flow.1, (p.frame_since_reset == 0) as u32];
    for (i, w) in words.iter().enumerate() {
        put(&mut blend.push, i * 4, *w);
    }
    let (oc, og) = spd(sizes.display.0, sizes.display.1);
    let mut pyramid = d(PASS_FI_INPAINTING_PYRAMID, og, vec![(0, Output), (1, Counters)]);
    pyramid.bind.extend((0..FI_PYRAMID_BINDINGS).map(|i| (2 + i, PyramidMip(i.min(sizes.pyramid_mips - 1)))));
    pyramid.pyramid_constants = Some(oc);
    vec![blend, pyramid, d(PASS_FI_INPAINTING, display, vec![(0, Scd), (1, Pyramid), (2, CurColor), (3, CurColor), (7, Output)])]
}

/// CPU reference of `fi_blend.comp`: per pixel, the candidates are zero, then the FSR 3 vectors of
/// the pixel's own 16×16 block and its edge neighbours; the one whose two 3×3 windows of FSR 3's
/// 8-bit luma agree best wins (zero on ties, which keeps a still HUD still), and a winner that
/// still disagrees takes the current frame.
pub fn fi_blend_reference(
    prev: &Image<Rgba8>,
    cur: &Image<Rgba8>,
    flow_raw: &Image<[i16; 2]>,
    prev_luma: &Image<u8>,
    cur_luma: &Image<u8>,
    reset: bool,
) -> Image<Rgba8> {
    let mut out = cur.clone();
    if reset {
        return out;
    }
    let luma_at = |img: &Image<u8>, x: i64, y: i64| {
        img.get(x.clamp(0, img.width as i64 - 1) as u32, y.clamp(0, img.height as i64 - 1) as u32) as i32
    };
    let window_cost = |ax: f32, ay: f32, bx: f32, by: f32| {
        let (pa, pb) = (((ax / 2.0).floor() as i64, (ay / 2.0).floor() as i64), ((bx / 2.0).floor() as i64, (by / 2.0).floor() as i64));
        let mut sum = 0u32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let weight = if dx == 0 && dy == 0 { 4 } else { 1 };
                sum += weight * luma_at(prev_luma, pa.0 + dx, pa.1 + dy).abs_diff(luma_at(cur_luma, pb.0 + dx, pb.1 + dy));
            }
        }
        sum
    };
    for y in 0..cur.height {
        for x in 0..cur.width {
            let (bx, by) = ((x / 16).min(flow_raw.width - 1) as i64, (y / 16).min(flow_raw.height - 1) as i64);
            let mut cands: Vec<[f32; 2]> = Vec::with_capacity(6);
            cands.push([0.0, 0.0]);
            for (ox, oy) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (bx + ox, by + oy);
                if nx >= 0 && ny >= 0 && nx < flow_raw.width as i64 && ny < flow_raw.height as i64 {
                    let v = crate::fsr3_flow::ffx_to_haq(flow_raw.get(nx as u32, ny as u32), 2.0);
                    if !cands.contains(&v) {
                        cands.push(v);
                    }
                }
            }
            let (px, py) = (x as f32, y as f32);
            let mut best = (u32::MAX, [0.0f32; 2]);
            for v in cands {
                let (ax, ay, bx2, by2) = (px - v[0] * 0.5, py - v[1] * 0.5, px + v[0] * 0.5, py + v[1] * 0.5);
                let cost = window_cost(ax, ay, bx2, by2);
                if cost < best.0 {
                    best = (cost, v);
                }
            }
            let v = best.1;
            let a = crate::flow::bilinear(prev, px - v[0] * 0.5, py - v[1] * 0.5);
            let b = crate::flow::bilinear(cur, px + v[0] * 0.5, py + v[1] * 0.5);
            let mut c = [0u8; 4];
            for i in 0..3 {
                let value = if best.0 > DISAGREE8 { b[i] } else { (a[i] + b[i]) * 0.5 };
                c[i] = (value + 0.5).floor().clamp(0.0, 255.0) as u8;
            }
            out.set(x, y, c);
        }
    }
    out
}

/// Texel bytes of the images interpolation owns (the output, the inpainting pyramid and the
/// counters); the frames, the flow, its luma and its scene-change words belong to others.
pub fn fi_memory_bytes(display_w: u32, display_h: u32) -> u64 {
    let s = fi_sizes(display_w, display_h);
    let area = |(w, h): (u32, u32)| w as u64 * h as u64;
    let mut pyramid = 0u64;
    for m in 0..s.pyramid_mips {
        pyramid += area(((s.pyramid.0 >> m).max(1), (s.pyramid.1 >> m).max(1))) * 8;
    }
    pyramid + area(s.display) * 4 + 8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(b: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(b[offset..offset + 4].try_into().unwrap())
    }

    fn float(b: &[u8], offset: usize) -> f32 {
        f32::from_bits(word(b, offset))
    }

    const P1: FiParams = FiParams { frame_since_reset: 1, delta_time_ms: 33.3 };

    #[test]
    fn fi_sizes_1280x800() {
        let s = fi_sizes(1280, 800);
        assert_eq!((s.display, s.render, s.flow), ((1280, 800), (640, 400), (80, 50)));
        assert_eq!((s.pyramid, s.pyramid_mips), ((640, 400), 10));
    }

    #[test]
    fn fi_sizes_1281x801() {
        let s = fi_sizes(1281, 801);
        assert_eq!((s.render, s.pyramid), ((641, 401), (640, 400)));
        assert_eq!(s.flow, of_sizes(641, 401).flow[0]);
    }

    #[test]
    fn fi_constants_layout() {
        let s = fi_sizes(1280, 800);
        let c = fi_constants(&s, &P1);
        assert_eq!(c.len(), 176);
        assert_eq!([word(&c, 0), word(&c, 4), word(&c, 8), word(&c, 12)], [640, 400, 1280, 800]);
        assert_eq!(float(&c, 16), 1.0 / 1280.0);
        assert_eq!([float(&c, 24), float(&c, 28)], [CAMERA_NEAR, CAMERA_FAR]);
        assert_eq!(word(&c, 44), 0);
        let q = CAMERA_FAR / (CAMERA_NEAR - CAMERA_FAR);
        assert_eq!([float(&c, 48), float(&c, 52)], [-q, q * CAMERA_NEAR]);
        assert_eq!(float(&c, 64), 33.3);
        assert_eq!([word(&c, 72), word(&c, 76)], [1, 1]);
        assert_eq!([float(&c, 80), float(&c, 84)], [1.0 / 640.0, 1.0 / 400.0]);
        assert_eq!(word(&c, 88), 8);
        assert_eq!([word(&c, 96), word(&c, 100)], [640, 400]);
        assert_eq!(word(&c, 108), 1);
        assert_eq!([word(&c, 112), word(&c, 116), word(&c, 120), word(&c, 124)], [0, 0, 1280, 800]);
        assert_eq!(word(&c, 140), 0);
        assert_eq!([float(&c, 144), float(&c, 148)], [0.0, 1.0]);
        let tan_half = (CAMERA_FOV_VERTICAL / 2.0).tan() * 640.0 / 400.0;
        assert!((float(&c, 152) - tan_half).abs() < 1e-5);
        assert_eq!(c[160..176], [0u8; 16]);
        let reset = fi_constants(&s, &FiParams { frame_since_reset: 0, delta_time_ms: 33.3 });
        assert_eq!(word(&reset, 44), 1);
    }

    fn luma8(img: &crate::Image<crate::Rgba8>) -> crate::Image<u8> {
        use crate::packed::{to_packed, PackedFormat};
        crate::fsr3_flow::luma_half_reference(PackedFormat::Rgba8, &to_packed(PackedFormat::Rgba8, img))
    }

    fn raw_flow(width: u32, height: u32, raw: [i16; 2]) -> crate::Image<[i16; 2]> {
        let s = fi_sizes(width, height);
        crate::Image::new(s.flow.0, s.flow.1, raw)
    }

    #[test]
    fn fi_blend_uniform_pan_follows_motion() {
        let prev = crate::noise(64, 48, 3);
        let cur = crate::shift(&prev, 8, -4);
        let out = fi_blend_reference(&prev, &cur, &raw_flow(64, 48, [-4, 2]), &luma8(&prev), &luma8(&cur), false);
        let want = crate::shift(&prev, 4, -2);
        for y in 12..36 {
            for x in 12..52 {
                assert_eq!(out.get(x, y)[..3], want.get(x, y)[..3], "({x}, {y})");
            }
        }
    }

    #[test]
    fn fi_blend_static_overlay_takes_zero() {
        let bar = crate::checker(16, 16, 2);
        let mut prev = crate::noise(64, 48, 5);
        let mut cur = crate::shift(&prev, 8, 0);
        let want_bg = crate::shift(&prev, 4, 0);
        for img in [&mut prev, &mut cur] {
            for y in 0..16 {
                for x in 0..16 {
                    img.set(24 + x, 16 + y, bar.get(x, y));
                }
            }
        }
        let out = fi_blend_reference(&prev, &cur, &raw_flow(64, 48, [-4, 0]), &luma8(&prev), &luma8(&cur), false);
        for y in 18..30 {
            for x in 26..38 {
                assert_eq!(out.get(x, y)[..3], bar.get(x - 24, y - 16)[..3], "bar ({x}, {y})");
            }
        }
        for y in 4..12 {
            for x in 8..56 {
                assert_eq!(out.get(x, y)[..3], want_bg.get(x, y)[..3], "background ({x}, {y})");
            }
        }
    }

    #[test]
    fn fi_blend_disagreement_takes_current() {
        let prev = crate::Image::new(32, 32, [0u8, 0, 0, 255]);
        let cur = crate::Image::new(32, 32, [255u8, 255, 255, 255]);
        let out = fi_blend_reference(&prev, &cur, &raw_flow(32, 32, [0, 0]), &luma8(&prev), &luma8(&cur), false);
        assert!(out.data.iter().all(|p| p[..3] == [255, 255, 255]));
    }

    #[test]
    fn fi_blend_reset_copies_current() {
        let prev = crate::noise(40, 24, 7);
        let cur = crate::noise(40, 24, 8);
        let out = fi_blend_reference(&prev, &cur, &raw_flow(40, 24, [-4, 2]), &luma8(&prev), &luma8(&cur), true);
        assert_eq!(out.data.iter().map(|p| [p[0], p[1], p[2]]).collect::<Vec<_>>(), cur.data.iter().map(|p| [p[0], p[1], p[2]]).collect::<Vec<_>>());
    }

    #[test]
    fn fi_schedule_order() {
        let s = fi_sizes(1280, 800);
        let passes = |f| fi_schedule(&s, &FiParams { frame_since_reset: f, delta_time_ms: 33.3 }).iter().map(|d| d.pass).collect::<Vec<_>>();
        let order = [PASS_FI_BLEND, PASS_FI_INPAINTING_PYRAMID, PASS_FI_INPAINTING];
        assert_eq!(passes(0), order);
        assert_eq!(passes(1), order);
    }

    #[test]
    fn fi_dispatch_sizes_1280x800() {
        let s = fi_sizes(1280, 800);
        let d = fi_schedule(&s, &P1);
        let by = |pass| d.iter().find(|x| x.pass == pass).unwrap();
        let blend = by(PASS_FI_BLEND);
        assert_eq!(blend.groups, [160, 100, 1]);
        assert_eq!(blend.clears, vec![(FiRes::Counters, 0)]);
        let push: Vec<u32> = blend.push.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(push, [1280, 800, 80, 50, 0]);
        assert_eq!(by(PASS_FI_INPAINTING).groups, [160, 100, 1]);
        let out = by(PASS_FI_INPAINTING_PYRAMID);
        assert_eq!(out.groups, [20, 13, 1]);
        let op = out.pyramid_constants.unwrap();
        assert_eq!([word(&op, 0), word(&op, 4)], [10, 260]);
    }

    #[test]
    fn fi_bindings_follow_amd() {
        let s = fi_sizes(1280, 800);
        let d = fi_schedule(&s, &P1);
        let by = |pass| d.iter().find(|x| x.pass == pass).unwrap();
        assert_eq!(
            by(PASS_FI_BLEND).bind,
            vec![
                (0, FiRes::PrevColor),
                (1, FiRes::CurColor),
                (2, FiRes::Flow),
                (3, FiRes::PrevLuma),
                (4, FiRes::CurLuma),
                (5, FiRes::Output)
            ]
        );
        let pyramid = &by(PASS_FI_INPAINTING_PYRAMID).bind;
        assert_eq!(pyramid[0], (0, FiRes::Output));
        assert_eq!(pyramid[2], (2, FiRes::PyramidMip(0)));
        assert_eq!(pyramid.last(), Some(&(14, FiRes::PyramidMip(9))));
        assert_eq!(pyramid.len(), 2 + FI_PYRAMID_BINDINGS as usize);
        assert_eq!(by(PASS_FI_INPAINTING).bind, vec![(0, FiRes::Scd), (1, FiRes::Pyramid), (2, FiRes::CurColor), (3, FiRes::CurColor), (7, FiRes::Output)]);
    }

    #[test]
    fn fi_schedule_reuse() {
        let s = fi_sizes(1280, 800);
        let reset = fi_schedule(&s, &FiParams { frame_since_reset: 0, delta_time_ms: 33.3 });
        let five = fi_schedule(&s, &FiParams { frame_since_reset: 5, delta_time_ms: 33.3 });
        assert_eq!(five, fi_schedule(&s, &FiParams { frame_since_reset: 7, delta_time_ms: 33.3 }));
        let mut c = fi_constants(&s, &FiParams { frame_since_reset: 5, delta_time_ms: 33.3 });
        let mut patched = five.clone();
        set_fi_frame(&mut c, &mut patched, 0);
        assert_eq!(patched, reset);
        assert_eq!(c, fi_constants(&s, &FiParams { frame_since_reset: 0, delta_time_ms: 33.3 }));
    }

    #[test]
    fn fi_memory_bounded() {
        let bytes = fi_memory_bytes(1280, 800);
        assert!(bytes <= 12 << 20, "{bytes}");
        assert!(bytes > fi_memory_bytes(640, 400));
    }
}
