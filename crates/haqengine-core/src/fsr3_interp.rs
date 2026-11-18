//! The FSR 3 frame interpolation schedule: what FidelityFX SDK v1.1.4's host code dispatches for
//! one interpolated frame, fed with zero game motion, a constant depth and half-resolution flow.

use crate::fsr3_flow::{half_size, of_sizes};

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
}

/// Per-frame inputs to the constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiParams {
    pub frame_since_reset: u32,
    pub delta_time_ms: f32,
}

/// One compute dispatch. `clears` run before it, as `(image, 32-bit clear value)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiDispatch {
    pub pass: usize,
    pub groups: [u32; 3],
    pub bind: Vec<(u32, FiRes)>,
    pub pyramid_constants: Option<[u8; 16]>,
    pub clears: Vec<(FiRes, u32)>,
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

/// Patches the only per-frame word, `reset`; the frame counter lives in the `Counters` buffer,
/// which the setup pass updates on the GPU.
pub fn set_fi_frame(constants: &mut [u8; FI_CONSTANTS_SIZE], frame_since_reset: u32) {
    put(constants, 44, (frame_since_reset == 0) as u32);
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

fn mips(sizes: &FiSizes, first: u32) -> impl Iterator<Item = (u32, FiRes)> + '_ {
    (0..FI_PYRAMID_BINDINGS).map(move |i| (first + i, FiRes::PyramidMip(i.min(sizes.pyramid_mips - 1))))
}

/// Every dispatch of one interpolated frame, in `ffxFrameInterpolationDispatch` order. The
/// prepare pass is left out: on a constant depth and zero motion its outputs are constant images.
pub fn fi_schedule(sizes: &FiSizes, p: &FiParams) -> Vec<FiDispatch> {
    use FiRes::*;
    let d = |pass, groups, bind: Vec<(u32, FiRes)>| FiDispatch { pass, groups, bind, pyramid_constants: None, clears: Vec::new() };
    let (render, display) = (groups8(sizes.render), groups8(sizes.display));
    let mut out = vec![d(
        PASS_FI_SETUP,
        render,
        vec![(0, Scd), (1, GameFieldX), (2, GameFieldY), (3, FlowFieldX), (4, FlowFieldY), (5, Disocclusion), (6, Counters)],
    )];
    if p.frame_since_reset != 0 {
        let mut rpd = d(
            PASS_FI_RECONSTRUCT_PREVIOUS_DEPTH,
            render,
            vec![(0, DilatedMotion), (1, DilatedDepth), (2, CurColor), (3, Distortion), (4, ReconstructedInterpDepth)],
        );
        rpd.clears.push((ReconstructedInterpDepth, 1.0f32.to_bits()));
        out.push(rpd);
        out.push(d(
            PASS_FI_GAME_MOTION_VECTOR_FIELD,
            render,
            vec![(0, DilatedMotion), (1, DilatedDepth), (2, PrevColor), (3, CurColor), (4, Distortion), (5, GameFieldX), (6, GameFieldY)],
        ));
        let (gc, gg) = spd(sizes.render.0, sizes.render.1);
        let mut game = d(PASS_FI_GAME_VECTOR_FIELD_INPAINTING_PYRAMID, gg, vec![(0, GameFieldX), (1, GameFieldY), (2, Counters)]);
        game.bind.extend(mips(sizes, 3));
        game.pyramid_constants = Some(gc);
        out.push(game);
        out.push(d(
            PASS_FI_OPTICAL_FLOW_VECTOR_FIELD,
            groups8(sizes.flow),
            vec![(0, Flow), (1, Confidence), (2, DilatedDepth), (3, PrevColor), (4, CurColor), (6, FlowFieldX), (7, FlowFieldY)],
        ));
        out.push(d(
            PASS_FI_DISOCCLUSION_MASK,
            render,
            vec![
                (0, GameFieldX),
                (1, GameFieldY),
                (2, ReconstructedPrevDepth),
                (3, DilatedDepth),
                (4, ReconstructedInterpDepth),
                (5, Pyramid),
                (6, Distortion),
                (7, Disocclusion),
            ],
        ));
    }
    out.push(d(
        PASS_FI_INTERPOLATION,
        display,
        vec![
            (0, GameFieldX),
            (1, GameFieldY),
            (2, FlowFieldX),
            (3, FlowFieldY),
            (4, PrevColor),
            (5, CurColor),
            (6, Disocclusion),
            (7, Pyramid),
            (8, Counters),
            (9, Output),
        ],
    ));
    let (oc, og) = spd(sizes.display.0, sizes.display.1);
    let mut pyramid = d(PASS_FI_INPAINTING_PYRAMID, og, vec![(0, Output), (1, Counters)]);
    pyramid.bind.extend(mips(sizes, 2));
    pyramid.pyramid_constants = Some(oc);
    out.push(pyramid);
    out.push(d(PASS_FI_INPAINTING, display, vec![(0, Scd), (1, Pyramid), (2, CurColor), (3, CurColor), (7, Output)]));
    out
}

/// AMD's prepare pass on its own, for checking that its outputs are the constant images.
pub fn fi_prepare_dispatch(sizes: &FiSizes) -> FiDispatch {
    use FiRes::*;
    FiDispatch {
        pass: PASS_FI_RECONSTRUCT_AND_DILATE,
        groups: groups8(sizes.render),
        bind: vec![(0, InputMotion), (1, InputDepth), (2, ReconstructedPrevDepth), (3, DilatedMotion), (4, DilatedDepth)],
        pyramid_constants: None,
        clears: Vec::new(),
    }
}

/// Texel bytes of the images interpolation owns: everything but the two frames, the flow and
/// the scene-change words, which belong to the frame generator and the optical flow.
pub fn fi_memory_bytes(display_w: u32, display_h: u32) -> u64 {
    let s = fi_sizes(display_w, display_h);
    let area = |(w, h): (u32, u32)| w as u64 * h as u64;
    let render = area(s.render);
    let mut pyramid = 0u64;
    for m in 0..s.pyramid_mips {
        pyramid += area(((s.pyramid.0 >> m).max(1), (s.pyramid.1 >> m).max(1))) * 8;
    }
    let render_images = render * (4 + 4 + 4 + 4 + 4 * 4 + 2);
    render_images + pyramid + area(s.display) * 4 + 8 + 2 + 4
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

    #[test]
    fn fi_reset_skips_preparation() {
        let s = fi_sizes(1280, 800);
        let passes = |f| fi_schedule(&s, &FiParams { frame_since_reset: f, delta_time_ms: 33.3 }).iter().map(|d| d.pass).collect::<Vec<_>>();
        assert_eq!(passes(0), [PASS_FI_SETUP, PASS_FI_INTERPOLATION, PASS_FI_INPAINTING_PYRAMID, PASS_FI_INPAINTING]);
        assert_eq!(
            passes(1),
            [
                PASS_FI_SETUP,
                PASS_FI_RECONSTRUCT_PREVIOUS_DEPTH,
                PASS_FI_GAME_MOTION_VECTOR_FIELD,
                PASS_FI_GAME_VECTOR_FIELD_INPAINTING_PYRAMID,
                PASS_FI_OPTICAL_FLOW_VECTOR_FIELD,
                PASS_FI_DISOCCLUSION_MASK,
                PASS_FI_INTERPOLATION,
                PASS_FI_INPAINTING_PYRAMID,
                PASS_FI_INPAINTING
            ]
        );
    }

    #[test]
    fn fi_dispatch_sizes_1280x800() {
        let s = fi_sizes(1280, 800);
        let d = fi_schedule(&s, &P1);
        let by = |pass| d.iter().find(|x| x.pass == pass).unwrap();
        assert_eq!(by(PASS_FI_SETUP).groups, [80, 50, 1]);
        assert_eq!(by(PASS_FI_INTERPOLATION).groups, [160, 100, 1]);
        assert_eq!(by(PASS_FI_INPAINTING).groups, [160, 100, 1]);
        assert_eq!(by(PASS_FI_OPTICAL_FLOW_VECTOR_FIELD).groups, [10, 7, 1]);
        let game = by(PASS_FI_GAME_VECTOR_FIELD_INPAINTING_PYRAMID);
        assert_eq!(game.groups, [10, 7, 1]);
        let gp = game.pyramid_constants.unwrap();
        assert_eq!([word(&gp, 0), word(&gp, 4)], [9, 70]);
        let out = by(PASS_FI_INPAINTING_PYRAMID);
        assert_eq!(out.groups, [20, 13, 1]);
        let op = out.pyramid_constants.unwrap();
        assert_eq!([word(&op, 0), word(&op, 4)], [10, 260]);
        let rpd = by(PASS_FI_RECONSTRUCT_PREVIOUS_DEPTH);
        assert_eq!(rpd.clears, vec![(FiRes::ReconstructedInterpDepth, 1.0f32.to_bits())]);
    }

    #[test]
    fn fi_bindings_follow_amd() {
        let s = fi_sizes(1280, 800);
        let d = fi_schedule(&s, &P1);
        let by = |pass| d.iter().find(|x| x.pass == pass).unwrap();
        assert_eq!(
            by(PASS_FI_INTERPOLATION).bind,
            vec![
                (0, FiRes::GameFieldX),
                (1, FiRes::GameFieldY),
                (2, FiRes::FlowFieldX),
                (3, FiRes::FlowFieldY),
                (4, FiRes::PrevColor),
                (5, FiRes::CurColor),
                (6, FiRes::Disocclusion),
                (7, FiRes::Pyramid),
                (8, FiRes::Counters),
                (9, FiRes::Output)
            ]
        );
        let pyramid = &by(PASS_FI_INPAINTING_PYRAMID).bind;
        assert_eq!(pyramid[0], (0, FiRes::Output));
        assert_eq!(pyramid[2], (2, FiRes::PyramidMip(0)));
        assert_eq!(pyramid.last(), Some(&(14, FiRes::PyramidMip(9))));
        assert_eq!(pyramid.len(), 2 + FI_PYRAMID_BINDINGS as usize);
        assert_eq!(by(PASS_FI_OPTICAL_FLOW_VECTOR_FIELD).bind[0], (0, FiRes::Flow));
        assert_eq!(by(PASS_FI_SETUP).bind[0], (0, FiRes::Scd));
        let prepare = fi_prepare_dispatch(&s);
        assert_eq!(prepare.pass, PASS_FI_RECONSTRUCT_AND_DILATE);
        assert_eq!(prepare.groups, [80, 50, 1]);
        assert_eq!(
            prepare.bind,
            vec![
                (0, FiRes::InputMotion),
                (1, FiRes::InputDepth),
                (2, FiRes::ReconstructedPrevDepth),
                (3, FiRes::DilatedMotion),
                (4, FiRes::DilatedDepth)
            ]
        );
    }

    #[test]
    fn fi_schedule_reuse() {
        let s = fi_sizes(1280, 800);
        let five = FiParams { frame_since_reset: 5, delta_time_ms: 33.3 };
        let seven = FiParams { frame_since_reset: 7, delta_time_ms: 33.3 };
        assert_eq!(fi_schedule(&s, &five), fi_schedule(&s, &seven));
        let mut c = fi_constants(&s, &five);
        set_fi_frame(&mut c, 7);
        assert_eq!(c, fi_constants(&s, &seven));
        set_fi_frame(&mut c, 0);
        assert_eq!(c, fi_constants(&s, &FiParams { frame_since_reset: 0, delta_time_ms: 33.3 }));
    }

    #[test]
    fn fi_memory_bounded() {
        let bytes = fi_memory_bytes(1280, 800);
        assert!(bytes <= 24 << 20, "{bytes}");
        assert!(bytes > fi_memory_bytes(640, 400));
    }
}
