//! HaqEngine's per-pixel blend between two real frames: zero motion first, then the flow vectors
//! of the pixel's block and its edge neighbours, chosen by a 3×3 window of 8-bit luma.

use crate::flow::Flow;
use crate::image::{Image, Rgba8};
use crate::frame_change::{is_cut, ChangeCounts};
use crate::packed::PackedFormat;

/// Above this weighted 3×3 sum of 8-bit luma differences the best candidate's samples show
/// different things, and the pixel takes the nearer real frame.
pub const DISAGREE8: u32 = 288;

/// The per-pixel choice: the winning candidate's cost and its vector (full-resolution pixels); the
/// samples disagree when the cost is above `DISAGREE8`. `vector_at` gives the vector of a 16×16 block of a `blocks_w × blocks_h` grid;
/// the luma closures read 8-bit half-resolution luma, clamping coordinates.
#[allow(clippy::too_many_arguments)]
pub fn choose_vector(
    x: u32,
    y: u32,
    blocks: (u32, u32),
    vector_at: impl Fn(u32, u32) -> [f32; 2],
    luma_prev: impl Fn(i64, i64) -> i32,
    luma_cur: impl Fn(i64, i64) -> i32,
    t: f32,
) -> (u32, [f32; 2]) {
    let window = |ax: f32, ay: f32, bx: f32, by: f32| {
        let (pa, pb) = (((ax / 2.0).floor() as i64, (ay / 2.0).floor() as i64), ((bx / 2.0).floor() as i64, (by / 2.0).floor() as i64));
        let mut sum = 0u32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let weight = if dx == 0 && dy == 0 { 4 } else { 1 };
                sum += weight * luma_prev(pa.0 + dx, pa.1 + dy).abs_diff(luma_cur(pb.0 + dx, pb.1 + dy));
            }
        }
        sum
    };
    let (bx, by) = ((x / 16).min(blocks.0 - 1) as i64, (y / 16).min(blocks.1 - 1) as i64);
    let mut cands = [[0.0f32; 2]; 6];
    let mut n = 1;
    for (ox, oy) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
        let (nx, ny) = (bx + ox, by + oy);
        if nx >= 0 && ny >= 0 && nx < blocks.0 as i64 && ny < blocks.1 as i64 {
            let v = vector_at(nx as u32, ny as u32);
            if !cands[..n].contains(&v) {
                cands[n] = v;
                n += 1;
            }
        }
    }
    let (px, py) = (x as f32, y as f32);
    let mut best = (u32::MAX, [0.0f32; 2]);
    for v in &cands[..n] {
        let cost = window(px - t * v[0], py - t * v[1], px + (1.0 - t) * v[0], py + (1.0 - t) * v[1]);
        if cost < best.0 {
            best = (cost, *v);
        }
    }
    best
}

fn luma8(img: &Image<f32>) -> impl Fn(i64, i64) -> i32 + '_ {
    move |x, y| {
        let l = img.get(x.clamp(0, img.width as i64 - 1) as u32, y.clamp(0, img.height as i64 - 1) as u32);
        (l.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as i32
    }
}

/// HaqEngine's blend on Battery's level-0 flow (half-resolution pixels per 16×16 block) at time
/// `t`, with its change counts; `t = 0` and `t = 1` return the real frames, and a scene cut
/// (`is_cut`) returns the current frame, as the GPU's fixup pass does.
pub fn battery_blend_reference(
    prev: &Image<Rgba8>,
    cur: &Image<Rgba8>,
    flow: &Flow,
    prev_luma: &Image<f32>,
    cur_luma: &Image<f32>,
    t: f32,
) -> (Image<Rgba8>, ChangeCounts) {
    let mut counts = ChangeCounts { pixels: cur.width * cur.height, ..ChangeCounts::default() };
    if t == 0.0 {
        return (prev.clone(), counts);
    }
    if t == 1.0 {
        return (cur.clone(), counts);
    }
    let mut out = cur.clone();
    for y in 0..cur.height {
        for x in 0..cur.width {
            let vector_at = |bx, by| {
                let v: [f32; 2] = flow.get(bx, by);
                [v[0] * 2.0, v[1] * 2.0]
            };
            let (cost, v) = choose_vector(x, y, (flow.width, flow.height), vector_at, luma8(prev_luma), luma8(cur_luma), t);
            let disagree = cost > DISAGREE8;
            counts.disagree += disagree as u32;
            counts.still += (cost == 0 && v == [0.0, 0.0]) as u32;
            let (px, py) = (x as f32, y as f32);
            let a = crate::flow::bilinear(prev, px - t * v[0], py - t * v[1]);
            let b = crate::flow::bilinear(cur, px + (1.0 - t) * v[0], py + (1.0 - t) * v[1]);
            let mut c = [0u8, 0, 0, 255];
            for i in 0..3 {
                let value = if disagree { if t >= 0.5 { b[i] } else { a[i] } } else { a[i] * (1.0 - t) + b[i] * t };
                c[i] = (value + 0.5).floor().clamp(0.0, 255.0) as u8;
            }
            out.set(x, y, c);
        }
    }
    if is_cut(&counts) {
        return (cur.clone(), counts);
    }
    (out, counts)
}

/// `battery_blend_reference` on packed pixels, rounding through `pack` as the shader does.
pub fn battery_blend_packed_reference(
    f: PackedFormat,
    prev: &Image<u32>,
    cur: &Image<u32>,
    flow: &Flow,
    prev_luma: &Image<f32>,
    cur_luma: &Image<f32>,
    t: f32,
) -> (Image<u32>, ChangeCounts) {
    use crate::packed::{bilinear_packed, pack};
    let mut counts = ChangeCounts { pixels: cur.width * cur.height, ..ChangeCounts::default() };
    if t == 0.0 {
        return (prev.clone(), counts);
    }
    if t == 1.0 {
        return (cur.clone(), counts);
    }
    let mut out = cur.clone();
    for y in 0..cur.height {
        for x in 0..cur.width {
            let vector_at = |bx, by| {
                let v: [f32; 2] = flow.get(bx, by);
                [v[0] * 2.0, v[1] * 2.0]
            };
            let (cost, v) = choose_vector(x, y, (flow.width, flow.height), vector_at, luma8(prev_luma), luma8(cur_luma), t);
            let disagree = cost > DISAGREE8;
            counts.disagree += disagree as u32;
            counts.still += (cost == 0 && v == [0.0, 0.0]) as u32;
            let (px, py) = (x as f32, y as f32);
            let a = bilinear_packed(f, prev, px - t * v[0], py - t * v[1]);
            let b = bilinear_packed(f, cur, px + (1.0 - t) * v[0], py + (1.0 - t) * v[1]);
            let mut c = [0.0f32, 0.0, 0.0, 1.0];
            for i in 0..3 {
                c[i] = if disagree { if t >= 0.5 { b[i] } else { a[i] } } else { a[i] * (1.0 - t) + b[i] * t };
            }
            out.set(x, y, pack(f, c));
        }
    }
    if is_cut(&counts) {
        return (cur.clone(), counts);
    }
    (out, counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::{to_packed, unpack};
    use crate::reference::luma_downsample;
    use crate::{checker, noise, shift};

    fn uniform(width: u32, height: u32, v: [f32; 2]) -> Flow {
        Image::new(width.div_ceil(16), height.div_ceil(16), v)
    }

    use crate::flow::{estimate_flow, pyramid};
    use crate::frame_change::is_cut;

    fn battery(prev: &Image<Rgba8>, cur: &Image<Rgba8>) -> (Image<Rgba8>, crate::frame_change::ChangeCounts) {
        let (pp, cp) = (pyramid(prev), pyramid(cur));
        battery_blend_reference(prev, cur, &estimate_flow(&pp, &cp), &pp[0], &cp[0], 0.5)
    }

    #[test]
    fn blend_counts_pan_cut_still() {
        let a = noise(256, 160, 21);
        let (_, pan) = battery(&a, &shift(&a, 8, -4));
        assert_eq!(pan.pixels, 256 * 160);
        assert!(pan.disagree * 20 < pan.pixels, "{pan:?}");
        assert_eq!(pan.still, 0);
        let b = crate::gradient(256, 160);
        let (out, cut) = battery(&a, &b);
        assert!(is_cut(&cut), "{cut:?}");
        assert_eq!(out, b);
        let (_, still) = battery(&a, &a);
        assert_eq!(still.still, still.pixels);
    }

    #[test]
    fn flash_is_not_a_cut() {
        let a = noise(256, 160, 23);
        let mut b = a.clone();
        for y in 0..160 {
            for x in 154..256 {
                let p = a.get(x, y);
                b.set(x, y, [p[0].saturating_add(80), p[1].saturating_add(80), p[2].saturating_add(80), 255]);
            }
        }
        let (_, c) = battery(&a, &b);
        assert!(!is_cut(&c), "{c:?}");
    }

    #[test]
    fn battery_blend_endpoints_exact() {
        let (prev, cur) = (noise(48, 32, 1), noise(48, 32, 2));
        let (lp, lc) = (luma_downsample(&prev), luma_downsample(&cur));
        let flow = uniform(48, 32, [3.0, -2.0]);
        assert_eq!(battery_blend_reference(&prev, &cur, &flow, &lp, &lc, 0.0).0, prev);
        assert_eq!(battery_blend_reference(&prev, &cur, &flow, &lp, &lc, 1.0).0, cur);
    }

    #[test]
    fn battery_blend_uniform_pan_matches_shift() {
        let prev = noise(64, 48, 3);
        let cur = shift(&prev, 8, -4);
        let (out, _) = battery_blend_reference(&prev, &cur, &uniform(64, 48, [4.0, -2.0]), &luma_downsample(&prev), &luma_downsample(&cur), 0.5);
        let want = shift(&prev, 4, -2);
        for y in 12..36 {
            for x in 12..52 {
                assert_eq!(out.get(x, y)[..3], want.get(x, y)[..3], "({x}, {y})");
            }
        }
    }

    #[test]
    fn battery_blend_static_overlay_takes_zero() {
        let bar = checker(16, 16, 2);
        let mut prev = noise(64, 48, 5);
        let mut cur = shift(&prev, 8, 0);
        for img in [&mut prev, &mut cur] {
            for y in 0..16 {
                for x in 0..16 {
                    img.set(24 + x, 16 + y, bar.get(x, y));
                }
            }
        }
        let (out, _) = battery_blend_reference(&prev, &cur, &uniform(64, 48, [4.0, 0.0]), &luma_downsample(&prev), &luma_downsample(&cur), 0.5);
        for y in 18..30 {
            for x in 26..38 {
                assert_eq!(out.get(x, y)[..3], bar.get(x - 24, y - 16)[..3], "bar ({x}, {y})");
            }
        }
    }

    #[test]
    fn battery_blend_disagreement_takes_nearer_frame() {
        let prev = noise(64, 64, 31);
        let mut cur = prev.clone();
        for y in 0..64 {
            for x in 16..32 {
                cur.set(x, y, [255, 255, 255, 255]);
            }
        }
        let (lp, lc) = (luma_downsample(&prev), luma_downsample(&cur));
        let flow = uniform(64, 64, [0.0, 0.0]);
        let (half, counts) = battery_blend_reference(&prev, &cur, &flow, &lp, &lc, 0.5);
        assert!(!crate::frame_change::is_cut(&counts));
        let (early, _) = battery_blend_reference(&prev, &cur, &flow, &lp, &lc, 0.4);
        for y in 4..60 {
            for x in 20..28 {
                assert_eq!(half.get(x, y)[..3], [255, 255, 255], "t=0.5 ({x}, {y})");
                assert_eq!(early.get(x, y)[..3], prev.get(x, y)[..3], "t=0.4 ({x}, {y})");
            }
        }
    }

    #[test]
    fn battery_blend_packed_matches_rgba() {
        let prev = noise(64, 48, 9);
        let cur = shift(&prev, 6, 2);
        let (lp, lc) = (luma_downsample(&prev), luma_downsample(&cur));
        let flow = uniform(64, 48, [3.0, 1.0]);
        let (rgba, _) = battery_blend_reference(&prev, &cur, &flow, &lp, &lc, 0.5);
        let f = PackedFormat::Rgba8;
        let (packed, _) = battery_blend_packed_reference(f, &to_packed(f, &prev), &to_packed(f, &cur), &flow, &lp, &lc, 0.5);
        for (e, g) in rgba.data.iter().zip(&packed.data) {
            let g = unpack(f, *g);
            for c in 0..3 {
                assert!((e[c] as f32 - g[c] * 255.0).abs() <= 1.01, "{e:?} vs {g:?}");
            }
        }
    }
}
