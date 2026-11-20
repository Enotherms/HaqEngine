//! HaqEngine's per-pixel blend between two real frames: zero motion first, then the flow vectors
//! of the pixel's block and its edge neighbours, chosen by a 3×3 window of 8-bit luma.

use crate::flow::Flow;
use crate::image::{Image, Rgba8};
use crate::frame_change::{is_cut, ChangeCounts};
use crate::packed::PackedFormat;

/// Above this weighted 3×3 sum of 8-bit luma differences the best candidate's samples show
/// different things, and the pixel takes the nearer real frame.
pub const DISAGREE8: u32 = 288;

/// A candidate whose cost is within this of the best (and not above `DISAGREE8`) explains the
/// pixel nearly as well; among those the smallest motion wins, so a near-static foreground (an ear
/// over a fast background) is not painted over by the background passing behind it.
pub const NEAR_BEST8: u32 = 96;

/// A smaller candidate must also match in shape: its 3×3 differences, with their weighted mean
/// removed, may sum to at most this. A brightness change (an ear turning) passes; texture that
/// only lines up badly (a dim pan seen as still) does not.
pub const SHAPE8: u32 = 16;

/// The per-pixel choice: the winning candidate's cost and its vector (full-resolution pixels): the
/// smallest motion among the candidates within `NEAR_BEST8` of the best cost whose shape matches
/// (`SHAPE8`), else the best one; above `DISAGREE8` the samples disagree. `vector_at` gives the vector of a 16×16 block of a `blocks_w × blocks_h` grid;
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
        let (mut sum, mut signed) = (0u32, 0i32);
        let mut diffs = [(0i32, 0i32); 9];
        let mut k = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let weight = if dx == 0 && dy == 0 { 4 } else { 1 };
                let d = luma_prev(pa.0 + dx, pa.1 + dy) - luma_cur(pb.0 + dx, pb.1 + dy);
                sum += weight as u32 * d.unsigned_abs();
                signed += weight * d;
                diffs[k] = (weight, d);
                k += 1;
            }
        }
        let shape12: u32 = diffs.iter().map(|&(w, d)| w as u32 * (12 * d - signed).unsigned_abs()).sum();
        (sum, shape12)
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
    let mut costs = [0u32; 6];
    let mut shapes = [0u32; 6];
    for i in 0..n {
        let v = cands[i];
        (costs[i], shapes[i]) = window(px - t * v[0], py - t * v[1], px + (1.0 - t) * v[0], py + (1.0 - t) * v[1]);
    }
    let best_cost = *costs[..n].iter().min().expect("the zero candidate");
    let first_best = costs[..n].iter().position(|&c| c == best_cost).expect("present");
    if best_cost > DISAGREE8 {
        return (best_cost, cands[first_best]);
    }
    let limit = (best_cost + NEAR_BEST8).min(DISAGREE8);
    let size = |v: &[f32; 2]| v[0].abs() + v[1].abs();
    let mut pick = first_best;
    for i in 0..n {
        if costs[i] <= limit && shapes[i] <= 12 * SHAPE8 && size(&cands[i]) < size(&cands[pick]) {
            pick = i;
        }
    }
    (costs[pick], cands[pick])
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

    fn grey(v: u8) -> Rgba8 {
        [v, v, v, 255]
    }

    #[test]
    fn thin_static_bar_survives_a_pan() {
        let mut prev = noise(256, 128, 41);
        let mut cur = shift(&prev, 24, 0);
        for y in 40..88 {
            for x in 100..112 {
                prev.set(x, y, grey(204));
                cur.set(x, y, grey(210));
            }
        }
        let (lp, lc) = (luma_downsample(&prev), luma_downsample(&cur));
        let (out, _) = battery_blend_reference(&prev, &cur, &uniform(256, 128, [12.0, 0.0]), &lp, &lc, 1.0 / 3.0);
        for y in 46..82 {
            for x in 103..109 {
                let p = out.get(x, y);
                assert!((0..3).all(|i| p[i].abs_diff(206) <= 8), "bar ({x}, {y}) is {p:?}");
            }
        }
    }

    #[test]
    fn dim_pan_keeps_moving() {
        let mut prev = noise(256, 128, 43);
        for y in 0..128 {
            for x in 0..256 {
                let n = prev.get(x, y)[0];
                prev.set(x, y, grey(124 + n % 9));
            }
        }
        let cur = shift(&prev, 12, 0);
        let (lp, lc) = (luma_downsample(&prev), luma_downsample(&cur));
        let (out, _) = battery_blend_reference(&prev, &cur, &uniform(256, 128, [6.0, 0.0]), &lp, &lc, 0.5);
        let truth = shift(&prev, 6, 0);
        let (mut sum, mut n) = (0u64, 0u64);
        for y in 16..112 {
            for x in 32..224 {
                sum += out.get(x, y)[0].abs_diff(truth.get(x, y)[0]) as u64;
                n += 1;
            }
        }
        let mean = sum as f64 / n as f64;
        assert!(mean <= 2.0, "dim pan mean error {mean:.3}");
    }

    #[test]
    fn smallest_acceptable_vector_wins_and_reports_its_cost() {
        let blocks = (4, 4);
        let vector_at = |_, _| [20.0f32, 0.0];
        let pick = |zero_cost: i32| {
            let prev = |_x: i64, _y: i64| 100;
            let cur = move |x: i64, y: i64| if (14..=16).contains(&x) && (14..=16).contains(&y) { 100 + zero_cost / 12 } else { 100 };
            choose_vector(30, 30, blocks, vector_at, prev, cur, 0.5)
        };
        assert_eq!(pick(60), (60, [0.0, 0.0]));
        assert_eq!(pick(120), (0, [20.0, 0.0]));
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
