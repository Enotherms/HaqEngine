//! Battery-mode frame generation on the CPU: luma pyramid, coarse-to-fine block matching and
//! warp-and-blend interpolation. These are the references the GPU shaders must match.

use crate::image::{Image, Rgba8};
use crate::reference::{luma_downsample, LUMA};

/// Block size in pixels of the level being matched.
pub const BLOCK: u32 = 8;
/// Pyramid levels: 0 is half resolution, each next level halves again.
pub const LEVELS: usize = 4;
/// Search radius at the coarsest level.
pub const RADIUS_COARSE: i32 = 4;
/// Search radius at the finer levels.
pub const RADIUS_FINE: i32 = 2;
/// Luma is matched as integers in `0..=QUANT`.
pub const QUANT: f32 = 1023.0;

/// One motion vector per block, in pixels of that level, from the previous frame to the current.
pub type Flow = Image<[f32; 2]>;

/// Halves a luma image (rounding sizes up), averaging the pixels that exist in each 2×2 block.
pub fn luma_reduce(input: &Image<f32>) -> Image<f32> {
    let (w, h) = (input.width.div_ceil(2), input.height.div_ceil(2));
    let mut out = Image::new(w, h, 0.0f32);
    for oy in 0..h {
        for ox in 0..w {
            let (mut sum, mut count) = (0.0f32, 0.0f32);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (x, y) = (ox * 2 + dx, oy * 2 + dy);
                if x < input.width && y < input.height {
                    sum += input.get(x, y);
                    count += 1.0;
                }
            }
            out.set(ox, oy, sum / count);
        }
    }
    out
}

/// The luma pyramid of a frame: level 0 from `luma_downsample`, then `luma_reduce`.
pub fn pyramid(frame: &Image<Rgba8>) -> Vec<Image<f32>> {
    let mut levels = vec![luma_downsample(frame)];
    while levels.len() < LEVELS {
        let next = luma_reduce(levels.last().expect("level 0 exists"));
        levels.push(next);
    }
    levels
}

/// Luma as an integer for matching.
pub fn quantize(l: f32) -> i32 {
    (l.clamp(0.0, 1.0) * QUANT + 0.5).floor() as i32
}

/// Block matching at one level, optionally refining a coarser level's flow.
pub fn block_match(prev: &Image<f32>, cur: &Image<f32>, coarse: Option<&Flow>, radius: i32) -> Flow {
    let mut flow = Image::new(prev.width.div_ceil(BLOCK), prev.height.div_ceil(BLOCK), [0.0f32; 2]);
    for by in 0..flow.height {
        for bx in 0..flow.width {
            let candidates = predictions(coarse, bx, by);
            let mut pred = candidates[0];
            let mut pred_score = i64::MAX;
            for &c in &candidates {
                let score = candidate_score(prev, cur, bx, by, c);
                if score < pred_score {
                    pred = c;
                    pred_score = score;
                }
            }
            let mut best = (i64::MAX, i32::MAX, pred);
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let offset = [pred[0] + dx, pred[1] + dy];
                    let sad = block_sad(prev, cur, bx, by, offset);
                    let d2 = dx * dx + dy * dy;
                    if sad < best.0 || (sad == best.0 && d2 < best.1) {
                        best = (sad, d2, offset);
                    }
                }
            }
            flow.set(bx, by, [best.2[0] as f32, best.2[1] as f32]);
        }
    }
    flow
}

/// Lowest SAD within ±1 pixel of a candidate, so a candidate half a pixel off the true motion
/// at the coarser level still scores well.
pub fn candidate_score(prev: &Image<f32>, cur: &Image<f32>, bx: u32, by: u32, c: [i32; 2]) -> i64 {
    let mut best = i64::MAX;
    for dy in -1..=1 {
        for dx in -1..=1 {
            best = best.min(block_sad(prev, cur, bx, by, [c[0] + dx, c[1] + dy]));
        }
    }
    best
}

/// Candidate predictions for a block, in order and without duplicates: its own coarse block, the
/// eight neighbouring coarse blocks (row-major, those that exist), then zero. Without `coarse`,
/// only zero.
pub fn predictions(coarse: Option<&Flow>, bx: u32, by: u32) -> Vec<[i32; 2]> {
    let Some(c) = coarse else { return vec![[0, 0]] };
    let scaled = |x: u32, y: u32| {
        let v = c.get(x, y);
        [(2.0 * v[0]).round() as i32, (2.0 * v[1]).round() as i32]
    };
    let (cx, cy) = ((bx / 2).min(c.width - 1) as i32, (by / 2).min(c.height - 1) as i32);
    let mut out = vec![scaled(cx as u32, cy as u32)];
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (x, y) = (cx + dx, cy + dy);
            if (dx, dy) != (0, 0) && x >= 0 && y >= 0 && x < c.width as i32 && y < c.height as i32 {
                out.push(scaled(x as u32, y as u32));
            }
        }
    }
    out.push([0, 0]);
    let mut unique: Vec<[i32; 2]> = Vec::with_capacity(out.len());
    for c in out {
        if !unique.contains(&c) {
            unique.push(c);
        }
    }
    unique
}

/// Integer SAD of block `(bx, by)` of `prev` against `cur` displaced by `offset` (clamped reads).
pub fn block_sad(prev: &Image<f32>, cur: &Image<f32>, bx: u32, by: u32, offset: [i32; 2]) -> i64 {
    let (w, h) = (prev.width as i32, prev.height as i32);
    let mut sad = 0i64;
    for y in (by * BLOCK) as i32..((by + 1) * BLOCK).min(prev.height) as i32 {
        for x in (bx * BLOCK) as i32..((bx + 1) * BLOCK).min(prev.width) as i32 {
            let (cx, cy) = ((x + offset[0]).clamp(0, w - 1), (y + offset[1]).clamp(0, h - 1));
            sad += (quantize(prev.get(x as u32, y as u32)) - quantize(cur.get(cx as u32, cy as u32))).abs() as i64;
        }
    }
    sad
}

/// Component-wise median of each block's 3×3 neighbourhood (cells that exist), so one wrong
/// coarse vector cannot mislead the finer level.
pub fn median_flow(flow: &Flow) -> Flow {
    let mut out = flow.clone();
    for by in 0..flow.height as i32 {
        for bx in 0..flow.width as i32 {
            let (mut xs, mut ys) = (Vec::with_capacity(9), Vec::with_capacity(9));
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (x, y) = (bx + dx, by + dy);
                    if x >= 0 && y >= 0 && x < flow.width as i32 && y < flow.height as i32 {
                        let v = flow.get(x as u32, y as u32);
                        xs.push(v[0]);
                        ys.push(v[1]);
                    }
                }
            }
            xs.sort_by(f32::total_cmp);
            ys.sort_by(f32::total_cmp);
            let mid = (xs.len() - 1) / 2;
            out.set(bx as u32, by as u32, [xs[mid], ys[mid]]);
        }
    }
    out
}

/// Level-0 flow between two pyramids, coarse to fine.
pub fn estimate_flow(prev: &[Image<f32>], cur: &[Image<f32>]) -> Flow {
    let top = prev.len() - 1;
    let mut flow = block_match(&prev[top], &cur[top], None, RADIUS_COARSE);
    for level in (0..top).rev() {
        flow = block_match(&prev[level], &cur[level], Some(&median_flow(&flow)), RADIUS_FINE);
    }
    flow
}

/// Motion candidates per output pixel: own block, left, right, up, down.
pub const CANDIDATES: usize = 5;
/// Quantised luma difference above which the two samples are treated as different content.
pub const DISAGREE: i32 = 96;

/// Full-resolution motion candidates for pixel `(x, y)` (own block, then left, right, up, down;
/// those that exist) and how many there are.
pub fn candidate_vectors(flow: &Flow, x: u32, y: u32) -> ([[f32; 2]; CANDIDATES], usize) {
    let full_block = BLOCK * 2;
    let bx = (x / full_block).min(flow.width - 1) as i32;
    let by = (y / full_block).min(flow.height - 1) as i32;
    let mut out = [[0.0f32; 2]; CANDIDATES];
    let mut n = 0;
    for (dx, dy) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
        let (cx, cy) = (bx + dx, by + dy);
        if cx >= 0 && cy >= 0 && cx < flow.width as i32 && cy < flow.height as i32 {
            let v = flow.get(cx as u32, cy as u32);
            out[n] = [v[0] * 2.0, v[1] * 2.0];
            n += 1;
        }
    }
    (out, n)
}

/// Index and cost of the candidate whose previous and current samples agree best.
pub fn pick(
    prev_luma_at: impl Fn(f32, f32) -> f32,
    cur_luma_at: impl Fn(f32, f32) -> f32,
    p: (f32, f32),
    t: f32,
    cands: &[[f32; 2]],
) -> (usize, i32) {
    let mut best = (0, i32::MAX);
    for (i, v) in cands.iter().enumerate() {
        let a = prev_luma_at(p.0 - t * v[0], p.1 - t * v[1]);
        let b = cur_luma_at(p.0 + (1.0 - t) * v[0], p.1 + (1.0 - t) * v[1]);
        let cost = (quantize(a) - quantize(b)).abs();
        if cost < best.1 {
            best = (i, cost);
        }
    }
    best
}

/// One output pixel at time `t`: picks the best candidate vector, then blends the two samples, or
/// takes the nearer frame's sample when they disagree. Samplers return channels in `0..=scale`.
pub fn interpolate_pixel(
    sample_prev: impl Fn(f32, f32) -> [f32; 4],
    sample_cur: impl Fn(f32, f32) -> [f32; 4],
    scale: f32,
    p: (f32, f32),
    t: f32,
    cands: &[[f32; 2]],
) -> [f32; 4] {
    let luma = |c: [f32; 4]| (LUMA[0] * c[0] + LUMA[1] * c[1] + LUMA[2] * c[2]) / scale;
    let (i, cost) = pick(|x, y| luma(sample_prev(x, y)), |x, y| luma(sample_cur(x, y)), p, t, cands);
    let v = cands[i];
    let a = sample_prev(p.0 - t * v[0], p.1 - t * v[1]);
    let b = sample_cur(p.0 + (1.0 - t) * v[0], p.1 + (1.0 - t) * v[1]);
    if cost > DISAGREE {
        return if t >= 0.5 { b } else { a };
    }
    let mut out = [0.0f32; 4];
    for c in 0..4 {
        out[c] = a[c] * (1.0 - t) + b[c] * t;
    }
    out
}

/// The frame at time `t` between `prev` (t = 0) and `cur` (t = 1), following `flow`.
pub fn warp_blend(prev: &Image<Rgba8>, cur: &Image<Rgba8>, flow: &Flow, t: f32) -> Image<Rgba8> {
    if t == 0.0 {
        return prev.clone();
    }
    if t == 1.0 {
        return cur.clone();
    }
    let mut out = prev.clone();
    for y in 0..prev.height {
        for x in 0..prev.width {
            let (cands, n) = candidate_vectors(flow, x, y);
            let c = interpolate_pixel(
                |sx, sy| bilinear(prev, sx, sy),
                |sx, sy| bilinear(cur, sx, sy),
                255.0,
                (x as f32, y as f32),
                t,
                &cands[..n],
            );
            out.set(x, y, c.map(|v| v.round().clamp(0.0, 255.0) as u8));
        }
    }
    out
}

/// Bilinear sample with coordinates clamped to the image, per channel in 0..=255.
pub fn bilinear(img: &Image<Rgba8>, x: f32, y: f32) -> [f32; 4] {
    let x = x.clamp(0.0, (img.width - 1) as f32);
    let y = y.clamp(0.0, (img.height - 1) as f32);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(img.width - 1), (y0 + 1).min(img.height - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let (p00, p10, p01, p11) = (img.get(x0, y0), img.get(x1, y0), img.get(x0, y1), img.get(x1, y1));
    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
        let bottom = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
        out[c] = top * (1.0 - fy) + bottom * fy;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{noise, shift};

    #[test]
    fn pyramid_odd_sizes() {
        let sizes: Vec<(u32, u32)> = pyramid(&noise(1281, 801, 1)).iter().map(|l| (l.width, l.height)).collect();
        assert_eq!(sizes, [(641, 401), (321, 201), (161, 101), (81, 51)]);
    }

    #[test]
    fn flat_image_gives_zero_flow() {
        let black = Image::new(256, 160, [0u8, 0, 0, 255]);
        let p = pyramid(&black);
        let flow = estimate_flow(&p, &p);
        assert!(flow.data.iter().all(|v| *v == [0.0, 0.0]));
    }

    #[test]
    fn recovers_known_shift() {
        let prev = noise(512, 320, 9);
        let cur = shift(&prev, 12, -8);
        let flow = estimate_flow(&pyramid(&prev), &pyramid(&cur));
        let (mut hits, mut total) = (0, 0);
        for by in 2..flow.height - 2 {
            for bx in 2..flow.width - 2 {
                total += 1;
                if flow.get(bx, by) == [6.0, -4.0] {
                    hits += 1;
                }
            }
        }
        assert!(hits * 100 >= total * 95, "{hits}/{total}");
    }

    #[test]
    fn median_removes_isolated_outlier() {
        let mut f = Image::new(5, 5, [2.0f32, -1.0]);
        f.set(2, 2, [30.0, 30.0]);
        assert!(median_flow(&f).data.iter().all(|v| *v == [2.0, -1.0]));
    }

    #[test]
    fn median_keeps_uniform_field() {
        let f = Image::new(4, 3, [6.0f32, -4.0]);
        assert_eq!(median_flow(&f), f);
    }

    #[test]
    fn median_edges_use_existing_cells() {
        let f = Image { width: 2, height: 1, data: vec![[1.0f32, 5.0], [3.0, 7.0]] };
        let m = median_flow(&f);
        assert_eq!(m.get(0, 0), [1.0, 5.0]);
        assert_eq!(m.get(1, 0), [1.0, 5.0]);
    }

    fn recovery(w: u32, h: u32, seed: u64, dx: i32, dy: i32) -> (u32, u32) {
        let prev = noise(w, h, seed);
        let cur = shift(&prev, dx, dy);
        let flow = estimate_flow(&pyramid(&prev), &pyramid(&cur));
        let want = [dx as f32 / 2.0, dy as f32 / 2.0];
        let (mut hits, mut total) = (0, 0);
        for by in 2..flow.height - 2 {
            for bx in 2..flow.width - 2 {
                total += 1;
                if flow.get(bx, by) == want {
                    hits += 1;
                }
            }
        }
        (hits, total)
    }

    #[test]
    fn recovers_half_pixel_coarse_shifts() {
        for (w, h, seed, dx, dy) in [(1281, 801, 8, -6, 4), (1280, 800, 21, 10, -6), (1280, 800, 6, 12, -8)] {
            let (hits, total) = recovery(w, h, seed, dx, dy);
            assert!(hits * 100 >= total * 95, "{w}x{h} seed {seed} shift ({dx},{dy}): {hits}/{total}");
        }
    }

    #[test]
    fn recovers_known_shift_1280_seed7() {
        let prev = noise(1280, 800, 7);
        let cur = shift(&prev, 12, -8);
        let flow = estimate_flow(&pyramid(&prev), &pyramid(&cur));
        let (mut hits, mut total) = (0, 0);
        for by in 2..flow.height - 2 {
            for bx in 2..flow.width - 2 {
                total += 1;
                if flow.get(bx, by) == [6.0, -4.0] {
                    hits += 1;
                }
            }
        }
        assert!(hits * 100 >= total * 95, "{hits}/{total}");
    }

    #[test]
    fn coarse_outlier_is_corrected_by_neighbours() {
        let prev = noise(256, 256, 5);
        let cur = shift(&prev, 8, 6);
        let (p, c) = (pyramid(&prev), pyramid(&cur));
        let mut coarse = Image::new(p[1].width.div_ceil(BLOCK), p[1].height.div_ceil(BLOCK), [2.0f32, 1.5]);
        coarse.set(3, 3, [-9.0, 9.0]);
        let fine = block_match(&p[0], &c[0], Some(&coarse), RADIUS_FINE);
        for by in 6..8 {
            for bx in 6..8 {
                assert_eq!(fine.get(bx, by), [4.0, 3.0], "block ({bx},{by})");
            }
        }
    }

    #[test]
    fn block_match_edge_blocks_stay_in_bounds() {
        let prev: Image<f32> = Image { width: 24, height: 24, data: (0..576).map(|i| (i % 7) as f32 / 7.0).collect() };
        let coarse = Image::new(2, 2, [10.0f32, 10.0]);
        let flow = block_match(&prev, &prev, Some(&coarse), 2);
        for v in &flow.data {
            assert!(v[0].is_finite() && v[1].is_finite());
            assert!(v[0].abs() <= 22.0 && v[1].abs() <= 22.0, "{v:?}");
        }
        assert!(flow.data.iter().all(|v| *v == [0.0, 0.0]), "identical frames: zero is the best candidate");
    }

    fn old_warp(prev: &Image<Rgba8>, cur: &Image<Rgba8>, flow: &Flow, t: f32) -> Image<Rgba8> {
        let mut out = prev.clone();
        for y in 0..prev.height {
            for x in 0..prev.width {
                let v = flow.get((x / 16).min(flow.width - 1), (y / 16).min(flow.height - 1));
                let (fx, fy) = (v[0] * 2.0, v[1] * 2.0);
                let a = bilinear(prev, x as f32 - t * fx, y as f32 - t * fy);
                let b = bilinear(cur, x as f32 + (1.0 - t) * fx, y as f32 + (1.0 - t) * fy);
                let mut px = [0u8; 4];
                for c in 0..4 {
                    px[c] = (a[c] * (1.0 - t) + b[c] * t).round().clamp(0.0, 255.0) as u8;
                }
                out.set(x, y, px);
            }
        }
        out
    }

    fn paste(dst: &mut Image<Rgba8>, src: &Image<Rgba8>, ox: u32, oy: u32) {
        for y in 0..src.height {
            for x in 0..src.width {
                dst.set(ox + x, oy + y, src.get(x, y));
            }
        }
    }

    fn err(a: Rgba8, b: Rgba8) -> i32 {
        (0..3).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap_or(0)
    }

    #[test]
    fn still_square_over_moving_background() {
        let bg = noise(256, 160, 1);
        let square = crate::image::checker(48, 48, 4);
        let (sx, sy) = (104u32, 56u32);
        let mut prev = bg.clone();
        paste(&mut prev, &square, sx, sy);
        let mut cur = shift(&bg, 16, 0);
        paste(&mut cur, &square, sx, sy);
        let mut flow = Image::new(16, 10, [8.0f32, 0.0]);
        for by in 3..=6 {
            for bx in 6..=9 {
                flow.set(bx, by, [0.0, 0.0]);
            }
        }
        let mut expected = shift(&bg, 8, 0);
        paste(&mut expected, &square, sx, sy);
        let new = warp_blend(&prev, &cur, &flow, 0.5);
        let old = old_warp(&prev, &cur, &flow, 0.5);
        for y in sy + 8..sy + 40 {
            for x in sx + 8..sx + 40 {
                assert_eq!(new.get(x, y), square.get(x - sx, y - sy), "square interior ({x},{y})");
            }
        }
        for y in 20..140 {
            for x in [40u32, 220] {
                assert!(err(new.get(x, y), expected.get(x, y)) <= 1, "background ({x},{y})");
            }
        }
        let band = |img: &Image<Rgba8>| {
            let mut bad = 0;
            for y in sy - 8..sy + 56 {
                for x in sx - 8..sx + 56 {
                    let inside = x >= sx && x < sx + 48 && y >= sy && y < sy + 48;
                    if !inside && err(img.get(x, y), expected.get(x, y)) > 40 {
                        bad += 1;
                    }
                }
            }
            bad
        };
        let (new_bad, old_bad) = (band(&new), band(&old));
        assert!(old_bad > 0, "the scene must break the single-vector warp");
        assert!(new_bad * 2 < old_bad, "edge band: new {new_bad} vs old {old_bad}");
    }

    #[test]
    fn uniform_shift_matches_m3a() {
        let prev = noise(128, 96, 2);
        let cur = shift(&prev, 8, 0);
        let flow = Image::new(8, 6, [4.0f32, 0.0]);
        let (new, old) = (warp_blend(&prev, &cur, &flow, 0.5), old_warp(&prev, &cur, &flow, 0.5));
        for y in 0..96 {
            for x in 16..112 {
                assert_eq!(new.get(x, y), old.get(x, y), "({x},{y})");
            }
        }
    }

    #[test]
    fn revealed_area_uses_one_frame() {
        let prev = Image::new(64, 48, [0u8, 0, 0, 255]);
        let cur = Image::new(64, 48, [255u8, 255, 255, 255]);
        let flow = Image::new(4, 3, [0.0f32, 0.0]);
        assert_eq!(warp_blend(&prev, &cur, &flow, 0.5), cur);
        assert_eq!(warp_blend(&prev, &cur, &flow, 0.4), prev);
    }

    #[test]
    fn border_pixels_use_existing_candidates() {
        let flow = Image::new(16, 10, [1.0f32, 0.0]);
        assert_eq!(candidate_vectors(&flow, 0, 0).1, 3);
        assert_eq!(candidate_vectors(&flow, 255, 159).1, 3);
        assert_eq!(candidate_vectors(&flow, 100, 80).1, 5);
        assert_eq!(candidate_vectors(&flow, 0, 0).0[0], [2.0, 0.0]);
    }

    #[test]
    fn warp_blend_endpoints_are_exact() {
        let prev = noise(64, 48, 1);
        let cur = noise(64, 48, 2);
        let flow = Image::new(4, 3, [5.0f32, 3.0]);
        assert_eq!(warp_blend(&prev, &cur, &flow, 0.0), prev);
        assert_eq!(warp_blend(&prev, &cur, &flow, 1.0), cur);
    }

    #[test]
    fn warp_blend_midpoint_of_pure_shift() {
        let prev = noise(256, 160, 3);
        let cur = shift(&prev, 8, 0);
        let flow = Image::new(16, 10, [4.0f32, 0.0]);
        let mid = warp_blend(&prev, &cur, &flow, 0.5);
        let expected = shift(&prev, 4, 0);
        for y in 8..152 {
            for x in 8..248 {
                let (a, b) = (mid.get(x, y), expected.get(x, y));
                for c in 0..4 {
                    assert!((a[c] as i32 - b[c] as i32).abs() <= 1, "({x},{y}) {a:?} vs {b:?}");
                }
            }
        }
    }
}
