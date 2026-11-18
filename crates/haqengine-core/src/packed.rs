//! Pixels stored as one 32-bit word, as in swapchain images, and the battery passes on them.

use crate::flow::Flow;
use crate::image::{Image, Rgba8};
use crate::reference::{CompareReport, LUMA};

/// The 32-bit layouts HaqEngine reads and writes; the code is passed to the shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PackedFormat {
    Rgba8 = 0,
    Bgra8 = 1,
    A2b10g10r10 = 2,
}

impl PackedFormat {
    pub fn code(self) -> u32 {
        self as u32
    }

    /// Largest value of a colour channel.
    pub fn max_rgb(self) -> f32 {
        match self {
            PackedFormat::A2b10g10r10 => 1023.0,
            _ => 255.0,
        }
    }
}

/// Channels in 0..=1, in R, G, B, A order.
pub fn unpack(f: PackedFormat, word: u32) -> [f32; 4] {
    let byte = |i: u32| ((word >> (8 * i)) & 0xff) as f32 / 255.0;
    match f {
        PackedFormat::Rgba8 => [byte(0), byte(1), byte(2), byte(3)],
        PackedFormat::Bgra8 => [byte(2), byte(1), byte(0), byte(3)],
        PackedFormat::A2b10g10r10 => {
            let ten = |shift: u32| ((word >> shift) & 0x3ff) as f32 / 1023.0;
            [ten(0), ten(10), ten(20), (word >> 30) as f32 / 3.0]
        }
    }
}

/// Packs R, G, B, A in 0..=1, rounding each channel to the nearest step.
pub fn pack(f: PackedFormat, rgba: [f32; 4]) -> u32 {
    let q = |c: f32, max: f32| (c.clamp(0.0, 1.0) * max + 0.5).floor() as u32;
    let [r, g, b, a] = rgba;
    match f {
        PackedFormat::Rgba8 => q(r, 255.0) | (q(g, 255.0) << 8) | (q(b, 255.0) << 16) | (q(a, 255.0) << 24),
        PackedFormat::Bgra8 => q(b, 255.0) | (q(g, 255.0) << 8) | (q(r, 255.0) << 16) | (q(a, 255.0) << 24),
        PackedFormat::A2b10g10r10 => q(r, 1023.0) | (q(g, 1023.0) << 10) | (q(b, 1023.0) << 20) | (q(a, 3.0) << 30),
    }
}

/// Compares packed images channel by channel, in steps of the format (1/255 or 1/1023).
pub fn compare_packed(f: PackedFormat, expected: &Image<u32>, actual: &Image<u32>, tol_steps: f32) -> CompareReport {
    crate::reference::compare(expected, actual, tol_steps, |e, a| {
        let (e, a) = (unpack(f, e), unpack(f, a));
        (0..4).map(|c| (e[c] - a[c]).abs() * f.max_rgb()).fold(0.0, f32::max)
    })
}

/// `img` stored in format `f`.
pub fn to_packed(f: PackedFormat, img: &Image<Rgba8>) -> Image<u32> {
    let data = img
        .data
        .iter()
        .map(|p| pack(f, [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0, p[3] as f32 / 255.0]))
        .collect();
    Image { width: img.width, height: img.height, data }
}

/// `luma_downsample` on packed pixels.
pub fn luma_downsample_packed(f: PackedFormat, input: &Image<u32>) -> Image<f32> {
    let (w, h) = (input.width.div_ceil(2), input.height.div_ceil(2));
    let mut out = Image::new(w, h, 0.0f32);
    for oy in 0..h {
        for ox in 0..w {
            let (mut sum, mut count) = (0.0f32, 0.0f32);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (x, y) = (ox * 2 + dx, oy * 2 + dy);
                if x < input.width && y < input.height {
                    let p = unpack(f, input.get(x, y));
                    sum += LUMA[0] * p[0] + LUMA[1] * p[1] + LUMA[2] * p[2];
                    count += 1.0;
                }
            }
            out.set(ox, oy, sum / count);
        }
    }
    out
}

/// `flow::warp_blend` on packed pixels.
pub fn warp_blend_packed(f: PackedFormat, prev: &Image<u32>, cur: &Image<u32>, flow: &Flow, t: f32) -> Image<u32> {
    if t == 0.0 {
        return prev.clone();
    }
    if t == 1.0 {
        return cur.clone();
    }
    let mut out = prev.clone();
    for y in 0..prev.height {
        for x in 0..prev.width {
            let (cands, n) = crate::flow::candidate_vectors(flow, x, y);
            let c = crate::flow::interpolate_pixel(
                |sx, sy| bilinear_packed(f, prev, sx, sy),
                |sx, sy| bilinear_packed(f, cur, sx, sy),
                1.0,
                (x as f32, y as f32),
                t,
                &cands[..n],
            );
            out.set(x, y, pack(f, c));
        }
    }
    out
}

/// Bilinear sample of unpacked channels with coordinates clamped to the image.
pub fn bilinear_packed(f: PackedFormat, img: &Image<u32>, x: f32, y: f32) -> [f32; 4] {
    let x = x.clamp(0.0, (img.width - 1) as f32);
    let y = y.clamp(0.0, (img.height - 1) as f32);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(img.width - 1), (y0 + 1).min(img.height - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let (p00, p10) = (unpack(f, img.get(x0, y0)), unpack(f, img.get(x1, y0)));
    let (p01, p11) = (unpack(f, img.get(x0, y1)), unpack(f, img.get(x1, y1)));
    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let top = p00[c] * (1.0 - fx) + p10[c] * fx;
        let bottom = p01[c] * (1.0 - fx) + p11[c] * fx;
        out[c] = top * (1.0 - fy) + bottom * fy;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::noise;
    use crate::reference::luma_downsample;

    #[test]
    fn pack_roundtrip_all_formats() {
        for f in [PackedFormat::Rgba8, PackedFormat::Bgra8] {
            for v in 0..=255u32 {
                let c = v as f32 / 255.0;
                let a = (255 - v) as f32 / 255.0;
                let word = pack(f, [c, 1.0 - c, c, a]);
                let back = unpack(f, word);
                assert_eq!(pack(f, back), word, "{f:?} {v}");
            }
        }
        let f = PackedFormat::A2b10g10r10;
        for v in 0..=1023u32 {
            let c = v as f32 / 1023.0;
            let word = pack(f, [c, 1.0 - c, c, (v % 4) as f32 / 3.0]);
            assert_eq!(pack(f, unpack(f, word)), word, "{v}");
            assert_eq!(word & 0x3ff, v);
        }
    }

    #[test]
    fn bgra_swaps_red_and_blue() {
        assert_eq!(unpack(PackedFormat::Bgra8, 0xFF00_00FF), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(unpack(PackedFormat::Rgba8, 0xFF00_00FF), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn a2_layout() {
        let word = (3u32 << 30) | (1023u32 << 20) | (512u32 << 10) | 1;
        let px = unpack(PackedFormat::A2b10g10r10, word);
        assert_eq!(px[3], 1.0);
        assert_eq!(px[2], 1.0);
        assert!((px[1] - 512.0 / 1023.0).abs() < 1e-7);
        assert!((px[0] - 1.0 / 1023.0).abs() < 1e-7);
    }

    #[test]
    fn luma_packed_matches_rgba_path() {
        let img = noise(65, 33, 4);
        let expected = luma_downsample(&img);
        for f in [PackedFormat::Rgba8, PackedFormat::Bgra8] {
            let got = luma_downsample_packed(f, &to_packed(f, &img));
            for (a, b) in expected.data.iter().zip(&got.data) {
                assert!((a - b).abs() < 1e-6, "{f:?}");
            }
        }
    }

    #[test]
    fn compare_packed_counts_steps() {
        let f = PackedFormat::A2b10g10r10;
        let a = Image::new(2, 1, pack(f, [0.5, 0.5, 0.5, 1.0]));
        let mut b = a.clone();
        b.set(1, 0, a.get(1, 0) + 2);
        assert_eq!(compare_packed(f, &a, &b, 1.0).over_tolerance, 1);
        assert_eq!(compare_packed(f, &a, &b, 2.0).over_tolerance, 0);
    }

    #[test]
    fn warp_packed_endpoints_exact() {
        let f = PackedFormat::A2b10g10r10;
        let prev = to_packed(f, &noise(64, 48, 1));
        let cur = to_packed(f, &noise(64, 48, 2));
        let flow = Image::new(4, 3, [5.0f32, 3.0]);
        assert_eq!(warp_blend_packed(f, &prev, &cur, &flow, 0.0), prev);
        assert_eq!(warp_blend_packed(f, &prev, &cur, &flow, 1.0), cur);
    }

    #[test]
    fn warp_packed_matches_rgba8_within_one() {
        let prev = noise(96, 64, 5);
        let cur = noise(96, 64, 6);
        let flow = Image::new(6, 4, [3.0f32, -2.0]);
        let expected = crate::flow::warp_blend(&prev, &cur, &flow, 0.5);
        let f = PackedFormat::Rgba8;
        let got = warp_blend_packed(f, &to_packed(f, &prev), &to_packed(f, &cur), &flow, 0.5);
        for (e, g) in expected.data.iter().zip(&got.data) {
            let g = unpack(f, *g);
            for c in 0..4 {
                assert!((e[c] as f32 - g[c] * 255.0).abs() <= 1.01, "{e:?} vs {g:?}");
            }
        }
    }
}
