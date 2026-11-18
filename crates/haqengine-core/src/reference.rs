//! CPU versions of HaqEngine's shaders, used as the expected results in GPU tests.

use crate::image::{Image, Rgba8};

/// Rec. 709 luma weights, the same as the shader's.
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Half-resolution luma: each output pixel averages the luma of the existing pixels in its 2×2 block.
pub fn luma_downsample(input: &Image<Rgba8>) -> Image<f32> {
    let (w, h) = (input.width.div_ceil(2), input.height.div_ceil(2));
    let mut out = Image::new(w, h, 0.0f32);
    for oy in 0..h {
        for ox in 0..w {
            let (mut sum, mut count) = (0.0f32, 0.0f32);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (x, y) = (ox * 2 + dx, oy * 2 + dy);
                if x < input.width && y < input.height {
                    let p = input.get(x, y);
                    sum += LUMA[0] * (p[0] as f32 / 255.0) + LUMA[1] * (p[1] as f32 / 255.0) + LUMA[2] * (p[2] as f32 / 255.0);
                    count += 1.0;
                }
            }
            out.set(ox, oy, sum / count);
        }
    }
    out
}

/// `a·(1−t) + b·t` per channel, rounded and clamped.
pub fn blend(a: &Image<Rgba8>, b: &Image<Rgba8>, t: f32) -> Image<Rgba8> {
    let mut out = a.clone();
    for (o, (pa, pb)) in out.data.iter_mut().zip(a.data.iter().zip(&b.data)) {
        for c in 0..4 {
            let v = pa[c] as f32 * (1.0 - t) + pb[c] as f32 * t;
            o[c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// One pixel that differs from the expected value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mismatch {
    pub x: u32,
    pub y: u32,
    pub diff: f32,
}

/// Result of comparing two images.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompareReport {
    pub max_diff: f32,
    pub worst: Option<Mismatch>,
    pub over_tolerance: u32,
}

/// Compares float images; NaN in `actual` always counts as over tolerance.
pub fn compare_f32(expected: &Image<f32>, actual: &Image<f32>, tol: f32) -> CompareReport {
    compare(expected, actual, tol, |e, a| if a.is_nan() { f32::INFINITY } else { (e - a).abs() })
}

/// Compares two-component float images (motion vectors); NaN in `actual` counts as over tolerance.
pub fn compare_rg32f(expected: &Image<[f32; 2]>, actual: &Image<[f32; 2]>, tol: f32) -> CompareReport {
    compare(expected, actual, tol, |e, a| {
        if a[0].is_nan() || a[1].is_nan() {
            f32::INFINITY
        } else {
            (e[0] - a[0]).abs().max((e[1] - a[1]).abs())
        }
    })
}

/// Compares RGBA8 images channel by channel.
pub fn compare_rgba8(expected: &Image<Rgba8>, actual: &Image<Rgba8>, tol_per_channel: u8) -> CompareReport {
    compare(expected, actual, tol_per_channel as f32, |e, a| {
        (0..4).map(|c| (e[c] as f32 - a[c] as f32).abs()).fold(0.0, f32::max)
    })
}

pub(crate) fn compare<T: Copy>(expected: &Image<T>, actual: &Image<T>, tol: f32, diff: impl Fn(T, T) -> f32) -> CompareReport {
    if (expected.width, expected.height) != (actual.width, actual.height) || expected.data.len() != actual.data.len() {
        return CompareReport { max_diff: f32::INFINITY, worst: None, over_tolerance: u32::MAX };
    }
    let mut report = CompareReport { max_diff: 0.0, worst: None, over_tolerance: 0 };
    for (i, (&e, &a)) in expected.data.iter().zip(&actual.data).enumerate() {
        let d = diff(e, a);
        if d > tol {
            report.over_tolerance += 1;
        }
        if d > report.max_diff || (report.worst.is_none() && d > tol) {
            report.max_diff = report.max_diff.max(d);
            let (x, y) = (i as u32 % expected.width, i as u32 / expected.width);
            report.worst = Some(Mismatch { x, y, diff: d });
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE: Rgba8 = [255, 255, 255, 255];
    const BLACK: Rgba8 = [0, 0, 0, 255];

    #[test]
    fn reference_downsample_even() {
        let out = luma_downsample(&Image::new(2, 2, WHITE));
        assert_eq!((out.width, out.height), (1, 1));
        assert!((out.get(0, 0) - 1.0).abs() < 1e-6);
        let mut img = Image::new(4, 2, BLACK);
        for y in 0..2 {
            for x in 2..4 {
                img.set(x, y, WHITE);
            }
        }
        let out = luma_downsample(&img);
        assert_eq!(out.data.len(), 2);
        assert!(out.get(0, 0).abs() < 1e-6 && (out.get(1, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reference_downsample_odd_size_edges() {
        let mut img = Image::new(3, 3, BLACK);
        img.set(2, 2, WHITE);
        let out = luma_downsample(&img);
        assert_eq!((out.width, out.height), (2, 2));
        assert!((out.get(1, 1) - 1.0).abs() < 1e-6);
        assert!(out.get(0, 0).abs() < 1e-6 && out.get(1, 0).abs() < 1e-6 && out.get(0, 1).abs() < 1e-6);
    }

    #[test]
    fn reference_blend_midpoint() {
        let a = Image::new(2, 1, [0, 0, 0, 0]);
        let b = Image::new(2, 1, [255, 255, 255, 255]);
        assert_eq!(blend(&a, &b, 0.5).get(0, 0), [128, 128, 128, 128]);
        assert_eq!(blend(&a, &b, 0.0), a);
        assert_eq!(blend(&a, &b, 1.0), b);
    }

    #[test]
    fn compare_detects_canary() {
        let expected = Image::new(4, 2, 0.5f32);
        let mut actual = expected.clone();
        actual.set(3, 1, f32::NAN);
        let r = compare_f32(&expected, &actual, 1e-5);
        assert_eq!(r.over_tolerance, 1);
        let w = r.worst.unwrap();
        assert_eq!((w.x, w.y), (3, 1));
    }

    #[test]
    fn compare_rg32f_exact_and_nan() {
        let expected = Image::new(3, 2, [6.0f32, -4.0]);
        assert_eq!(compare_rg32f(&expected, &expected, 0.0).over_tolerance, 0);
        let mut actual = expected.clone();
        actual.set(1, 1, [6.0, f32::NAN]);
        actual.set(2, 0, [5.0, -4.0]);
        let r = compare_rg32f(&expected, &actual, 0.0);
        assert_eq!(r.over_tolerance, 2);
        assert_eq!(r.worst.map(|w| (w.x, w.y)), Some((1, 1)));
    }

    #[test]
    fn compare_size_mismatch_fails() {
        assert_eq!(compare_f32(&Image::new(2, 2, 0.0), &Image::new(2, 1, 0.0), 1.0).over_tolerance, u32::MAX);
        assert_eq!(compare_rgba8(&Image::new(2, 2, BLACK), &Image::new(1, 2, BLACK), 1).over_tolerance, u32::MAX);
    }

    #[test]
    fn compare_rgba8_tolerance() {
        let a = Image::new(2, 1, [10, 10, 10, 10]);
        let mut b = a.clone();
        b.set(1, 0, [11, 10, 10, 10]);
        assert_eq!(compare_rgba8(&a, &b, 1).over_tolerance, 0);
        assert_eq!(compare_rgba8(&a, &b, 0).over_tolerance, 1);
    }
}
