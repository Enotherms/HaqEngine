//! The GPU test cases: each runs one shader on a test image and compares with the CPU reference.

use crate::gpu::{push_bytes, Gpu};
use ash::vk;
use haqengine_core::{blend, checker, compare_f32, compare_rgba8, gradient, luma_downsample, noise, CompareReport, Image, Rgba8};
use haqengine_shaders::{BLEND, LUMA_DOWNSAMPLE};

pub const LUMA_TOLERANCE: f32 = 1e-5;
pub const BLEND_TOLERANCE: u8 = 1;
const CANARY_FLOAT: [f32; 4] = [f32::NAN, 0.0, 0.0, 0.0];
const CANARY_RGBA8: [f32; 4] = [0xDE as f32 / 255.0; 4];

pub struct CaseResult {
    pub report: CompareReport,
    pub gpu_us: f64,
    pub tolerance: f32,
}

pub type CaseFn = Box<dyn Fn(&Gpu) -> Result<CaseResult, String>>;

pub struct Case {
    pub name: String,
    pub run: CaseFn,
}

fn groups(w: u32, h: u32) -> [u32; 2] {
    [w.div_ceil(8), h.div_ceil(8)]
}

fn luma_case(gpu: &Gpu, input: &Image<Rgba8>) -> Result<CaseResult, String> {
    let src = gpu.upload_rgba8(input)?;
    let (ow, oh) = (input.width.div_ceil(2), input.height.div_ceil(2));
    let dst = gpu.create_storage_image(vk::Format::R32_SFLOAT, ow, oh)?;
    let result = (|| {
        gpu.clear(&dst, CANARY_FLOAT)?;
        let push = push_bytes(&[input.width.to_le_bytes(), input.height.to_le_bytes()]);
        let gpu_us = gpu.run(&LUMA_DOWNSAMPLE, &[&src, &dst], &push, groups(ow, oh))?;
        let actual = gpu.read_r32f(&dst)?;
        let report = compare_f32(&luma_downsample(input), &actual, LUMA_TOLERANCE);
        Ok(CaseResult { report, gpu_us, tolerance: LUMA_TOLERANCE })
    })();
    gpu.destroy_image(src);
    gpu.destroy_image(dst);
    result
}

fn blend_case(gpu: &Gpu, a: &Image<Rgba8>, b: &Image<Rgba8>, t: f32, reference_t: f32) -> Result<CaseResult, String> {
    let ga = gpu.upload_rgba8(a)?;
    let gb = gpu.upload_rgba8(b)?;
    let dst = gpu.create_storage_image(vk::Format::R8G8B8A8_UNORM, a.width, a.height)?;
    let result = (|| {
        gpu.clear(&dst, CANARY_RGBA8)?;
        let push = push_bytes(&[a.width.to_le_bytes(), a.height.to_le_bytes(), t.to_le_bytes()]);
        let gpu_us = gpu.run(&BLEND, &[&ga, &gb, &dst], &push, groups(a.width, a.height))?;
        let actual = gpu.read_rgba8(&dst)?;
        let report = compare_rgba8(&blend(a, b, reference_t), &actual, BLEND_TOLERANCE);
        Ok(CaseResult { report, gpu_us, tolerance: BLEND_TOLERANCE as f32 })
    })();
    gpu.destroy_image(ga);
    gpu.destroy_image(gb);
    gpu.destroy_image(dst);
    result
}

fn case(name: &str, run: impl Fn(&Gpu) -> Result<CaseResult, String> + 'static) -> Case {
    Case { name: name.to_string(), run: Box::new(run) }
}

/// Every case; `self_test_fail` adds one that must fail, proving failures are reported.
pub fn all_cases(self_test_fail: bool) -> Vec<Case> {
    let mut cases = vec![
        case("luma_downsample 1280x800 noise", |g| luma_case(g, &noise(1280, 800, 1))),
        case("luma_downsample 1281x801 gradient", |g| luma_case(g, &gradient(1281, 801))),
        case("luma_downsample 8x8 checker", |g| luma_case(g, &checker(8, 8, 1))),
        case("blend 1280x800 t=0.5 (noise, gradient)", |g| {
            blend_case(g, &noise(1280, 800, 2), &gradient(1280, 800), 0.5, 0.5)
        }),
        case("blend 1280x800 t=0.25", |g| blend_case(g, &noise(1280, 800, 3), &gradient(1280, 800), 0.25, 0.25)),
    ];
    if self_test_fail {
        cases.push(case("blend self-test (must fail)", |g| {
            blend_case(g, &noise(64, 64, 4), &gradient(64, 64), 0.5, 0.6)
        }));
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_are_named_and_unique() {
        let cases = all_cases(false);
        let mut names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
        assert!(names.iter().all(|n| !n.is_empty()));
        assert!(names.iter().any(|n| n.starts_with("luma_downsample")));
        assert!(names.iter().any(|n| n.starts_with("blend")));
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total);
        assert!(!names.iter().any(|n| n.contains("must fail")));
        assert!(all_cases(true).iter().any(|c| c.name.contains("must fail")));
    }
}
