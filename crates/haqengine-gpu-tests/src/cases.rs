//! The GPU test cases: each runs one shader on a test image and compares with the CPU reference.

use crate::gpu::{push_bytes, Gpu, StorageImage};
use ash::vk;
use haqengine_core::flow::{self, Flow, BLOCK, LEVELS, RADIUS_COARSE, RADIUS_FINE};
use haqengine_core::{
    blend, checker, compare_f32, compare_rg32f, compare_rgba8, gradient, luma_downsample, noise, shift, CompareReport,
    Image, Rgba8,
};
use haqengine_shaders::{BLEND, BLOCK_MATCH, FLOW_MEDIAN, LUMA_DOWNSAMPLE, LUMA_REDUCE, WARP_BLEND};

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

fn block_groups(blocks_w: u32, blocks_h: u32) -> [u32; 2] {
    [blocks_w, blocks_h]
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

fn reduce_case(gpu: &Gpu, input: &Image<f32>) -> Result<CaseResult, String> {
    let src = gpu.upload_r32f(input)?;
    let (ow, oh) = (input.width.div_ceil(2), input.height.div_ceil(2));
    let dst = gpu.create_storage_image(vk::Format::R32_SFLOAT, ow, oh)?;
    let result = (|| {
        gpu.clear(&dst, CANARY_FLOAT)?;
        let push = push_bytes(&[input.width.to_le_bytes(), input.height.to_le_bytes()]);
        let gpu_us = gpu.run(&LUMA_REDUCE, &[&src, &dst], &push, groups(ow, oh))?;
        let report = compare_f32(&flow::luma_reduce(input), &gpu.read_r32f(&dst)?, LUMA_TOLERANCE);
        Ok(CaseResult { report, gpu_us, tolerance: LUMA_TOLERANCE })
    })();
    gpu.destroy_image(src);
    gpu.destroy_image(dst);
    result
}

/// GPU images for one frame and its luma pyramid, freed together.
struct Frame {
    color: StorageImage,
    levels: Vec<StorageImage>,
}

impl Frame {
    fn destroy(self, gpu: &Gpu) {
        gpu.destroy_image(self.color);
        for l in self.levels {
            gpu.destroy_image(l);
        }
    }
}

fn gpu_frame(gpu: &Gpu, frame: &Image<Rgba8>, time: &mut f64) -> Result<Frame, String> {
    let color = gpu.upload_rgba8(frame)?;
    let mut levels: Vec<StorageImage> = Vec::new();
    let (mut w, mut h) = (frame.width, frame.height);
    for level in 0..LEVELS {
        let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
        let dst = gpu.create_storage_image(vk::Format::R32_SFLOAT, ow, oh)?;
        gpu.clear(&dst, CANARY_FLOAT)?;
        let push = push_bytes(&[w.to_le_bytes(), h.to_le_bytes()]);
        let (shader, src) = if level == 0 { (&LUMA_DOWNSAMPLE, &color) } else { (&LUMA_REDUCE, &levels[level - 1]) };
        *time += gpu.run(shader, &[src, &dst], &push, groups(ow, oh))?;
        levels.push(dst);
        (w, h) = (ow, oh);
    }
    Ok(Frame { color, levels })
}

/// GPU flow for every level (index = level), each checked exactly against the CPU fed with the
/// GPU's own luma and coarser flow.
struct Flows {
    images: Vec<StorageImage>,
    checks: Vec<CompareReport>,
}

fn gpu_flows(gpu: &Gpu, prev: &Frame, cur: &Frame, time: &mut f64) -> Result<Flows, String> {
    let dummy = gpu.create_storage_image(vk::Format::R32G32_SFLOAT, 1, 1)?;
    let mut by_level: Vec<Option<StorageImage>> = (0..LEVELS).map(|_| None).collect();
    let mut medians: Vec<StorageImage> = Vec::new();
    let mut checks = vec![CompareReport { max_diff: 0.0, worst: None, over_tolerance: 0 }; LEVELS];
    let result = (|| {
        let mut coarse: Option<usize> = None;
        for level in (0..LEVELS).rev() {
            let (pl, cl) = (&prev.levels[level], &cur.levels[level]);
            let (bw, bh) = (pl.width.div_ceil(BLOCK), pl.height.div_ceil(BLOCK));
            let out = gpu.create_storage_image(vk::Format::R32G32_SFLOAT, bw, bh)?;
            gpu.clear(&out, CANARY_FLOAT)?;
            let coarse_img = coarse.map(|i| &medians[i]);
            let radius = if coarse_img.is_some() { RADIUS_FINE } else { RADIUS_COARSE };
            let (cw, ch) = coarse_img.map_or((1, 1), |c| (c.width, c.height));
            let push = push_bytes(&[
                pl.width.to_le_bytes(),
                pl.height.to_le_bytes(),
                cw.to_le_bytes(),
                ch.to_le_bytes(),
                radius.to_le_bytes(),
                (coarse_img.is_some() as u32).to_le_bytes(),
            ]);
            *time += gpu.run(&BLOCK_MATCH, &[pl, cl, coarse_img.unwrap_or(&dummy), &out], &push, block_groups(bw, bh))?;
            let cpu_coarse: Option<Flow> = coarse_img.map(|c| gpu.read_rg32f(c)).transpose()?;
            let gpu_flow = gpu.read_rg32f(&out)?;
            let expected = flow::block_match(&gpu.read_r32f(pl)?, &gpu.read_r32f(cl)?, cpu_coarse.as_ref(), radius);
            let mut report = compare_rg32f(&expected, &gpu_flow, 0.0);
            if level > 0 {
                let median = gpu.create_storage_image(vk::Format::R32G32_SFLOAT, bw, bh)?;
                gpu.clear(&median, CANARY_FLOAT)?;
                let push = push_bytes(&[bw.to_le_bytes(), bh.to_le_bytes()]);
                *time += gpu.run(&FLOW_MEDIAN, &[&out, &median], &push, groups(bw, bh))?;
                let median_check = compare_rg32f(&flow::median_flow(&gpu_flow), &gpu.read_rg32f(&median)?, 0.0);
                report = merge(&[report, median_check]);
                medians.push(median);
                coarse = Some(medians.len() - 1);
            }
            checks[level] = report;
            by_level[level] = Some(out);
        }
        Ok(())
    })();
    gpu.destroy_image(dummy);
    for m in medians {
        gpu.destroy_image(m);
    }
    let images: Vec<StorageImage> = by_level.into_iter().flatten().collect();
    if let Err(e) = result {
        for img in images {
            gpu.destroy_image(img);
        }
        return Err(e);
    }
    Ok(Flows { images, checks })
}

fn merge(reports: &[CompareReport]) -> CompareReport {
    let mut out = CompareReport { max_diff: 0.0, worst: None, over_tolerance: 0 };
    for r in reports {
        out.over_tolerance = out.over_tolerance.saturating_add(r.over_tolerance);
        if r.max_diff >= out.max_diff {
            out.max_diff = r.max_diff;
            if r.worst.is_some() {
                out.worst = r.worst;
            }
        }
    }
    out
}

fn interior_hits(flow: &Flow, want: [f32; 2]) -> (u32, u32) {
    let (mut hits, mut total) = (0, 0);
    for by in 2..flow.height.saturating_sub(2) {
        for bx in 2..flow.width.saturating_sub(2) {
            total += 1;
            if flow.get(bx, by) == want {
                hits += 1;
            }
        }
    }
    (hits, total)
}

/// Runs the pyramid and flow for a frame pair and returns the per-level checks and level-0 flow.
fn flow_case(
    gpu: &Gpu,
    prev: &Image<Rgba8>,
    cur: &Image<Rgba8>,
    levels: &[usize],
    expect: Option<[f32; 2]>,
    finite_only: bool,
) -> Result<CaseResult, String> {
    let mut time = 0.0;
    let fp = gpu_frame(gpu, prev, &mut time)?;
    let fc = match gpu_frame(gpu, cur, &mut time) {
        Ok(f) => f,
        Err(e) => {
            fp.destroy(gpu);
            return Err(e);
        }
    };
    let result = (|| {
        let flows = gpu_flows(gpu, &fp, &fc, &mut time)?;
        let level0 = gpu.read_rg32f(&flows.images[0]);
        for img in flows.images {
            gpu.destroy_image(img);
        }
        let level0 = level0?;
        if finite_only {
            let bad = level0.data.iter().filter(|v| !v[0].is_finite() || !v[1].is_finite()).count() as u32;
            let report = CompareReport { max_diff: 0.0, worst: None, over_tolerance: bad };
            return Ok(CaseResult { report, gpu_us: time, tolerance: 0.0 });
        }
        let picked: Vec<CompareReport> = levels.iter().map(|&l| flows.checks[l]).collect();
        let report = merge(&picked);
        if let Some(want) = expect {
            let (hits, total) = interior_hits(&level0, want);
            if hits * 100 < total * 95 {
                return Err(format!("motion recovered in {hits}/{total} interior blocks, need 95%"));
            }
        }
        Ok(CaseResult { report, gpu_us: time, tolerance: 0.0 })
    })();
    fp.destroy(gpu);
    fc.destroy(gpu);
    result
}

fn warp_case(gpu: &Gpu, prev: &Image<Rgba8>, cur: &Image<Rgba8>, flow_cpu: &Flow, ts: &[f32], tol: u8) -> Result<CaseResult, String> {
    let gp = gpu.upload_rgba8(prev)?;
    let gc = gpu.upload_rgba8(cur)?;
    let gf = gpu.upload_rg32f(flow_cpu)?;
    let dst = gpu.create_storage_image(vk::Format::R8G8B8A8_UNORM, prev.width, prev.height)?;
    let result = (|| {
        let mut reports = Vec::new();
        let mut time = 0.0;
        for &t in ts {
            gpu.clear(&dst, CANARY_RGBA8)?;
            let push = push_bytes(&[
                prev.width.to_le_bytes(),
                prev.height.to_le_bytes(),
                flow_cpu.width.to_le_bytes(),
                flow_cpu.height.to_le_bytes(),
                t.to_le_bytes(),
            ]);
            time += gpu.run(&WARP_BLEND, &[&gp, &gc, &gf, &dst], &push, groups(prev.width, prev.height))?;
            reports.push(compare_rgba8(&flow::warp_blend(prev, cur, flow_cpu, t), &gpu.read_rgba8(&dst)?, tol));
        }
        Ok(CaseResult { report: merge(&reports), gpu_us: time, tolerance: tol as f32 })
    })();
    for img in [gp, gc, gf, dst] {
        gpu.destroy_image(img);
    }
    result
}

fn battery_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let prev = noise(1280, 800, 21);
    let cur = shift(&prev, 10, -6);
    let mut setup_time = 0.0;
    let fp = gpu_frame(gpu, &prev, &mut setup_time)?;
    let mut time = 0.0;
    let fc = match gpu_frame(gpu, &cur, &mut time) {
        Ok(f) => f,
        Err(e) => {
            fp.destroy(gpu);
            return Err(e);
        }
    };
    let result = (|| {
        let flows = gpu_flows(gpu, &fp, &fc, &mut time)?;
        let level0 = gpu.read_rg32f(&flows.images[0]);
        let mut reports = flows.checks.clone();
        let dst = gpu.create_storage_image(vk::Format::R8G8B8A8_UNORM, prev.width, prev.height)?;
        let warp = (|| {
            gpu.clear(&dst, CANARY_RGBA8)?;
            let push = push_bytes(&[
                prev.width.to_le_bytes(),
                prev.height.to_le_bytes(),
                flows.images[0].width.to_le_bytes(),
                flows.images[0].height.to_le_bytes(),
                0.5f32.to_le_bytes(),
            ]);
            time += gpu.run(&WARP_BLEND, &[&fp.color, &fc.color, &flows.images[0], &dst], &push, groups(prev.width, prev.height))?;
            gpu.read_rgba8(&dst)
        })();
        gpu.destroy_image(dst);
        for img in flows.images {
            gpu.destroy_image(img);
        }
        let level0 = level0?;
        reports.push(compare_rgba8(&flow::warp_blend(&prev, &cur, &level0, 0.5), &warp?, BLEND_TOLERANCE));
        println!("      battery pipeline total {time:.1} us (budget 2000 us)");
        Ok(CaseResult { report: merge(&reports), gpu_us: time, tolerance: BLEND_TOLERANCE as f32 })
    })();
    fp.destroy(gpu);
    fc.destroy(gpu);
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
    cases.extend([
        case("luma_reduce 640x400 noise", |g| reduce_case(g, &luma_downsample(&noise(1280, 800, 5)))),
        case("luma_reduce 641x401", |g| reduce_case(g, &luma_downsample(&gradient(1281, 801)))),
        case("block_match level3 coarse", |g| {
            let prev = noise(1280, 800, 6);
            flow_case(g, &prev, &shift(&prev, 12, -8), &[3], None, false)
        }),
        case("block_match level0 with coarse", |g| {
            let prev = noise(1280, 800, 6);
            flow_case(g, &prev, &shift(&prev, 12, -8), &[0], None, false)
        }),
        case("flow pipeline 1280x800 shift(12,-8)", |g| {
            let prev = noise(1280, 800, 7);
            flow_case(g, &prev, &shift(&prev, 12, -8), &[0, 1, 2, 3], Some([6.0, -4.0]), false)
        }),
        case("flow pipeline 1281x801 shift(-6,4)", |g| {
            let prev = noise(1281, 801, 8);
            flow_case(g, &prev, &shift(&prev, -6, 4), &[0, 1, 2, 3], None, false)
        }),
        case("flow flat black", |g| {
            let black = Image::new(640, 400, [0u8, 0, 0, 255]);
            flow_case(g, &black, &black, &[0, 1, 2, 3], Some([0.0, 0.0]), false)
        }),
        case("flow large shift 96px", |g| {
            let prev = noise(1280, 800, 10);
            flow_case(g, &prev, &shift(&prev, 96, 0), &[], None, true)
        }),
        case("warp_blend 1280x800 t=0.5 shift(8,0)", |g| {
            let prev = noise(1280, 800, 11);
            let flow = Image::new(80, 50, [4.0f32, 0.0]);
            warp_case(g, &prev, &shift(&prev, 8, 0), &flow, &[0.5], BLEND_TOLERANCE)
        }),
        case("warp_blend endpoints t=0 and t=1", |g| {
            let flow = Image::new(80, 50, [5.0f32, 3.0]);
            warp_case(g, &noise(1280, 800, 12), &noise(1280, 800, 13), &flow, &[0.0, 1.0], 0)
        }),
        case("battery pipeline 1280x800 total", battery_case),
    ]);
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
        for required in [
            "luma_reduce 640x400 noise",
            "luma_reduce 641x401",
            "block_match level3 coarse",
            "block_match level0 with coarse",
            "flow pipeline 1280x800 shift(12,-8)",
            "flow pipeline 1281x801 shift(-6,4)",
            "flow flat black",
            "flow large shift 96px",
            "warp_blend 1280x800 t=0.5 shift(8,0)",
            "warp_blend endpoints t=0 and t=1",
            "battery pipeline 1280x800 total",
        ] {
            assert!(names.contains(&required), "missing case {required}");
        }
        assert!(all_cases(true).iter().any(|c| c.name.contains("must fail")));
    }
}
