//! The GPU test cases: each runs one shader on a test image and compares with the CPU reference.

use crate::fsr3::run_sequence;
use crate::gpu::{push_bytes, Bind, Gpu, StorageImage};
use ash::vk;
use haqengine_core::flow::{self, level_radius, Flow, BLOCK, LEVELS};
use haqengine_core::{
    blend, checker, compare_f32, compare_rg32f, compare_rgba8, gradient, luma_downsample, noise, shift, CompareReport,
    Image, Rgba8,
};
use haqengine_core::packed::{self, PackedFormat};
use haqengine_shaders::{
    BLEND, BLOCK_MATCH, FLOW_MEDIAN, LUMA_DOWNSAMPLE, LUMA_DOWNSAMPLE_PACKED, LUMA_REDUCE, WARP_BLEND, WARP_BLEND_PACKED,
};

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

const CANARY_U32: u32 = 0xDEAD_BEEF;

fn gpu_frame(gpu: &Gpu, frame: &Image<Rgba8>, time: &mut f64) -> Result<Frame, String> {
    let color = gpu.upload_rgba8(frame)?;
    build_pyramid(gpu, color, None, time)
}

#[allow(clippy::too_many_arguments)]
fn run_warp_packed(
    gpu: &Gpu,
    f: PackedFormat,
    prev: &StorageImage,
    cur: &StorageImage,
    flow_img: &StorageImage,
    prev_luma: &StorageImage,
    cur_luma: &StorageImage,
    t: f32,
) -> Result<(Image<u32>, f64), String> {
    run_warp_packed_with(gpu, &WARP_BLEND_PACKED, f, prev, cur, flow_img, prev_luma, cur_luma, t)
}

fn gpu_frame_packed(gpu: &Gpu, f: PackedFormat, frame: &Image<u32>, time: &mut f64) -> Result<Frame, String> {
    let color = gpu.upload_u32(frame)?;
    build_pyramid(gpu, color, Some(f), time)
}

fn build_pyramid(gpu: &Gpu, color: StorageImage, packed: Option<PackedFormat>, time: &mut f64) -> Result<Frame, String> {
    let (frame_w, frame_h) = (color.width, color.height);
    let mut levels: Vec<StorageImage> = Vec::new();
    let (mut w, mut h) = (frame_w, frame_h);
    let result = (|| {
        for level in 0..LEVELS {
            let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
            let dst = gpu.create_storage_image(vk::Format::R32_SFLOAT, ow, oh)?;
            gpu.clear(&dst, CANARY_FLOAT)?;
            let mut words = vec![w.to_le_bytes(), h.to_le_bytes()];
            let (shader, src) = match (level, packed) {
                (0, Some(f)) => {
                    words.push(f.code().to_le_bytes());
                    (&LUMA_DOWNSAMPLE_PACKED, &color)
                }
                (0, None) => (&LUMA_DOWNSAMPLE, &color),
                _ => (&LUMA_REDUCE, &levels[level - 1]),
            };
            let t = gpu.run(shader, &[src, &dst], &push_bytes(&words), groups(ow, oh));
            levels.push(dst);
            *time += t?;
            (w, h) = (ow, oh);
        }
        Ok(())
    })();
    let frame = Frame { color, levels };
    match result {
        Ok(()) => Ok(frame),
        Err(e) => {
            frame.destroy(gpu);
            Err(e)
        }
    }
}

fn luma_packed_case(gpu: &Gpu, f: PackedFormat, input: &Image<u32>) -> Result<CaseResult, String> {
    let src = gpu.upload_u32(input)?;
    let (ow, oh) = (input.width.div_ceil(2), input.height.div_ceil(2));
    let dst = gpu.create_storage_image(vk::Format::R32_SFLOAT, ow, oh)?;
    let result = (|| {
        gpu.clear(&dst, CANARY_FLOAT)?;
        let push = push_bytes(&[input.width.to_le_bytes(), input.height.to_le_bytes(), f.code().to_le_bytes()]);
        let gpu_us = gpu.run(&LUMA_DOWNSAMPLE_PACKED, &[&src, &dst], &push, groups(ow, oh))?;
        let report = compare_f32(&packed::luma_downsample_packed(f, input), &gpu.read_r32f(&dst)?, LUMA_TOLERANCE);
        Ok(CaseResult { report, gpu_us, tolerance: LUMA_TOLERANCE })
    })();
    gpu.destroy_image(src);
    gpu.destroy_image(dst);
    result
}

#[allow(clippy::too_many_arguments)]
fn run_warp_packed_with(
    gpu: &Gpu,
    shader: &haqengine_shaders::Shader,
    f: PackedFormat,
    prev: &StorageImage,
    cur: &StorageImage,
    flow_img: &StorageImage,
    prev_luma: &StorageImage,
    cur_luma: &StorageImage,
    t: f32,
) -> Result<(Image<u32>, f64), String> {
    let dst = gpu.create_storage_image(vk::Format::R32_UINT, prev.width, prev.height)?;
    let result = (|| {
        gpu.clear_u32(&dst, CANARY_U32)?;
        let push = push_bytes(&[
            prev.width.to_le_bytes(),
            prev.height.to_le_bytes(),
            flow_img.width.to_le_bytes(),
            flow_img.height.to_le_bytes(),
            t.to_le_bytes(),
            f.code().to_le_bytes(),
        ]);
        let us = gpu.run(
            shader,
            &[prev, cur, flow_img, &dst, prev_luma, cur_luma],
            &push,
            groups(prev.width, prev.height),
        )?;
        Ok((gpu.read_u32(&dst)?, us))
    })();
    gpu.destroy_image(dst);
    result
}

fn warp_packed_case(
    gpu: &Gpu,
    f: PackedFormat,
    prev: &Image<u32>,
    cur: &Image<u32>,
    flow_cpu: &Flow,
    ts: &[f32],
    tol_steps: f32,
) -> Result<CaseResult, String> {
    let (lp, lc) = (packed::luma_downsample_packed(f, prev), packed::luma_downsample_packed(f, cur));
    let gp = gpu.upload_u32(prev)?;
    let gc = gpu.upload_u32(cur)?;
    let gf = gpu.upload_rg32f(flow_cpu)?;
    let glp = gpu.upload_r32f(&lp)?;
    let glc = gpu.upload_r32f(&lc)?;
    let result = (|| {
        let (mut reports, mut time) = (Vec::new(), 0.0);
        for &t in ts {
            let (actual, us) = run_warp_packed(gpu, f, &gp, &gc, &gf, &glp, &glc, t)?;
            time += us;
            let expected = packed::warp_blend_packed(f, prev, cur, &lp, &lc, flow_cpu, t);
            let report = packed::compare_packed(f, &expected, &actual, tol_steps);
            reports.push(report);
        }
        Ok(CaseResult { report: merge(&reports), gpu_us: time, tolerance: tol_steps })
    })();
    for img in [gp, gc, gf, glp, glc] {
        gpu.destroy_image(img);
    }
    result
}

fn battery_packed_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let f = PackedFormat::Bgra8;
    let prev = noise(1280, 800, 31);
    let cur = shift(&prev, 10, -6);
    let (pp, cp) = (packed::to_packed(f, &prev), packed::to_packed(f, &cur));
    let mut setup = 0.0;
    let fp = gpu_frame_packed(gpu, f, &pp, &mut setup)?;
    let mut time = 0.0;
    let fc = match gpu_frame_packed(gpu, f, &cp, &mut time) {
        Ok(frame) => frame,
        Err(e) => {
            fp.destroy(gpu);
            return Err(e);
        }
    };
    let result = (|| {
        let flows = gpu_flows(gpu, &fp, &fc, &mut time)?;
        let level0 = gpu.read_rg32f(&flows.images[0]);
        let warp = run_warp_packed(gpu, f, &fp.color, &fc.color, &flows.images[0], &fp.levels[0], &fc.levels[0], 0.5);
        let lumas = gpu.read_r32f(&fp.levels[0]).and_then(|a| Ok((a, gpu.read_r32f(&fc.levels[0])?)));
        let mut reports = flows.checks.clone();
        for img in flows.images {
            gpu.destroy_image(img);
        }
        let level0 = level0?;
        let (actual, us) = warp?;
        time += us;
        end_to_end(&prev, &cur, &level0)?;
        require_recovery(&level0, [5.0, -3.0])?;
        let (lp, lc) = lumas?;
        let report = packed::compare_packed(f, &packed::warp_blend_packed(f, &pp, &cp, &lp, &lc, &level0, 0.5), &actual, 1.0);
        reports.push(report);
        println!("      battery packed pipeline total {time:.1} us (budget 2000 us)");
        Ok(CaseResult { report: merge(&reports), gpu_us: time, tolerance: 1.0 })
    })();
    fp.destroy(gpu);
    fc.destroy(gpu);
    result
}

/// GPU flow for every level (index = level), each checked exactly against the CPU fed with the
/// GPU's own luma and coarser flow.
struct Flows {
    images: Vec<StorageImage>,
    checks: Vec<CompareReport>,
    pass_us: Vec<(String, f64)>,
}

fn gpu_flows(gpu: &Gpu, prev: &Frame, cur: &Frame, time: &mut f64) -> Result<Flows, String> {
    let mut pass_us: Vec<(String, f64)> = Vec::new();
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
            let radius = level_radius(level);
            let (cw, ch) = coarse_img.map_or((1, 1), |c| (c.width, c.height));
            let push = push_bytes(&[
                pl.width.to_le_bytes(),
                pl.height.to_le_bytes(),
                cw.to_le_bytes(),
                ch.to_le_bytes(),
                radius.to_le_bytes(),
                (coarse_img.is_some() as u32).to_le_bytes(),
            ]);
            let us = gpu.run(&BLOCK_MATCH, &[pl, cl, coarse_img.unwrap_or(&dummy), &out], &push, block_groups(bw, bh))?;
            *time += us;
            pass_us.push((format!("match L{level}"), us));
            let cpu_coarse: Option<Flow> = coarse_img.map(|c| gpu.read_rg32f(c)).transpose()?;
            let gpu_flow = gpu.read_rg32f(&out)?;
            let expected = flow::block_match(&gpu.read_r32f(pl)?, &gpu.read_r32f(cl)?, cpu_coarse.as_ref(), radius);
            let mut report = compare_rg32f(&expected, &gpu_flow, 0.0);
            if level > 0 {
                let median = gpu.create_storage_image(vk::Format::R32G32_SFLOAT, bw, bh)?;
                gpu.clear(&median, CANARY_FLOAT)?;
                let push = push_bytes(&[bw.to_le_bytes(), bh.to_le_bytes()]);
                let us = gpu.run(&FLOW_MEDIAN, &[&out, &median], &push, groups(bw, bh))?;
                *time += us;
                pass_us.push((format!("median L{level}"), us));
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
    Ok(Flows { images, checks, pass_us })
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

/// Blocks where the GPU's level-0 flow differs from a pure-CPU run; at most 0.5% may differ,
/// because CPU and GPU luma differ by ~1e-7 and can round to different integers.
fn end_to_end(prev: &Image<Rgba8>, cur: &Image<Rgba8>, gpu_level0: &Flow) -> Result<(), String> {
    let cpu = flow::estimate_flow(&flow::pyramid(prev), &flow::pyramid(cur));
    let differ = compare_rg32f(&cpu, gpu_level0, 0.0).over_tolerance;
    let total = gpu_level0.data.len() as u32;
    if differ == u32::MAX || differ * 200 > total {
        return Err(format!("GPU flow differs from the pure-CPU run in {differ}/{total} blocks, max 0.5%"));
    }
    Ok(())
}

fn require_recovery(flow: &Flow, want: [f32; 2]) -> Result<(), String> {
    let (hits, total) = interior_hits(flow, want);
    if hits * 100 < total * 95 {
        return Err(format!("motion recovered in {hits}/{total} interior blocks, need 95%"));
    }
    Ok(())
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
        end_to_end(prev, cur, &level0)?;
        if let Some(want) = expect {
            require_recovery(&level0, want)?;
        }
        Ok(CaseResult { report, gpu_us: time, tolerance: 0.0 })
    })();
    fp.destroy(gpu);
    fc.destroy(gpu);
    result
}

fn warp_case(gpu: &Gpu, prev: &Image<Rgba8>, cur: &Image<Rgba8>, flow_cpu: &Flow, ts: &[f32], tol: u8) -> Result<CaseResult, String> {
    let (lp, lc) = (luma_downsample(prev), luma_downsample(cur));
    let gp = gpu.upload_rgba8(prev)?;
    let gc = gpu.upload_rgba8(cur)?;
    let gf = gpu.upload_rg32f(flow_cpu)?;
    let glp = gpu.upload_r32f(&lp)?;
    let glc = gpu.upload_r32f(&lc)?;
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
            time += gpu.run(&WARP_BLEND, &[&gp, &gc, &gf, &dst, &glp, &glc], &push, groups(prev.width, prev.height))?;
            let report = compare_rgba8(&flow::warp_blend(prev, cur, &lp, &lc, flow_cpu, t), &gpu.read_rgba8(&dst)?, tol);
            reports.push(report);
        }
        Ok(CaseResult { report: merge(&reports), gpu_us: time, tolerance: tol as f32 })
    })();
    for img in [gp, gc, gf, dst, glp, glc] {
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
            time += gpu.run(
                &WARP_BLEND,
                &[&fp.color, &fc.color, &flows.images[0], &dst, &fp.levels[0], &fc.levels[0]],
                &push,
                groups(prev.width, prev.height),
            )?;
            gpu.read_rgba8(&dst)
        })();
        gpu.destroy_image(dst);
        for img in flows.images {
            gpu.destroy_image(img);
        }
        let level0 = level0?;
        end_to_end(&prev, &cur, &level0)?;
        require_recovery(&level0, [5.0, -3.0])?;
        let (lp, lc) = (gpu.read_r32f(&fp.levels[0])?, gpu.read_r32f(&fc.levels[0])?);
        let report = compare_rgba8(&flow::warp_blend(&prev, &cur, &lp, &lc, &level0, 0.5), &warp?, BLEND_TOLERANCE);
        reports.push(report);
        println!("      battery pipeline total {time:.1} us (budget 2000 us)");
        Ok(CaseResult { report: merge(&reports), gpu_us: time, tolerance: BLEND_TOLERANCE as f32 })
    })();
    fp.destroy(gpu);
    fc.destroy(gpu);
    result
}

fn still_square_scene() -> (Image<Rgba8>, Image<Rgba8>, Flow) {
    let bg = noise(1280, 800, 51);
    let square = checker(96, 96, 6);
    let (sx, sy) = (592u32, 352u32);
    let paste = |dst: &mut Image<Rgba8>| {
        for y in 0..square.height {
            for x in 0..square.width {
                dst.set(sx + x, sy + y, square.get(x, y));
            }
        }
    };
    let mut prev = bg.clone();
    paste(&mut prev);
    let mut cur = shift(&bg, 16, 0);
    paste(&mut cur);
    let mut flow = Image::new(80, 50, [8.0f32, 0.0]);
    for by in sy / 16..=(sy + 95) / 16 {
        for bx in sx / 16..=(sx + 95) / 16 {
            flow.set(bx, by, [0.0, 0.0]);
        }
    }
    (prev, cur, flow)
}

/// A level-0 flow for a `width`×`height` frame whose vectors change from block to block.
fn varied_flow(width: u32, height: u32) -> Flow {
    let (bw, bh) = (width.div_ceil(2).div_ceil(BLOCK), height.div_ceil(2).div_ceil(BLOCK));
    let mut flow = Image::new(bw, bh, [0.0f32, 0.0]);
    for by in 0..bh {
        for bx in 0..bw {
            let x = ((bx * 7 + by * 3) % 9) as f32 - 4.0;
            let y = ((bx * 5 + by * 11) % 7) as f32 - 3.0;
            flow.set(bx, by, [x, y]);
        }
    }
    flow
}

/// `n + 1` frames of a pan: frame `k` is the base noise moved by `k·(dx, dy)` pixels.
fn pan_frames(width: u32, height: u32, dx: i32, dy: i32, n: u32, seed: u64) -> Vec<Image<Rgba8>> {
    let base = noise(width + n * dx.unsigned_abs(), height + n * dy.unsigned_abs(), seed);
    let (cx, cy) = (n as i32 * dx.max(0), n as i32 * dy.max(0));
    (0..=n as i32)
        .map(|k| {
            let mut f = Image::new(width, height, [0u8, 0, 0, 255]);
            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    f.set(x as u32, y as u32, base.get((x - k * dx + cx) as u32, (y - k * dy + cy) as u32));
                }
            }
            f
        })
        .collect()
}

const FSR3_FRAMES: u32 = 7;
const FSR3_BLOCK: u32 = 16;
const FSR3_FORMAT: PackedFormat = PackedFormat::Bgra8;

fn fsr3_run(gpu: &Gpu, frames: &[Image<Rgba8>]) -> Result<crate::fsr3::SequenceResult, String> {
    let packed: Vec<Image<u32>> = frames.iter().map(|f| packed::to_packed(FSR3_FORMAT, f)).collect();
    run_sequence(gpu, FSR3_FORMAT, &packed)
}

fn near(v: [f32; 2], want: [f32; 2]) -> bool {
    (v[0] - want[0]).abs() <= 1.0 && (v[1] - want[1]).abs() <= 1.0
}

fn fsr3_hits(flow: &Flow, want: [f32; 2], keep: impl Fn(u32, u32) -> bool) -> (u32, u32) {
    let (mut hits, mut total) = (0, 0);
    for by in 2..flow.height.saturating_sub(2) {
        for bx in 2..flow.width.saturating_sub(2) {
            if keep(bx, by) {
                total += 1;
                hits += near(flow.get(bx, by), want) as u32;
            }
        }
    }
    (hits, total)
}

fn require_fsr3(flow: &Flow, want: [f32; 2], what: &str, keep: impl Fn(u32, u32) -> bool) -> Result<(), String> {
    let (hits, total) = fsr3_hits(flow, want, keep);
    if total == 0 || hits * 100 < total * 95 {
        return Err(format!("{what}: {hits}/{total} blocks within 1 px of {want:?}, need 95%; most common {}", common(flow)));
    }
    Ok(())
}

fn common(flow: &Flow) -> String {
    let mut counts: Vec<([f32; 2], u32)> = Vec::new();
    for v in &flow.data {
        match counts.iter_mut().find(|(c, _)| c == v) {
            Some((_, n)) => *n += 1,
            None => counts.push((*v, 1)),
        }
    }
    counts.sort_by_key(|c| std::cmp::Reverse(c.1));
    counts.iter().take(3).map(|(v, n)| format!("{v:?}x{n}")).collect::<Vec<_>>().join(" ")
}

/// True when the 16×16 block's content was inside the previous frame of a `(dx, dy)` pan.
fn seen_before(bx: u32, by: u32, width: u32, height: u32, dx: i32, dy: i32) -> bool {
    let b = FSR3_BLOCK as i32;
    let (x0, y0) = (bx as i32 * b - dx, by as i32 * b - dy);
    x0 >= 0 && y0 >= 0 && x0 + b <= width as i32 && y0 + b <= height as i32
}

/// Blocks in the outer two rows and columns, which the interior checks skip, must reach 90%.
fn require_border(flow: &Flow, want: [f32; 2], keep: impl Fn(u32, u32) -> bool) -> Result<(), String> {
    let (mut hits, mut total) = (0, 0);
    for by in 0..flow.height {
        for bx in 0..flow.width {
            let border = bx < 2 || by < 2 || bx + 2 >= flow.width || by + 2 >= flow.height;
            if border && keep(bx, by) {
                total += 1;
                hits += near(flow.get(bx, by), want) as u32;
            }
        }
    }
    if total == 0 || hits * 100 < total * 90 {
        return Err(format!("border: {hits}/{total} blocks within 1 px of {want:?}, need 90%"));
    }
    println!("      border blocks {hits}/{total} within 1 px");
    Ok(())
}

fn fsr3_ready(gpu: &Gpu) -> Result<(), String> {
    fsr3_requirements(gpu.subgroup_ok, gpu.subgroup_size, gpu.extended_formats, gpu.robust_images)?;
    if !gpu.write_without_format {
        return Err("the device lacks shaderStorageImage{Read,Write}WithoutFormat".into());
    }
    Ok(())
}

/// What FSR 3's passes need from the device: subgroup basic, arithmetic and quad operations in
/// compute at size 32 or 64 (the search's reduction assumes one of those), `r8ui`/`rg16i` storage
/// images, and robust image access because AMD's passes read and write past image edges.
fn fsr3_requirements(subgroup_ok: bool, subgroup_size: u32, extended_formats: bool, robust_images: bool) -> Result<(), String> {
    if !subgroup_ok {
        return Err("the device lacks subgroup basic, arithmetic or quad operations in compute".into());
    }
    if subgroup_size != 32 && subgroup_size != 64 {
        return Err(format!("subgroup size {subgroup_size} is not 32 or 64"));
    }
    if !extended_formats {
        return Err("the device lacks shaderStorageImageExtendedFormats".into());
    }
    if !robust_images {
        return Err("robust image access is not enabled".into());
    }
    Ok(())
}

fn fsr3_result(gpu_us: f64) -> CaseResult {
    CaseResult { report: CompareReport { max_diff: 0.0, worst: None, over_tolerance: 0 }, gpu_us, tolerance: 1.0 }
}

fn fsr3_shift_case(gpu: &Gpu, width: u32, height: u32, dx: i32, dy: i32, frames: u32) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let (flow, _, us, _) = fsr3_run(gpu, &pan_frames(width, height, dx, dy, frames, 61))?;
    require_fsr3(&flow, [dx as f32, dy as f32], "pan", |bx, by| seen_before(bx, by, width, height, dx, dy))?;
    require_border(&flow, [dx as f32, dy as f32], |bx, by| seen_before(bx, by, width, height, dx, dy))?;
    Ok(fsr3_result(us))
}

fn fsr3_black_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let black = Image::new(1280, 800, [0u8, 0, 0, 255]);
    let frames: Vec<_> = (0..=FSR3_FRAMES).map(|_| black.clone()).collect();
    let (flow, scd, us, _) = fsr3_run(gpu, &frames)?;
    if let Some(v) = flow.data.iter().find(|v| **v != [0.0, 0.0]) {
        return Err(format!("flat black gave vector {v:?}"));
    }
    if scd[1] & 0xf != 0 {
        return Err(format!("flat black still counts as a scene change (history {:#x}), so the search never ran", scd[1]));
    }
    Ok(fsr3_result(us))
}

fn fsr3_square_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let square = checker(96, 96, 6);
    let (sx, sy) = (592u32, 352u32);
    let frames: Vec<_> = pan_frames(1280, 800, 16, 0, FSR3_FRAMES, 51)
        .into_iter()
        .map(|mut f| {
            for y in 0..square.height {
                for x in 0..square.width {
                    f.set(sx + x, sy + y, square.get(x, y));
                }
            }
            f
        })
        .collect();
    let (flow, _, us, _) = fsr3_run(gpu, &frames)?;
    let b = FSR3_BLOCK;
    let inside = |bx: u32, by: u32| bx * b >= sx + 8 && bx * b + b <= sx + 88 && by * b >= sy + 8 && by * b + b <= sy + 88;
    let far = |bx: u32, by: u32| {
        let (x0, y0, x1, y1) = (bx * b, by * b, bx * b + b, by * b + b);
        x1 + 32 <= sx || x0 >= sx + 96 + 32 || y1 + 32 <= sy || y0 >= sy + 96 + 32
    };
    require_fsr3(&flow, [0.0, 0.0], "square", inside)?;
    require_fsr3(&flow, [16.0, 0.0], "background", far)?;
    Ok(fsr3_result(us))
}

fn fsr3_scene_cut_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let mut frames = pan_frames(1280, 800, 12, -8, FSR3_FRAMES, 71);
    let (_, calm, _, _) = fsr3_run(gpu, &frames)?;
    frames.push(gradient(1280, 800));
    let (_, cut, us, _) = fsr3_run(gpu, &frames)?;
    let value = |s: [u32; 3]| f32::from_bits(s[0]);
    println!(
        "      scene change value calm {:.3} history {:#x}, cut {:.3} history {:#x}",
        value(calm),
        calm[1],
        value(cut),
        cut[1]
    );
    if calm[1] & 1 != 0 || cut[1] & 1 == 0 {
        return Err(format!("history bits calm {:#x} cut {:#x}: expected the cut only on the new frame", calm[1], cut[1]));
    }
    Ok(fsr3_result(us))
}

fn fsr3_pipeline_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let frames = pan_frames(1280, 800, 12, -8, FSR3_FRAMES, 81);
    let (flow, _, us, passes) = fsr3_run(gpu, &frames)?;
    require_fsr3(&flow, [12.0, -8.0], "pan", |_, _| true)?;
    let names = ["prepare luma", "pyramid", "scd histogram", "scd divergence", "search", "filter", "scale", "luma half"];
    let detail: Vec<String> = names.iter().zip(passes.passes).map(|(n, t)| format!("{n} {t:.0}")).collect();
    println!("      fsr3 flow pipeline total {us:.1} us (budget 2000 us): {}", detail.join(", "));
    let levels: Vec<String> = passes.search_by_level.iter().enumerate().map(|(l, t)| format!("L{l} {t:.0}")).collect();
    println!("      fsr3 search by level: {}", levels.join(", "));
    let (prev, cur) = (&frames[frames.len() - 2], &frames[frames.len() - 1]);
    let battery = flow::estimate_flow(&flow::pyramid(prev), &flow::pyramid(cur));
    let (bh, bt) = interior_hits(&battery, [6.0, -4.0]);
    let (fh, ft) = fsr3_hits(&flow, [12.0, -8.0], |_, _| true);
    println!("      recovery fsr3 {fh}/{ft} (within 1 px), battery {bh}/{bt} (exact), both on 16 px blocks");
    Ok(fsr3_result(us))
}

/// Odd full-resolution motion is between two half-resolution vectors, so neither engine can hit
/// it exactly; FSR 3 must keep its M4a level (84% of blocks within 1 px), and Battery's result on
/// the same frames is printed next to it.
fn fsr3_odd_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let frames = pan_frames(1280, 800, 7, -3, FSR3_FRAMES, 63);
    let (flow, _, us, _) = fsr3_run(gpu, &frames)?;
    let (prev, cur) = (&frames[frames.len() - 2], &frames[frames.len() - 1]);
    let battery = flow::estimate_flow(&flow::pyramid(prev), &flow::pyramid(cur));
    let battery_px = Image { width: battery.width, height: battery.height, data: battery.data.iter().map(|v| [v[0] * 2.0, v[1] * 2.0]).collect() };
    let keep = |bx: u32, by: u32| seen_before(bx, by, 1280, 800, 7, -3);
    let (fh, ft) = fsr3_hits(&flow, [7.0, -3.0], keep);
    let (bh, bt) = fsr3_hits(&battery_px, [7.0, -3.0], keep);
    println!("      odd shift within 1 px: fsr3 {fh}/{ft}, battery {bh}/{bt}; fsr3 most common {}", common(&flow));
    if fh * 100 < ft * 84 {
        return Err(format!("fsr3 {fh}/{ft} is below 84% (battery {bh}/{bt})"));
    }
    Ok(fsr3_result(us))
}

/// The flow's own images, measured as the driver allocates them, stay within 1.5 MiB at 1280×800;
/// the packed frame is borrowed from the layer and not counted.
fn fsr3_memory_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let flow = crate::fsr3::Fsr3Flow::new(gpu, 1280, 800, FSR3_FORMAT)?;
    let (owned, count) = (flow.owned_bytes(), flow.owned_images());
    let texels = haqengine_core::fsr3_flow::of_memory_bytes(640, 400);
    flow.destroy(gpu);
    println!("      fsr3 flow owns {owned} bytes in {count} images ({texels} bytes of texels)");
    if owned > 3 << 19 {
        return Err(format!("flow owns {owned} bytes, budget 1.5 MiB"));
    }
    Ok(fsr3_result(0.0))
}

const CANARIES: [[f32; 4]; 2] = [[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]];

fn canary_rgb(c: [f32; 4]) -> [u8; 3] {
    [(c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8]
}

/// Runs every frame through a fresh interpolation, pre-filling the output with `canary`, and
/// returns each frame's output and the last frame's times.
fn interp_run(
    gpu: &Gpu,
    frames: &[Image<Rgba8>],
    canary: [f32; 4],
) -> Result<(Vec<Image<Rgba8>>, f64, crate::fsr3_interp::InterpTimes), String> {
    let first = frames.first().ok_or("no frames")?;
    let mut fi = crate::fsr3_interp::Fsr3Interp::new(gpu, first.width, first.height, FSR3_FORMAT)?;
    let result = (|| {
        let (mut outs, mut last) = (Vec::with_capacity(frames.len()), 0.0);
        for f in frames {
            fi.fill_output(gpu, canary)?;
            last = fi.run_frame(gpu, f)?;
            outs.push(fi.read_output(gpu)?);
        }
        Ok((outs, last, fi.times))
    })();
    fi.destroy(gpu);
    result
}

/// Pixels both canary runs left untouched, which no pass wrote.
fn unwritten(a: &Image<Rgba8>, b: &Image<Rgba8>) -> usize {
    let (ca, cb) = (canary_rgb(CANARIES[0]), canary_rgb(CANARIES[1]));
    a.data.iter().zip(&b.data).filter(|(x, y)| x[..3] == ca && y[..3] == cb).count()
}

/// The blend equals the CPU reference within one 8-bit step, on the still-square scene and at
/// an odd size.
fn blend_parity_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let square = checker(96, 96, 6);
    let mut reports = Vec::new();
    let mut time = 0.0;
    for (w, h, dx, dy) in [(1280u32, 800u32, 16i32, 0i32), (1281, 801, -6, 4)] {
        let mut frames = pan_frames(w, h, dx, dy, QUALITY_STEPS, 151);
        for f in &mut frames {
            paste(f, &square, (592, 352));
        }
        let mut fi = crate::fsr3_interp::Fsr3Interp::new(gpu, w, h, FSR3_FORMAT)?;
        let result = (|| -> Result<CompareReport, String> {
            for f in &frames {
                time = fi.run_frame(gpu, f)?;
            }
            let (raw, lp, lc) = fi.read_blend_inputs(gpu)?;
            let n = frames.len();
            let expected = haqengine_core::fsr3_interp::fi_blend_reference(&frames[n - 2], &frames[n - 1], &raw, &lp, &lc, false);
            let mut got = fi.read_output(gpu)?;
            for (g, e) in got.data.iter_mut().zip(&expected.data) {
                g[3] = e[3];
            }
            Ok(compare_rgba8(&expected, &got, BLEND_TOLERANCE))
        })();
        fi.destroy(gpu);
        reports.push(result?);
    }
    Ok(CaseResult { report: merge(&reports), gpu_us: time, tolerance: BLEND_TOLERANCE as f32 })
}

/// The packed blend, run on the frames and flow the layer would have, equals its CPU reference
/// within one step of the format.
fn blend_packed_case(gpu: &Gpu, f: PackedFormat) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let square = checker(96, 96, 6);
    let mut frames = pan_frames(1280, 800, 16, 0, QUALITY_STEPS, 171);
    for fr in &mut frames {
        paste(fr, &square, (592, 352));
    }
    let packed_frames: Vec<Image<u32>> = frames.iter().map(|fr| packed::to_packed(f, fr)).collect();
    let mut flow = crate::fsr3::Fsr3Flow::new(gpu, 1280, 800, f)?;
    let mut images: Vec<StorageImage> = Vec::new();
    let result = (|| -> Result<CaseResult, String> {
        for fr in &packed_frames {
            flow.run_frame(gpu, fr)?;
        }
        let n = packed_frames.len();
        let (prev, cur) = (&packed_frames[n - 2], &packed_frames[n - 1]);
        images.push(gpu.upload_u32(prev)?);
        images.push(gpu.upload_u32(cur)?);
        images.push(gpu.create_storage_image(vk::Format::R32_UINT, 1280, 800)?);
        let (ps, cs) = flow.luma_slots();
        let raw = flow.read_raw(gpu)?;
        let lp = gpu.read_r8(flow.image(haqengine_core::fsr3_flow::OfRes::Input(ps, 0))?)?;
        let lc = gpu.read_r8(flow.image(haqengine_core::fsr3_flow::OfRes::Input(cs, 0))?)?;
        let binds = [
            (0, Bind::Image(&images[0])),
            (1, Bind::Image(&images[1])),
            (2, Bind::Image(flow.image(haqengine_core::fsr3_flow::OfRes::Output)?)),
            (3, Bind::Image(flow.image(haqengine_core::fsr3_flow::OfRes::Input(ps, 0))?)),
            (4, Bind::Image(flow.image(haqengine_core::fsr3_flow::OfRes::Input(cs, 0))?)),
            (5, Bind::Image(&images[2])),
            (6, Bind::Image(flow.image(haqengine_core::fsr3_flow::OfRes::ScdOutput)?)),
        ];
        let push = push_bytes(&[
            1280u32.to_le_bytes(),
            800u32.to_le_bytes(),
            raw.width.to_le_bytes(),
            raw.height.to_le_bytes(),
            0u32.to_le_bytes(),
            f.code().to_le_bytes(),
        ]);
        let us = gpu.run_fsr3(&haqengine_shaders::FI_BLEND_PACKED, &binds, &[], &push, [160, 100, 1])?;
        let expected = haqengine_core::fsr3_interp::fi_blend_packed_reference(f, prev, cur, &raw, &lp, &lc, false);
        let got = gpu.read_u32(&images[2])?;
        Ok(CaseResult { report: packed::compare_packed(f, &expected, &got, 1.0), gpu_us: us, tolerance: 1.0 })
    })();
    for img in images {
        gpu.destroy_image(img);
    }
    flow.destroy(gpu);
    result
}

fn interp_reset_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let frames = pan_frames(1280, 800, 12, -8, 9, 101);
    let (a, us, _) = interp_run(gpu, &frames, CANARIES[0])?;
    let (b, _, _) = interp_run(gpu, &frames, CANARIES[1])?;
    let differs = a[0].data.iter().zip(&frames[0].data).filter(|(o, f)| o[..3] != f[..3]).count();
    if differs > 0 {
        return Err(format!("frame 0 differs from the current frame in {differs} pixels"));
    }
    for k in 1..frames.len() {
        let n = unwritten(&a[k], &b[k]);
        if n > 0 {
            return Err(format!("frame {k}: {n} pixels were never written"));
        }
    }
    Ok(fsr3_result(us))
}

/// Interpolation's own images, measured as the driver allocates them, stay within 12 MiB at
/// 1280×800; the two real frames and the flow are counted elsewhere.
fn interp_memory_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let fi = crate::fsr3_interp::Fsr3Interp::new(gpu, 1280, 800, FSR3_FORMAT)?;
    let (owned, count) = (fi.owned_bytes(), fi.owned_images());
    fi.destroy(gpu);
    let texels = haqengine_core::fsr3_interp::fi_memory_bytes(1280, 800);
    println!("      fsr3 interp owns {owned} bytes in {count} images ({texels} bytes of texels)");
    if owned > 12 << 20 {
        return Err(format!("interpolation owns {owned} bytes, budget 12 MiB"));
    }
    Ok(fsr3_result(0.0))
}

const QUALITY_STEPS: u32 = 11;

/// The pan scene `half_steps / 2` steps in: the exact frame between two real ones when
/// `half_steps` is odd and both motions are even.
fn pan_at(width: u32, height: u32, dx: i32, dy: i32, n: u32, seed: u64, half_steps: i32) -> Image<Rgba8> {
    let base = noise(width + n * dx.unsigned_abs(), height + n * dy.unsigned_abs(), seed);
    let (cx, cy) = (n as i32 * dx.max(0), n as i32 * dy.max(0));
    let (ox, oy) = (half_steps * dx / 2, half_steps * dy / 2);
    let mut f = Image::new(width, height, [0u8, 0, 0, 255]);
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            f.set(x as u32, y as u32, base.get((x - ox + cx) as u32, (y - oy + cy) as u32));
        }
    }
    f
}

fn paste(dst: &mut Image<Rgba8>, src: &Image<Rgba8>, at: (u32, u32)) {
    for y in 0..src.height {
        for x in 0..src.width {
            dst.set(at.0 + x, at.1 + y, src.get(x, y));
        }
    }
}

/// Mean absolute RGB difference over the kept pixels, in 8-bit steps.
fn mean_err(a: &Image<Rgba8>, b: &Image<Rgba8>, keep: impl Fn(u32, u32) -> bool) -> f64 {
    let (mut sum, mut n) = (0u64, 0u64);
    for y in 0..a.height {
        for x in 0..a.width {
            if keep(x, y) {
                let (p, q) = (a.get(x, y), b.get(x, y));
                sum += (0..3).map(|c| p[c].abs_diff(q[c]) as u64).sum::<u64>();
                n += 3;
            }
        }
    }
    if n == 0 { f64::NAN } else { sum as f64 / n as f64 }
}

/// Battery's frame between the last two frames, computed exactly as its GPU passes do.
fn battery_mid(prev: &Image<Rgba8>, cur: &Image<Rgba8>) -> Image<Rgba8> {
    let (pp, cp) = (flow::pyramid(prev), flow::pyramid(cur));
    let f = flow::estimate_flow(&pp, &cp);
    flow::warp_blend(prev, cur, &pp[0], &cp[0], &f, 0.5)
}

/// FSR 3's last interpolated frame for a sequence, checked for unwritten pixels with two canaries.
fn fsr3_mid(gpu: &Gpu, frames: &[Image<Rgba8>]) -> Result<(Image<Rgba8>, f64, crate::fsr3_interp::InterpTimes), String> {
    let (a, us, times) = interp_run(gpu, frames, CANARIES[0])?;
    let (b, _, _) = interp_run(gpu, frames, CANARIES[1])?;
    let (la, lb) = (a.last().ok_or("no output")?, b.last().ok_or("no output")?);
    let n = unwritten(la, lb);
    if n > 0 {
        return Err(format!("{n} pixels were never written"));
    }
    Ok((la.clone(), us, times))
}

fn compare_to_battery(what: &str, fsr3: f64, battery: f64, bar: f64, slack: f64) -> Result<(), String> {
    println!("      {what}: fsr3 {fsr3:.3}, battery {battery:.3} (8-bit steps)");
    if fsr3.is_nan() || fsr3 > bar || fsr3 > battery + slack {
        return Err(format!("{what}: fsr3 {fsr3:.3} vs battery {battery:.3}, bar {bar} and battery + {slack}"));
    }
    Ok(())
}

fn interp_pan_case(gpu: &Gpu, width: u32, height: u32, dx: i32, dy: i32) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let frames = pan_frames(width, height, dx, dy, QUALITY_STEPS, 111);
    let truth = pan_at(width, height, dx, dy, QUALITY_STEPS, 111, 2 * QUALITY_STEPS as i32 - 1);
    let (out, us, _) = fsr3_mid(gpu, &frames)?;
    let n = frames.len();
    let battery = battery_mid(&frames[n - 2], &frames[n - 1]);
    let margin = 48;
    let keep = |x: u32, y: u32| x >= margin && y >= margin && x + margin < width && y + margin < height;
    compare_to_battery("pan", mean_err(&out, &truth, keep), mean_err(&battery, &truth, keep), 2.0, 0.5)?;
    Ok(fsr3_result(us))
}

fn interp_square_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let square = checker(96, 96, 6);
    let at = (592u32, 352u32);
    let mut frames = pan_frames(1280, 800, 16, 0, QUALITY_STEPS, 51);
    for f in &mut frames {
        paste(f, &square, at);
    }
    let mut truth = pan_at(1280, 800, 16, 0, QUALITY_STEPS, 51, 2 * QUALITY_STEPS as i32 - 1);
    paste(&mut truth, &square, at);
    let (out, us, _) = fsr3_mid(gpu, &frames)?;
    let n = frames.len();
    let battery = battery_mid(&frames[n - 2], &frames[n - 1]);
    let dist = |x: u32, y: u32| {
        let dx = (at.0 as i32 - x as i32).max(x as i32 - (at.0 + 95) as i32).max(0);
        let dy = (at.1 as i32 - y as i32).max(y as i32 - (at.1 + 95) as i32).max(0);
        dx.max(dy)
    };
    let inside = |x: u32, y: u32| x >= at.0 + 8 && x < at.0 + 88 && y >= at.1 + 8 && y < at.1 + 88;
    let far = |x: u32, y: u32| dist(x, y) >= 32 && x >= 48 && y >= 48 && x + 48 < 1280 && y + 48 < 800;
    let band = |x: u32, y: u32| !inside(x, y) && dist(x, y) < 24;
    compare_to_battery("square", mean_err(&out, &truth, inside), mean_err(&battery, &truth, inside), 2.0, 0.5)?;
    compare_to_battery("background", mean_err(&out, &truth, far), mean_err(&battery, &truth, far), 2.0, 0.5)?;
    compare_to_battery("edge band", mean_err(&out, &truth, band), mean_err(&battery, &truth, band), 255.0, 0.0)?;
    Ok(fsr3_result(us))
}

fn interp_overlay_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let mut bar = Image::new(200, 40, [255u8, 255, 255, 255]);
    for y in 0..40 {
        for x in (0..200).filter(|x| x % 8 < 2) {
            bar.set(x, y, [0, 0, 0, 255]);
        }
    }
    let at = (100u32, 700u32);
    let mut frames = pan_frames(1280, 800, 12, -8, QUALITY_STEPS, 121);
    for f in &mut frames {
        paste(f, &bar, at);
    }
    let mut truth = pan_at(1280, 800, 12, -8, QUALITY_STEPS, 121, 2 * QUALITY_STEPS as i32 - 1);
    paste(&mut truth, &bar, at);
    let (out, us, _) = fsr3_mid(gpu, &frames)?;
    let n = frames.len();
    let battery = battery_mid(&frames[n - 2], &frames[n - 1]);
    let on_bar = |x: u32, y: u32| x >= at.0 && x < at.0 + 200 && y >= at.1 && y < at.1 + 40;
    compare_to_battery("overlay", mean_err(&out, &truth, on_bar), mean_err(&battery, &truth, on_bar), 5.0, 0.0)?;
    Ok(fsr3_result(us))
}

/// A black square moving 16 px per real frame over still noise. Where it covers or reveals the
/// background, FSR 3's frame must be at least as close to the true midpoint as Battery's.
fn interp_revealed_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let bg = noise(1280, 800, 131);
    let square = Image::new(200, 200, [0u8, 0, 0, 255]);
    let frames: Vec<Image<Rgba8>> = (0..=QUALITY_STEPS)
        .map(|k| {
            let mut f = bg.clone();
            paste(&mut f, &square, (200 + 16 * k, 300));
            f
        })
        .collect();
    let (out, us, _) = fsr3_mid(gpu, &frames)?;
    let n = frames.len();
    let (prev, cur) = (&frames[n - 2], &frames[n - 1]);
    let mut truth = bg.clone();
    paste(&mut truth, &square, (200 + 16 * (QUALITY_STEPS - 1) + 8, 300));
    let battery = battery_mid(prev, cur);
    let band = |x: u32, y: u32| (300..500).contains(&y) && prev.get(x, y) != cur.get(x, y);
    compare_to_battery("revealed band", mean_err(&out, &truth, band), mean_err(&battery, &truth, band), 255.0, 0.0)?;
    Ok(fsr3_result(us))
}

/// A hard cut between two dark scenes: the generated frame must be the new frame itself, never a
/// double exposure of both scenes.
fn interp_cut_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let dim = |img: Image<Rgba8>| Image { width: img.width, height: img.height, data: img.data.iter().map(|p| [p[0] / 8, p[1] / 8, p[2] / 8, 255]).collect() };
    let mut frames: Vec<Image<Rgba8>> = pan_frames(1280, 800, 12, -8, QUALITY_STEPS, 161).into_iter().map(dim).collect();
    frames.push(dim(gradient(1280, 800)));
    let (out, us, _) = fsr3_mid(gpu, &frames)?;
    let cur = frames.last().ok_or("no frames")?;
    let differs = out.data.iter().zip(&cur.data).filter(|(o, c)| o[..3] != c[..3]).count();
    println!("      scene cut: {differs} pixels differ from the new frame");
    if differs > 0 {
        return Err(format!("scene cut: {differs} pixels are not the new frame"));
    }
    Ok(fsr3_result(us))
}

/// A black-and-white checker square moving 16 px per frame over flat grey 200. Every true colour
/// is 0, 200 or 255 on all channels, so a pixel far from all three is a blend of two things: a
/// ghost. At most 0.1% of the pixels the square covers or reveals may be one.
fn interp_ghost_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let bg = Image::new(1280, 800, [200u8, 200, 200, 255]);
    let square = checker(200, 200, 10);
    let frames: Vec<Image<Rgba8>> = (0..=QUALITY_STEPS)
        .map(|k| {
            let mut f = bg.clone();
            paste(&mut f, &square, (200 + 16 * k, 300));
            f
        })
        .collect();
    let (out, us, _) = fsr3_mid(gpu, &frames)?;
    let n = frames.len();
    let (prev, cur) = (&frames[n - 2], &frames[n - 1]);
    let (mut ghosts, mut total) = (0usize, 0usize);
    for y in 300..500 {
        for x in 0..1280 {
            if prev.get(x, y) == cur.get(x, y) {
                continue;
            }
            total += 1;
            let o = out.get(x, y);
            let near = |v: u8| (0..3).all(|i| o[i].abs_diff(v) <= 8);
            ghosts += (!near(0) && !near(200) && !near(255)) as usize;
        }
    }
    println!("      ghosts: {ghosts}/{total} changed pixels are a blend");
    if total == 0 || ghosts * 1000 > total {
        return Err(format!("ghosts: {ghosts}/{total}, at most 0.1%"));
    }
    Ok(fsr3_result(us))
}

fn interp_black_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let black = Image::new(1280, 800, [0u8, 0, 0, 255]);
    let frames: Vec<_> = (0..=QUALITY_STEPS).map(|_| black.clone()).collect();
    let (out, us, _) = fsr3_mid(gpu, &frames)?;
    if let Some(p) = out.data.iter().find(|p| p[..3].iter().any(|c| *c > 1)) {
        return Err(format!("flat black gave {p:?}"));
    }
    Ok(fsr3_result(us))
}

fn interp_pipeline_case(gpu: &Gpu) -> Result<CaseResult, String> {
    fsr3_ready(gpu)?;
    let frames = pan_frames(1280, 800, 12, -8, QUALITY_STEPS, 141);
    let (_, us, times) = interp_run(gpu, &frames, CANARIES[0])?;
    let names = [
        "setup", "prepare", "prev depth", "game field", "game pyramid", "flow field", "disocclusion", "interpolation", "pyramid",
        "inpainting", "blend",
    ];
    let detail: Vec<String> =
        names.iter().zip(times.passes).filter(|(_, t)| *t > 0.0).map(|(n, t)| format!("{n} {t:.0}")).collect();
    let interp: f64 = times.passes.iter().sum();
    println!(
        "      fsr3 quality pipeline total {us:.1} us (budget 3000 us): flow {:.0}, interpolation {interp:.0} ({})",
        times.flow,
        detail.join(", ")
    );
    Ok(fsr3_result(us))
}

/// Named GPU pass times in microseconds.
type PassTimes = Vec<(String, f64)>;

/// One blend output and its change counts.
type BlendOutput = (Image<u32>, haqengine_core::frame_change::ChangeCounts);

/// The new blend with its change counters, then the cut fixup, once per time in `times`: output
/// `k` blends at `times[k]` and counts into counter row `k`. Returns each output with its counts,
/// and the GPU time of all passes.
#[allow(clippy::too_many_arguments)]
fn run_battery_blends(
    gpu: &Gpu,
    f: PackedFormat,
    prev: &StorageImage,
    cur: &StorageImage,
    flow_img: &StorageImage,
    prev_luma: &StorageImage,
    cur_luma: &StorageImage,
    times: &[f32],
) -> Result<(Vec<BlendOutput>, f64, Image<u32>), String> {
    if times.is_empty() || times.len() > haqengine_core::MAX_GENERATED {
        return Err(format!("{} blend times", times.len()));
    }
    let counters = gpu.create_storage_image(vk::Format::R32_UINT, 2, haqengine_core::MAX_GENERATED as u32)?;
    let tile = haqengine_core::blend::TILE;
    let tiles = gpu.create_storage_image(
        vk::Format::R32_UINT,
        prev.width.div_ceil(tile),
        prev.height.div_ceil(tile) * haqengine_core::MAX_GENERATED as u32,
    )?;
    let mut dsts: Vec<StorageImage> = Vec::with_capacity(times.len());
    let result = (|| {
        for _ in times {
            let dst = gpu.create_storage_image(vk::Format::R32_UINT, prev.width, prev.height)?;
            gpu.clear_u32(&dst, CANARY_U32)?;
            dsts.push(dst);
        }
        gpu.clear_u32(&counters, 0)?;
        gpu.clear_u32(&tiles, 0)?;
        let pixels = prev.width * prev.height;
        let threshold = haqengine_core::frame_change::cut_threshold_pixels(pixels);
        let mut us = 0.0;
        for (k, (&t, dst)) in times.iter().zip(&dsts).enumerate() {
            let row = k as u32;
            let push = push_bytes(&[
                prev.width.to_le_bytes(),
                prev.height.to_le_bytes(),
                flow_img.width.to_le_bytes(),
                flow_img.height.to_le_bytes(),
                t.to_le_bytes(),
                f.code().to_le_bytes(),
                row.to_le_bytes(),
            ]);
            us += gpu.run(
                &haqengine_shaders::BATTERY_BLEND_PACKED,
                &[prev, cur, flow_img, dst, prev_luma, cur_luma, &counters, &tiles],
                &push,
                groups(prev.width, prev.height),
            )?;
            let fix_push = push_bytes(&[
                prev.width.to_le_bytes(),
                prev.height.to_le_bytes(),
                threshold.to_le_bytes(),
                row.to_le_bytes(),
                t.to_le_bytes(),
                f.code().to_le_bytes(),
            ]);
            us += gpu.run(&haqengine_shaders::CUT_FIXUP, &[&counters, cur, dst, prev, &tiles], &fix_push, groups(prev.width, prev.height))?;
        }
        let c = gpu.read_u32(&counters)?;
        let mut outputs = Vec::with_capacity(times.len());
        for (k, dst) in dsts.iter().enumerate() {
            let counts = haqengine_core::frame_change::ChangeCounts { disagree: c.data[2 * k], still: c.data[2 * k + 1], pixels };
            outputs.push((gpu.read_u32(dst)?, counts));
        }
        Ok((outputs, us, gpu.read_u32(&tiles)?))
    })();
    for dst in dsts {
        gpu.destroy_image(dst);
    }
    gpu.destroy_image(counters);
    gpu.destroy_image(tiles);
    result
}

/// `run_battery_blends` at one time.
#[allow(clippy::too_many_arguments)]
fn run_battery_blend(
    gpu: &Gpu,
    f: PackedFormat,
    prev: &StorageImage,
    cur: &StorageImage,
    flow_img: &StorageImage,
    prev_luma: &StorageImage,
    cur_luma: &StorageImage,
    t: f32,
) -> Result<(Image<u32>, haqengine_core::frame_change::ChangeCounts, f64), String> {
    let (mut outputs, us, _) = run_battery_blends(gpu, f, prev, cur, flow_img, prev_luma, cur_luma, &[t])?;
    let (out, counts) = outputs.remove(0);
    Ok((out, counts, us))
}

/// Battery's flow on the GPU for one frame pair, then `run` with both frames and the flow; the
/// frames and flow images are destroyed afterwards. Returns `run`'s result, the luma and flow time,
/// and the per-pass times.
fn with_battery_flow<R>(
    gpu: &Gpu,
    f: PackedFormat,
    prev: &Image<Rgba8>,
    cur: &Image<Rgba8>,
    run: impl FnOnce(&Frame, &Frame, &Flows) -> Result<R, String>,
) -> Result<(R, f64, PassTimes), String> {
    let (pp, cp) = (packed::to_packed(f, prev), packed::to_packed(f, cur));
    let mut setup = 0.0;
    let fp = gpu_frame_packed(gpu, f, &pp, &mut setup)?;
    let mut flow_us = 0.0;
    let fc = match gpu_frame_packed(gpu, f, &cp, &mut flow_us) {
        Ok(x) => x,
        Err(e) => {
            fp.destroy(gpu);
            return Err(e);
        }
    };
    let result = (|| {
        let luma_us = flow_us;
        let flows = gpu_flows(gpu, &fp, &fc, &mut flow_us)?;
        let mut pass_us = vec![("luma + reduce".to_string(), luma_us)];
        pass_us.extend(flows.pass_us.iter().cloned());
        let r = run(&fp, &fc, &flows);
        for img in flows.images {
            gpu.destroy_image(img);
        }
        Ok((r?, flow_us, pass_us))
    })();
    fp.destroy(gpu);
    fc.destroy(gpu);
    result
}

/// Several blends of one frame pair on one flow, as 3x/4x does.
struct BatteryMulti {
    outputs: Vec<BlendOutput>,
    /// The tile map, one band of `height.div_ceil(TILE)` rows per output.
    tiles: Image<u32>,
    flow: Flow,
    prev_luma: Image<f32>,
    cur_luma: Image<f32>,
    flow_us: f64,
    blend_us: f64,
}

fn battery_multi(gpu: &Gpu, f: PackedFormat, prev: &Image<Rgba8>, cur: &Image<Rgba8>, times: &[f32]) -> Result<BatteryMulti, String> {
    let ((outputs, blend_us, tiles, flow, prev_luma, cur_luma), flow_us, _) = with_battery_flow(gpu, f, prev, cur, |fp, fc, flows| {
        let (outputs, us, tiles) = run_battery_blends(gpu, f, &fp.color, &fc.color, &flows.images[0], &fp.levels[0], &fc.levels[0], times)?;
        Ok((outputs, us, tiles, gpu.read_rg32f(&flows.images[0])?, gpu.read_r32f(&fp.levels[0])?, gpu.read_r32f(&fc.levels[0])?))
    })?;
    Ok(BatteryMulti { outputs, tiles, flow, prev_luma, cur_luma, flow_us, blend_us })
}

/// One frame pair through Battery on the GPU: its flow, then M3c's warp and the new blend.
struct BatteryPair {
    old: Image<Rgba8>,
    new: Image<u32>,
    new_rgba: Image<Rgba8>,
    flow: Flow,
    prev_luma: Image<f32>,
    cur_luma: Image<f32>,
    flow_us: f64,
    blend_us: f64,
    counts: haqengine_core::frame_change::ChangeCounts,
    pass_us: Vec<(String, f64)>,
}

fn unpack_image(f: PackedFormat, img: &Image<u32>) -> Image<Rgba8> {
    let data = img
        .data
        .iter()
        .map(|w| {
            let c = packed::unpack(f, *w);
            [0, 1, 2, 3].map(|i| (c[i] * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8)
        })
        .collect();
    Image { width: img.width, height: img.height, data }
}

fn battery_pair(gpu: &Gpu, f: PackedFormat, prev: &Image<Rgba8>, cur: &Image<Rgba8>) -> Result<BatteryPair, String> {
    battery_pair_at(gpu, f, prev, cur, 0.5)
}

fn battery_pair_at(gpu: &Gpu, f: PackedFormat, prev: &Image<Rgba8>, cur: &Image<Rgba8>, t: f32) -> Result<BatteryPair, String> {
    let ((old, new, counts, blend_us, flow, prev_luma, cur_luma), flow_us, mut pass_us) = with_battery_flow(gpu, f, prev, cur, |fp, fc, flows| {
        let (old, _) = run_warp_packed(gpu, f, &fp.color, &fc.color, &flows.images[0], &fp.levels[0], &fc.levels[0], t)?;
        let (new, counts, blend_us) = run_battery_blend(gpu, f, &fp.color, &fc.color, &flows.images[0], &fp.levels[0], &fc.levels[0], t)?;
        Ok((old, new, counts, blend_us, gpu.read_rg32f(&flows.images[0])?, gpu.read_r32f(&fp.levels[0])?, gpu.read_r32f(&fc.levels[0])?))
    })?;
    pass_us.push(("blend + fixup".to_string(), blend_us));
    Ok(BatteryPair { old: unpack_image(f, &old), new_rgba: unpack_image(f, &new), new, flow, prev_luma, cur_luma, flow_us, blend_us, counts, pass_us })
}

fn battery_parity_case(gpu: &Gpu, f: PackedFormat, width: u32, height: u32, t: f32) -> Result<CaseResult, String> {
    let square = checker(96, 96, 6);
    let mut frames = pan_frames(width, height, 12, -8, 1, 201);
    for fr in &mut frames {
        paste(fr, &square, (592, 352));
    }
    let pair = battery_pair_at(gpu, f, &frames[0], &frames[1], t)?;
    let (pp, cp) = (packed::to_packed(f, &frames[0]), packed::to_packed(f, &frames[1]));
    let (expected, _) = haqengine_core::blend::battery_blend_packed_reference(f, &pp, &cp, &pair.flow, &pair.prev_luma, &pair.cur_luma, t);
    Ok(CaseResult { report: packed::compare_packed(f, &expected, &pair.new, 1.0), gpu_us: pair.blend_us, tolerance: 1.0 })
}

/// GPU counts and output must equal the CPU reference fed with the GPU's flow and luma.
fn battery_counts_parity(gpu: &Gpu, prev: &Image<Rgba8>, cur: &Image<Rgba8>) -> Result<CompareReport, String> {
    let f = BF;
    let pair = battery_pair(gpu, f, prev, cur)?;
    let (pp, cp) = (packed::to_packed(f, prev), packed::to_packed(f, cur));
    let (expected, counts) = haqengine_core::blend::battery_blend_packed_reference(f, &pp, &cp, &pair.flow, &pair.prev_luma, &pair.cur_luma, 0.5);
    if counts != pair.counts {
        return Err(format!("counts: GPU {:?}, CPU {counts:?}", pair.counts));
    }
    Ok(packed::compare_packed(f, &expected, &pair.new, 1.0))
}

fn battery_counts_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let mut reports = Vec::new();
    for (w, h) in [(1280u32, 800u32), (1283, 797)] {
        let pan = pan_frames(w, h, 12, -8, 1, 261);
        reports.push(battery_counts_parity(gpu, &pan[0], &pan[1])?);
        reports.push(battery_counts_parity(gpu, &pan[0], &gradient(w, h))?);
        reports.push(battery_counts_parity(gpu, &pan[0], &pan[0])?);
    }
    Ok(CaseResult { report: merge(&reports), gpu_us: 0.0, tolerance: 1.0 })
}

fn battery_cut_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let a = pan_frames(1280, 800, 12, -8, 1, 271).remove(0);
    let b = gradient(1280, 800);
    let pair = battery_pair(gpu, BF, &a, &b)?;
    println!("      cut: {:?}", pair.counts);
    if !haqengine_core::frame_change::is_cut(&pair.counts) {
        return Err(format!("not detected as a cut: {:?}", pair.counts));
    }
    let differs = pair.new_rgba.data.iter().zip(&b.data).filter(|(o, c)| o[..3] != c[..3]).count();
    if differs > 0 {
        return Err(format!("cut: {differs} pixels are not the new frame"));
    }
    Ok(battery_result(pair.blend_us))
}

const THIRDS: [f32; 2] = [1.0 / 3.0, 2.0 / 3.0];

fn battery_3x_parity_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let f = BF;
    let mut reports = Vec::new();
    for (w, h) in [(1280u32, 800u32), (1283, 797)] {
        let pan = pan_frames(w, h, 12, -8, 1, 311);
        let m = battery_multi(gpu, f, &pan[0], &pan[1], &THIRDS)?;
        let (pp, cp) = (packed::to_packed(f, &pan[0]), packed::to_packed(f, &pan[1]));
        for (k, (out, counts)) in m.outputs.iter().enumerate() {
            let (expected, want) = haqengine_core::blend::battery_blend_packed_reference(f, &pp, &cp, &m.flow, &m.prev_luma, &m.cur_luma, THIRDS[k]);
            if *counts != want {
                return Err(format!("{w}x{h} output {k}: counts GPU {counts:?}, CPU {want:?}"));
            }
            reports.push(packed::compare_packed(f, &expected, out, 1.0));
        }
    }
    Ok(CaseResult { report: merge(&reports), gpu_us: 0.0, tolerance: 1.0 })
}

/// A pan with a 192×192 white flash in the new frame (an ultimate's burst at 1280×800 scale).
fn flash_frames(width: u32, height: u32) -> Vec<Image<Rgba8>> {
    let mut frames = pan_frames(width, height, 8, 0, 1, 321);
    paste(&mut frames[1], &Image::new(192, 192, [255u8, 255, 255, 255]), (544, 304));
    frames
}

/// Inside a region without matching motion each generated frame shows the nearer real frame.
fn battery_fallback_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let frames = flash_frames(1280, 800);
    let m = battery_multi(gpu, BF, &frames[0], &frames[1], &THIRDS)?;
    let inside = |x: u32, y: u32| (568..712).contains(&x) && (328..472).contains(&y);
    let mut reports = Vec::new();
    for (k, nearer) in [&frames[0], &frames[1]].into_iter().enumerate() {
        let got = unpack_image(BF, &m.outputs[k].0);
        let err = mean_err(&got, nearer, inside);
        println!("      output {k}: mean error to the nearer real frame {err:.3}");
        if err > 1.0 {
            return Err(format!("output {k}: {err:.3} from the nearer real frame inside the flash"));
        }
        reports.push(packed::compare_packed(BF, &packed::to_packed(BF, nearer), &m.outputs[k].0, 255.0));
    }
    Ok(CaseResult { report: merge(&reports), gpu_us: m.blend_us, tolerance: 255.0 })
}

/// GPU output and tile counts equal the CPU reference on the flash scene, at 2x and 3x and at an
/// odd size whose edge tiles are partial.
fn battery_fallback_parity_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let f = BF;
    let tile = haqengine_core::blend::TILE;
    let mut reports = Vec::new();
    for (w, h) in [(1280u32, 800u32), (1283, 797)] {
        let frames = flash_frames(w, h);
        let (pp, cp) = (packed::to_packed(f, &frames[0]), packed::to_packed(f, &frames[1]));
        for times in [&[0.5f32][..], &THIRDS[..]] {
            let m = battery_multi(gpu, f, &frames[0], &frames[1], times)?;
            let band = h.div_ceil(tile);
            for (k, (out, counts)) in m.outputs.iter().enumerate() {
                let (expected, want, tiles) =
                    haqengine_core::blend::battery_blend_packed_with_tiles(f, &pp, &cp, &m.flow, &m.prev_luma, &m.cur_luma, times[k]);
                if *counts != want {
                    return Err(format!("{w}x{h} t={} counts GPU {counts:?}, CPU {want:?}", times[k]));
                }
                for ty in 0..band {
                    for tx in 0..tiles.width {
                        let (gpu_n, cpu_n) = (m.tiles.get(tx, ty + k as u32 * band), tiles.get(tx, ty));
                        if gpu_n != cpu_n {
                            return Err(format!("{w}x{h} t={} tile ({tx}, {ty}): GPU {gpu_n}, CPU {cpu_n}", times[k]));
                        }
                    }
                }
                reports.push(packed::compare_packed(f, &expected, out, 1.0));
            }
        }
    }
    Ok(CaseResult { report: merge(&reports), gpu_us: 0.0, tolerance: 1.0 })
}

fn battery_3x_cut_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let a = pan_frames(1280, 800, 12, -8, 1, 271).remove(0);
    let b = gradient(1280, 800);
    let m = battery_multi(gpu, BF, &a, &b, &THIRDS)?;
    let want = packed::to_packed(BF, &b);
    for (k, (out, counts)) in m.outputs.iter().enumerate() {
        if !haqengine_core::frame_change::is_cut(counts) {
            return Err(format!("output {k} not a cut: {counts:?}"));
        }
        let differs = out.data.iter().zip(&want.data).filter(|(o, c)| o != c).count();
        if differs > 0 {
            return Err(format!("output {k}: {differs} pixels are not the new frame"));
        }
    }
    Ok(battery_result(m.blend_us))
}

fn battery_3x_rows_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let pan = pan_frames(1280, 800, 12, -8, 1, 321);
    let one = battery_multi(gpu, BF, &pan[0], &pan[1], &THIRDS[..1])?;
    let two = battery_multi(gpu, BF, &pan[0], &pan[1], &THIRDS)?;
    if one.outputs[0].1 != two.outputs[0].1 {
        return Err(format!("row 0 changed by a second blend: {:?} vs {:?}", one.outputs[0].1, two.outputs[0].1));
    }
    Ok(battery_result(two.blend_us))
}

fn battery_3x_total_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let pan = pan_frames(1280, 800, 12, -8, 1, 331);
    let two = battery_multi(gpu, BF, &pan[0], &pan[1], &[0.5])?;
    let three = battery_multi(gpu, BF, &pan[0], &pan[1], &THIRDS)?;
    println!(
        "      battery 2x total {:.0} us, 3x total {:.0} us (budget 3500 us)",
        two.flow_us + two.blend_us,
        three.flow_us + three.blend_us
    );
    Ok(battery_result(three.flow_us + three.blend_us))
}

fn battery_flash_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let a = pan_frames(1280, 800, 0, 0, 1, 281).remove(0);
    let mut b = a.clone();
    for y in 0..800 {
        for x in 768..1280 {
            let p = a.get(x, y);
            b.set(x, y, [p[0].saturating_add(80), p[1].saturating_add(80), p[2].saturating_add(80), 255]);
        }
    }
    let pair = battery_pair(gpu, BF, &a, &b)?;
    println!("      flash: {:?}", pair.counts);
    if haqengine_core::frame_change::is_cut(&pair.counts) {
        return Err(format!("a 40% flash counted as a cut: {:?}", pair.counts));
    }
    Ok(battery_result(pair.blend_us))
}

fn battery_still_pan_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let frames = pan_frames(1280, 800, 1, 0, 1, 291);
    let pair = battery_pair(gpu, BF, &frames[0], &frames[1])?;
    println!("      1 px pan: {:?}", pair.counts);
    if haqengine_core::frame_change::stays_still(&pair.counts) {
        return Err(format!("a 1 px pan counted as still: {:?}", pair.counts));
    }
    let same = battery_pair(gpu, BF, &frames[0], &frames[0])?;
    if !haqengine_core::frame_change::is_still(&same.counts) {
        return Err(format!("identical frames not still: {:?}", same.counts));
    }
    Ok(battery_result(pair.blend_us))
}

fn battery_result(us: f64) -> CaseResult {
    CaseResult { report: CompareReport { max_diff: 0.0, worst: None, over_tolerance: 0 }, gpu_us: us, tolerance: 1.0 }
}

fn battery_compare(what: &str, new: f64, old: f64, bar: f64, slack: f64) -> Result<(), String> {
    println!("      {what}: new blend {new:.3}, M3c warp {old:.3} (8-bit steps)");
    if new.is_nan() || new > bar || new > old + slack {
        return Err(format!("{what}: new {new:.3} vs old {old:.3}, bar {bar} and old + {slack}"));
    }
    Ok(())
}

const BF: PackedFormat = PackedFormat::Bgra8;

fn battery_pan_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let frames = pan_frames(1280, 800, 12, -8, 1, 211);
    let truth = pan_at(1280, 800, 12, -8, 1, 211, 1);
    let pair = battery_pair(gpu, BF, &frames[0], &frames[1])?;
    let keep = |x: u32, y: u32| x >= 48 && y >= 48 && x + 48 < 1280 && y + 48 < 800;
    battery_compare("pan", mean_err(&pair.new_rgba, &truth, keep), mean_err(&pair.old, &truth, keep), 2.0, 0.5)?;
    Ok(battery_result(pair.blend_us))
}

fn battery_overlay_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let mut bar = Image::new(200, 40, [255u8, 255, 255, 255]);
    for y in 0..40 {
        for x in (0..200).filter(|x| x % 8 < 2) {
            bar.set(x, y, [0, 0, 0, 255]);
        }
    }
    let at = (100u32, 700u32);
    let mut frames = pan_frames(1280, 800, 12, -8, 1, 221);
    for f in &mut frames {
        paste(f, &bar, at);
    }
    let mut truth = pan_at(1280, 800, 12, -8, 1, 221, 1);
    paste(&mut truth, &bar, at);
    let pair = battery_pair(gpu, BF, &frames[0], &frames[1])?;
    let on_bar = |x: u32, y: u32| x >= at.0 && x < at.0 + 200 && y >= at.1 && y < at.1 + 40;
    battery_compare("overlay", mean_err(&pair.new_rgba, &truth, on_bar), mean_err(&pair.old, &truth, on_bar), 5.0, 0.0)?;
    Ok(battery_result(pair.blend_us))
}

fn battery_square_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let square = checker(96, 96, 6);
    let at = (592u32, 352u32);
    let mut frames = pan_frames(1280, 800, 16, 0, 1, 231);
    for f in &mut frames {
        paste(f, &square, at);
    }
    let mut truth = pan_at(1280, 800, 16, 0, 1, 231, 1);
    paste(&mut truth, &square, at);
    let pair = battery_pair(gpu, BF, &frames[0], &frames[1])?;
    let dist = |x: u32, y: u32| {
        let dx = (at.0 as i32 - x as i32).max(x as i32 - (at.0 + 95) as i32).max(0);
        let dy = (at.1 as i32 - y as i32).max(y as i32 - (at.1 + 95) as i32).max(0);
        dx.max(dy)
    };
    let inside = |x: u32, y: u32| x >= at.0 + 8 && x < at.0 + 88 && y >= at.1 + 8 && y < at.1 + 88;
    let band = |x: u32, y: u32| !inside(x, y) && dist(x, y) < 24;
    battery_compare("square", mean_err(&pair.new_rgba, &truth, inside), mean_err(&pair.old, &truth, inside), 2.0, 0.5)?;
    battery_compare("edge band", mean_err(&pair.new_rgba, &truth, band), mean_err(&pair.old, &truth, band), 255.0, 0.5)?;
    Ok(battery_result(pair.blend_us))
}

fn battery_slow_object_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let head = noise(40, 40, 181);
    let mut frames = pan_frames(1280, 800, 24, 0, 1, 241);
    for (k, f) in frames.iter_mut().enumerate() {
        paste(f, &head, (600 + 2 * k as u32, 380));
    }
    let mut truth = pan_at(1280, 800, 24, 0, 1, 241, 1);
    paste(&mut truth, &head, (601, 380));
    let pair = battery_pair(gpu, BF, &frames[0], &frames[1])?;
    let on_head = |x: u32, y: u32| (603..637).contains(&x) && (382..418).contains(&y);
    let rim = |x: u32, y: u32| !on_head(x, y) && x + 8 >= 601 && x < 649 && y + 8 >= 380 && y < 428;
    battery_compare("slow head", mean_err(&pair.new_rgba, &truth, on_head), mean_err(&pair.old, &truth, on_head), 255.0, 0.5)?;
    battery_compare("slow head rim", mean_err(&pair.new_rgba, &truth, rim), mean_err(&pair.old, &truth, rim), 255.0, 5.0)?;
    Ok(battery_result(pair.blend_us))
}

/// A 12×48 bar standing still over a 24 px/frame pan, a little brighter in the new frame (an ear
/// turning in front of a fast background): it must stay a bar, not take the background's motion.
fn thin_bar_frames() -> Vec<Image<Rgba8>> {
    let mut frames = pan_frames(1280, 800, 24, 0, 1, 251);
    for (k, f) in frames.iter_mut().enumerate() {
        let v = 204 + 6 * k as u8;
        paste(f, &Image::new(12, 48, [v, v, v, 255]), (600, 380));
    }
    frames
}

fn dim(img: &Image<Rgba8>) -> Image<Rgba8> {
    let mut out = img.clone();
    for y in 0..img.height {
        for x in 0..img.width {
            let v = 124 + img.get(x, y)[0] % 9;
            out.set(x, y, [v, v, v, 255]);
        }
    }
    out
}

fn battery_thin_bar_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let at = (600u32, 380u32);
    let frames = thin_bar_frames();
    let mut truth = pan_at(1280, 800, 24, 0, 1, 251, 1);
    paste(&mut truth, &Image::new(12, 48, [207u8, 207, 207, 255]), at);
    let pair = battery_pair(gpu, BF, &frames[0], &frames[1])?;
    let on_bar = |x: u32, y: u32| (at.0 + 3..at.0 + 9).contains(&x) && (at.1 + 6..at.1 + 42).contains(&y);
    battery_compare("thin bar", mean_err(&pair.new_rgba, &truth, on_bar), mean_err(&pair.old, &truth, on_bar), 8.0, 0.5)?;
    Ok(battery_result(pair.blend_us))
}

/// A low-contrast pan (luma 124..132, 12 px/frame): every candidate costs little, and the blend
/// must still move it rather than treat it as still.
fn battery_dim_pan_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let frames: Vec<Image<Rgba8>> = pan_frames(1280, 800, 12, 0, 1, 261).iter().map(dim).collect();
    let truth = dim(&pan_at(1280, 800, 12, 0, 1, 261, 1));
    let pair = battery_pair(gpu, BF, &frames[0], &frames[1])?;
    let keep = |x: u32, y: u32| x >= 48 && y >= 48 && x + 48 < 1280 && y + 48 < 800;
    battery_compare("dim pan", mean_err(&pair.new_rgba, &truth, keep), mean_err(&pair.old, &truth, keep), 2.0, 0.5)?;
    Ok(battery_result(pair.blend_us))
}

/// GPU output equals the CPU reference on scenes where the smallest-motion rule decides pixels.
fn battery_rule_parity_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let bar = thin_bar_frames();
    let dim_pan: Vec<Image<Rgba8>> = pan_frames(1280, 800, 12, 0, 1, 261).iter().map(dim).collect();
    let mut reports = Vec::new();
    for (frames, t) in [(&bar, 0.5f32), (&bar, 1.0 / 3.0), (&dim_pan, 0.5)] {
        let pair = battery_pair_at(gpu, BF, &frames[0], &frames[1], t)?;
        let (pp, cp) = (packed::to_packed(BF, &frames[0]), packed::to_packed(BF, &frames[1]));
        let (expected, _) = haqengine_core::blend::battery_blend_packed_reference(BF, &pp, &cp, &pair.flow, &pair.prev_luma, &pair.cur_luma, t);
        reports.push(packed::compare_packed(BF, &expected, &pair.new, 1.0));
    }
    Ok(CaseResult { report: merge(&reports), gpu_us: 0.0, tolerance: 1.0 })
}

fn battery_ghost_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let bg = Image::new(1280, 800, [200u8, 200, 200, 255]);
    let square = checker(200, 200, 10);
    let frames: Vec<Image<Rgba8>> = (0..2)
        .map(|k| {
            let mut f = bg.clone();
            paste(&mut f, &square, (200 + 16 * k, 300));
            f
        })
        .collect();
    let pair = battery_pair(gpu, BF, &frames[0], &frames[1])?;
    let (mut ghosts, mut total) = (0usize, 0usize);
    for y in 300..500 {
        for x in 0..1280 {
            if frames[0].get(x, y) == frames[1].get(x, y) {
                continue;
            }
            total += 1;
            let o = pair.new_rgba.get(x, y);
            let near = |v: u8| (0..3).all(|i| o[i].abs_diff(v) <= 8);
            ghosts += (!near(0) && !near(200) && !near(255)) as usize;
        }
    }
    println!("      ghosts: {ghosts}/{total} changed pixels are a blend");
    if total == 0 || ghosts * 1000 > total {
        return Err(format!("ghosts: {ghosts}/{total}, at most 0.1%"));
    }
    Ok(battery_result(pair.blend_us))
}

/// A menu-like still screen (light text bars on a dark panel) whose brightness drops 10%: every
/// output pixel stays between the two frames' values, within 2 steps.
fn battery_menu_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let mut menu = Image::new(1280, 800, [30u8, 34, 48, 255]);
    for row in 0..12u32 {
        for y in 120 + row * 48..120 + row * 48 + 18 {
            for x in 160..(160 + 300 + (row * 37) % 400) {
                let v = if (x / 6 + y / 3) % 3 == 0 { 120 } else { 235 };
                menu.set(x, y, [v, v, v, 255]);
            }
        }
    }
    let dim = Image {
        width: 1280,
        height: 800,
        data: menu.data.iter().map(|p| [0, 1, 2].map(|i| (p[i] as u16 * 9 / 10) as u8)).map(|c| [c[0], c[1], c[2], 255]).collect(),
    };
    let pair = battery_pair(gpu, BF, &menu, &dim)?;
    let mut outside = 0usize;
    for y in 0..800 {
        for x in 0..1280 {
            let (a, b, o) = (menu.get(x, y), dim.get(x, y), pair.new_rgba.get(x, y));
            outside += (0..3).any(|i| o[i].saturating_add(2) < a[i].min(b[i]) || o[i] > a[i].max(b[i]).saturating_add(2)) as usize;
        }
    }
    let old_outside = (0..800u32)
        .flat_map(|y| (0..1280u32).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let (a, b, o) = (menu.get(x, y), dim.get(x, y), pair.old.get(x, y));
            (0..3).any(|i| o[i].saturating_add(2) < a[i].min(b[i]) || o[i] > a[i].max(b[i]).saturating_add(2))
        })
        .count();
    println!("      menu fade: {outside} pixels outside the two frames (M3c warp {old_outside})");
    if outside > 0 {
        return Err(format!("menu fade: {outside} pixels outside the two frames, M3c warp {old_outside}"));
    }
    Ok(battery_result(pair.blend_us))
}

/// Reports Battery's GPU time with the new blend against the budget; a report, not a gate (the
/// in-game `gpu` stat decides).
fn battery_total_case(gpu: &Gpu) -> Result<CaseResult, String> {
    let frames = pan_frames(1280, 800, 12, -8, 1, 251);
    let pair = battery_pair(gpu, BF, &frames[0], &frames[1])?;
    let total = pair.flow_us + pair.blend_us;
    println!("      battery with the new blend: total {total:.1} us (budget 2500 us): flow {:.0}, blend {:.0}", pair.flow_us, pair.blend_us);
    let parts: Vec<String> = pair.pass_us.iter().map(|(n, t)| format!("{n} {t:.0}")).collect();
    println!("      battery passes: {}", parts.join(", "));
    Ok(battery_result(total))
}

fn luma_half_case(gpu: &Gpu, f: PackedFormat, img: &Image<Rgba8>) -> Result<CaseResult, String> {
    let input = packed::to_packed(f, img);
    let expected = haqengine_core::fsr3_flow::luma_half_reference(f, &input);
    let src = gpu.upload_u32(&input)?;
    let dst = match gpu.create_storage_image(vk::Format::R8_UINT, expected.width, expected.height) {
        Ok(d) => d,
        Err(e) => {
            gpu.destroy_image(src);
            return Err(e);
        }
    };
    let result = (|| {
        gpu.clear_u32(&dst, 0xAB)?;
        let push = push_bytes(&[img.width.to_le_bytes(), img.height.to_le_bytes(), f.code().to_le_bytes()]);
        let us = gpu.run_fsr3(&haqengine_shaders::OF_LUMA_HALF, &[(0, Bind::Image(&src)), (1, Bind::Image(&dst))], &[], &push, [
            expected.width.div_ceil(8),
            expected.height.div_ceil(8),
            1,
        ])?;
        let got = gpu.read_r8(&dst)?;
        let bad = expected.data.iter().zip(&got.data).filter(|(a, b)| a != b).count() as u32;
        let max = expected.data.iter().zip(&got.data).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        Ok(CaseResult { report: CompareReport { max_diff: max as f32, worst: None, over_tolerance: bad }, gpu_us: us, tolerance: 0.0 })
    })();
    gpu.destroy_image(src);
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
            flow_case(g, &prev, &shift(&prev, -6, 4), &[0, 1, 2, 3], Some([-3.0, 2.0]), false)
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
    cases.extend([
        case("luma packed bgra8 1280x800", |g| {
            luma_packed_case(g, PackedFormat::Bgra8, &packed::to_packed(PackedFormat::Bgra8, &noise(1280, 800, 41)))
        }),
        case("luma packed rgba8 1281x801", |g| {
            luma_packed_case(g, PackedFormat::Rgba8, &packed::to_packed(PackedFormat::Rgba8, &gradient(1281, 801)))
        }),
        case("luma packed a2b10g10r10 1280x800", |g| {
            let f = PackedFormat::A2b10g10r10;
            luma_packed_case(g, f, &packed::to_packed(f, &noise(1280, 800, 42)))
        }),
        case("warp packed bgra8 t=0.5", |g| {
            let f = PackedFormat::Bgra8;
            let prev = noise(1280, 800, 43);
            let flow = Image::new(80, 50, [4.0f32, 0.0]);
            warp_packed_case(g, f, &packed::to_packed(f, &prev), &packed::to_packed(f, &shift(&prev, 8, 0)), &flow, &[0.5], 1.0)
        }),
        case("warp packed a2b10g10r10 t=0.5", |g| {
            let f = PackedFormat::A2b10g10r10;
            let prev = noise(1280, 800, 44);
            let flow = Image::new(80, 50, [3.0f32, -2.0]);
            warp_packed_case(g, f, &packed::to_packed(f, &prev), &packed::to_packed(f, &noise(1280, 800, 45)), &flow, &[0.5], 1.0)
        }),
        case("warp packed endpoints", |g| {
            let f = PackedFormat::A2b10g10r10;
            let flow = Image::new(80, 50, [5.0f32, 3.0]);
            let (a, b) = (packed::to_packed(f, &noise(1280, 800, 46)), packed::to_packed(f, &noise(1280, 800, 47)));
            warp_packed_case(g, f, &a, &b, &flow, &[0.0, 1.0], 0.0)
        }),
        case("battery packed pipeline bgra8 total", battery_packed_case),
        case("warp still square over moving background", |g| {
            let (prev, cur, flow) = still_square_scene();
            warp_case(g, &prev, &cur, &flow, &[0.5], BLEND_TOLERANCE)
        }),
        case("warp_blend 1281x801 t=0.5 varied flow", |g| {
            let prev = noise(1281, 801, 52);
            warp_case(g, &prev, &shift(&prev, -6, 4), &varied_flow(1281, 801), &[0.5], BLEND_TOLERANCE)
        }),
        case("warp packed bgra8 1281x801 t=0.5 varied flow", |g| {
            let f = PackedFormat::Bgra8;
            let prev = noise(1281, 801, 53);
            let cur = shift(&prev, -6, 4);
            warp_packed_case(g, f, &packed::to_packed(f, &prev), &packed::to_packed(f, &cur), &varied_flow(1281, 801), &[0.5], 1.0)
        }),
    ]);
    cases.extend([
        case("of_luma_half matches CPU", |g| {
            luma_half_case(g, PackedFormat::Bgra8, &noise(1281, 801, 91))?;
            luma_half_case(g, PackedFormat::A2b10g10r10, &noise(1281, 801, 92))?;
            luma_half_case(g, PackedFormat::Rgba8, &gradient(1280, 800))
        }),
        case("fsr3 flow 1280x800 shift(12,-8)", |g| fsr3_shift_case(g, 1280, 800, 12, -8, FSR3_FRAMES)),
        case("fsr3 flow 1281x801 shift(-6,4)", |g| fsr3_shift_case(g, 1281, 801, -6, 4, FSR3_FRAMES)),
        case("fsr3 flow large shift 96px", |g| fsr3_shift_case(g, 1280, 800, 96, 0, FSR3_FRAMES)),
        case("fsr3 flow flat black", fsr3_black_case),
        case("fsr3 flow still square over moving background", fsr3_square_case),
        case("fsr3 flow second frame parity", |g| fsr3_shift_case(g, 1280, 800, 12, -8, FSR3_FRAMES + 1)),
        case("fsr3 scene cut detection", fsr3_scene_cut_case),
        case("fsr3 flow pipeline 1280x800 total", fsr3_pipeline_case),
        case("fsr3 flow odd shift(7,-3)", fsr3_odd_case),
        case("fsr3 flow memory within budget", fsr3_memory_case),
        case("fi_blend matches CPU", blend_parity_case),
        case("fi_blend_packed matches CPU bgra8", |g| blend_packed_case(g, PackedFormat::Bgra8)),
        case("fi_blend_packed matches CPU a2b10g10r10", |g| blend_packed_case(g, PackedFormat::A2b10g10r10)),
        case("fsr3 interp first frames after reset", interp_reset_case),
        case("fsr3 interp memory within budget", interp_memory_case),
        case("fsr3 interp pan", |g| interp_pan_case(g, 1280, 800, 12, -8)),
        case("fsr3 interp 1281x801 pan", |g| interp_pan_case(g, 1281, 801, -6, 4)),
        case("fsr3 interp still square over moving background", interp_square_case),
        case("fsr3 interp static overlay", interp_overlay_case),
        case("fsr3 interp revealed area", interp_revealed_case),
        case("fsr3 interp flat black", interp_black_case),
        case("fsr3 interp pipeline total", interp_pipeline_case),
        case("fsr3 interp scene cut takes current frame", interp_cut_case),
        case("fsr3 interp no ghosts behind a moving object", interp_ghost_case),
        case("battery blend matches CPU bgra8", |g| battery_parity_case(g, PackedFormat::Bgra8, 1280, 800, 0.5)),
        case("battery blend matches CPU a2b10g10r10", |g| battery_parity_case(g, PackedFormat::A2b10g10r10, 1280, 800, 0.5)),
        case("battery blend matches CPU 1283x797", |g| battery_parity_case(g, PackedFormat::Bgra8, 1283, 797, 0.5)),
        case("battery blend matches CPU t=0.25 and 0.75", |g| {
            battery_parity_case(g, PackedFormat::Bgra8, 1283, 797, 0.25)?;
            battery_parity_case(g, PackedFormat::Rgba8, 1280, 800, 0.75)
        }),
        case("battery blend pan", battery_pan_case),
        case("battery blend static overlay", battery_overlay_case),
        case("battery blend still square", battery_square_case),
        case("battery blend small slow object", battery_slow_object_case),
        case("battery blend thin bar over a pan", battery_thin_bar_case),
        case("battery blend dim pan", battery_dim_pan_case),
        case("battery blend matches CPU where motion size decides", battery_rule_parity_case),
        case("battery fallback takes the nearer frame", battery_fallback_case),
        case("battery fallback matches CPU", battery_fallback_parity_case),
        case("battery blend no ghosts", battery_ghost_case),
        case("battery blend menu fade", battery_menu_case),
        case("battery blend pipeline total", battery_total_case),
        case("battery counts match CPU", battery_counts_case),
        case("battery cut takes the new frame", battery_cut_case),
        case("battery 3x outputs match CPU", battery_3x_parity_case),
        case("battery 3x cut takes the new frame in both outputs", battery_3x_cut_case),
        case("battery 3x rows stay separate", battery_3x_rows_case),
        case("battery 3x pipeline total", battery_3x_total_case),
        case("battery cut flash is not a cut", battery_flash_case),
        case("battery still counts on a 1 px pan", battery_still_pan_case),
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
    fn fsr3_requirements_reject_unsafe_devices() {
        assert!(fsr3_requirements(true, 64, true, true).is_ok());
        assert!(fsr3_requirements(true, 32, true, true).is_ok());
        assert!(fsr3_requirements(false, 64, true, true).is_err());
        assert!(fsr3_requirements(true, 16, true, true).is_err());
        assert!(fsr3_requirements(true, 64, false, true).is_err());
        assert!(fsr3_requirements(true, 64, true, false).is_err());
    }

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
            "luma packed bgra8 1280x800",
            "luma packed rgba8 1281x801",
            "luma packed a2b10g10r10 1280x800",
            "warp packed bgra8 t=0.5",
            "warp packed a2b10g10r10 t=0.5",
            "warp packed endpoints",
            "battery packed pipeline bgra8 total",
            "warp still square over moving background",
            "warp_blend 1281x801 t=0.5 varied flow",
            "warp packed bgra8 1281x801 t=0.5 varied flow",
            "fsr3 flow 1280x800 shift(12,-8)",
            "fsr3 flow 1281x801 shift(-6,4)",
            "fsr3 flow large shift 96px",
            "fsr3 flow flat black",
            "fsr3 flow still square over moving background",
            "fsr3 flow second frame parity",
            "fsr3 scene cut detection",
            "fsr3 flow pipeline 1280x800 total",
            "of_luma_half matches CPU",
            "fsr3 flow odd shift(7,-3)",
            "fsr3 flow memory within budget",
            "fi_blend matches CPU",
            "fsr3 interp first frames after reset",
            "fsr3 interp memory within budget",
            "fsr3 interp pan",
            "fsr3 interp 1281x801 pan",
            "fsr3 interp still square over moving background",
            "fsr3 interp static overlay",
            "fsr3 interp revealed area",
            "fsr3 interp flat black",
            "fsr3 interp pipeline total",
            "fsr3 interp scene cut takes current frame",
            "fsr3 interp no ghosts behind a moving object",
            "fi_blend_packed matches CPU bgra8",
            "fi_blend_packed matches CPU a2b10g10r10",
            "battery blend matches CPU bgra8",
            "battery blend matches CPU a2b10g10r10",
            "battery blend matches CPU 1283x797",
            "battery blend matches CPU t=0.25 and 0.75",
            "battery blend pan",
            "battery blend static overlay",
            "battery blend still square",
            "battery blend small slow object",
            "battery blend thin bar over a pan",
            "battery blend dim pan",
            "battery blend matches CPU where motion size decides",
            "battery fallback takes the nearer frame",
            "battery fallback matches CPU",
            "battery blend no ghosts",
            "battery blend menu fade",
            "battery blend pipeline total",
            "battery counts match CPU",
            "battery cut takes the new frame",
            "battery 3x outputs match CPU",
            "battery 3x cut takes the new frame in both outputs",
            "battery 3x rows stay separate",
            "battery 3x pipeline total",
            "battery cut flash is not a cut",
            "battery still counts on a 1 px pan",
        ] {
            assert!(names.contains(&required), "missing case {required}");
        }
        assert!(all_cases(true).iter().any(|c| c.name.contains("must fail")));
    }
}
