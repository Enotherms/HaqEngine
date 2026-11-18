//! The GPU test cases: each runs one shader on a test image and compares with the CPU reference.

use crate::fsr3::run_sequence;
use crate::gpu::{push_bytes, Gpu, StorageImage};
use ash::vk;
use haqengine_core::flow::{self, Flow, BLOCK, LEVELS, RADIUS_COARSE, RADIUS_FINE};
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
            &WARP_BLEND_PACKED,
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
    fsr3_requirements(gpu.subgroup_ok, gpu.subgroup_size, gpu.extended_formats, gpu.robust_images)
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
        let us = gpu.run_fsr3(&haqengine_shaders::OF_LUMA_HALF, &[(0, &src), (1, &dst)], &[], &push, [
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
        ] {
            assert!(names.contains(&required), "missing case {required}");
        }
        assert!(all_cases(true).iter().any(|c| c.name.contains("must fail")));
    }
}
