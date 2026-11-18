//! FSR 3 frame interpolation on the GPU after the half-resolution flow. Every image, the counters
//! buffer and both schedules (reset and normal) are created once; the two real frames swap roles
//! by parity instead of being copied.

use crate::fsr3::Fsr3Flow;
use crate::gpu::{Bind, Gpu, GpuBuffer, StorageImage};
use ash::vk;
use haqengine_core::fsr3_flow::OfRes;
use haqengine_core::fsr3_interp::{
    fi_constants, fi_prepare_dispatch, fi_schedule, fi_sizes, set_fi_frame, FiDispatch, FiParams, FiRes, FiSizes,
    FI_CONSTANTS_SIZE,
};
use haqengine_core::packed::{self, PackedFormat};
use haqengine_core::{Image, Rgba8};
use haqengine_shaders::{BindingKind, FI_PASSES};

const DELTA_TIME_MS: f32 = 33.3;

/// GPU time of one frame: the flow, and each interpolation pass, in µs.
#[derive(Debug, Clone, Copy, Default)]
pub struct InterpTimes {
    pub flow: f64,
    pub passes: [f64; 10],
}

pub struct Fsr3Interp {
    pub width: u32,
    pub height: u32,
    format: PackedFormat,
    flow: Fsr3Flow,
    images: Vec<(FiRes, StorageImage)>,
    colors: Vec<StorageImage>,
    counters: Option<GpuBuffer>,
    schedules: [Vec<FiDispatch>; 2],
    constants: [u8; FI_CONSTANTS_SIZE],
    frames: u32,
    pub times: InterpTimes,
}

/// An image interpolation owns: what it holds, its format, size, mip count and initial 32-bit fill.
type ImageSpec = (FiRes, vk::Format, (u32, u32), u32, u32);

fn owned_images(s: &FiSizes) -> Vec<ImageSpec> {
    let one = 1.0f32.to_bits();
    vec![
        (FiRes::Output, vk::Format::R8G8B8A8_UNORM, s.display, 1, 0),
        (FiRes::DilatedDepth, vk::Format::R32_SFLOAT, s.render, 1, one),
        (FiRes::DilatedMotion, vk::Format::R16G16_SFLOAT, s.render, 1, 0),
        (FiRes::ReconstructedPrevDepth, vk::Format::R32_UINT, s.render, 1, one),
        (FiRes::ReconstructedInterpDepth, vk::Format::R32_UINT, s.render, 1, one),
        (FiRes::GameFieldX, vk::Format::R32_UINT, s.render, 1, 0),
        (FiRes::GameFieldY, vk::Format::R32_UINT, s.render, 1, 0),
        (FiRes::FlowFieldX, vk::Format::R32_UINT, s.render, 1, 0),
        (FiRes::FlowFieldY, vk::Format::R32_UINT, s.render, 1, 0),
        (FiRes::Disocclusion, vk::Format::R8G8_UNORM, s.render, 1, 0),
        (FiRes::Pyramid, vk::Format::R16G16B16A16_SFLOAT, s.pyramid, s.pyramid_mips, 0),
        (FiRes::Distortion, vk::Format::R8G8_UNORM, (1, 1), 1, 0),
        (FiRes::Confidence, vk::Format::R32_UINT, (1, 1), 1, 0),
    ]
}

impl Fsr3Interp {
    pub fn new(gpu: &Gpu, width: u32, height: u32, format: PackedFormat) -> Result<Fsr3Interp, String> {
        let sizes = fi_sizes(width, height);
        let flow = Fsr3Flow::new(gpu, width, height, format)?;
        let params = |f| FiParams { frame_since_reset: f, delta_time_ms: DELTA_TIME_MS };
        let mut me = Fsr3Interp {
            width,
            height,
            format,
            flow,
            images: Vec::new(),
            colors: Vec::new(),
            counters: None,
            schedules: [fi_schedule(&sizes, &params(0)), fi_schedule(&sizes, &params(1))],
            constants: fi_constants(&sizes, &params(0)),
            frames: 0,
            times: InterpTimes::default(),
        };
        if let Err(e) = me.allocate(gpu, &sizes) {
            me.destroy(gpu);
            return Err(e);
        }
        Ok(me)
    }

    fn allocate(&mut self, gpu: &Gpu, sizes: &FiSizes) -> Result<(), String> {
        for (res, format, (w, h), mips, fill) in owned_images(sizes) {
            let img = gpu.create_image(format, w, h, mips)?;
            let filled = gpu.clear_u32(&img, fill);
            self.images.push((res, img));
            filled?;
        }
        for _ in 0..2 {
            let img = gpu.create_storage_image(vk::Format::R8G8B8A8_UNORM, sizes.display.0, sizes.display.1)?;
            let cleared = gpu.clear_u32(&img, 0);
            self.colors.push(img);
            cleared?;
        }
        self.counters = Some(gpu.create_storage_buffer(8)?);
        Ok(())
    }

    fn owned(&self, res: FiRes) -> Result<&StorageImage, String> {
        self.images.iter().find(|(r, _)| *r == res).map(|(_, i)| i).ok_or_else(|| format!("no image for {res:?}"))
    }

    fn bind(&self, res: FiRes) -> Result<Bind<'_>, String> {
        let cur = (self.frames & 1) as usize;
        Ok(match res {
            FiRes::CurColor => Bind::Image(&self.colors[cur]),
            FiRes::PrevColor => Bind::Image(&self.colors[1 - cur]),
            FiRes::Flow => Bind::Image(self.flow.image(OfRes::Output)?),
            FiRes::Scd => Bind::Image(self.flow.image(OfRes::ScdOutput)?),
            FiRes::Counters => Bind::Buffer(self.counters.as_ref().ok_or("no counters")?),
            FiRes::PyramidMip(m) => Bind::Mip(self.owned(FiRes::Pyramid)?, m),
            other => Bind::Image(self.owned(other)?),
        })
    }

    /// Runs the flow and the interpolation for the next real frame and returns the GPU time in µs.
    /// The output is the frame midway between the previous real frame and this one.
    pub fn run_frame(&mut self, gpu: &Gpu, frame: &Image<Rgba8>) -> Result<f64, String> {
        if (frame.width, frame.height) != (self.width, self.height) {
            return Err("frame size differs from the interpolation size".into());
        }
        self.times = InterpTimes { flow: self.flow.run_frame(gpu, &packed::to_packed(self.format, frame))?, ..InterpTimes::default() };
        gpu.write_rgba8(&self.colors[(self.frames & 1) as usize], frame)?;
        set_fi_frame(&mut self.constants, self.frames);
        let mut total = self.times.flow;
        for d in &self.schedules[(self.frames > 0) as usize] {
            for (res, value) in &d.clears {
                gpu.clear_u32(self.owned(*res)?, *value)?;
            }
            let binds: Vec<(u32, Bind)> = d.bind.iter().map(|(b, r)| self.bind(*r).map(|x| (*b, x))).collect::<Result<_, _>>()?;
            let pass = FI_PASSES[d.pass];
            let wanted = pass.bindings.iter().filter(|b| b.1 == BindingKind::UniformBuffer).count();
            let blocks: Vec<&[u8]> = match (wanted, &d.pyramid_constants) {
                (1, None) => vec![&self.constants],
                (2, Some(p)) => vec![&self.constants, p],
                _ => return Err(format!("{} takes {wanted} constant blocks; the schedule disagrees", pass.name)),
            };
            let us = gpu.run_fsr3(pass, &binds, &blocks, &[], d.groups)?;
            self.times.passes[d.pass] += us;
            total += us;
        }
        self.frames = self.frames.saturating_add(1);
        Ok(total)
    }

    /// Pre-fills the output, so pixels a pass never writes can be found.
    pub fn fill_output(&self, gpu: &Gpu, rgba: [f32; 4]) -> Result<(), String> {
        gpu.clear(self.owned(FiRes::Output)?, rgba)
    }

    pub fn read_output(&self, gpu: &Gpu) -> Result<Image<Rgba8>, String> {
        gpu.read_rgba8(self.owned(FiRes::Output)?)
    }

    /// Device memory interpolation owns, as the driver sized it: everything but the two real
    /// frames (the layer keeps those already) and the flow, which reports its own.
    pub fn owned_bytes(&self) -> u64 {
        self.images.iter().map(|(_, i)| i.bytes).sum::<u64>() + self.counters.as_ref().map_or(0, |c| c.size)
    }

    pub fn owned_images(&self) -> usize {
        self.images.len()
    }

    pub fn destroy(self, gpu: &Gpu) {
        for (_, img) in self.images {
            gpu.destroy_image(img);
        }
        for img in self.colors {
            gpu.destroy_image(img);
        }
        if let Some(c) = self.counters {
            gpu.destroy_buffer(c);
        }
        self.flow.destroy(gpu);
    }
}

/// Runs AMD's prepare pass on a constant far depth and zero motion and checks that its three
/// outputs are exactly the images `Fsr3Interp::new` fills in its place.
pub fn prepare_is_constant(gpu: &Gpu, width: u32, height: u32) -> Result<f64, String> {
    let sizes = fi_sizes(width, height);
    let one = 1.0f32.to_bits();
    let (rw, rh) = sizes.render;
    let specs = [
        (FiRes::InputDepth, vk::Format::R32_SFLOAT, one),
        (FiRes::InputMotion, vk::Format::R16G16_SFLOAT, 0),
        (FiRes::ReconstructedPrevDepth, vk::Format::R32_UINT, one),
        (FiRes::DilatedMotion, vk::Format::R16G16_SFLOAT, 0xDEAD_BEEF),
        (FiRes::DilatedDepth, vk::Format::R32_SFLOAT, 0xDEAD_BEEF),
    ];
    let mut images: Vec<(FiRes, StorageImage)> = Vec::new();
    let result = (|| {
        for (res, format, fill) in specs {
            let img = gpu.create_storage_image(format, rw, rh)?;
            let cleared = gpu.clear_u32(&img, fill);
            images.push((res, img));
            cleared?;
        }
        let d = fi_prepare_dispatch(&sizes);
        let find = |r: FiRes| images.iter().find(|(x, _)| *x == r).map(|(_, i)| i).ok_or(format!("no {r:?}"));
        let binds: Vec<(u32, Bind)> = d.bind.iter().map(|(b, r)| find(*r).map(|i| (*b, Bind::Image(i)))).collect::<Result<_, _>>()?;
        let constants = fi_constants(&sizes, &FiParams { frame_since_reset: 1, delta_time_ms: DELTA_TIME_MS });
        let us = gpu.run_fsr3(FI_PASSES[d.pass], &binds, &[&constants], &[], d.groups)?;
        let words = |r: FiRes| -> Result<Vec<u32>, String> {
            Ok(gpu.read_raw(find(r)?, 4)?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        for (res, want) in [(FiRes::DilatedDepth, one), (FiRes::DilatedMotion, 0), (FiRes::ReconstructedPrevDepth, one)] {
            let got = words(res)?;
            if let Some((i, w)) = got.iter().enumerate().find(|(_, w)| **w != want) {
                return Err(format!("{res:?} texel {i} is {w:#x}, expected {want:#x}"));
            }
        }
        Ok(us)
    })();
    for (_, img) in images {
        gpu.destroy_image(img);
    }
    result
}
