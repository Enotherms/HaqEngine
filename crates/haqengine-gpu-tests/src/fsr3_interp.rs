//! Quality-mode interpolation on the GPU: our blend on FSR 3's half-resolution flow. The output
//! and the schedule are created once; per frame only the reset word is patched, and the two real
//! frames swap roles by parity instead of copying.

use crate::fsr3::Fsr3Flow;
use crate::gpu::{Bind, Gpu, StorageImage};
use ash::vk;
use haqengine_core::fsr3_flow::OfRes;
use haqengine_core::fsr3_interp::{fi_schedule, fi_sizes, set_fi_frame, FiDispatch, FiParams, FiRes, FiSizes};
use haqengine_core::packed::{self, PackedFormat};
use haqengine_core::{Image, Rgba8};
use haqengine_shaders::{BindingKind, FI_PASSES};

const DELTA_TIME_MS: f32 = 33.3;

/// Raw FSR 3 flow, previous 8-bit luma and current 8-bit luma.
pub type BlendInputs = (Image<[i16; 2]>, Image<u8>, Image<u8>);

/// GPU time of one frame: the flow, and each interpolation pass, in µs.
#[derive(Debug, Clone, Copy, Default)]
pub struct InterpTimes {
    pub flow: f64,
    pub passes: [f64; 11],
}

pub struct Fsr3Interp {
    pub width: u32,
    pub height: u32,
    format: PackedFormat,
    flow: Fsr3Flow,
    images: Vec<(FiRes, StorageImage)>,
    colors: Vec<StorageImage>,
    schedule: Vec<FiDispatch>,
    frames: u32,
    pub times: InterpTimes,
}

/// An image interpolation owns: what it holds, its format, size, mip count and initial 32-bit fill.
type ImageSpec = (FiRes, vk::Format, (u32, u32), u32, u32);

fn owned_images(s: &FiSizes) -> Vec<ImageSpec> {
    vec![(FiRes::Output, vk::Format::R8G8B8A8_UNORM, s.display, 1, 0)]
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
            schedule: fi_schedule(&sizes, &params(0)),
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
            FiRes::PrevLuma => Bind::Image(self.flow.image(OfRes::Input(self.flow.luma_slots().0, 0))?),
            FiRes::CurLuma => Bind::Image(self.flow.image(OfRes::Input(self.flow.luma_slots().1, 0))?),
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
        set_fi_frame(&mut self.schedule, self.frames);
        let mut total = self.times.flow;
        for d in &self.schedule {
            for (res, value) in &d.clears {
                gpu.clear_u32(self.owned(*res)?, *value)?;
            }
            let binds: Vec<(u32, Bind)> = d.bind.iter().map(|(b, r)| self.bind(*r).map(|x| (*b, x))).collect::<Result<_, _>>()?;
            let pass = FI_PASSES[d.pass];
            let wanted = pass.bindings.iter().filter(|b| b.1 == BindingKind::UniformBuffer).count();
            if wanted != 0 || d.pyramid_constants.is_some() {
                return Err(format!("{} takes {wanted} constant blocks; the schedule has none", pass.name));
            }
            let blocks: Vec<&[u8]> = Vec::new();
            let us = gpu.run_fsr3(pass, &binds, &blocks, &d.push[..pass.push_size as usize], d.groups)?;
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

    /// What the last frame's blend read: raw flow and the previous and current 8-bit luma.
    pub fn read_blend_inputs(&self, gpu: &Gpu) -> Result<BlendInputs, String> {
        let (p, c) = self.flow.luma_slots();
        Ok((
            self.flow.read_raw(gpu)?,
            gpu.read_r8(self.flow.image(OfRes::Input(p, 0))?)?,
            gpu.read_r8(self.flow.image(OfRes::Input(c, 0))?)?,
        ))
    }

    pub fn read_output(&self, gpu: &Gpu) -> Result<Image<Rgba8>, String> {
        gpu.read_rgba8(self.owned(FiRes::Output)?)
    }

    /// Device memory interpolation owns, as the driver sized it: everything but the two real
    /// frames (the layer keeps those already) and the flow, which reports its own.
    pub fn owned_bytes(&self) -> u64 {
        self.images.iter().map(|(_, i)| i.bytes).sum::<u64>()
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
        self.flow.destroy(gpu);
    }
}
