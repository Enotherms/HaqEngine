//! FSR 3 optical flow on the GPU at half resolution: every image and both schedules are created
//! once, and each frame only uploads the packed frame and patches the frame number.

use crate::gpu::{Bind, Gpu, StorageImage};
use ash::vk;
use haqengine_core::fsr3_flow::{
    ffx_to_haq, half_size, of_schedule, of_sizes, set_frame_index, OfDispatch, OfInput, OfRes, OF_LEVELS, PASS_SEARCH, SCD_HISTOGRAM_WIDTH,
};
use haqengine_core::packed::PackedFormat;
use haqengine_core::Image;
use haqengine_shaders::{BindingKind, OF_PASSES};

/// The scene-change words FSR 3 writes: value (as `f32` bits), history bits, completed groups.
pub type ScdOutput = [u32; 3];

/// GPU time of one frame per pass, and of the search per pyramid level, in µs.
#[derive(Debug, Clone, Copy, Default)]
pub struct PassTimes {
    pub passes: [f64; 8],
    pub search_by_level: [f64; OF_LEVELS],
}

/// The optical flow state for packed frames of one size.
pub struct Fsr3Flow {
    pub width: u32,
    pub height: u32,
    images: Vec<(OfRes, StorageImage)>,
    schedules: [Vec<OfDispatch>; 2],
    frame_index: u32,
    odd: bool,
    pub times: PassTimes,
}

impl Fsr3Flow {
    pub fn new(gpu: &Gpu, width: u32, height: u32, format: PackedFormat) -> Result<Fsr3Flow, String> {
        let (hw, hh) = half_size(width, height);
        let sizes = of_sizes(hw, hh);
        let mut wanted: Vec<(OfRes, vk::Format, (u32, u32))> = vec![
            (OfRes::Raw, vk::Format::R32_UINT, (width, height)),
            (OfRes::Output, vk::Format::R16G16_SINT, sizes.flow[0]),
            (OfRes::ScdHistogram, vk::Format::R32_UINT, (SCD_HISTOGRAM_WIDTH, 1)),
            (OfRes::ScdPreviousHistogram, vk::Format::R32_SFLOAT, (SCD_HISTOGRAM_WIDTH, 1)),
            (OfRes::ScdTemp, vk::Format::R32_UINT, (3, 1)),
            (OfRes::ScdOutput, vk::Format::R32_UINT, (3, 1)),
        ];
        for slot in 0..2 {
            for level in 0..OF_LEVELS {
                wanted.push((OfRes::Input(slot, level as u32), vk::Format::R8_UINT, sizes.luma[level]));
                wanted.push((OfRes::Flow(slot, level as u32), vk::Format::R16G16_SINT, sizes.flow[level]));
            }
        }
        let input = OfInput::PackedHalf { format: format.code(), full_width: width, full_height: height };
        let mut flow = Fsr3Flow {
            width,
            height,
            images: Vec::with_capacity(wanted.len()),
            schedules: [of_schedule(hw, hh, 0, input), of_schedule(hw, hh, 1, input)],
            frame_index: 0,
            odd: false,
            times: PassTimes::default(),
        };
        for (res, format, (w, h)) in wanted {
            let made = gpu.create_storage_image(format, w, h).and_then(|img| match gpu.clear_u32(&img, 0) {
                Ok(()) => Ok(img),
                Err(e) => {
                    gpu.destroy_image(img);
                    Err(e)
                }
            });
            match made {
                Ok(img) => flow.images.push((res, img)),
                Err(e) => {
                    flow.destroy(gpu);
                    return Err(e);
                }
            }
        }
        Ok(flow)
    }

    pub fn image(&self, res: OfRes) -> Result<&StorageImage, String> {
        self.images.iter().find(|(r, _)| *r == res).map(|(_, i)| i).ok_or_else(|| format!("no image for {res:?}"))
    }

    /// Runs the schedule for the next packed frame and returns its GPU time in µs.
    pub fn run_frame(&mut self, gpu: &Gpu, frame: &Image<u32>) -> Result<f64, String> {
        if (frame.width, frame.height) != (self.width, self.height) {
            return Err("frame size differs from the flow size".into());
        }
        gpu.write_u32(self.image(OfRes::Raw)?, frame)?;
        let mut total = 0.0;
        for d in &self.schedules[self.odd as usize] {
            let images: Vec<(u32, Bind)> =
                d.bind.iter().map(|(b, r)| self.image(*r).map(|i| (*b, Bind::Image(i)))).collect::<Result<_, _>>()?;
            let mut constants = d.constants;
            set_frame_index(&mut constants, self.frame_index);
            let pass = OF_PASSES[d.pass];
            let wanted = pass.bindings.iter().filter(|b| b.1 == BindingKind::UniformBuffer).count();
            let blocks: Vec<&[u8]> = match (wanted, &d.spd_constants) {
                (0, None) => Vec::new(),
                (1, None) => vec![&constants],
                (2, Some(spd)) => vec![&constants, spd],
                _ => return Err(format!("{} takes {wanted} constant blocks; the schedule disagrees", pass.name)),
            };
            let us = gpu.run_fsr3(pass, &images, &blocks, &d.push[..pass.push_size as usize], d.groups)?;
            self.times.passes[d.pass] += us;
            if d.pass == PASS_SEARCH {
                let level = u32::from_le_bytes([d.constants[8], d.constants[9], d.constants[10], d.constants[11]]);
                self.times.search_by_level[level as usize] += us;
            }
            total += us;
        }
        self.frame_index = self.frame_index.saturating_add(1);
        self.odd = !self.odd;
        Ok(total)
    }

    /// Level-0 flow, one vector per 16×16 block of the frame, in HaqEngine's convention and
    /// full-resolution pixels.
    pub fn read_flow(&self, gpu: &Gpu) -> Result<Image<[f32; 2]>, String> {
        let raw = gpu.read_rg16i(self.image(OfRes::Output)?)?;
        Ok(Image { width: raw.width, height: raw.height, data: raw.data.iter().map(|v| ffx_to_haq(*v, 2.0)).collect() })
    }

    pub fn read_raw(&self, gpu: &Gpu) -> Result<Image<[i16; 2]>, String> {
        gpu.read_rg16i(self.image(OfRes::Output)?)
    }

    /// The `(previous, current)` input slots of the last frame run; their level 0 is 8-bit luma.
    pub fn luma_slots(&self) -> (usize, usize) {
        let cur = (!self.odd) as usize;
        (1 - cur, cur)
    }

    pub fn read_scd(&self, gpu: &Gpu) -> Result<ScdOutput, String> {
        let img = gpu.read_u32(self.image(OfRes::ScdOutput)?)?;
        Ok([img.data[0], img.data[1], img.data[2]])
    }

    /// Device memory of the images the flow owns, as the driver sized them. The packed frame is
    /// excluded: in the layer it is the raw copy the frame generator already keeps.
    pub fn owned_bytes(&self) -> u64 {
        self.images.iter().filter(|(r, _)| *r != OfRes::Raw).map(|(_, i)| i.bytes).sum()
    }

    pub fn owned_images(&self) -> usize {
        self.images.iter().filter(|(r, _)| *r != OfRes::Raw).count()
    }

    pub fn destroy(self, gpu: &Gpu) {
        for (_, img) in self.images {
            gpu.destroy_image(img);
        }
    }
}

/// The last frame of a sequence: flow, scene-change output, GPU time and time per pass.
pub type SequenceResult = (Image<[f32; 2]>, ScdOutput, f64, PassTimes);

/// Runs every packed frame through a fresh flow state and reports on the last one.
pub fn run_sequence(gpu: &Gpu, format: PackedFormat, frames: &[Image<u32>]) -> Result<SequenceResult, String> {
    let first = frames.first().ok_or("no frames")?;
    let mut flow = Fsr3Flow::new(gpu, first.width, first.height, format)?;
    let result = (|| {
        let mut last = 0.0;
        for f in frames {
            flow.times = PassTimes::default();
            last = flow.run_frame(gpu, f)?;
        }
        Ok((flow.read_flow(gpu)?, flow.read_scd(gpu)?, last, flow.times))
    })();
    flow.destroy(gpu);
    result
}
