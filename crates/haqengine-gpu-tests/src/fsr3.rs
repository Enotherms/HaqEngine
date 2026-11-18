//! FSR 3 optical flow on the GPU: every image of the schedule, created once and reused per frame.

use crate::gpu::{Gpu, StorageImage};
use ash::vk;
use haqengine_core::fsr3_flow::{ffx_to_haq, of_schedule, of_sizes, OfRes, OF_LEVELS, PASS_SEARCH};
use haqengine_core::{Image, Rgba8};
use haqengine_shaders::OF_PASSES;

const SCD_HISTOGRAM_WIDTH: u32 = 256 * 9;

/// The optical flow state of one frame size.
pub struct Fsr3Flow {
    pub width: u32,
    pub height: u32,
    images: Vec<(OfRes, StorageImage)>,
    frame_index: u32,
    pub times: PassTimes,
}

/// GPU time of one frame per pass, and of the search per pyramid level, in µs.
#[derive(Debug, Clone, Copy, Default)]
pub struct PassTimes {
    pub passes: [f64; 7],
    pub search_by_level: [f64; OF_LEVELS],
}

/// The scene-change words FSR 3 writes: value (as `f32` bits), history bits, completed groups.
pub type ScdOutput = [u32; 3];

impl Fsr3Flow {
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Result<Fsr3Flow, String> {
        let sizes = of_sizes(width, height);
        let mut wanted: Vec<(OfRes, vk::Format, (u32, u32))> = vec![
            (OfRes::Color, vk::Format::R8G8B8A8_UNORM, (width, height)),
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
        let mut flow = Fsr3Flow { width, height, images: Vec::new(), frame_index: 0, times: PassTimes::default() };
        for (res, format, (w, h)) in wanted {
            match gpu.create_storage_image(format, w, h).and_then(|img| gpu.clear_u32(&img, 0).map(|_| img)) {
                Ok(img) => flow.images.push((res, img)),
                Err(e) => {
                    flow.destroy(gpu);
                    return Err(e);
                }
            }
        }
        Ok(flow)
    }

    fn image(&self, res: OfRes) -> Result<&StorageImage, String> {
        self.images.iter().find(|(r, _)| *r == res).map(|(_, i)| i).ok_or_else(|| format!("no image for {res:?}"))
    }

    /// Runs the whole schedule for the next frame and returns its GPU time in µs.
    pub fn run_frame(&mut self, gpu: &Gpu, frame: &Image<Rgba8>) -> Result<f64, String> {
        if (frame.width, frame.height) != (self.width, self.height) {
            return Err("frame size differs from the flow size".into());
        }
        gpu.write_rgba8(self.image(OfRes::Color)?, frame)?;
        let mut total = 0.0;
        for d in of_schedule(self.width, self.height, self.frame_index) {
            let images: Vec<(u32, &StorageImage)> =
                d.bind.iter().map(|(b, r)| self.image(*r).map(|i| (*b, i))).collect::<Result<_, _>>()?;
            let mut constants: Vec<&[u8]> = vec![&d.constants];
            if let Some(spd) = &d.spd_constants {
                constants.push(spd);
            }
            let us = gpu.run_fsr3(OF_PASSES[d.pass], &images, &constants, d.groups)?;
            self.times.passes[d.pass] += us;
            if d.pass == PASS_SEARCH {
                let level = u32::from_le_bytes([d.constants[8], d.constants[9], d.constants[10], d.constants[11]]);
                self.times.search_by_level[level as usize] += us;
            }
            total += us;
        }
        self.frame_index += 1;
        Ok(total)
    }

    /// Level-0 flow, one vector per 8×8 block of the frame, in HaqEngine's convention and pixels.
    pub fn read_flow(&self, gpu: &Gpu) -> Result<Image<[f32; 2]>, String> {
        let raw = gpu.read_rg16i(self.image(OfRes::Output)?)?;
        Ok(Image { width: raw.width, height: raw.height, data: raw.data.iter().map(|v| ffx_to_haq(*v)).collect() })
    }

    pub fn read_scd(&self, gpu: &Gpu) -> Result<ScdOutput, String> {
        let img = gpu.read_u32(self.image(OfRes::ScdOutput)?)?;
        Ok([img.data[0], img.data[1], img.data[2]])
    }

    pub fn destroy(self, gpu: &Gpu) {
        for (_, img) in self.images {
            gpu.destroy_image(img);
        }
    }
}

/// The last frame of a sequence: flow, scene-change output, GPU time and time per pass.
pub type SequenceResult = (Image<[f32; 2]>, ScdOutput, f64, PassTimes);

/// Runs every frame through a fresh flow state and reports on the last one.
pub fn run_sequence(gpu: &Gpu, frames: &[Image<Rgba8>]) -> Result<SequenceResult, String> {
    let first = frames.first().ok_or("no frames")?;
    let mut flow = Fsr3Flow::new(gpu, first.width, first.height)?;
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
