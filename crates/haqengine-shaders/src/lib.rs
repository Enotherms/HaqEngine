//! HaqEngine's compute shaders, compiled to SPIR-V at build time and embedded.

/// One compiled compute shader.
#[derive(Debug, Clone, Copy)]
pub struct Shader {
    pub name: &'static str,
    pub spirv: &'static [u8],
    pub local_size: [u32; 2],
}

/// Half-resolution luma: bindings 0 = `rgba8` input, 1 = `r32f` output; push `{ in_w, in_h }`.
pub const LUMA_DOWNSAMPLE: Shader = Shader {
    name: "luma_downsample",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/luma_downsample.spv")),
    local_size: [8, 8],
};

/// `mix(a, b, t)`: bindings 0 = `rgba8` a, 1 = `rgba8` b, 2 = `rgba8` output; push `{ w, h, t }`.
pub const BLEND: Shader = Shader {
    name: "blend",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/blend.spv")),
    local_size: [8, 8],
};

/// Halves an `r32f` luma level: bindings 0 = input, 1 = output; push `{ in_w, in_h }`.
pub const LUMA_REDUCE: Shader = Shader {
    name: "luma_reduce",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/luma_reduce.spv")),
    local_size: [8, 8],
};

/// One motion vector per 8×8 block: bindings 0 = prev `r32f`, 1 = cur `r32f`, 2 = coarse `rg32f`,
/// 3 = output `rg32f`; push `{ w, h, coarse_w, coarse_h, radius: i32, use_coarse }`. One workgroup
/// (8×8 threads) per block: dispatch `blocks_w × blocks_h` groups. `radius` is at most 4.
pub const BLOCK_MATCH: Shader = Shader {
    name: "block_match",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/block_match.spv")),
    local_size: [8, 8],
};

/// The frame at time `t`: bindings 0 = prev `rgba8`, 1 = cur `rgba8`, 2 = level-0 flow `rg32f`,
/// 3 = output `rgba8`; push `{ w, h, flow_w, flow_h, t: f32 }`.
pub const WARP_BLEND: Shader = Shader {
    name: "warp_blend",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/warp_blend.spv")),
    local_size: [8, 8],
};

/// 3×3 component-wise median of a flow field: bindings 0 = input `rg32f`, 1 = output `rg32f`;
/// push `{ w, h }`.
pub const FLOW_MEDIAN: Shader = Shader {
    name: "flow_median",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/flow_median.spv")),
    local_size: [8, 8],
};

/// Half-resolution luma from packed pixels: bindings 0 = `r32ui` input, 1 = `r32f` output;
/// push `{ in_w, in_h, format }` with the `PackedFormat` code.
pub const LUMA_DOWNSAMPLE_PACKED: Shader = Shader {
    name: "luma_downsample_packed",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/luma_downsample_packed.spv")),
    local_size: [8, 8],
};

/// The frame at time `t` on packed pixels: bindings 0 = prev `r32ui`, 1 = cur `r32ui`, 2 = level-0
/// flow `rg32f`, 3 = output `r32ui`; push `{ w, h, flow_w, flow_h, t: f32, format }`.
pub const WARP_BLEND_PACKED: Shader = Shader {
    name: "warp_blend_packed",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/warp_blend_packed.spv")),
    local_size: [8, 8],
};

/// How a descriptor binding of an FSR 3 pass is declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingKind {
    StorageImage,
    SampledImage,
    UniformBuffer,
    Sampler,
    StorageBuffer,
}

/// One FidelityFX FSR 3 compute pass (SDK v1.1.4), compiled as SPIR-V 1.3.
#[derive(Debug, Clone, Copy)]
pub struct Fsr3Pass {
    pub name: &'static str,
    pub spirv: &'static [u8],
    pub local_size: [u32; 3],
    pub bindings: &'static [(u32, BindingKind)],
    pub push_size: u32,
}

use BindingKind::{SampledImage as Srv, Sampler as Smp, StorageBuffer as Buf, StorageImage as Uav, UniformBuffer as Cb};

/// Colour to 8-bit luma: 0 = colour, 1 = luma level 0 (`r8ui`), 2 = constants.
pub const OF_PREPARE_LUMA: Fsr3Pass = Fsr3Pass {
    name: "of_prepare_luma",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_prepare_luma.spv")),
    local_size: [16, 16, 1],
    bindings: &[(0, Srv), (1, Uav), (2, Cb)],
    push_size: 0,
};

/// Luma levels 1 to 6 from level 0: 0..=6 = luma levels (`r8ui`), 7 = constants, 8 = SPD constants.
pub const OF_LUMINANCE_PYRAMID: Fsr3Pass = Fsr3Pass {
    name: "of_luminance_pyramid",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_luminance_pyramid.spv")),
    local_size: [256, 1, 1],
    bindings: &[(0, Uav), (1, Uav), (2, Uav), (3, Uav), (4, Uav), (5, Uav), (6, Uav), (7, Cb), (8, Cb)],
    push_size: 0,
};

/// Scene-change histograms: 0 = luma, 1 = histogram (`r32ui`), 2 = constants.
pub const OF_SCD_HISTOGRAM: Fsr3Pass = Fsr3Pass {
    name: "of_scd_histogram",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_scd_histogram.spv")),
    local_size: [32, 8, 1],
    bindings: &[(0, Srv), (1, Uav), (2, Cb)],
    push_size: 0,
};

/// Scene-change verdict: 0 = histogram, 1 = previous histogram (`r32f`), 2 = temp, 3 = output
/// (`r32ui`), 4 = constants.
pub const OF_SCD_DIVERGENCE: Fsr3Pass = Fsr3Pass {
    name: "of_scd_divergence",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_scd_divergence.spv")),
    local_size: [256, 1, 1],
    bindings: &[(0, Uav), (1, Uav), (2, Uav), (3, Uav), (4, Cb)],
    push_size: 0,
};

/// Block search for one level: 0 = current luma, 1 = previous luma, 2 = flow (`rg16i`), 3 = scene
/// change output, 4 = constants.
pub const OF_SEARCH: Fsr3Pass = Fsr3Pass {
    name: "of_search",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_search.spv")),
    local_size: [64, 1, 1],
    bindings: &[(0, Srv), (1, Srv), (2, Uav), (3, Uav), (4, Cb)],
    push_size: 0,
};

/// Flow filter for one level: 0 = searched flow, 1 = filtered flow (`rg16i`), 2 = constants.
pub const OF_FILTER: Fsr3Pass = Fsr3Pass {
    name: "of_filter",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_filter.spv")),
    local_size: [16, 4, 1],
    bindings: &[(0, Srv), (1, Uav), (2, Cb)],
    push_size: 0,
};

/// Flow of one level scaled to the next finer level: 0 = current luma, 1 = previous luma, 2 = flow,
/// 3 = next level flow, 4 = scene change output, 5 = constants.
pub const OF_SCALE: Fsr3Pass = Fsr3Pass {
    name: "of_scale",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_scale.spv")),
    local_size: [4, 4, 4],
    bindings: &[(0, Srv), (1, Srv), (2, Srv), (3, Uav), (4, Uav), (5, Cb)],
    push_size: 0,
};

/// HaqEngine's replacement for prepare luma at half resolution (SPIR-V 1.0): 0 = packed frame
/// (`r32ui`), 1 = luma level 0 (`r8ui`); push `{ in_w, in_h, format }`.
pub const OF_LUMA_HALF: Fsr3Pass = Fsr3Pass {
    name: "of_luma_half",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_luma_half.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Uav), (1, Uav)],
    push_size: 12,
};

/// The optical flow passes; indices are the schedule's `PASS_*` constants.
pub const OF_PASSES: [&Fsr3Pass; 8] = [
    &OF_PREPARE_LUMA,
    &OF_LUMINANCE_PYRAMID,
    &OF_SCD_HISTOGRAM,
    &OF_SCD_DIVERGENCE,
    &OF_SEARCH,
    &OF_FILTER,
    &OF_SCALE,
    &OF_LUMA_HALF,
];

/// Clears the vector fields and masks: 0 = scene change, 1..=4 = game and flow fields, 5 = disocclusion, 6 = counters, 7 = constants.
pub const FI_SETUP: Fsr3Pass = Fsr3Pass {
    name: "fi_setup",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_setup.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Srv), (1, Uav), (2, Uav), (3, Uav), (4, Uav), (5, Uav), (6, Buf), (7, Cb), (1000, Smp)],
    push_size: 0,
};

/// AMD's prepare pass: 0 = game motion, 1 = depth, 2 = reconstructed previous depth, 3 = dilated motion, 4 = dilated depth.
pub const FI_RECONSTRUCT_AND_DILATE: Fsr3Pass = Fsr3Pass {
    name: "fi_reconstruct_and_dilate",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_reconstruct_and_dilate.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Srv), (1, Srv), (2, Uav), (3, Uav), (4, Uav), (5, Cb), (1000, Smp)],
    push_size: 0,
};

/// Depth at the interpolated time: 0 = dilated motion, 1 = dilated depth, 2 = current frame, 3 = distortion, 4 = output.
pub const FI_RECONSTRUCT_PREVIOUS_DEPTH: Fsr3Pass = Fsr3Pass {
    name: "fi_reconstruct_previous_depth",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_reconstruct_previous_depth.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Srv), (1, Srv), (2, Srv), (3, Srv), (4, Uav), (5, Cb), (1000, Smp)],
    push_size: 0,
};

/// Game vector field: 0 = dilated motion, 1 = dilated depth, 2 = previous, 3 = current, 4 = distortion, 5/6 = field x/y.
pub const FI_GAME_MOTION_VECTOR_FIELD: Fsr3Pass = Fsr3Pass {
    name: "fi_game_motion_vector_field",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_game_motion_vector_field.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Srv), (1, Srv), (2, Srv), (3, Srv), (4, Srv), (5, Uav), (6, Uav), (7, Cb), (1000, Smp)],
    push_size: 0,
};

/// Game-field inpainting pyramid: 0/1 = field x/y, 2 = counters, 3..=15 = mips, 16 = constants, 17 = pyramid constants.
pub const FI_GAME_VECTOR_FIELD_INPAINTING_PYRAMID: Fsr3Pass = Fsr3Pass {
    name: "fi_game_vector_field_inpainting_pyramid",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_game_vector_field_inpainting_pyramid.spv")),
    local_size: [256, 1, 1],
    bindings: &[(0, Srv), (1, Srv), (2, Buf), (3, Uav), (4, Uav), (5, Uav), (6, Uav), (7, Uav), (8, Uav), (9, Uav), (10, Uav), (11, Uav), (12, Uav), (13, Uav), (14, Uav), (15, Uav), (16, Cb), (17, Cb), (1000, Smp)],
    push_size: 0,
};

/// Flow vector field: 0 = optical flow, 1 = confidence, 2 = dilated depth, 3 = previous, 4 = current, 6/7 = field x/y.
pub const FI_OPTICAL_FLOW_VECTOR_FIELD: Fsr3Pass = Fsr3Pass {
    name: "fi_optical_flow_vector_field",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_optical_flow_vector_field.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Srv), (1, Srv), (2, Srv), (3, Srv), (4, Srv), (6, Uav), (7, Uav), (8, Cb), (1000, Smp)],
    push_size: 0,
};

/// Disocclusion: 0/1 = game field x/y, 2/3/4 = depths, 5 = pyramid, 6 = distortion, 7 = mask.
pub const FI_DISOCCLUSION_MASK: Fsr3Pass = Fsr3Pass {
    name: "fi_disocclusion_mask",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_disocclusion_mask.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Srv), (1, Srv), (2, Srv), (3, Srv), (4, Srv), (5, Srv), (6, Srv), (7, Uav), (8, Cb), (1000, Smp)],
    push_size: 0,
};

/// The interpolated frame: 0..=3 = fields, 4 = previous, 5 = current, 6 = disocclusion, 7 = pyramid, 8 = counters, 9 = output.
pub const FI_INTERPOLATION: Fsr3Pass = Fsr3Pass {
    name: "fi_interpolation",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_interpolation.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Srv), (1, Srv), (2, Srv), (3, Srv), (4, Srv), (5, Srv), (6, Srv), (7, Srv), (8, Buf), (9, Uav), (10, Cb), (1000, Smp)],
    push_size: 0,
};

/// Output inpainting pyramid: 0 = output, 1 = counters, 2..=14 = mips, 15 = constants, 16 = pyramid constants.
pub const FI_INPAINTING_PYRAMID: Fsr3Pass = Fsr3Pass {
    name: "fi_inpainting_pyramid",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_inpainting_pyramid.spv")),
    local_size: [256, 1, 1],
    bindings: &[(0, Srv), (1, Buf), (2, Uav), (3, Uav), (4, Uav), (5, Uav), (6, Uav), (7, Uav), (8, Uav), (9, Uav), (10, Uav), (11, Uav), (12, Uav), (13, Uav), (14, Uav), (15, Cb), (16, Cb), (1000, Smp)],
    push_size: 0,
};

/// Fills holes: 0 = scene change, 1 = pyramid, 2 = present frame, 3 = current, 7 = output.
pub const FI_INPAINTING: Fsr3Pass = Fsr3Pass {
    name: "fi_inpainting",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_inpainting.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Srv), (1, Srv), (2, Srv), (3, Srv), (7, Uav), (8, Cb), (1000, Smp)],
    push_size: 0,
};

/// HaqEngine's blend on FSR 3's flow (SPIR-V 1.0), replacing AMD's interpolation pass:
/// 0/1 = previous/current frame (`rgba8`), 2 = flow (`rg16i`), 3/4 = previous/current 8-bit luma,
/// 5 = output (`rgba8`), 6 = FSR 3 scene-change words (`r32ui`); push `{ w, h, flow_w, flow_h, reset }`.
pub const FI_BLEND: Fsr3Pass = Fsr3Pass {
    name: "fi_blend",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_blend.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Uav), (1, Uav), (2, Uav), (3, Uav), (4, Uav), (5, Uav), (6, Uav)],
    push_size: 20,
};

/// `FI_BLEND` on packed pixels (`r32ui` frames and output, `PackedFormat` code in the push),
/// for the layer; push `{ w, h, flow_w, flow_h, reset, format }`.
pub const FI_BLEND_PACKED: Fsr3Pass = Fsr3Pass {
    name: "fi_blend_packed",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/fi_blend_packed.spv")),
    local_size: [8, 8, 1],
    bindings: &[(0, Uav), (1, Uav), (2, Uav), (3, Uav), (4, Uav), (5, Uav), (6, Uav)],
    push_size: 24,
};

/// The interpolation passes; indices are `haqengine_core::fsr3_interp`'s `PASS_FI_*` constants.
pub const FI_PASSES: [&Fsr3Pass; 11] = [
    &FI_SETUP,
    &FI_RECONSTRUCT_AND_DILATE,
    &FI_RECONSTRUCT_PREVIOUS_DEPTH,
    &FI_GAME_MOTION_VECTOR_FIELD,
    &FI_GAME_VECTOR_FIELD_INPAINTING_PYRAMID,
    &FI_OPTICAL_FLOW_VECTOR_FIELD,
    &FI_DISOCCLUSION_MASK,
    &FI_INTERPOLATION,
    &FI_INPAINTING_PYRAMID,
    &FI_INPAINTING,
    &FI_BLEND,
];

/// HaqEngine's zero-first blend on Battery's flow at time `t`, packed: bindings 0 = prev `r32ui`,
/// 1 = cur `r32ui`, 2 = level-0 flow `rg32f`, 3 = output `r32ui`, 4/5 = prev/cur level-0 luma
/// `r32f`, 6 = change counters `r32ui` 2 wide, one row per output (column 0 = disagreeing pixels,
/// 1 = still pixels, added with atomics; cleared before the first blend); push
/// `{ w, h, flow_w, flow_h, t: f32, format, row }`.
pub const BATTERY_BLEND_PACKED: Shader = Shader {
    name: "battery_blend_packed",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/battery_blend_packed.spv")),
    local_size: [8, 8],
};

/// Shows the current frame on a scene cut: bindings 0 = change counters `r32ui`, 1 = current
/// frame `r32ui`, 2 = output `r32ui`; push `{ w, h, cut_threshold, row }` (pixels; more
/// disagreeing pixels than that in counter row `row` is a cut).
pub const CUT_FIXUP: Shader = Shader {
    name: "cut_fixup",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/cut_fixup.spv")),
    local_size: [8, 8],
};

/// Every shader HaqEngine ships.
pub fn all() -> &'static [Shader] {
    &[
        LUMA_DOWNSAMPLE,
        BLEND,
        LUMA_REDUCE,
        BLOCK_MATCH,
        WARP_BLEND,
        FLOW_MEDIAN,
        LUMA_DOWNSAMPLE_PACKED,
        WARP_BLEND_PACKED,
        BATTERY_BLEND_PACKED,
        CUT_FIXUP,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_has_magic_and_words() {
        assert!(!all().is_empty());
        for s in all() {
            assert!(!s.spirv.is_empty() && s.spirv.len() % 4 == 0, "{}", s.name);
            assert_eq!(u32::from_le_bytes(s.spirv[..4].try_into().unwrap()), 0x0723_0203, "{}", s.name);
        }
    }

    #[test]
    fn spirv_is_version_1_0_for_vulkan_1_0_apps() {
        for s in all() {
            let version = u32::from_le_bytes(s.spirv[4..8].try_into().unwrap());
            assert_eq!(version, 0x0001_0000, "{} is SPIR-V {:#x}", s.name, version);
        }
    }

    fn words(spirv: &[u8]) -> Vec<u32> {
        spirv.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn decorated_bindings(spirv: &[u8]) -> Vec<u32> {
        let w = words(spirv);
        let mut out = Vec::new();
        let mut i = 5;
        while i < w.len() {
            let (count, op) = ((w[i] >> 16) as usize, w[i] & 0xffff);
            if op == 71 && count >= 4 && w[i + 2] == 33 {
                out.push(w[i + 3]);
            }
            i += count.max(1);
        }
        out.sort_unstable();
        out
    }

    fn binding_kinds(spirv: &[u8]) -> Vec<(u32, BindingKind)> {
        use std::collections::HashMap;
        let w = words(spirv);
        let (mut binding, mut block, mut buffer_block) = (HashMap::new(), Vec::new(), Vec::new());
        let (mut pointers, mut images, mut samplers, mut vars) = (HashMap::new(), HashMap::new(), Vec::new(), Vec::new());
        let mut i = 5;
        while i < w.len() {
            let (count, op) = ((w[i] >> 16) as usize, w[i] & 0xffff);
            let a = &w[i..i + count.max(1)];
            match op {
                71 if a.len() >= 3 && a[2] == 33 => {
                    binding.insert(a[1], a[3]);
                }
                71 if a.len() >= 3 && a[2] == 2 => block.push(a[1]),
                71 if a.len() >= 3 && a[2] == 3 => buffer_block.push(a[1]),
                25 => {
                    images.insert(a[1], a[7]);
                }
                26 => samplers.push(a[1]),
                32 => {
                    pointers.insert(a[1], (a[2], a[3]));
                }
                59 => vars.push((a[2], a[1])),
                _ => {}
            }
            i += count.max(1);
        }
        let mut out = Vec::new();
        for (var, ptr) in vars {
            let Some(&b) = binding.get(&var) else { continue };
            let Some(&(class, pointee)) = pointers.get(&ptr) else { continue };
            let kind = match class {
                0 if samplers.contains(&pointee) => BindingKind::Sampler,
                0 if images.get(&pointee) == Some(&1) => BindingKind::SampledImage,
                0 if images.get(&pointee) == Some(&2) => BindingKind::StorageImage,
                2 if buffer_block.contains(&pointee) => BindingKind::StorageBuffer,
                2 if block.contains(&pointee) => BindingKind::UniformBuffer,
                12 => BindingKind::StorageBuffer,
                _ => panic!("binding {b}: unknown storage class {class}"),
            };
            out.push((b, kind));
        }
        out.sort_by_key(|x| x.0);
        out
    }

    fn table(p: &Fsr3Pass) -> Vec<(u32, BindingKind)> {
        let mut t = p.bindings.to_vec();
        t.sort_by_key(|x| x.0);
        t
    }

    #[test]
    fn fi_passes_are_spirv_1_3() {
        assert_eq!(FI_PASSES.len(), 11);
        assert_eq!(words(FI_BLEND.spirv)[1], 0x0001_0000);
        for p in &FI_PASSES[..10] {
            assert_eq!(words(p.spirv)[1], 0x0001_0300, "{}", p.name);
        }
    }

    #[test]
    fn fi_bindings_match_spirv() {
        for p in FI_PASSES {
            let numbers: Vec<u32> = table(p).iter().map(|b| b.0).collect();
            assert_eq!(decorated_bindings(p.spirv), numbers, "{}", p.name);
        }
    }

    #[test]
    fn binding_kinds_match_spirv() {
        for p in OF_PASSES.iter().chain(FI_PASSES.iter()) {
            assert_eq!(binding_kinds(p.spirv), table(p), "{}", p.name);
        }
    }

    #[test]
    fn fi_pass_order() {
        let names: Vec<&str> = FI_PASSES.iter().map(|p| p.name).collect();
        assert_eq!(
            names,
            [
                "fi_setup",
                "fi_reconstruct_and_dilate",
                "fi_reconstruct_previous_depth",
                "fi_game_motion_vector_field",
                "fi_game_vector_field_inpainting_pyramid",
                "fi_optical_flow_vector_field",
                "fi_disocclusion_mask",
                "fi_interpolation",
                "fi_inpainting_pyramid",
                "fi_inpainting",
                "fi_blend"
            ]
        );
    }

    #[test]
    fn fsr3_passes_are_spirv_1_3() {
        assert_eq!(OF_PASSES.len(), 8);
        for p in &OF_PASSES[..7] {
            assert_eq!(words(p.spirv)[0], 0x0723_0203, "{}", p.name);
            assert_eq!(words(p.spirv)[1], 0x0001_0300, "{}", p.name);
        }
    }

    #[test]
    fn fsr3_bindings_match_spirv() {
        for p in OF_PASSES {
            let mut table: Vec<u32> = p.bindings.iter().map(|b| b.0).collect();
            table.sort_unstable();
            assert_eq!(decorated_bindings(p.spirv), table, "{}", p.name);
        }
    }

    #[test]
    fn fsr3_pass_order() {
        let names: Vec<&str> = OF_PASSES.iter().map(|p| p.name).collect();
        assert_eq!(names, ["of_prepare_luma", "of_luminance_pyramid", "of_scd_histogram", "of_scd_divergence", "of_search", "of_filter", "of_scale", "of_luma_half"]);
    }

    #[test]
    fn packed_blend_is_spirv_1_0_with_kinds() {
        assert_eq!(words(FI_BLEND_PACKED.spirv)[1], 0x0001_0000);
        assert_eq!(binding_kinds(FI_BLEND_PACKED.spirv), table(&FI_BLEND_PACKED));
    }

    #[test]
    fn own_luma_half_pass_is_spirv_1_0_with_push() {
        assert_eq!(words(OF_LUMA_HALF.spirv)[1], 0x0001_0000);
        assert_eq!(OF_LUMA_HALF.push_size, 12);
        assert!(OF_PASSES[..7].iter().all(|p| p.push_size == 0));
    }

    #[test]
    fn all_lists_every_shader() {
        let names: Vec<&str> = all().iter().map(|s| s.name).collect();
        assert_eq!(names, ["luma_downsample", "blend", "luma_reduce", "block_match", "warp_blend", "flow_median", "luma_downsample_packed", "warp_blend_packed", "battery_blend_packed", "cut_fixup"]);
    }
}
