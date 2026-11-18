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
}

/// One FidelityFX FSR 3 compute pass (SDK v1.1.4), compiled as SPIR-V 1.3.
#[derive(Debug, Clone, Copy)]
pub struct Fsr3Pass {
    pub name: &'static str,
    pub spirv: &'static [u8],
    pub local_size: [u32; 3],
    pub bindings: &'static [(u32, BindingKind)],
}

use BindingKind::{SampledImage as Srv, StorageImage as Uav, UniformBuffer as Cb};

/// Colour to 8-bit luma: 0 = colour, 1 = luma level 0 (`r8ui`), 2 = constants.
pub const OF_PREPARE_LUMA: Fsr3Pass = Fsr3Pass {
    name: "of_prepare_luma",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_prepare_luma.spv")),
    local_size: [16, 16, 1],
    bindings: &[(0, Srv), (1, Uav), (2, Cb)],
};

/// Luma levels 1 to 6 from level 0: 0..=6 = luma levels (`r8ui`), 7 = constants, 8 = SPD constants.
pub const OF_LUMINANCE_PYRAMID: Fsr3Pass = Fsr3Pass {
    name: "of_luminance_pyramid",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_luminance_pyramid.spv")),
    local_size: [256, 1, 1],
    bindings: &[(0, Uav), (1, Uav), (2, Uav), (3, Uav), (4, Uav), (5, Uav), (6, Uav), (7, Cb), (8, Cb)],
};

/// Scene-change histograms: 0 = luma, 1 = histogram (`r32ui`), 2 = constants.
pub const OF_SCD_HISTOGRAM: Fsr3Pass = Fsr3Pass {
    name: "of_scd_histogram",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_scd_histogram.spv")),
    local_size: [32, 8, 1],
    bindings: &[(0, Srv), (1, Uav), (2, Cb)],
};

/// Scene-change verdict: 0 = histogram, 1 = previous histogram (`r32f`), 2 = temp, 3 = output
/// (`r32ui`), 4 = constants.
pub const OF_SCD_DIVERGENCE: Fsr3Pass = Fsr3Pass {
    name: "of_scd_divergence",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_scd_divergence.spv")),
    local_size: [256, 1, 1],
    bindings: &[(0, Uav), (1, Uav), (2, Uav), (3, Uav), (4, Cb)],
};

/// Block search for one level: 0 = current luma, 1 = previous luma, 2 = flow (`rg16i`), 3 = scene
/// change output, 4 = constants.
pub const OF_SEARCH: Fsr3Pass = Fsr3Pass {
    name: "of_search",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_search.spv")),
    local_size: [64, 1, 1],
    bindings: &[(0, Srv), (1, Srv), (2, Uav), (3, Uav), (4, Cb)],
};

/// Flow filter for one level: 0 = searched flow, 1 = filtered flow (`rg16i`), 2 = constants.
pub const OF_FILTER: Fsr3Pass = Fsr3Pass {
    name: "of_filter",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_filter.spv")),
    local_size: [16, 4, 1],
    bindings: &[(0, Srv), (1, Uav), (2, Cb)],
};

/// Flow of one level scaled to the next finer level: 0 = current luma, 1 = previous luma, 2 = flow,
/// 3 = next level flow, 4 = scene change output, 5 = constants.
pub const OF_SCALE: Fsr3Pass = Fsr3Pass {
    name: "of_scale",
    spirv: include_bytes!(concat!(env!("OUT_DIR"), "/of_scale.spv")),
    local_size: [4, 4, 4],
    bindings: &[(0, Srv), (1, Srv), (2, Srv), (3, Uav), (4, Uav), (5, Cb)],
};

/// The FSR 3 optical flow passes, in dispatch order; indices are used by the schedule.
pub const OF_PASSES: [&Fsr3Pass; 7] =
    [&OF_PREPARE_LUMA, &OF_LUMINANCE_PYRAMID, &OF_SCD_HISTOGRAM, &OF_SCD_DIVERGENCE, &OF_SEARCH, &OF_FILTER, &OF_SCALE];

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

    #[test]
    fn fsr3_passes_are_spirv_1_3() {
        assert_eq!(OF_PASSES.len(), 7);
        for p in OF_PASSES {
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
        assert_eq!(names, ["of_prepare_luma", "of_luminance_pyramid", "of_scd_histogram", "of_scd_divergence", "of_search", "of_filter", "of_scale"]);
    }

    #[test]
    fn all_lists_every_shader() {
        let names: Vec<&str> = all().iter().map(|s| s.name).collect();
        assert_eq!(names, ["luma_downsample", "blend", "luma_reduce", "block_match", "warp_blend", "flow_median", "luma_downsample_packed", "warp_blend_packed"]);
    }
}
