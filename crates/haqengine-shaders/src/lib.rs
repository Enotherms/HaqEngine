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

    #[test]
    fn all_lists_every_shader() {
        let names: Vec<&str> = all().iter().map(|s| s.name).collect();
        assert_eq!(names, ["luma_downsample", "blend", "luma_reduce", "block_match", "warp_blend", "flow_median", "luma_downsample_packed", "warp_blend_packed"]);
    }
}
