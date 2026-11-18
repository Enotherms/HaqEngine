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

/// Every shader HaqEngine ships.
pub fn all() -> &'static [Shader] {
    &[LUMA_DOWNSAMPLE, BLEND]
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
    fn all_lists_both_shaders() {
        let names: Vec<&str> = all().iter().map(|s| s.name).collect();
        assert_eq!(names, ["luma_downsample", "blend"]);
    }
}
