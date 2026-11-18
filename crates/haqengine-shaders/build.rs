use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const SHADERS: &[&str] = &["luma_downsample", "blend", "luma_reduce", "block_match", "warp_blend", "flow_median", "luma_downsample_packed", "warp_blend_packed"];

fn compiler() -> PathBuf {
    if let Ok(p) = env::var("GLSLANG_VALIDATOR") {
        return PathBuf::from(p);
    }
    let brew = Path::new("/opt/homebrew/bin/glslangValidator");
    if brew.exists() {
        return brew.to_path_buf();
    }
    PathBuf::from("glslangValidator")
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders");
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let glslang = compiler();
    for name in SHADERS {
        let src = root.join(format!("{name}.comp"));
        println!("cargo:rerun-if-changed={}", src.display());
        let dst = out.join(format!("{name}.spv"));
        let status = Command::new(&glslang)
            .args(["-V", "--target-env", "vulkan1.1", "-o"])
            .arg(&dst)
            .arg(&src)
            .status()
            .unwrap_or_else(|e| panic!("cannot run {}: {e} (install it with `brew install glslang`)", glslang.display()));
        assert!(status.success(), "glslangValidator failed for {}", src.display());
    }
    println!("cargo:rerun-if-env-changed=GLSLANG_VALIDATOR");
}
