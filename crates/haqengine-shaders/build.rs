use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const SHADERS: &[&str] = &["luma_downsample", "blend", "luma_reduce", "block_match", "warp_blend", "flow_median", "luma_downsample_packed", "warp_blend_packed"];

const FSR3_PASSES: &[(&str, &str)] = &[
    ("of_prepare_luma", "ffx_opticalflow_prepare_luma_pass"),
    ("of_luminance_pyramid", "ffx_opticalflow_compute_luminance_pyramid_pass"),
    ("of_scd_histogram", "ffx_opticalflow_generate_scd_histogram_pass"),
    ("of_scd_divergence", "ffx_opticalflow_compute_scd_divergence_pass"),
    ("of_search", "ffx_opticalflow_compute_optical_flow_advanced_pass_v5"),
    ("of_filter", "ffx_opticalflow_filter_optical_flow_pass_v5"),
    ("of_scale", "ffx_opticalflow_scale_optical_flow_advanced_pass_v5"),
];

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
            .args(["-V", "--target-env", "vulkan1.0", "-o"])
            .arg(&dst)
            .arg(&src)
            .status()
            .unwrap_or_else(|e| panic!("cannot run {}: {e} (install it with `brew install glslang`)", glslang.display()));
        assert!(status.success(), "glslangValidator failed for {}", src.display());
    }
    let ffx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/fidelityfx");
    println!("cargo:rerun-if-changed={}", ffx.display());
    let (gpu, of) = (ffx.join("gpu"), ffx.join("gpu/opticalflow"));
    for (name, file) in FSR3_PASSES {
        let src = ffx.join("vk/opticalflow").join(format!("{file}.glsl"));
        let dst = out.join(format!("{name}.spv"));
        let status = Command::new(&glslang)
            .args(["-V", "--target-env", "vulkan1.1", "-S", "comp", "-DFFX_GPU=1", "-DFFX_GLSL=1"])
            .arg("-DFFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT=0")
            .arg(format!("-I{}", gpu.display()))
            .arg(format!("-I{}", of.display()))
            .arg("-o")
            .arg(&dst)
            .arg(&src)
            .status()
            .unwrap_or_else(|e| panic!("cannot run {}: {e}", glslang.display()));
        assert!(status.success(), "glslangValidator failed for {}", src.display());
    }
    println!("cargo:rerun-if-env-changed=GLSLANG_VALIDATOR");
}
