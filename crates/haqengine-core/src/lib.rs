//! Platform-independent parts of HaqEngine: settings, logging and handle bookkeeping.

pub mod blend;
pub mod config;
pub mod flow;
pub mod frame_change;
pub mod fsr3_flow;
pub mod fsr3_interp;
pub mod handle_map;
pub mod image;
pub mod log;
pub mod packed;
pub mod pacing;
pub mod quality;
pub mod reference;
pub mod settings;
pub mod stats;

pub use handle_map::HandleMap;
pub use image::{checker, gradient, noise, shift, Image, Rgba8};
pub use log::{Logger, LOG_CAP_BYTES};
pub use packed::PackedFormat;
pub use pacing::{blend_times, MAX_GENERATED, MAX_MULTIPLIER};
pub use reference::{blend, compare_f32, compare_rg32f, compare_rgba8, luma_downsample, CompareReport, Mismatch};
pub use settings::{process_name, Mode, Settings};
pub use stats::{should_log, FrameStats, Summary, STATS_CAPACITY};
