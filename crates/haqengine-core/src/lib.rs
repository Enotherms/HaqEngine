//! Platform-independent parts of HaqEngine: settings, logging and handle bookkeeping.

pub mod flow;
pub mod handle_map;
pub mod image;
pub mod log;
pub mod reference;
pub mod settings;
pub mod stats;

pub use handle_map::HandleMap;
pub use image::{checker, gradient, noise, shift, Image, Rgba8};
pub use log::{Logger, LOG_CAP_BYTES};
pub use reference::{blend, compare_f32, compare_rgba8, luma_downsample, CompareReport, Mismatch};
pub use settings::{process_name, Settings};
pub use stats::{should_log, FrameStats, Summary, STATS_CAPACITY};
