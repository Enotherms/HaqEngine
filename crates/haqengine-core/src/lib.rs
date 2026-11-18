//! Platform-independent parts of HaqEngine: settings, logging and handle bookkeeping.

pub mod handle_map;
pub mod log;
pub mod settings;
pub mod stats;

pub use handle_map::HandleMap;
pub use log::{Logger, LOG_CAP_BYTES};
pub use settings::{process_name, Settings};
pub use stats::{should_log, FrameStats, Summary, STATS_CAPACITY};
