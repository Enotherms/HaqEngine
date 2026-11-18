//! Platform-independent parts of HaqEngine: settings, logging and handle bookkeeping.

pub mod handle_map;
pub mod log;
pub mod settings;

pub use handle_map::HandleMap;
pub use log::{Logger, LOG_CAP_BYTES};
pub use settings::{process_name, Settings};
