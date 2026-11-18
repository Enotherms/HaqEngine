//! A small, size-capped log file. Never fails and never panics: any I/O error turns it off.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Largest log file HaqEngine writes before it stops logging.
pub const LOG_CAP_BYTES: u64 = 1_048_576;

const LIMIT_LINE: &[u8] = b"log size limit reached\n";

struct Sink {
    file: File,
    written: u64,
    cap: u64,
}

/// Appends timestamped lines to one file, up to a size cap.
pub struct Logger {
    sink: Mutex<Option<Sink>>,
}

impl Logger {
    /// Opens (or creates) the log at `path`. With `None`, or on any error, the logger is off.
    pub fn open(path: Option<&Path>, cap_bytes: u64) -> Logger {
        let sink = path.and_then(|p| {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).ok()?;
            }
            let file = OpenOptions::new().create(true).append(true).open(p).ok()?;
            let written = file.metadata().ok()?.len();
            Some(Sink { file, written, cap: cap_bytes })
        });
        Logger { sink: Mutex::new(sink) }
    }

    /// A logger that writes nothing.
    pub fn disabled() -> Logger {
        Logger { sink: Mutex::new(None) }
    }

    /// Writes one line. Does nothing when the logger is off.
    pub fn log(&self, msg: &str) {
        let mut guard = self.sink.lock().unwrap_or_else(|e| e.into_inner());
        let Some(sink) = guard.as_mut() else { return };
        let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let line = format!("{secs} {} {msg}\n", std::process::id());
        let ok = if sink.written + line.len() as u64 > sink.cap {
            let _ = sink.file.write_all(LIMIT_LINE);
            false
        } else {
            sink.written += line.len() as u64;
            sink.file.write_all(line.as_bytes()).is_ok()
        };
        if !ok {
            *guard = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("haqengine-test-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn logger_writes_lines_and_creates_parent() {
        let path = scratch("write").join("sub").join("a.log");
        let log = Logger::open(Some(&path), LOG_CAP_BYTES);
        log.log("hello");
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.ends_with(" hello\n"), "{text:?}");
    }

    #[test]
    fn logger_stops_at_cap() {
        let path = scratch("cap").join("a.log");
        let log = Logger::open(Some(&path), 64);
        for _ in 0..100 {
            log.log("0123456789");
        }
        let text = fs::read_to_string(&path).unwrap();
        let limit = "log size limit reached\n";
        assert!(text.len() as u64 <= 64 + limit.len() as u64 + 32, "len {}", text.len());
        assert_eq!(text.matches("log size limit reached").count(), 1);
    }

    #[test]
    fn logger_unwritable_path_is_silent() {
        let dir = scratch("unwritable");
        let file = dir.join("file.txt");
        fs::write(&file, "x").unwrap();
        let log = Logger::open(Some(&file.join("a.log")), LOG_CAP_BYTES);
        log.log("x");
        Logger::disabled().log("x");
        Logger::open(None, LOG_CAP_BYTES).log("x");
    }
}
