//! Settings read from the environment at load time.

use std::path::PathBuf;

/// How generated frames are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Motion-compensated interpolation (block matching + warp).
    Battery,
    /// The real frame shown twice (M1).
    Copy,
}

/// Settings HaqEngine reads once when it loads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// Where to write the log, or `None` for no log.
    pub log_path: Option<PathBuf>,
    /// Frames shown per real frame: 1 (pass-through) or 2.
    pub multiplier: u32,
    /// Set when the requested multiplier was clamped.
    pub multiplier_note: Option<String>,
    pub mode: Mode,
    /// Set when `HAQENGINE_MODE` held an unknown value.
    pub mode_note: Option<String>,
}

impl Settings {
    /// Builds settings from an environment lookup (normally `std::env::var`).
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Settings {
        let non_empty = |k: &str| get(k).filter(|v| !v.is_empty());
        let log_path = match non_empty("HAQENGINE_LOG").as_deref() {
            Some("0") | Some("off") => None,
            Some(path) => Some(PathBuf::from(path)),
            None => non_empty("XDG_STATE_HOME")
                .map(PathBuf::from)
                .or_else(|| non_empty("HOME").map(|h| PathBuf::from(h).join(".local/state")))
                .map(|base| base.join("haqengine/haqengine.log")),
        };
        let (multiplier, multiplier_note) = match get("HAQENGINE_MULTIPLIER").as_deref().map(str::trim) {
            Some("2") => (2, None),
            Some(v @ ("3" | "4")) => (2, Some(format!("multiplier {v} requested; 3x/4x arrive in M5, using 2"))),
            _ => (1, None),
        };
        let (mode, mode_note) = match get("HAQENGINE_MODE").map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("battery") => (Mode::Battery, None),
            Some("copy") => (Mode::Copy, None),
            Some(other) => (Mode::Battery, Some(format!("unknown HAQENGINE_MODE {other:?}; using battery"))),
        };
        Settings { log_path, multiplier, multiplier_note, mode, mode_note }
    }
}

/// Turns the contents of `/proc/self/comm` into a process name.
pub fn process_name(comm: &str) -> String {
    let name = comm.trim();
    if name.is_empty() { "unknown".to_string() } else { name.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
    }

    #[test]
    fn settings_log_off() {
        assert_eq!(Settings::from_lookup(env(&[("HAQENGINE_LOG", "off"), ("HOME", "/home/deck")])).log_path, None);
        assert_eq!(Settings::from_lookup(env(&[("HAQENGINE_LOG", "0"), ("HOME", "/home/deck")])).log_path, None);
    }

    #[test]
    fn settings_log_explicit_path() {
        assert_eq!(
            Settings::from_lookup(env(&[("HAQENGINE_LOG", "/tmp/x.log")])).log_path,
            Some(PathBuf::from("/tmp/x.log"))
        );
    }

    #[test]
    fn settings_log_default_prefers_xdg_state() {
        assert_eq!(
            Settings::from_lookup(env(&[("XDG_STATE_HOME", "/s"), ("HOME", "/home/deck")])).log_path,
            Some(PathBuf::from("/s/haqengine/haqengine.log"))
        );
        assert_eq!(
            Settings::from_lookup(env(&[("HOME", "/home/deck")])).log_path,
            Some(PathBuf::from("/home/deck/.local/state/haqengine/haqengine.log"))
        );
        assert_eq!(Settings::from_lookup(env(&[])).log_path, None);
    }

    #[test]
    fn multiplier_defaults_to_1() {
        assert_eq!(Settings::from_lookup(env(&[])).multiplier, 1);
    }

    #[test]
    fn multiplier_accepts_1_and_2() {
        assert_eq!(Settings::from_lookup(env(&[("HAQENGINE_MULTIPLIER", "2")])).multiplier, 2);
        assert_eq!(Settings::from_lookup(env(&[("HAQENGINE_MULTIPLIER", "1")])).multiplier, 1);
    }

    #[test]
    fn multiplier_clamps_3_and_4_with_note() {
        let s = Settings::from_lookup(env(&[("HAQENGINE_MULTIPLIER", "4")]));
        assert_eq!(s.multiplier, 2);
        assert!(s.multiplier_note.unwrap().contains("M5"));
    }

    #[test]
    fn multiplier_rejects_garbage() {
        for v in ["0", "9", "x", "", "-2"] {
            assert_eq!(Settings::from_lookup(env(&[("HAQENGINE_MULTIPLIER", v)])).multiplier, 1, "{v}");
        }
    }

    #[test]
    fn mode_defaults_to_battery() {
        let s = Settings::from_lookup(env(&[]));
        assert_eq!(s.mode, Mode::Battery);
        assert!(s.mode_note.is_none());
    }

    #[test]
    fn mode_parses_copy_and_battery() {
        assert_eq!(Settings::from_lookup(env(&[("HAQENGINE_MODE", "copy")])).mode, Mode::Copy);
        assert_eq!(Settings::from_lookup(env(&[("HAQENGINE_MODE", "COPY")])).mode, Mode::Copy);
        assert_eq!(Settings::from_lookup(env(&[("HAQENGINE_MODE", "Battery")])).mode, Mode::Battery);
    }

    #[test]
    fn mode_rejects_garbage_with_note() {
        let s = Settings::from_lookup(env(&[("HAQENGINE_MODE", "turbo")]));
        assert_eq!(s.mode, Mode::Battery);
        assert!(s.mode_note.unwrap().contains("turbo"));
    }

    #[test]
    fn process_name_trims() {
        assert_eq!(process_name("MonsterHunterWi\n"), "MonsterHunterWi");
        assert_eq!(process_name("\n"), "unknown");
    }
}
