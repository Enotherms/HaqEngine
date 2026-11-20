//! Settings read from the environment at load time.

use std::path::PathBuf;

/// How generated frames are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Motion-compensated interpolation (block matching + warp).
    Battery,
    /// The real frame shown twice (M1).
    Copy,
    /// FSR 3 optical flow at half resolution with HaqEngine's per-pixel blend (M4).
    Quality,
}

/// Settings HaqEngine reads once when it loads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// Where to write the log, or `None` for no log.
    pub log_path: Option<PathBuf>,
    /// Frames shown per real frame: 1 (pass-through) to `MAX_MULTIPLIER`.
    pub multiplier: u32,
    /// Set when the requested multiplier was clamped.
    pub multiplier_note: Option<String>,
    pub mode: Mode,
    /// Set when `HAQENGINE_MODE` held an unknown value.
    pub mode_note: Option<String>,
    /// Notes from reading `conf.toml` at start.
    pub config_notes: Vec<String>,
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
            None => (1, None),
            Some(v) => match v.parse::<u32>() {
                Ok(m @ 1..=crate::pacing::MAX_MULTIPLIER) => (m, None),
                _ => (1, Some(format!("unknown HAQENGINE_MULTIPLIER {v:?}; using 1"))),
            },
        };
        let (mode, mode_note) = match get("HAQENGINE_MODE").map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("battery") => (Mode::Battery, None),
            Some("copy") => (Mode::Copy, None),
            Some("quality") => (Mode::Quality, None),
            Some(other) => (Mode::Battery, Some(format!("unknown HAQENGINE_MODE {other:?}; using battery"))),
        };
        Settings { log_path, multiplier, multiplier_note, mode, mode_note, config_notes: Vec::new() }
    }
}

impl Settings {
    /// Settings from the environment over the config file's profile for `process`: a variable that
    /// is set wins, otherwise the game's section, then `[default]`, then the built-in values.
    pub fn from_lookup_and_file(get: impl Fn(&str) -> Option<String>, file: Option<&crate::config::FileConfig>, process: &str) -> Settings {
        let profile = crate::config::resolve(file, process, get("HAQENGINE_PROFILE").as_deref().filter(|p| !p.is_empty()));
        let set = |k: &str| get(k).is_some_and(|v| !v.trim().is_empty());
        let (mode_set, multiplier_set) = (set("HAQENGINE_MODE"), set("HAQENGINE_MULTIPLIER"));
        let mut s = Settings::from_lookup(get);
        if !mode_set {
            s.mode = profile.mode.unwrap_or(Mode::Battery);
        }
        if !multiplier_set {
            s.multiplier = profile.multiplier.unwrap_or(1);
        }
        s
    }
}

impl Settings {
    /// Warnings about the environment, to log once when HaqEngine loads.
    pub fn notes(&self) -> Vec<String> {
        self.mode_note.iter().chain(self.multiplier_note.iter()).chain(self.config_notes.iter()).cloned().collect()
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
    fn multiplier_accepts_1_to_4() {
        for m in 1..=4u32 {
            let s = Settings::from_lookup(env(&[("HAQENGINE_MULTIPLIER", &m.to_string())]));
            assert_eq!((s.multiplier, s.multiplier_note), (m, None), "{m}");
        }
    }

    #[test]
    fn multiplier_rejects_5_and_garbage_with_note() {
        for v in ["5", "0", "x"] {
            let s = Settings::from_lookup(env(&[("HAQENGINE_MULTIPLIER", v)]));
            assert_eq!(s.multiplier, 1, "{v}");
            assert!(s.multiplier_note.is_some(), "{v}");
        }
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
    fn mode_parses_quality() {
        assert_eq!(Settings::from_lookup(env(&[("HAQENGINE_MODE", "quality")])).mode, Mode::Quality);
        assert_eq!(Settings::from_lookup(env(&[("HAQENGINE_MODE", " Quality ")])).mode, Mode::Quality);
        assert!(Settings::from_lookup(env(&[("HAQENGINE_MODE", "quality")])).mode_note.is_none());
    }

    #[test]
    fn mode_rejects_garbage_with_note() {
        let s = Settings::from_lookup(env(&[("HAQENGINE_MODE", "turbo")]));
        assert_eq!(s.mode, Mode::Battery);
        assert!(s.mode_note.unwrap().contains("turbo"));
    }

    #[test]
    fn notes_report_mode_and_multiplier_problems() {
        let s = Settings::from_lookup(env(&[("HAQENGINE_MODE", "cpy"), ("HAQENGINE_MULTIPLIER", "9")]));
        let notes = s.notes();
        assert_eq!(notes.len(), 2);
        assert!(notes.iter().any(|n| n.contains("cpy")));
        assert!(notes.iter().any(|n| n.contains("\"9\"")));
        assert!(Settings::from_lookup(env(&[])).notes().is_empty());
    }

    #[test]
    fn process_name_trims() {
        assert_eq!(process_name("MonsterHunterWi\n"), "MonsterHunterWi");
        assert_eq!(process_name("\n"), "unknown");
    }
}
