//! Settings read from the environment at load time.

use std::path::PathBuf;

/// Settings HaqEngine reads once when it loads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// Where to write the log, or `None` for no log.
    pub log_path: Option<PathBuf>,
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
        Settings { log_path }
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
    fn process_name_trims() {
        assert_eq!(process_name("MonsterHunterWi\n"), "MonsterHunterWi");
        assert_eq!(process_name("\n"), "unknown");
    }
}
