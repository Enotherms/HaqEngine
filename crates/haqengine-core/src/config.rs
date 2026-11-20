//! `conf.toml`: per-game mode and multiplier, read at start and again whenever the file changes.

use crate::pacing::MAX_MULTIPLIER;
use crate::settings::Mode;

/// Settings one section of the file gives; unset keys fall through to the next level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Profile {
    pub mode: Option<Mode>,
    pub multiplier: Option<u32>,
}

impl Profile {
    fn over(self, base: Profile) -> Profile {
        Profile { mode: self.mode.or(base.mode), multiplier: self.multiplier.or(base.multiplier) }
    }
}

/// The parsed file: `[default]` and each `[game."name"]` in file order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FileConfig {
    pub default: Profile,
    pub games: Vec<(String, Profile)>,
}

/// The mode name used in the file and in `HAQENGINE_MODE`.
pub fn parse_mode(name: &str) -> Option<Mode> {
    match name.trim().to_ascii_lowercase().as_str() {
        "battery" => Some(Mode::Battery),
        "copy" => Some(Mode::Copy),
        "quality" => Some(Mode::Quality),
        _ => None,
    }
}

enum Value {
    Text(String),
    Number(i64),
}

fn strip_comment(line: &str) -> &str {
    let mut quoted = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '#' if !quoted => return &line[..i],
            _ => {}
        }
    }
    line
}

fn parse_value(raw: &str) -> Option<Value> {
    let raw = raw.trim();
    if let Some(rest) = raw.strip_prefix('"') {
        let text = rest.strip_suffix('"')?;
        return (!text.contains('"')).then(|| Value::Text(text.to_string()));
    }
    raw.parse::<i64>().ok().map(Value::Number)
}

/// Parses the supported subset of TOML: `[default]` and `[game."name"]` sections holding
/// `mode = "battery" | "copy" | "quality"` and `multiplier = 1..=4`, with `#` comments. Unknown
/// sections and keys, and out-of-range values, are skipped with a note; a malformed line is an
/// error naming its line number.
pub fn parse_config(text: &str) -> Result<(FileConfig, Vec<String>), String> {
    let mut config = FileConfig::default();
    let mut notes = Vec::new();
    let mut section: Option<Option<usize>> = None;
    for (n, line) in text.lines().enumerate() {
        let n = n + 1;
        let line = strip_comment(line).trim();
        if line.is_empty() {
            continue;
        }
        if let Some(head) = line.strip_prefix('[') {
            let name = head.strip_suffix(']').ok_or(format!("line {n}: section header without ']'"))?.trim();
            section = if name == "default" {
                Some(None)
            } else if let Some(game) = name.strip_prefix("game.").and_then(|g| g.strip_prefix('"')).and_then(|g| g.strip_suffix('"')) {
                config.games.push((game.to_string(), Profile::default()));
                Some(Some(config.games.len() - 1))
            } else {
                notes.push(format!("config line {n}: unknown section [{name}] ignored"));
                None
            };
            continue;
        }
        let (key, raw) = line.split_once('=').ok_or(format!("line {n}: expected key = value"))?;
        let key = key.trim();
        let value = parse_value(raw).ok_or(format!("line {n}: bad value {:?}", raw.trim()))?;
        let Some(target) = section else { continue };
        let profile = match target {
            None => &mut config.default,
            Some(i) => &mut config.games[i].1,
        };
        match (key, value) {
            ("mode", Value::Text(t)) => match parse_mode(&t) {
                Some(m) => profile.mode = Some(m),
                None => notes.push(format!("config line {n}: unknown mode {t:?} ignored")),
            },
            ("multiplier", Value::Number(m)) if (1..=MAX_MULTIPLIER as i64).contains(&m) => profile.multiplier = Some(m as u32),
            ("mode" | "multiplier", _) => notes.push(format!("config line {n}: bad {key} ignored")),
            _ => notes.push(format!("config line {n}: unknown key {key:?} ignored")),
        }
    }
    Ok((config, notes))
}

/// The profile for this process: `[default]`, overlaid by the game section chosen by
/// `HAQENGINE_PROFILE` (exact name) or else the first whose name starts with `process` (Linux
/// truncates process names to 15 bytes).
pub fn resolve(file: Option<&FileConfig>, process: &str, profile_env: Option<&str>) -> Profile {
    let Some(file) = file else { return Profile::default() };
    let game = match profile_env {
        Some(name) => file.games.iter().find(|(g, _)| g == name),
        None => file.games.iter().find(|(g, _)| !process.is_empty() && g.starts_with(process)),
    };
    game.map_or(file.default, |(_, p)| p.over(file.default))
}

/// The settings that can change while a game runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Live {
    pub mode: Mode,
    pub multiplier: u32,
}

/// Applies a re-read file to the running settings. A file that failed to parse keeps `current`; a
/// multiplier above `cap` (what the swapchain was created for) is held at `cap` until the game
/// recreates its swapchain. Unset keys fall back to the built-in defaults (Battery, 1). Returns the
/// new settings and a note to log, if any.
pub fn apply_reload(current: Live, reloaded: Result<Profile, String>, cap: u32) -> (Live, Option<String>) {
    let profile = match reloaded {
        Ok(p) => p,
        Err(e) => return (current, Some(format!("config not applied ({e}); keeping mode {:?} multiplier {}", current.mode, current.multiplier))),
    };
    let wanted = profile.multiplier.unwrap_or(1);
    let multiplier = wanted.min(cap.max(1));
    let note = (multiplier < wanted).then(|| format!("multiplier {wanted} applies after the game recreates its swapchain; using {multiplier}"));
    (Live { mode: profile.mode.unwrap_or(Mode::Battery), multiplier }, note)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{Mode, Settings};

    const SAMPLE: &str = r#"
# HaqEngine settings
[default]
mode = "battery"
multiplier = 2

[game."Client-Win64-Shipping"]
multiplier = 3

[game."MonsterHunterWilds.exe"]
mode = "quality"
"#;

    fn parsed(text: &str) -> FileConfig {
        parse_config(text).expect("parses").0
    }

    #[test]
    fn parses_default_and_game_sections() {
        let c = parsed(SAMPLE);
        assert_eq!(c.default, Profile { mode: Some(Mode::Battery), multiplier: Some(2) });
        assert_eq!(c.games.len(), 2);
        assert_eq!(c.games[0], ("Client-Win64-Shipping".to_string(), Profile { mode: None, multiplier: Some(3) }));
        assert_eq!(c.games[1].1.mode, Some(Mode::Quality));
    }

    #[test]
    fn comments_blank_lines_and_unknown_keys() {
        let (c, notes) = parse_config("\n# hi\n[default]\nmultiplier = 3 # inline\nspeed = 9\n[other]\nx = 1\n").expect("parses");
        assert_eq!(c.default.multiplier, Some(3));
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(notes.iter().any(|n| n.contains("speed")));
        assert!(notes.iter().any(|n| n.contains("other")));
    }

    #[test]
    fn bad_values_are_notes_not_errors() {
        let (c, notes) = parse_config("[default]\nmultiplier = 7\nmode = \"turbo\"\n").expect("parses");
        assert_eq!(c.default, Profile::default());
        assert_eq!(notes.len(), 2, "{notes:?}");
    }

    #[test]
    fn syntax_error_reports_line() {
        let e = parse_config("[default]\nmultiplier = 2\nmode = \"battery\n").unwrap_err();
        assert!(e.contains("line 3"), "{e}");
        assert!(parse_config("[default\n").unwrap_err().contains("line 1"));
        assert!(parse_config("multiplier\n").unwrap_err().contains("line 1"));
    }

    #[test]
    fn empty_file_means_defaults() {
        let (c, notes) = parse_config("").expect("parses");
        assert_eq!(c, FileConfig::default());
        assert!(notes.is_empty());
        assert_eq!(resolve(Some(&c), "game", None), Profile::default());
    }

    #[test]
    fn game_matches_truncated_comm() {
        let c = parsed(SAMPLE);
        assert_eq!(resolve(Some(&c), "Client-Win64-Sh", None), Profile { mode: Some(Mode::Battery), multiplier: Some(3) });
        assert_eq!(resolve(Some(&c), "vkcube", None), Profile { mode: Some(Mode::Battery), multiplier: Some(2) });
        assert_eq!(resolve(None, "vkcube", None), Profile::default());
    }

    #[test]
    fn profile_env_selects_by_exact_name() {
        let c = parsed(SAMPLE);
        assert_eq!(resolve(Some(&c), "vkcube", Some("MonsterHunterWilds.exe")).mode, Some(Mode::Quality));
        assert_eq!(resolve(Some(&c), "vkcube", Some("MonsterHunter")).mode, Some(Mode::Battery), "exact name only");
    }

    #[test]
    fn env_overrides_file_at_start() {
        let c = parsed(SAMPLE);
        let none = |_: &str| None;
        let s = Settings::from_lookup_and_file(none, Some(&c), "Client-Win64-Sh");
        assert_eq!((s.mode, s.multiplier), (Mode::Battery, 3));
        let env = |k: &str| (k == "HAQENGINE_MULTIPLIER").then(|| "2".to_string());
        let s = Settings::from_lookup_and_file(env, Some(&c), "Client-Win64-Sh");
        assert_eq!(s.multiplier, 2);
        let env = |k: &str| (k == "HAQENGINE_PROFILE").then(|| "MonsterHunterWilds.exe".to_string());
        assert_eq!(Settings::from_lookup_and_file(env, Some(&c), "vkcube").mode, Mode::Quality);
        let s = Settings::from_lookup_and_file(none, None, "vkcube");
        assert_eq!((s.mode, s.multiplier), (Mode::Battery, 1));
    }

    #[test]
    fn bad_config_keeps_previous() {
        let current = Live { mode: Mode::Battery, multiplier: 3 };
        let (live, note) = apply_reload(current, Err("line 2: missing =".into()), 3);
        assert_eq!(live, current);
        assert!(note.unwrap().contains("line 2"));
    }

    #[test]
    fn reload_applies_profile_and_falls_back_to_built_in() {
        let current = Live { mode: Mode::Battery, multiplier: 3 };
        let (live, note) = apply_reload(current, Ok(Profile { mode: Some(Mode::Copy), multiplier: Some(2) }), 3);
        assert_eq!((live, note), (Live { mode: Mode::Copy, multiplier: 2 }, None));
        let (live, _) = apply_reload(current, Ok(Profile::default()), 3);
        assert_eq!(live, Live { mode: Mode::Battery, multiplier: 1 });
    }

    #[test]
    fn multiplier_above_cap_is_clamped_with_note() {
        let current = Live { mode: Mode::Battery, multiplier: 2 };
        let (live, note) = apply_reload(current, Ok(Profile { mode: None, multiplier: Some(4) }), 2);
        assert_eq!(live.multiplier, 2);
        assert!(note.unwrap().contains("multiplier 4 applies after the game recreates its swapchain"));
    }
}
