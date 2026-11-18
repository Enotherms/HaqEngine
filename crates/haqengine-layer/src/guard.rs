//! Keeps panics inside HaqEngine from ever unwinding into the game.

use crate::state::LOG;
use haqengine_core::HandleMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::LazyLock;

static REPORTED: LazyLock<HandleMap<()>> = LazyLock::new(HandleMap::new);

/// Runs `f`; if it panics, logs once per `name` and returns `fallback` instead.
pub fn guard<R>(name: &'static str, fallback: R, f: impl FnOnce() -> R) -> R {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            let key = name.as_ptr() as usize;
            if REPORTED.get(key).is_none() {
                REPORTED.insert(key, ());
                let what = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                LOG.log(&format!("error: panic in {name}: {what}; passing through"));
            }
            fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_returns_value() {
        assert_eq!(guard("t", 0, || 5), 5);
    }

    #[test]
    fn guard_returns_fallback_on_panic() {
        assert_eq!(guard("t", 7, || -> i32 { panic!("boom") }), 7);
    }
}
