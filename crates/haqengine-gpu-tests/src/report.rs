//! Turns case results into PASS/FAIL lines and an exit code.

use crate::cases::CaseResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Pass,
    Fail,
}

pub fn outcome(r: &Result<CaseResult, String>) -> Outcome {
    match r {
        Ok(c) if c.report.over_tolerance == 0 => Outcome::Pass,
        _ => Outcome::Fail,
    }
}

pub fn format_line(name: &str, r: &Result<CaseResult, String>) -> String {
    let c = match r {
        Err(e) => return format!("FAIL  {name:<42} {e}"),
        Ok(c) => c,
    };
    let gpu = if c.gpu_us.is_nan() { "gpu n/a".to_string() } else { format!("gpu {:.1} us", c.gpu_us) };
    match c.report.worst {
        _ if c.report.over_tolerance == 0 => format!("PASS  {name:<42} max {:.1e}  {gpu}", c.report.max_diff),
        Some(w) => format!(
            "FAIL  {name:<42} {} over tolerance {}, worst ({},{}) diff {:.3}  {gpu}",
            c.report.over_tolerance, c.tolerance, w.x, w.y, w.diff
        ),
        None => format!("FAIL  {name:<42} size mismatch  {gpu}"),
    }
}

/// True when the GPU can no longer be trusted (hang or device loss): the run must stop.
pub fn is_fatal(r: &Result<CaseResult, String>) -> bool {
    matches!(r, Err(e) if e.contains("GPU timeout") || e.contains("ERROR_DEVICE_LOST"))
}

pub fn exit_code(results: &[Outcome], validation_errors: u32) -> i32 {
    if validation_errors == 0 && results.iter().all(|o| *o == Outcome::Pass) {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use haqengine_core::{CompareReport, Mismatch};

    fn result(over: u32) -> Result<CaseResult, String> {
        Ok(CaseResult {
            report: CompareReport {
                max_diff: if over > 0 { 0.2 } else { 3.0e-7 },
                worst: (over > 0).then_some(Mismatch { x: 12, y: 7, diff: 0.2 }),
                over_tolerance: over,
            },
            gpu_us: 41.2,
            tolerance: 1e-5,
        })
    }

    #[test]
    fn fence_timeout_is_failure() {
        let r = Err("GPU timeout".to_string());
        assert_eq!(outcome(&r), Outcome::Fail);
        assert!(format_line("x", &r).contains("GPU timeout"));
    }

    #[test]
    fn fatal_errors_stop_the_run() {
        assert!(is_fatal(&Err("GPU timeout".into())));
        assert!(is_fatal(&Err("submitting: ERROR_DEVICE_LOST".into())));
        assert!(!is_fatal(&Err("creating an image: ERROR_OUT_OF_DEVICE_MEMORY".into())));
        assert!(!is_fatal(&result(3)));
    }

    #[test]
    fn exit_code_rules() {
        assert_eq!(exit_code(&[Outcome::Pass, Outcome::Pass], 0), 0);
        assert_eq!(exit_code(&[Outcome::Pass, Outcome::Fail], 0), 1);
        assert_eq!(exit_code(&[Outcome::Pass], 2), 1);
    }

    #[test]
    fn format_line_shows_worst_pixel() {
        let pass = format_line("luma", &result(0));
        assert!(pass.starts_with("PASS") && pass.contains("41.2 us"), "{pass}");
        let fail = format_line("luma", &result(3));
        assert!(fail.starts_with("FAIL") && fail.contains("(12,7)"), "{fail}");
    }
}
