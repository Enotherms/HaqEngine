//! Runs HaqEngine's shaders on the GPU and checks them against CPU references.
//!
//! Usage: `haqengine-gpu-tests [--validation] [--filter <text>] [--self-test-fail]`

mod cases;
mod fsr3;
mod gpu;
mod report;

use report::{exit_code, format_line, is_fatal, outcome};

struct Args {
    validation: bool,
    filter: Option<String>,
    self_test_fail: bool,
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args { validation: false, filter: None, self_test_fail: false };
    let mut args = args.peekable();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--validation" => parsed.validation = true,
            "--self-test-fail" => parsed.self_test_fail = true,
            "--filter" => parsed.filter = Some(args.next().ok_or("--filter needs a value")?),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(parsed)
}

fn main() {
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let gpu = match gpu::Gpu::new(args.validation) {
        Ok(g) => g,
        Err(e) => {
            println!("FAIL  setup: {e}");
            std::process::exit(1);
        }
    };
    println!("device: {}", gpu.device_name);
    println!("subgroup size {} (arithmetic in compute: {})", gpu.subgroup_size, gpu.subgroup_ok);
    let mut outcomes = Vec::new();
    for case in cases::all_cases(args.self_test_fail) {
        if args.filter.as_deref().is_some_and(|f| !case.name.contains(f)) {
            continue;
        }
        let result = (case.run)(&gpu);
        println!("{}", format_line(&case.name, &result));
        outcomes.push(outcome(&result));
        if is_fatal(&result) {
            println!("stopping: the GPU is no longer usable");
            break;
        }
    }
    let passed = outcomes.iter().filter(|o| **o == report::Outcome::Pass).count();
    let errors = gpu.validation_errors();
    println!("{passed} passed, {} failed, validation errors {errors}", outcomes.len() - passed);
    let code = exit_code(&outcomes, errors);
    drop(gpu);
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_reads_flags() {
        let a = parse_args(["--validation", "--filter", "blend"].map(String::from).into_iter()).unwrap();
        assert!(a.validation && !a.self_test_fail);
        assert_eq!(a.filter.as_deref(), Some("blend"));
        assert!(parse_args(["--filter"].map(String::from).into_iter()).is_err());
        assert!(parse_args(["--nope"].map(String::from).into_iter()).is_err());
    }
}
