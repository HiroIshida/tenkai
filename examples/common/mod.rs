use std::ffi::OsString;
use std::iter::Peekable;

use rkern::{cuda, scheduling};

pub fn options(
    args: &mut Peekable<impl Iterator<Item = OsString>>,
) -> Result<(cuda::Options, cuda::Precision), Box<dyn std::error::Error>> {
    let mut precision = cuda::Precision::default();
    let mut sfu = None;
    let mut scheduling = scheduling::Mode::default();
    loop {
        match args.peek().and_then(|arg| arg.to_str()) {
            Some("--float") => precision = cuda::Precision::F32,
            Some("--double") => precision = cuda::Precision::F64,
            Some("--sfu") => sfu = Some(true),
            Some("--no-sfu") => sfu = Some(false),
            Some("--scheduling") => {
                scheduling = scheduling::Mode::Live(scheduling::Options::default());
            }
            Some("--no-scheduling") => scheduling = scheduling::Mode::DagOrder,
            Some("--live-budget") => {
                args.next();
                let live_budget = args.next().ok_or("--live-budget requires an integer")?;
                let live_budget = live_budget.to_str().ok_or("invalid live budget")?.parse()?;
                scheduling = scheduling::Mode::Live(scheduling::Options { live_budget });
                continue;
            }
            _ => break,
        }
        args.next();
    }
    let math = match (precision, sfu.unwrap_or(precision == cuda::Precision::F32)) {
        (cuda::Precision::F64, true) => {
            return Err("--sfu requires float; CUDA has no double sin/cos intrinsics".into());
        }
        (cuda::Precision::F32, true) => cuda::MathMode::F32Sfu,
        (cuda::Precision::F32, false) => cuda::MathMode::F32,
        (cuda::Precision::F64, false) => cuda::MathMode::F64,
    };
    Ok((cuda::Options { math, scheduling }, precision))
}
