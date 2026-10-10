use std::io::{self, Write};

use rkern::{cuda, fk::FkDag, scheduling, urdf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1).peekable();
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
    let selection = args
        .next()
        .ok_or("usage: cargo run --example generate_cuda -- [--float|--double] [--sfu|--no-sfu] [--scheduling|--no-scheduling|--live-budget N] <link-index|--list> [robot.urdf]")?;
    let robot = match args.next() {
        Some(path) => urdf::read_file(path)?,
        None => urdf::read_from_string(include_str!("../assets/fetch/fetch.urdf"))?,
    };
    if args.next().is_some() {
        return Err("expected a link index and an optional URDF path".into());
    }
    let fk = FkDag::from_robot(&robot)?;
    if selection == "--list" {
        println!("Input joints: {:?}", fk.joint_names());
        for (index, pose) in fk.link_poses().iter().enumerate() {
            println!("{index}: {}", pose.name);
        }
        return Ok(());
    }
    let link_index = selection
        .to_str()
        .ok_or("link index must be an integer")?
        .parse()?;
    let source = cuda::generate_fk_kernel(&fk, link_index, cuda::Options { math, scheduling })?;
    eprintln!("Link {link_index}: {}", fk.link_poses()[link_index].name);
    eprintln!("Math: {math:?}");
    eprintln!("Scheduling: {scheduling:?}");
    eprintln!("Input joints: {:?}", fk.joint_names());
    io::stdout().lock().write_all(source.as_bytes())?;
    Ok(())
}
