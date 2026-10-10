use std::io::{self, Write};

use rkern::{cuda, fk::FkDag, urdf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1).peekable();
    let precision = match args.peek().and_then(|arg| arg.to_str()) {
        Some("--float") => {
            args.next();
            cuda::Precision::F32
        }
        Some("--double") => {
            args.next();
            cuda::Precision::F64
        }
        _ => cuda::Precision::default(),
    };
    let selection = args
        .next()
        .ok_or("usage: cargo run --example generate_cuda -- [--float|--double] <link-index|--list> [robot.urdf]")?;
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
    let source = cuda::generate_fk_kernel_with_precision(&fk, link_index, precision)?;
    eprintln!("Link {link_index}: {}", fk.link_poses()[link_index].name);
    eprintln!("Input joints: {:?}", fk.joint_names());
    io::stdout().lock().write_all(source.as_bytes())?;
    Ok(())
}
