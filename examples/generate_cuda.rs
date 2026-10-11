use std::io::{self, Write};

use rkern::{cuda, fk::FkDag, urdf};

mod common;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1).peekable();
    let (options, _) = common::options(&mut args)?;
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
    let source = cuda::generate_fk_kernel(&fk, link_index, options)?;
    eprintln!("Link {link_index}: {}", fk.link_poses()[link_index].name);
    eprintln!("Math: {:?}", options.math);
    eprintln!("Scheduling: {:?}", options.scheduling);
    eprintln!("Input joints: {:?}", fk.joint_names());
    io::stdout().lock().write_all(source.as_bytes())?;
    Ok(())
}
