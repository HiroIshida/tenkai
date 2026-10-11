use std::io::{self, Write};

use rkern::{collision, cuda, fk::FkDag, urdf};

mod common;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1).peekable();
    let (options, _) = common::options(&mut args)?;
    let (robot, yaml) = match (args.next(), args.next()) {
        (None, None) => (
            urdf::read_from_string(include_str!("../assets/fetch/fetch.urdf"))?,
            include_str!("../assets/fetch/collision.yaml").to_owned(),
        ),
        (Some(urdf_path), Some(yaml_path)) => (
            urdf::read_file(urdf_path)?,
            std::fs::read_to_string(yaml_path)?,
        ),
        _ => return Err("provide both robot.urdf and collision.yaml, or neither".into()),
    };
    if args.next().is_some() {
        return Err("usage: cargo run --example generate_collision_cuda -- [--float|--double] [--sfu|--no-sfu] [--scheduling|--no-scheduling|--live-budget N] [robot.urdf collision.yaml]".into());
    }
    let fk = FkDag::from_robot(&robot)?;
    let spheres = collision::read_spheres(&yaml, &fk)?;
    let source = cuda::generate_collision_kernel(&fk, &spheres, options)?;
    eprintln!(
        "Math: {:?}; scheduling: {:?}",
        options.math, options.scheduling
    );
    eprintln!("Input joints: {:?}", fk.joint_names());
    eprintln!(
        "Robot spheres: {}; output: [count] uint8 (1 = collision)",
        spheres.len()
    );
    io::stdout().lock().write_all(source.as_bytes())?;
    Ok(())
}
