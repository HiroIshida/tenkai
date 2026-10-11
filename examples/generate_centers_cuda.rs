use std::io::{self, Write};

use rkern::{cuda, fk::FkDag, urdf};

const DEMO: &str = r#"
<robot name="sphere_demo">
  <link name="base"/>
  <link name="arm">
    <collision><origin xyz="1 0 0"/><geometry><sphere radius="0.1"/></geometry></collision>
    <collision><origin xyz="0 2 0"/><geometry><sphere radius="0.2"/></geometry></collision>
  </link>
  <joint name="yaw" type="continuous">
    <parent link="base"/><child link="arm"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
  </joint>
</robot>
"#;

mod common;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1).peekable();
    let (options, precision) = common::options(&mut args)?;
    let robot = match args.next() {
        Some(path) => urdf::read_file(path)?,
        None => urdf::read_from_string(DEMO)?,
    };
    if args.next().is_some() {
        return Err(
            "usage: cargo run --example generate_centers_cuda -- [--float|--double] [--sfu|--no-sfu] [--optimization none|basic|algebraic] [--scheduling|--no-scheduling|--live-budget N] [sphere_robot.urdf]".into(),
        );
    }
    let fk = FkDag::from_robot(&robot)?;
    let mut spheres = Vec::new();
    for link in &robot.links {
        let link_index = fk
            .link_poses()
            .iter()
            .position(|pose| pose.name == link.name)
            .ok_or("URDF link missing from FK")?;
        for collision in &link.collision {
            if !matches!(collision.geometry, urdf::Geometry::Sphere { .. }) {
                return Err(format!(
                    "{}: this example requires sphere collision geometry",
                    link.name
                )
                .into());
            }
            spheres.push(cuda::SphereCenter {
                link_index,
                position: collision.origin.xyz.0,
            });
            eprintln!(
                "Sphere {}: {} at {:?}",
                spheres.len() - 1,
                link.name,
                collision.origin.xyz.0
            );
        }
    }
    if spheres.is_empty() {
        return Err("URDF contains no collision spheres".into());
    }
    let source = cuda::generate_centers_kernel(&fk, &spheres, options)?;
    eprintln!("Math: {:?}", options.math);
    eprintln!("Optimization: {:?}", options.optimization);
    eprintln!("Scheduling: {:?}", options.scheduling);
    eprintln!("Input joints: {:?}", fk.joint_names());
    eprintln!("Output: [count][{}][3] {precision:?}", spheres.len());
    io::stdout().lock().write_all(source.as_bytes())?;
    Ok(())
}
