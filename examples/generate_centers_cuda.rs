use std::io::{self, Write};

use rkern::{cuda, fk::FkDag, scheduling, urdf};

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
    let robot = match args.next() {
        Some(path) => urdf::read_file(path)?,
        None => urdf::read_from_string(DEMO)?,
    };
    if args.next().is_some() {
        return Err(
            "usage: cargo run --example generate_centers_cuda -- [--float|--double] [--sfu|--no-sfu] [--scheduling|--no-scheduling|--live-budget N] [sphere_robot.urdf]".into(),
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
    let source = cuda::generate_centers_kernel(&fk, &spheres, cuda::Options { math, scheduling })?;
    eprintln!("Math: {math:?}");
    eprintln!("Scheduling: {scheduling:?}");
    eprintln!("Input joints: {:?}", fk.joint_names());
    eprintln!("Output: [count][{}][3] {precision:?}", spheres.len());
    io::stdout().lock().write_all(source.as_bytes())?;
    Ok(())
}
