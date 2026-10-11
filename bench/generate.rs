use std::fs;
use std::path::PathBuf;

use rkern::{collision, cuda, fk::FkDag, urdf};
use serde::Deserialize;
use serde_json::json;

#[path = "../examples/common/mod.rs"]
mod common;

#[derive(Deserialize)]
struct Controls {
    control_joint_names: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1).peekable();
    let (options, precision) = common::options(&mut args)?;
    let directory = PathBuf::from(args.next().ok_or("missing output directory")?);
    if args.next().is_some() {
        return Err("expected codegen options followed by one output directory".into());
    }
    let robot = urdf::read_from_string(include_str!("../assets/fetch/fetch.urdf"))?;
    let yaml = include_str!("../assets/fetch/collision.yaml");
    let fk = FkDag::from_robot(&robot)?;
    let spheres = collision::read_spheres(yaml, &fk)?;
    fs::create_dir_all(&directory)?;
    fs::write(
        directory.join("collision.cuh"),
        cuda::generate_collision_kernel(&fk, &spheres, options)?,
    )?;

    let controls: Controls = serde_saphyr::from_str(yaml)?;
    let mut bounds = Vec::new();
    for name in &controls.control_joint_names {
        let index = fk
            .joint_names()
            .iter()
            .position(|joint| joint == name)
            .ok_or_else(|| format!("control joint {name} is not an FK input"))?;
        let joint = robot
            .joints
            .iter()
            .find(|joint| &joint.name == name)
            .ok_or_else(|| format!("control joint {name} is not in the URDF"))?;
        let (lower, upper) = if joint.joint_type == urdf::JointType::Continuous {
            (-std::f64::consts::TAU, std::f64::consts::TAU)
        } else {
            (joint.limit.lower, joint.limit.upper)
        };
        if !lower.is_finite() || !upper.is_finite() || lower >= upper {
            return Err(format!("invalid sampling bounds for {name}").into());
        }
        bounds.push(json!({"name": name, "index": index, "lower": lower, "upper": upper}));
    }
    let model = json!({
        "input_joint_names": fk.joint_names(),
        "control_joints": bounds,
        "sphere_count": spheres.len(),
        "scalar": match precision { cuda::Precision::F32 => "float", cuda::Precision::F64 => "double" },
        "math": format!("{:?}", options.math),
        "optimization": format!("{:?}", options.optimization),
        "scheduling": format!("{:?}", options.scheduling),
    });
    fs::write(
        directory.join("model.json"),
        serde_json::to_vec_pretty(&model)?,
    )?;
    Ok(())
}
