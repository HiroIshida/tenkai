use std::{fs, process::Command};

use rkern::{cuda, fk::FkDag, urdf};

#[path = "common/compile.rs"]
mod compile;
use compile::{cuda_compiler, directory, run};

fn robot() -> FkDag {
    FkDag::from_robot(
        &urdf::read_from_string(
            r#"<robot name="cartesian">
<link name="root"/><link name="x"/><link name="y"/><link name="z"/>
<joint name="x" type="prismatic"><parent link="root"/><child link="x"/><axis xyz="1 0 0"/></joint>
<joint name="y" type="prismatic"><parent link="x"/><child link="y"/><axis xyz="0 1 0"/></joint>
<joint name="z" type="prismatic"><parent link="y"/><child link="z"/><axis xyz="0 0 1"/></joint>
</robot>"#,
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn controls_must_be_nonempty_unique_fk_inputs() {
    let fk = robot();
    for controls in [&[][..], &[3][..], &[0, 0][..]] {
        assert!(cuda::generate_rrt_kernel(&fk, &[], controls, cuda::Options::default()).is_err());
    }
}

#[test]
#[ignore = "requires CUDA toolkit and GPU; set NVCC and CXX if needed"]
fn batched_rrt_solves_and_reports_failures_for_each_precision() {
    let fk = robot();
    let spheres = [rkern::collision::Sphere {
        center: cuda::SphereCenter {
            link_index: 3,
            position: [0.0; 3],
        },
        radius: 0.05,
    }];
    for math in [
        cuda::MathMode::F32,
        cuda::MathMode::F64,
        cuda::MathMode::F32Sfu,
    ] {
        let directory = directory(&format!("rrt-{math:?}"));
        fs::write(
            directory.join("rrt.cuh"),
            cuda::generate_rrt_kernel(
                &fk,
                &spheres,
                &[1, 0],
                cuda::Options {
                    math,
                    ..cuda::Options::default()
                },
            )
            .unwrap(),
        )
        .unwrap();
        fs::write(directory.join("main.cu"), include_str!("fixtures/rrt.cu")).unwrap();
        let namespace = if math == cuda::MathMode::F64 {
            "rkern_f64"
        } else {
            "rkern_f32"
        };
        let mut command = cuda_compiler("c++14");
        command.arg("-O2");
        run(command
            .arg(format!("-DRKERN_MODEL_NAMESPACE={namespace}"))
            .arg(directory.join("main.cu"))
            .arg("-o")
            .arg(directory.join("check")));
        run(&mut Command::new(directory.join("check")));
    }
}
