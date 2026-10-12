use std::{fs, process::Command};

use rkern::{collision::Sphere, cuda, fk::FkDag, urdf};

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
fn batched_rrt_and_connect_solve_and_report_failures_for_each_precision() {
    let fk = robot();
    let spheres = [
        rkern::collision::Sphere {
            center: cuda::SphereCenter {
                link_index: 3,
                position: [0.0; 3],
            },
            radius: 0.05,
        },
        // The contained sphere exercises group bounds without changing the obstacle model.
        rkern::collision::Sphere {
            center: cuda::SphereCenter {
                link_index: 3,
                position: [0.0, 0.0, 0.025],
            },
            radius: 0.025,
        },
    ];
    for math in [
        cuda::MathMode::F32,
        cuda::MathMode::F64,
        cuda::MathMode::F32Sfu,
    ] {
        check_planner(
            &fk,
            &spheres,
            &[1, 0],
            math,
            "rrt",
            include_str!("fixtures/rrt.cu"),
        );
    }
}

#[test]
#[ignore = "requires CUDA toolkit and GPU; set NVCC and CXX if needed"]
fn fixed_spheres_reject_collisions_for_each_environment_and_solve() {
    let fk = FkDag::from_robot(
        &urdf::read_from_string(
            r#"<robot name="fixed_spheres">
<link name="root"/><link name="turn"/><link name="passive"/><link name="tip"/>
<joint name="yaw" type="continuous"><parent link="root"/><child link="turn"/><axis xyz="0 0 1"/></joint>
<joint name="passive" type="prismatic"><parent link="root"/><child link="passive"/><origin xyz="5 0 0"/><axis xyz="1 0 0"/></joint>
<joint name="child" type="prismatic"><parent link="turn"/><child link="tip"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/></joint>
</robot>"#,
        )
        .unwrap(),
    )
    .unwrap();
    let spheres = [
        ("root", [-5.0, 0.0, 0.0]),
        ("passive", [0.0; 3]),
        ("turn", [0.0, 0.0, 0.5]),
        ("tip", [0.0; 3]),
    ]
    .map(|(name, position)| Sphere {
        center: cuda::SphereCenter {
            link_index: fk
                .link_poses()
                .iter()
                .position(|link| link.name == name)
                .unwrap(),
            position,
        },
        radius: 0.1,
    });
    for math in [
        cuda::MathMode::F32,
        cuda::MathMode::F64,
        cuda::MathMode::F32Sfu,
    ] {
        for count in [0, 3, 4] {
            let fixture = format!(
                "#define TEST_SPHERE_COUNT {count}\n{}",
                include_str!("fixtures/rrt_static.cu"),
            );
            check_planner(
                &fk,
                &spheres[..count],
                &[0],
                math,
                &format!("rrt-static-{count}"),
                &fixture,
            );
        }
    }
}

fn check_planner(
    fk: &FkDag,
    spheres: &[Sphere],
    controls: &[usize],
    math: cuda::MathMode,
    name: &str,
    fixture: &str,
) {
    let directory = directory(&format!("{name}-{math:?}"));
    fs::write(
        directory.join("rrt.cuh"),
        cuda::generate_rrt_kernel(
            fk,
            spheres,
            controls,
            cuda::Options {
                math,
                ..cuda::Options::default()
            },
        )
        .unwrap(),
    )
    .unwrap();
    fs::write(directory.join("main.cu"), fixture).unwrap();
    fs::write(
        directory.join("broad_phase_checks.h"),
        include_str!("fixtures/broad_phase_checks.h"),
    )
    .unwrap();
    let namespace = if math == cuda::MathMode::F64 {
        "rkern_f64"
    } else {
        "rkern_f32"
    };
    let mut command = cuda_compiler("c++14");
    run(command
        .arg("-O2")
        .arg(format!("-DRKERN_MODEL_NAMESPACE={namespace}"))
        .arg(directory.join("main.cu"))
        .arg("-o")
        .arg(directory.join("check")));
    run(&mut Command::new(directory.join("check")));
}
