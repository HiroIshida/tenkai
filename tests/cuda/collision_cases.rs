use std::fmt::Write;

use rkern::{
    collision::{self, Sphere},
    cuda::{self, MathMode, SphereCenter},
    fk::FkDag,
    urdf,
};

use super::{cases::Case, cases::sphere_center, common};

pub(super) fn suite(options: cuda::Options) -> Vec<Case> {
    let root = FkDag::from_robot(
        &urdf::read_from_string(r#"<robot name="one"><link name="root"/></robot>"#).unwrap(),
    )
    .unwrap();
    let sphere = Sphere {
        center: SphereCenter {
            link_index: 0,
            position: [0.0; 3],
        },
        radius: 0.25,
    };
    let mut cases = Vec::new();
    // Each table includes contact, separation, containment, and an empty environment.
    for (name, setup) in [
        (
            "sphere",
            r#"
const sdf::Sphere shapes[] = {{{0,0,0},1}, {{0.5,0,0},0.25}, {{0.75,0,0},0.25}, {{0,0,0},0.25}};
const sdf::Environment envs[] = {
    {{0,0},{1,1},{0,0},{0,0}}, {{0,0},{2,1},{0,0},{0,0}},
    {{0,0},{3,1},{0,0},{0,0}}, {}
};
const sdf::Scene scene = {envs, nullptr, shapes, nullptr, nullptr};
"#,
        ),
        (
            "rotated_box",
            r#"
const sdf::Box shapes[] = {
    {{{0,0,0},{1,0,0,0,1,0,0,0,1}}, {2,4,6}},
    {{{2.25,0,0},{0,1,0,-1,0,0,0,0,1}}, {2,4,6}},
    {{{2.5,0,0},{0,1,0,-1,0,0,0,0,1}}, {2,4,6}},
    {{{1,0,0},{0,1,0,-1,0,0,0,0,1}}, {2,4,6}}
};
const sdf::Environment envs[] = {
    {{1,1},{0,0},{0,0},{0,0}}, {{2,1},{0,0},{0,0},{0,0}},
    {{3,1},{0,0},{0,0},{0,0}}, {}
};
const sdf::Scene scene = {envs, shapes, nullptr, nullptr, nullptr};
"#,
        ),
        (
            "rotated_cylinder",
            r#"
const sdf::Cylinder shapes[] = {
    {{{0,0,0},{1,0,0,0,1,0,0,0,1}}, 1,4},
    {{{2.25,0,0},{0,0,-1,0,1,0,1,0,0}}, 1,4},
    {{{2.5,0,0},{0,0,-1,0,1,0,1,0,0}}, 1,4},
    {{{1,0,0},{0,0,-1,0,1,0,1,0,0}}, 1,4}
};
const sdf::Environment envs[] = {
    {{0,0},{0,0},{1,1},{0,0}}, {{0,0},{0,0},{2,1},{0,0}},
    {{0,0},{0,0},{3,1},{0,0}}, {}
};
const sdf::Scene scene = {envs, nullptr, nullptr, shapes, nullptr};
"#,
        ),
        (
            "ground",
            r#"
const sdf::Ground shapes[] = {{-1},{0.25},{0.5},{-0.5}};
const sdf::Environment envs[] = {
    {{0,0},{0,0},{0,0},{1,1}}, {{0,0},{0,0},{0,0},{2,1}},
    {{0,0},{0,0},{0,0},{3,1}}, {}
};
const sdf::Scene scene = {envs, nullptr, nullptr, nullptr, shapes};
"#,
        ),
    ] {
        cases.push(Case {
            name: format!("collision_{name}"),
            source: cuda::generate_collision_kernel(&root, &[sphere], options).unwrap(),
            entry: "rkern_collision",
            components: 1,
            count: 5,
            inputs: vec![],
            expected: vec![0.0, 1.0, 0.0, 1.0, 1.0],
            collision_setup: Some(format!(
                "{setup}\nconst uint32_t environment_ids[] = {{3,0,1,2,0}};\n"
            )),
        });
    }
    cases.push(Case {
        name: "collision_empty_robot".into(),
        source: cuda::generate_collision_kernel(&root, &[], options).unwrap(),
        entry: "rkern_collision",
        components: 1,
        count: 5,
        inputs: vec![],
        expected: vec![0.0; 5],
        collision_setup: Some(
            "const uint32_t* environment_ids = nullptr; const sdf::Scene scene = {};\n".into(),
        ),
    });
    for count in [15, 16, 17, 32, 33] {
        cases.push(group_boundary_case(&root, count, options));
    }
    cases.push(sparse_group_case(&root, options));

    let chain = FkDag::from_robot(
        &urdf::read_from_string(include_str!("../fixtures/fk_chain.urdf")).unwrap(),
    )
    .unwrap();
    let spheres: Vec<_> = (0..chain.link_poses().len())
        .map(|link_index| Sphere {
            center: SphereCenter {
                link_index,
                position: [0.2, -0.3, 0.4],
            },
            radius: 0.1,
        })
        .collect();
    cases.push(fk_case("chain", &chain, &spheres, options));
    let fetch = FkDag::from_robot(
        &urdf::read_from_string(include_str!("../../assets/fetch/fetch.urdf")).unwrap(),
    )
    .unwrap();
    let spheres =
        collision::read_spheres(include_str!("../../assets/fetch/collision.yaml"), &fetch).unwrap();
    cases.push(fk_case("fetch", &fetch, &spheres, options));
    cases
}

fn sparse_group_case(fk: &FkDag, options: cuda::Options) -> Case {
    let spheres = [-2.0, 2.0].map(|x| Sphere {
        center: SphereCenter {
            link_index: 0,
            position: [x, 0.0, 0.0],
        },
        radius: 0.25,
    });
    Case {
        name: "collision_obstacles_in_group_gap_are_not_collisions".into(),
        source: cuda::generate_collision_kernel(fk, &spheres, options).unwrap(),
        entry: "rkern_collision",
        components: 1,
        count: 6,
        inputs: vec![],
        expected: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        collision_setup: Some(
            r#"
const sdf::Pose origin = {{0,0,0},{0,1,0,-1,0,0,0,0,1}};
const sdf::Pose touching = {{Scalar(2.75),0,0},{0,1,0,-1,0,0,0,0,1}};
const sdf::Box boxes[] = {{origin,{1,1,1}}, {touching,{1,1,1}}};
const sdf::Sphere obstacles[] = {{{0,0,0},Scalar(0.5)}, {{Scalar(2.75),0,0},Scalar(0.5)}};
const sdf::Cylinder cylinders[] = {{origin,Scalar(0.5),1}, {touching,Scalar(0.5),1}};
const sdf::Environment envs[] = {
    {{0,1},{0,0},{0,0},{0,0}}, {{0,0},{0,1},{0,0},{0,0}}, {{0,0},{0,0},{0,1},{0,0}},
    {{1,1},{0,0},{0,0},{0,0}}, {{0,0},{1,1},{0,0},{0,0}}, {{0,0},{0,0},{1,1},{0,0}}
};
const sdf::Scene scene = {envs, boxes, obstacles, cylinders, nullptr};
const uint32_t environment_ids[] = {0,1,2,3,4,5};
"#
            .into(),
        ),
    }
}

fn group_boundary_case(fk: &FkDag, count: usize, options: cuda::Options) -> Case {
    let spheres: Vec<_> = (0..count)
        .map(|index| Sphere {
            center: SphereCenter {
                link_index: 0,
                position: [4.0 * index as f64, 0.0, 10.0],
            },
            radius: (index % 3) as f64 * 0.125,
        })
        .collect();
    let contact_x = |index: usize| {
        let sphere = spheres[index.min(count - 1)];
        sphere.center.position[0] + sphere.radius + 0.25
    };
    let setup = r#"
const sdf::Pose pose = {{0,0,10},{1,0,0,0,1,0,0,0,1}};
sdf::Box boxes[] = {{pose,{0.5,0.5,0.5}}, {pose,{0.5,0.5,0.5}}, {pose,{0.5,0.5,0.5}}};
boxes[1].pose.position.x = 1000;
boxes[2].pose.position.x = @BOX_X@;
const sdf::Sphere obstacles[] = {{{0,0,10},0.25}, {{1000,0,10},0.25}, {{@SPHERE_X@,0,10},0.25}};
sdf::Cylinder cylinders[] = {{pose,0.25,2}, {pose,0.25,2}, {pose,0.25,2}};
cylinders[1].pose.position.x = 1000;
cylinders[2].pose.position.x = @CYLINDER_X@;
const sdf::Ground grounds[] = {{-10},{100},{-10}};
const sdf::Environment envs[] = {
    {},
    {{1,1},{1,1},{1,1},{1,1}},
    {{1,2},{1,1},{1,1},{1,1}},
    {{1,1},{1,2},{1,1},{1,1}},
    {{1,1},{1,1},{1,2},{1,1}},
    {{1,1},{1,1},{1,1},{1,2}}
};
const sdf::Scene scene = {envs, boxes, obstacles, cylinders, grounds};
const uint32_t environment_ids[] = {0,1,2,3,4,5,1};
"#
    .replace("@BOX_X@", &contact_x(15).to_string())
    .replace("@SPHERE_X@", &contact_x(16).to_string())
    .replace("@CYLINDER_X@", &contact_x(count - 1).to_string());
    Case {
        name: format!("collision_group_boundary_{count}"),
        source: cuda::generate_collision_kernel(fk, &spheres, options).unwrap(),
        entry: "rkern_collision",
        components: 1,
        count: 7,
        inputs: vec![],
        expected: vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        collision_setup: Some(setup),
    }
}

fn fk_case(name: &str, fk: &FkDag, spheres: &[Sphere], options: cuda::Options) -> Case {
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    let mut obstacles = String::new();
    for batch in 0..5 {
        let q: Vec<_> = (0..fk.joint_names().len())
            .map(|joint| {
                let value = 0.5 * (batch as f64 * 0.7 + joint as f64 * 0.31).sin();
                match options.math {
                    MathMode::F64 => value,
                    _ => f64::from(value as f32),
                }
            })
            .collect();
        let values = common::evaluate(fk, &q);
        let centers: Vec<_> = spheres
            .iter()
            .map(|sphere| sphere_center(fk, &values, &sphere.center))
            .collect();
        let obstacle = match batch {
            1 => centers[spheres.len() - 1],
            2 => centers[spheres.len() / 2],
            _ => [100.0, -100.0, 100.0],
        };
        let hit = centers.iter().zip(spheres).any(|(center, sphere)| {
            center
                .iter()
                .zip(obstacle)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                <= sphere.radius + 0.02
        });
        expected.push(if hit { 1.0 } else { 0.0 });
        writeln!(
            obstacles,
            "{{{{Scalar({:.17e}), Scalar({:.17e}), Scalar({:.17e})}}, Scalar(0.02)}},",
            obstacle[0], obstacle[1], obstacle[2]
        )
        .unwrap();
        inputs.extend(q);
    }
    // Both entries in each range must be considered; only the second can collide.
    let setup = r#"
const sdf::Sphere obstacles[] = {{{100,100,100},1}, @OBSTACLES@};
const sdf::Box boxes[] = {{{{100,100,100},{1,0,0,0,1,0,0,0,1}},{1,1,1}}};
const sdf::Cylinder cylinders[] = {{{{100,100,100},{1,0,0,0,1,0,0,0,1}},1,2}};
const sdf::Ground grounds[] = {{100}};
const sdf::Environment envs[] = {
    {{0,1},{0,2},{0,1},{0,1}},
    {{0,1},{1,2},{0,1},{0,1}},
    {{0,1},{2,2},{0,1},{0,1}},
    {},
    {{0,1},{5,1},{0,1},{0,1}}
};
const sdf::Scene scene = {envs, boxes, obstacles, cylinders, grounds};
const uint32_t environment_ids[] = {0,1,2,3,4};
"#
    .replace("@OBSTACLES@", &obstacles);
    Case {
        name: format!("collision_{name}"),
        source: cuda::generate_collision_kernel(fk, spheres, options).unwrap(),
        entry: "rkern_collision",
        components: 1,
        count: 5,
        inputs,
        expected,
        collision_setup: Some(setup),
    }
}
