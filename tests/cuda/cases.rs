use super::common;
use rkern::{
    cuda::{self, MathMode, SphereCenter},
    fk::FkDag,
    urdf,
};
use std::f64::consts::FRAC_PI_2;

const CHAIN: &str = include_str!("../fixtures/fk_chain.urdf");

enum Output {
    Link(usize),
    Centers(Vec<SphereCenter>),
}

pub(super) struct Case {
    pub(super) name: String,
    pub(super) source: String,
    pub(super) entry: &'static str,
    pub(super) components: usize,
    pub(super) count: usize,
    pub(super) inputs: Vec<f64>,
    pub(super) expected: Vec<f64>,
    pub(super) collision_setup: Option<String>,
}

impl Case {
    fn new(
        name: String,
        fk: &FkDag,
        output: Output,
        configurations: &[Vec<f64>],
        options: cuda::Options,
    ) -> Self {
        let (entry, source, components) = match &output {
            Output::Link(link) => (
                "rkern_fk",
                cuda::generate_fk_kernel(fk, *link, options).unwrap(),
                7,
            ),
            Output::Centers(spheres) => (
                "rkern_centers",
                cuda::generate_centers_kernel(fk, spheres, options).unwrap(),
                3 * spheres.len(),
            ),
        };
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for configuration in configurations {
            let q: Vec<_> = configuration
                .iter()
                .map(|&v| match options.math {
                    MathMode::F32 | MathMode::F32Sfu => f64::from(v as f32),
                    MathMode::F64 => v,
                })
                .collect();
            let values = common::evaluate(fk, &q);
            match &output {
                Output::Link(link) => {
                    let pose = &fk.link_poses()[*link];
                    expected.extend(
                        pose.translation
                            .iter()
                            .chain(&pose.rotation)
                            .map(|id| values[id.index()]),
                    );
                }
                Output::Centers(spheres) => {
                    for sphere in spheres {
                        expected.extend(sphere_center(fk, &values, sphere));
                    }
                }
            }
            inputs.extend(q);
        }
        Case {
            name,
            source,
            entry,
            components,
            count: configurations.len(),
            inputs,
            expected,
            collision_setup: None,
        }
    }
}

pub(super) fn suite(options: cuda::Options) -> Vec<Case> {
    let chain = FkDag::from_robot(&urdf::read_from_string(CHAIN).unwrap()).unwrap();
    let configurations = [
        vec![0.0, 0.0, 0.0, 0.0],
        vec![FRAC_PI_2, FRAC_PI_2, FRAC_PI_2, 0.5],
        vec![0.31, -0.72, 1.17, 0.24],
        vec![-1.2, 0.8, -0.4, -0.5],
        vec![std::f64::consts::PI, -FRAC_PI_2, 0.4, 0.3],
    ];
    let mut cases = Vec::new();
    for (link, pose) in chain.link_poses().iter().enumerate() {
        cases.push(Case::new(
            format!("chain_fk_{}", pose.name),
            &chain,
            Output::Link(link),
            &configurations,
            options,
        ));
    }
    let spheres = (0..chain.link_poses().len())
        .rev()
        .flat_map(|link_index| {
            [[0.0; 3], [0.2, -0.3, 0.4], [-0.5, 0.1, 0.7]].map(|position| SphereCenter {
                link_index,
                position,
            })
        })
        .collect();
    cases.push(Case::new(
        "chain_centers".into(),
        &chain,
        Output::Centers(spheres),
        &configurations,
        options,
    ));
    cases.push(Case::new(
        "empty_centers".into(),
        &chain,
        Output::Centers(vec![]),
        &configurations,
        options,
    ));

    let root = FkDag::from_robot(
        &urdf::read_from_string(r#"<robot name="one"><link name="root"/></robot>"#).unwrap(),
    )
    .unwrap();
    let no_inputs = vec![vec![]; configurations.len()];
    cases.push(Case::new(
        "root_fk".into(),
        &root,
        Output::Link(0),
        &no_inputs,
        options,
    ));
    cases.push(Case::new(
        "root_centers".into(),
        &root,
        Output::Centers(vec![SphereCenter {
            link_index: 0,
            position: [0.25, -0.5, 1.0],
        }]),
        &no_inputs,
        options,
    ));

    let fetch = FkDag::from_robot(
        &urdf::read_from_string(include_str!("../../assets/fetch/fetch.urdf")).unwrap(),
    )
    .unwrap();
    let gripper = fetch
        .link_poses()
        .iter()
        .position(|pose| pose.name == "gripper_link")
        .unwrap();
    let inputs: Vec<Vec<f64>> = (0..5)
        .map(|batch| {
            (0..fetch.joint_names().len())
                .map(|joint| 0.5 * (batch as f64 * 0.7 + joint as f64 * 0.31).sin())
                .collect()
        })
        .collect();
    cases.push(Case::new(
        "fetch_gripper".into(),
        &fetch,
        Output::Link(gripper),
        &inputs,
        options,
    ));
    cases.extend(super::collision_cases::suite(options));
    cases
}

pub(super) fn sphere_center(fk: &FkDag, values: &[f64], sphere: &SphereCenter) -> [f64; 3] {
    let pose = &fk.link_poses()[sphere.link_index];
    let [x, y, z, w] = pose.rotation.map(|id| values[id.index()]);
    // Use a matrix so the reference does not repeat the codegen quaternion-vector formula.
    let rotation = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    [0, 1, 2].map(|i| {
        values[pose.translation[i].index()]
            + rotation[i]
                .iter()
                .zip(sphere.position)
                .map(|(a, b)| a * b)
                .sum::<f64>()
    })
}
