use std::f64::consts::FRAC_PI_2;

use rkern::dag::{Node, NodeId};
use rkern::fk::FkDag;
use rkern::urdf::{self, JointType};

const CHAIN: &str = include_str!("fixtures/fk_chain.urdf");
const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const RX90: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
const RY90: [[f64; 3]; 3] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]];
const RZ90: [[f64; 3]; 3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

fn evaluate(fk: &FkDag, inputs: &[f64]) -> Vec<f64> {
    assert_eq!(inputs.len(), fk.joint_names().len());
    let mut values: Vec<f64> = Vec::with_capacity(fk.dag().nodes().len());
    for (index, node) in fk.dag().nodes().iter().enumerate() {
        let get = |id: NodeId| {
            assert!(id.index() < index, "operand must precede its user");
            values[id.index()]
        };
        let value = match *node {
            Node::Constant(value) => value,
            Node::Input(index) => inputs[index],
            Node::Add(a, b) => get(a) + get(b),
            Node::Sub(a, b) => get(a) - get(b),
            Node::Mul(a, b) => get(a) * get(b),
            Node::Sin(angle) => get(angle).sin(),
            Node::Cos(angle) => get(angle).cos(),
        };
        assert!(value.is_finite());
        values.push(value);
    }
    values
}

fn evaluate_chain(fk: &FkDag, [yaw, pitch, roll, slide]: [f64; 4]) -> Vec<f64> {
    let inputs: Vec<_> = fk
        .joint_names()
        .iter()
        .map(|name| match name.as_str() {
            "yaw" => yaw,
            "pitch" => pitch,
            "roll" => roll,
            "slide" => slide,
            name => panic!("unexpected joint {name}"),
        })
        .collect();
    evaluate(fk, &inputs)
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

fn assert_pose(
    fk: &FkDag,
    values: &[f64],
    name: &str,
    position: [f64; 3],
    rotation: [[f64; 3]; 3],
) {
    let pose = fk
        .link_poses()
        .iter()
        .find(|pose| pose.name == name)
        .unwrap();
    for (id, expected) in pose.translation.into_iter().zip(position) {
        assert_close(values[id.index()], expected);
    }
    let [x, y, z, w] = pose.rotation.map(|id| values[id.index()]);
    assert_close(x * x + y * y + z * z + w * w, 1.0);
    let actual = [
        [
            w * w + x * x - y * y - z * z,
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ];
    for (actual, expected) in actual
        .into_iter()
        .flatten()
        .zip(rotation.into_iter().flatten())
    {
        assert_close(actual, expected);
    }
}

fn closed_form_rotation(yaw: f64, pitch: f64, roll: f64) -> [[f64; 3]; 3] {
    let (sa, ca) = yaw.sin_cos();
    let (sb, cb) = pitch.sin_cos();
    let (sc, cc) = roll.sin_cos();
    [
        [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
        [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
        [-sb, cb * sc, cb * cc],
    ]
}

fn assert_chain(fk: &FkDag, [yaw, pitch, roll, slide]: [f64; 4]) {
    let values = evaluate_chain(fk, [yaw, pitch, roll, slide]);
    let (sa, ca) = yaw.sin_cos();
    let (sb, cb) = pitch.sin_cos();
    let wrist_rotation = closed_form_rotation(yaw, pitch, roll);
    let wrist_position = [
        1.0 + 2.0 * ca + 3.0 * ca * cb,
        2.0 * sa + 3.0 * sa * cb,
        -3.0 * sb,
    ];
    let slide_position = [
        1.0 + 2.0 * ca + (3.0 + slide) * ca * cb,
        2.0 * sa + (3.0 + slide) * sa * cb,
        -(3.0 + slide) * sb,
    ];
    let tool_position = [0, 1, 2].map(|i| slide_position[i] + wrist_rotation[i][1]);
    let tool_rotation = wrist_rotation.map(|[x, y, z]| [y, -x, z]);

    assert_eq!(fk.link_poses().len(), 7);
    assert_pose(fk, &values, "base", [0.0; 3], IDENTITY);
    assert_pose(
        fk,
        &values,
        "yaw_link",
        [1.0, 0.0, 0.0],
        closed_form_rotation(yaw, 0.0, 0.0),
    );
    assert_pose(
        fk,
        &values,
        "pitch_link",
        [1.0 + 2.0 * ca, 2.0 * sa, 0.0],
        closed_form_rotation(yaw, pitch, 0.0),
    );
    assert_pose(fk, &values, "roll_link", wrist_position, wrist_rotation);
    assert_pose(fk, &values, "slide_link", slide_position, wrist_rotation);
    assert_pose(fk, &values, "tool", tool_position, tool_rotation);
    assert_pose(fk, &values, "sensor", [0.0, 0.0, 2.0], RX90);
}

#[test]
fn zero_angles_place_the_links_on_the_x_axis() {
    let robot = urdf::read_from_string(CHAIN).unwrap();
    let fk = FkDag::from_robot(&robot).unwrap();
    assert_eq!(fk.joint_names(), ["yaw", "pitch", "roll", "slide"]);
    let values = evaluate_chain(&fk, [0.0, 0.0, 0.0, 0.5]);
    for (name, x) in [
        ("base", 0.0),
        ("yaw_link", 1.0),
        ("pitch_link", 3.0),
        ("roll_link", 6.0),
        ("slide_link", 6.5),
    ] {
        assert_pose(&fk, &values, name, [x, 0.0, 0.0], IDENTITY);
    }
    assert_pose(&fk, &values, "tool", [6.5, 1.0, 0.0], RZ90);
    assert_pose(&fk, &values, "sensor", [0.0, 0.0, 2.0], RX90);
}

#[test]
fn quarter_turns_have_obvious_positions_and_orientations() {
    let robot = urdf::read_from_string(CHAIN).unwrap();
    let fk = FkDag::from_robot(&robot).unwrap();
    for (configuration, slide_position, tool_position, rotation) in [
        (
            [FRAC_PI_2, 0.0, 0.0, 0.5],
            [1.0, 5.5, 0.0],
            [0.0, 5.5, 0.0],
            RZ90,
        ),
        (
            [0.0, FRAC_PI_2, 0.0, 0.5],
            [3.0, 0.0, -3.5],
            [3.0, 1.0, -3.5],
            RY90,
        ),
        (
            [0.0, 0.0, FRAC_PI_2, 0.5],
            [6.5, 0.0, 0.0],
            [6.5, 0.0, 1.0],
            RX90,
        ),
        (
            [FRAC_PI_2, FRAC_PI_2, FRAC_PI_2, 0.5],
            [1.0, 2.0, -3.5],
            [1.0, 3.0, -3.5],
            RY90,
        ),
    ] {
        let values = evaluate_chain(&fk, configuration);
        assert_pose(&fk, &values, "slide_link", slide_position, rotation);
        let tool_rotation = rotation.map(|[x, y, z]| [y, -x, z]);
        assert_pose(&fk, &values, "tool", tool_position, tool_rotation);
    }
}

#[test]
fn every_link_matches_closed_form_at_mixed_angles() {
    let robot = urdf::read_from_string(CHAIN).unwrap();
    let fk = FkDag::from_robot(&robot).unwrap();
    for configuration in [
        [0.0, 0.0, 0.0, 0.0],
        [FRAC_PI_2, 0.0, 0.0, 0.5],
        [0.0, FRAC_PI_2, 0.0, 0.5],
        [0.0, 0.0, FRAC_PI_2, 0.5],
        [FRAC_PI_2, FRAC_PI_2, FRAC_PI_2, 0.5],
        [0.31, -0.72, 1.17, 0.24],
        [-1.2, 0.8, -0.4, -0.5],
        [
            std::f64::consts::PI,
            -FRAC_PI_2,
            -std::f64::consts::PI,
            -0.3,
        ],
    ] {
        assert_chain(&fk, configuration);
    }
}

#[test]
fn prismatic_axis_is_rotated_by_its_joint_origin() {
    let mut robot = urdf::read_from_string(CHAIN).unwrap();
    robot.joints[3].origin.xyz.0 = [0.0, 0.0, 1.0];
    robot.joints[3].origin.rpy.0 = [0.0, FRAC_PI_2, 0.0];
    let fk = FkDag::from_robot(&robot).unwrap();
    let values = evaluate_chain(&fk, [0.0, 0.0, 0.0, 0.5]);
    assert_pose(&fk, &values, "slide_link", [6.0, 0.0, 0.5], RY90);
}

#[test]
fn revolute_motion_follows_the_origin_rotation() {
    let mut robot = urdf::read_from_string(CHAIN).unwrap();
    robot.joints[0].origin.rpy.0 = [FRAC_PI_2, 0.0, 0.0];
    let fk = FkDag::from_robot(&robot).unwrap();
    let values = evaluate_chain(&fk, [FRAC_PI_2, 0.0, 0.0, 0.0]);
    let rotation = [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]];
    assert_pose(&fk, &values, "yaw_link", [1.0, 0.0, 0.0], rotation);
    assert_pose(&fk, &values, "pitch_link", [1.0, 0.0, 2.0], rotation);
    assert_pose(&fk, &values, "slide_link", [1.0, 0.0, 5.0], rotation);
}

#[test]
fn origin_rpy_uses_roll_then_pitch_then_yaw() {
    let mut robot = urdf::read_from_string(CHAIN).unwrap();
    robot.joints[5].origin.rpy.0 = [0.2, -0.4, 0.7];
    let fk = FkDag::from_robot(&robot).unwrap();
    let values = evaluate_chain(&fk, [0.0; 4]);
    assert_pose(
        &fk,
        &values,
        "sensor",
        [0.0, 0.0, 2.0],
        closed_form_rotation(0.7, -0.4, 0.2),
    );
}

#[test]
fn urdf_order_does_not_need_to_follow_the_tree() {
    let mut robot = urdf::read_from_string(CHAIN).unwrap();
    robot.links.reverse();
    robot.joints.reverse();
    let fk = FkDag::from_robot(&robot).unwrap();
    assert_eq!(fk.joint_names(), ["slide", "roll", "pitch", "yaw"]);
    assert_chain(&fk, [0.31, -0.72, 1.17, 0.24]);
}

#[test]
fn single_link_has_identity_pose_without_inputs() {
    let robot = urdf::read_from_string(r#"<robot name="one"><link name="root"/></robot>"#).unwrap();
    let fk = FkDag::from_robot(&robot).unwrap();
    assert!(fk.joint_names().is_empty());
    assert_eq!(fk.link_poses().len(), 1);
    assert_pose(&fk, &evaluate(&fk, &[]), "root", [0.0; 3], IDENTITY);
}

#[test]
fn invalid_models_return_errors() {
    type ModelChange = fn(&mut urdf::Robot);
    let cases: &[(&str, ModelChange)] = &[
        ("no links", |r| {
            r.links.clear();
            r.joints.clear();
        }),
        ("duplicate link", |r| r.links.push(r.links[0].clone())),
        ("duplicate joint", |r| {
            r.joints[1].name = r.joints[0].name.clone()
        }),
        ("unknown parent", |r| {
            r.joints[0].parent.link = "missing".into()
        }),
        ("unknown child", |r| {
            r.joints[0].child.link = "missing".into()
        }),
        ("multiple parents", |r| {
            r.joints[1].child.link = "yaw_link".into()
        }),
        ("multiple roots", |r| {
            r.joints.remove(0);
        }),
        ("cycle including root", |r| {
            r.joints[5].child.link = "base".into()
        }),
        ("disconnected cycle", |r| {
            r.joints[0].parent.link = "tool".into()
        }),
        ("zero axis", |r| r.joints[0].axis.xyz.0 = [0.0; 3]),
        ("nonfinite axis", |r| r.joints[0].axis.xyz.0[0] = f64::NAN),
        ("nonfinite translation", |r| {
            r.joints[0].origin.xyz.0[0] = f64::INFINITY
        }),
        ("nonfinite rotation", |r| {
            r.joints[0].origin.rpy.0[0] = f64::NAN
        }),
        ("mimic", |r| {
            r.joints[1].mimic = Some(urdf::Mimic {
                joint: "yaw".into(),
                ..Default::default()
            })
        }),
    ];
    for (name, change) in cases {
        let mut robot = urdf::read_from_string(CHAIN).unwrap();
        change(&mut robot);
        assert!(FkDag::from_robot(&robot).is_err(), "accepted {name}");
    }
    for kind in [JointType::Floating, JointType::Planar, JointType::Spherical] {
        let mut robot = urdf::read_from_string(CHAIN).unwrap();
        robot.joints[0].joint_type = kind;
        assert!(FkDag::from_robot(&robot).is_err());
    }
}

#[test]
fn bundled_fetch_has_outputs_for_all_links() {
    let robot = urdf::read_from_string(include_str!("../assets/fetch/fetch.urdf")).unwrap();
    let fk = FkDag::from_robot(&robot).unwrap();
    assert_eq!(fk.joint_names().len(), 15);
    assert_eq!(fk.link_poses().len(), 26);
    let values = evaluate(&fk, &vec![0.0; fk.joint_names().len()]);
    for link in &robot.links {
        let pose = fk
            .link_poses()
            .iter()
            .find(|pose| pose.name == link.name)
            .unwrap();
        let norm_squared: f64 = pose
            .rotation
            .iter()
            .map(|id| values[id.index()].powi(2))
            .sum();
        assert_close(norm_squared, 1.0);
    }
    assert_pose(&fk, &values, "base_link", [0.0; 3], IDENTITY);
}
