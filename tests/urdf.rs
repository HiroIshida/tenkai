use rkern::urdf::{self, Geometry, JointType};

const ARM: &str = include_str!("fixtures/arm.urdf");

#[test]
fn reads_joint_parameters_and_mimic_relationship() {
    let robot = urdf::read_from_string(ARM).unwrap();
    assert_eq!(robot.name, "arm");
    assert_eq!(robot.links.len(), 4);
    assert_eq!(robot.joints.len(), 3);

    let joint = &robot.joints[0];
    assert_eq!(joint.name, "shoulder");
    assert_eq!(joint.joint_type, JointType::Revolute);
    assert_eq!(joint.parent.link, "base");
    assert_eq!(joint.child.link, "arm");
    assert_eq!(*joint.origin.xyz, [0.1, 0.2, 0.3]);
    assert_eq!(*joint.origin.rpy, [0.4, 0.5, 0.6]);
    assert_eq!(*joint.axis.xyz, [0.0, 0.0, 1.0]);
    assert_eq!(joint.limit.lower, -1.5);
    assert_eq!(joint.limit.upper, 1.5);
    assert_eq!(joint.limit.effort, 10.0);
    assert_eq!(joint.limit.velocity, 2.0);

    let mimic = robot.joints[2].mimic.as_ref().unwrap();
    assert_eq!(mimic.joint, "shoulder");
    assert_eq!(mimic.multiplier, Some(-1.0));
    assert_eq!(mimic.offset, Some(0.1));
}

#[test]
fn omitted_origin_and_axis_use_urdf_defaults() {
    let robot = urdf::read_from_string(ARM).unwrap();
    let mount = &robot.joints[1];
    assert_eq!(mount.joint_type, JointType::Fixed);
    assert_eq!(*mount.origin.xyz, [0.0; 3]);
    assert_eq!(*mount.origin.rpy, [0.0; 3]);
    assert_eq!(*mount.axis.xyz, [1.0, 0.0, 0.0]);
    assert!(mount.mimic.is_none());
}

#[test]
fn reads_file_and_preserves_geometry_without_loading_meshes() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/arm.urdf");
    let robot = urdf::read_file(path).unwrap();
    assert_eq!(robot.name, "arm");
    assert!(matches!(
        robot.links[0].collision[0].geometry,
        Geometry::Sphere { radius } if radius == 0.2
    ));
    match &robot.links[0].visual[0].geometry {
        Geometry::Mesh { filename, .. } => assert_eq!(filename, "package://arm/base.stl"),
        geometry => panic!("expected mesh, got {geometry:?}"),
    }
}

#[test]
fn malformed_input_returns_errors() {
    for xml in [
        "<robot>",
        &ARM.replace("type=\"revolute\"", "type=\"unknown\""),
        &ARM.replace("xyz=\"0 0 1\"", "xyz=\"0 1\""),
        &ARM.replace("velocity=\"2\"", "velocity=\"invalid\""),
    ] {
        assert!(urdf::read_from_string(xml).is_err(), "accepted {xml}");
    }
}

#[test]
fn invalid_file_path_returns_an_error() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/arm.urdf/nonexistent.urdf");
    assert!(urdf::read_file(path).is_err());
}
