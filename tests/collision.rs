use rkern::{collision, cuda, fk::FkDag, urdf};

fn root() -> FkDag {
    FkDag::from_robot(
        &urdf::read_from_string(r#"<robot name="test"><link name="root"/></robot>"#).unwrap(),
    )
    .unwrap()
}

#[test]
fn bundled_fetch_spheres_resolve_to_fk_links() {
    let fk = FkDag::from_robot(
        &urdf::read_from_string(include_str!("../assets/fetch/fetch.urdf")).unwrap(),
    )
    .unwrap();
    let spheres =
        collision::read_spheres(include_str!("../assets/fetch/collision.yaml"), &fk).unwrap();
    assert_eq!(spheres.len(), 69);
    assert!(
        spheres
            .windows(2)
            .all(|pair| pair[0].center.link_index <= pair[1].center.link_index)
    );
    for (name, count) in [
        ("gripper_link", 9),
        ("head_pan_link", 16),
        ("torso_lift_link", 2),
    ] {
        assert_eq!(
            spheres
                .iter()
                .filter(|sphere| fk.link_poses()[sphere.center.link_index].name == name)
                .count(),
            count
        );
    }
    let shoulder = spheres
        .iter()
        .find(|sphere| fk.link_poses()[sphere.center.link_index].name == "shoulder_pan_link")
        .unwrap();
    assert_eq!(shoulder.center.position, [0.0, 0.0, -0.14]);
    assert_eq!(shoulder.radius, 0.06);
}

#[test]
fn sphere_coordinates_and_local_order_are_preserved() {
    let yaml = "collision_spheres: {root: {spheres: [[+0.02, -0.0, 1., 0.1], [2, 3, 4, 0]]}}";
    let spheres = collision::read_spheres(yaml, &root()).unwrap();
    assert_eq!(spheres.len(), 2);
    assert_eq!(spheres[0].center.position, [0.02, -0.0, 1.0]);
    assert_eq!(spheres[1].center.position, [2.0, 3.0, 4.0]);
    assert_eq!(spheres[1].radius, 0.0);
    assert!(
        collision::read_spheres("collision_spheres: {}", &root())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn malformed_or_ambiguous_sphere_definitions_are_rejected() {
    for yaml in [
        "{}",
        "collision_spheres: {missing: {spheres: []}}",
        "collision_spheres: {root: {sphere: []}}",
        "collision_spheres: {root: {spheres: [[0, 0, 1]]}}",
        "collision_spheres: {root: {spheres: [[0, 0, 1, 1, 1]]}}",
        "collision_spheres: {root: {spheres: []}, root: {spheres: []}}",
        "collision_spheres: {root: {spheres: [[.nan, 0, 0, 1]]}}",
        "collision_spheres: {root: {spheres: [[0, .inf, 0, 1]]}}",
        "collision_spheres: {root: {spheres: [[0, 0, 0, -1]]}}",
        "collision_spheres: {root: {spheres: [[0, 0, 0, .inf]]}}",
    ] {
        assert!(
            collision::read_spheres(yaml, &root()).is_err(),
            "accepted {yaml}"
        );
    }
}

#[test]
fn invalid_programmatic_spheres_are_rejected_by_codegen() {
    let fk = root();
    for math in [
        cuda::MathMode::F32,
        cuda::MathMode::F32Sfu,
        cuda::MathMode::F64,
    ] {
        let options = cuda::Options {
            math,
            ..Default::default()
        };
        for (link_index, position, radius) in [
            (usize::MAX, [0.0; 3], 0.1),
            (0, [f64::NAN, 0.0, 0.0], 0.1),
            (0, [0.0; 3], -0.1),
            (0, [0.0; 3], f64::INFINITY),
            (0, [0.0; 3], f64::NAN),
        ] {
            let sphere = collision::Sphere {
                center: cuda::SphereCenter {
                    link_index,
                    position,
                },
                radius,
            };
            assert!(cuda::generate_collision_kernel(&fk, &[sphere], options).is_err());
        }
    }
}
