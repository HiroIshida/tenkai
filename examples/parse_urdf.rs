use rkern::urdf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let robot = match std::env::args_os().nth(1) {
        Some(path) => urdf::read_file(path)?,
        None => urdf::read_from_string(include_str!("../assets/fetch/fetch.urdf"))?,
    };

    println!(
        "{}: {} links, {} joints",
        robot.name,
        robot.links.len(),
        robot.joints.len()
    );
    for joint in &robot.joints {
        println!(
            "{} ({:?}): {} -> {}",
            joint.name, joint.joint_type, joint.parent.link, joint.child.link
        );
    }
    Ok(())
}
