//! URDF parsing through `urdf-rs`.
//!
//! Returns URDF data, including geometry and inertial properties. Parsing does
//! not establish that the links and joints form a valid kinematic tree. Mesh
//! paths are preserved without loading meshes; Xacro must be expanded first.
//!
//! ```
//! let robot = rkern::urdf::read_from_string(
//!     r#"<robot name="example"><link name="base"/></robot>"#,
//! )?;
//! assert_eq!(robot.links[0].name, "base");
//! # Ok::<(), rkern::urdf::UrdfError>(())
//! ```

pub use urdf_rs::{
    Geometry, Joint, JointLimit, JointType, Link, Mimic, Pose, Robot, UrdfError, read_file,
    read_from_string,
};
