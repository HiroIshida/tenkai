//! Scalar FK expressions for all links, relative to an identity root pose.
//!
//! Joint inputs follow movable joints in URDF order; link outputs follow tree
//! traversal order. Angles are in radians and translations are in meters.
//! Fixed, revolute, continuous, and prismatic joints are supported. Mimic and
//! joints with multiple degrees of freedom are rejected.

use std::collections::VecDeque;
use std::fmt;

use crate::dag::{Dag, NodeId};
use crate::transform::{PoseNodes, quaternion_product};
use crate::urdf::{Joint, JointType, Robot};

mod centers;
pub(crate) use centers::center_expressions;
pub use centers::{CentersError, InvalidLinkIndex, SphereCenter};

mod tree;
use tree::LinkTree;

#[derive(Debug)]
pub struct LinkPose {
    pub name: String,
    pub translation: [NodeId; 3],
    /// Quaternion components in `[x, y, z, w]` order.
    pub rotation: [NodeId; 4],
}

#[derive(Debug)]
pub struct FkDag {
    dag: Dag,
    joint_names: Vec<String>,
    link_poses: Vec<LinkPose>,
}

#[derive(Debug)]
pub struct BuildError(String);

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BuildError {}

impl FkDag {
    pub fn from_robot(robot: &Robot) -> Result<Self, BuildError> {
        let tree = LinkTree::from_robot(robot)?;
        let mut dag = Dag::default();
        let (joint_names, joint_inputs) = joint_inputs(&mut dag, &robot.joints);
        let identity = PoseNodes::identity(&mut dag);
        let mut pending = VecDeque::from([(tree.root, identity)]);
        let mut link_poses = Vec::with_capacity(robot.links.len());
        while let Some((parent_index, parent_pose)) = pending.pop_front() {
            link_poses.push(LinkPose {
                name: robot.links[parent_index].name.clone(),
                translation: parent_pose.translation,
                rotation: parent_pose.rotation,
            });
            for &(joint_index, child_index) in &tree.outgoing[parent_index] {
                let pose = joint_pose(
                    &mut dag,
                    parent_pose,
                    &robot.joints[joint_index],
                    joint_inputs[joint_index],
                )?;
                pending.push_back((child_index, pose));
            }
        }
        if link_poses.len() != robot.links.len() {
            return Err(BuildError("links contain a disconnected cycle".into()));
        }
        Ok(Self {
            dag,
            joint_names,
            link_poses,
        })
    }

    pub fn dag(&self) -> &Dag {
        &self.dag
    }

    /// `Node::Input(i)` refers to the joint named by `joint_names()[i]`.
    pub fn joint_names(&self) -> &[String] {
        &self.joint_names
    }

    pub fn link_poses(&self) -> &[LinkPose] {
        &self.link_poses
    }
}

fn joint_inputs(dag: &mut Dag, joints: &[Joint]) -> (Vec<String>, Vec<Option<NodeId>>) {
    let mut names = Vec::new();
    let inputs = joints
        .iter()
        .map(|joint| {
            if joint.joint_type == JointType::Fixed {
                None
            } else {
                let input = dag.input(names.len());
                names.push(joint.name.clone());
                Some(input)
            }
        })
        .collect();
    (names, inputs)
}

fn joint_pose(
    dag: &mut Dag,
    parent: PoseNodes,
    joint: &Joint,
    input: Option<NodeId>,
) -> Result<PoseNodes, BuildError> {
    if !joint
        .origin
        .xyz
        .iter()
        .chain(joint.origin.rpy.iter())
        .all(|v| v.is_finite())
    {
        return Err(BuildError(format!(
            "joint {:?}: origin must be finite",
            joint.name
        )));
    }
    let origin = PoseNodes::from_origin(dag, &joint.origin);
    let mut world = parent.compose(dag, origin);
    if let Some(input) = input {
        let [x, y, z] = joint.axis.xyz.0;
        let norm = x.hypot(y).hypot(z);
        if !norm.is_finite() || norm == 0.0 {
            return Err(BuildError(format!(
                "joint {:?}: axis must be finite and nonzero",
                joint.name
            )));
        }
        let axis = [x, y, z].map(|v| dag.constant(v / norm));
        if joint.joint_type == JointType::Prismatic {
            let local = axis.map(|v| dag.mul(v, input));
            world.translation = world.transform_point(dag, local);
        } else {
            let half = dag.constant(0.5);
            let angle = dag.mul(input, half);
            let sin = dag.sin(angle);
            let cos = dag.cos(angle);
            let [x, y, z] = axis.map(|v| dag.mul(v, sin));
            world.rotation = quaternion_product(dag, world.rotation, [x, y, z, cos]);
        }
    }
    Ok(world)
}
