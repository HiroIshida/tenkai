//! Scalar FK expressions for all links, relative to an identity root pose.
//!
//! Joint inputs follow movable joints in URDF order; link outputs follow tree
//! traversal order. Angles are in radians and translations are in meters.
//! Fixed, revolute, continuous, and prismatic joints are supported. Mimic and
//! joints with multiple degrees of freedom are rejected.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::dag::{Dag, NodeId};
use crate::urdf::{Joint, JointType, Robot};

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
        let mut link_indices = HashMap::new();
        for (index, link) in robot.links.iter().enumerate() {
            if link.name.is_empty() || link_indices.insert(link.name.as_str(), index).is_some() {
                return Err(BuildError(format!(
                    "empty or duplicate link name: {:?}",
                    link.name
                )));
            }
        }

        let mut dag = Dag::default();
        let mut joint_names = Vec::new();
        let mut joint_inputs = Vec::new();
        let mut names = HashSet::new();
        let mut incoming = vec![false; robot.links.len()];
        let mut outgoing = vec![Vec::new(); robot.links.len()];

        for (index, joint) in robot.joints.iter().enumerate() {
            if joint.name.is_empty() || !names.insert(joint.name.as_str()) {
                return Err(BuildError(format!(
                    "empty or duplicate joint name: {:?}",
                    joint.name
                )));
            }
            if !matches!(
                joint.joint_type,
                JointType::Fixed
                    | JointType::Revolute
                    | JointType::Continuous
                    | JointType::Prismatic
            ) {
                return Err(BuildError(format!(
                    "joint {:?}: unsupported type {:?}",
                    joint.name, joint.joint_type
                )));
            }
            if joint.mimic.is_some() {
                return Err(BuildError(format!(
                    "joint {:?}: mimic is not supported",
                    joint.name
                )));
            }
            let parent = *link_indices
                .get(joint.parent.link.as_str())
                .ok_or_else(|| {
                    BuildError(format!(
                        "joint {:?}: unknown parent link {:?}",
                        joint.name, joint.parent.link
                    ))
                })?;
            let child = *link_indices.get(joint.child.link.as_str()).ok_or_else(|| {
                BuildError(format!(
                    "joint {:?}: unknown child link {:?}",
                    joint.name, joint.child.link
                ))
            })?;
            if incoming[child] {
                return Err(BuildError(format!(
                    "link {:?} has multiple parent joints",
                    joint.child.link
                )));
            }
            incoming[child] = true;
            outgoing[parent].push((index, child));
            let input = if joint.joint_type == JointType::Fixed {
                None
            } else {
                let input = dag.input(joint_names.len());
                joint_names.push(joint.name.clone());
                Some(input)
            };
            joint_inputs.push(input);
        }

        let mut roots = incoming
            .iter()
            .enumerate()
            .filter(|(_, has_parent)| !**has_parent);
        let root = roots
            .next()
            .map(|(index, _)| index)
            .ok_or_else(|| BuildError("expected one root link; found none".into()))?;
        if roots.next().is_some() {
            return Err(BuildError("expected one root link; found multiple".into()));
        }

        let zero = dag.constant(0.0);
        let one = dag.constant(1.0);
        let identity = PoseNodes {
            translation: [zero; 3],
            rotation: [zero, zero, zero, one],
        };
        let mut pending = VecDeque::from([(root, identity)]);
        let mut link_poses = Vec::with_capacity(robot.links.len());
        while let Some((parent_index, parent_pose)) = pending.pop_front() {
            link_poses.push(LinkPose {
                name: robot.links[parent_index].name.clone(),
                translation: parent_pose.translation,
                rotation: parent_pose.rotation,
            });
            for &(joint_index, child_index) in &outgoing[parent_index] {
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

#[derive(Clone, Copy)]
struct PoseNodes {
    translation: [NodeId; 3],
    rotation: [NodeId; 4],
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
    let origin = PoseNodes {
        translation: joint.origin.xyz.0.map(|v| dag.constant(v)),
        rotation: rpy_quaternion(joint.origin.rpy.0).map(|v| dag.constant(v)),
    };
    let mut world = compose(dag, parent, origin);
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
            let offset = rotate_vector(dag, world.rotation, local);
            world.translation = [0, 1, 2].map(|i| dag.add(world.translation[i], offset[i]));
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

fn rpy_quaternion([roll, pitch, yaw]: [f64; 3]) -> [f64; 4] {
    let (sr, cr) = (roll * 0.5).sin_cos();
    let (sp, cp) = (pitch * 0.5).sin_cos();
    let (sy, cy) = (yaw * 0.5).sin_cos();
    [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]
}

fn compose(dag: &mut Dag, parent: PoseNodes, local: PoseNodes) -> PoseNodes {
    let offset = rotate_vector(dag, parent.rotation, local.translation);
    PoseNodes {
        translation: [0, 1, 2].map(|i| dag.add(parent.translation[i], offset[i])),
        rotation: quaternion_product(dag, parent.rotation, local.rotation),
    }
}

fn cross(dag: &mut Dag, a: [NodeId; 3], b: [NodeId; 3]) -> [NodeId; 3] {
    [[1, 2], [2, 0], [0, 1]].map(|[j, k]| {
        let positive = dag.mul(a[j], b[k]);
        let negative = dag.mul(a[k], b[j]);
        dag.sub(positive, negative)
    })
}

fn quaternion_product(dag: &mut Dag, a: [NodeId; 4], b: [NodeId; 4]) -> [NodeId; 4] {
    let cross = cross(dag, [a[0], a[1], a[2]], [b[0], b[1], b[2]]);
    let [x, y, z] = [0, 1, 2].map(|i| {
        let left = dag.mul(a[3], b[i]);
        let right = dag.mul(b[3], a[i]);
        let sum = dag.add(left, right);
        dag.add(sum, cross[i])
    });
    let products = [0, 1, 2, 3].map(|i| dag.mul(a[i], b[i]));
    let sum = dag.add(products[0], products[1]);
    let dot = dag.add(sum, products[2]);
    let w = dag.sub(products[3], dot);
    [x, y, z, w]
}

fn rotate_vector(dag: &mut Dag, q: [NodeId; 4], v: [NodeId; 3]) -> [NodeId; 3] {
    let imaginary = [q[0], q[1], q[2]];
    let two = dag.constant(2.0);
    let twice_cross = cross(dag, imaginary, v).map(|value| dag.mul(two, value));
    let correction = cross(dag, imaginary, twice_cross);
    [0, 1, 2].map(|i| {
        let scaled = dag.mul(q[3], twice_cross[i]);
        let sum = dag.add(v[i], scaled);
        dag.add(sum, correction[i])
    })
}
