use std::collections::{HashMap, HashSet};

use super::BuildError;
use crate::urdf::{JointType, Robot};

pub(super) struct LinkTree {
    pub(super) root: usize,
    /// Each edge stores (URDF joint index, child link index) in URDF joint order.
    pub(super) outgoing: Vec<Vec<(usize, usize)>>,
}

impl LinkTree {
    pub(super) fn from_robot(robot: &Robot) -> Result<Self, BuildError> {
        let mut link_indices = HashMap::new();
        for (index, link) in robot.links.iter().enumerate() {
            if link.name.is_empty() || link_indices.insert(link.name.as_str(), index).is_some() {
                return Err(BuildError(format!(
                    "empty or duplicate link name: {:?}",
                    link.name
                )));
            }
        }

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

        // Reachability is checked during FK traversal; a separate walk would repeat it.
        Ok(Self { root, outgoing })
    }
}
