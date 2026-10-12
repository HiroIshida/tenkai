use std::fmt;

use super::FkDag;
use crate::dag::{Dag, NodeId, Optimization, Precision};
use crate::transform::PoseNodes;

#[derive(Debug, PartialEq, Eq)]
pub struct InvalidLinkIndex {
    pub index: usize,
    pub link_count: usize,
}

impl fmt::Display for InvalidLinkIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "link index {} is out of range for {} links",
            self.index, self.link_count
        )
    }
}

impl std::error::Error for InvalidLinkIndex {}

/// A collision sphere center in the coordinate frame of `link_poses()[link_index]`.
/// Radius is unnecessary for computing centers.
#[derive(Debug, Clone, Copy)]
pub struct SphereCenter {
    pub link_index: usize,
    pub position: [f64; 3],
}

#[derive(Debug, PartialEq, Eq)]
pub enum CentersError {
    InvalidLink {
        sphere_index: usize,
        source: InvalidLinkIndex,
    },
    NonFinitePosition {
        sphere_index: usize,
    },
}

impl fmt::Display for CentersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLink {
                sphere_index,
                source,
            } => write!(f, "sphere {sphere_index}: {source}"),
            Self::NonFinitePosition { sphere_index } => {
                write!(f, "sphere {sphere_index}: position must be finite")
            }
        }
    }
}

impl std::error::Error for CentersError {}

impl SphereCenter {
    pub(crate) fn validate(&self, fk: &FkDag, sphere_index: usize) -> Result<(), CentersError> {
        if self.link_index >= fk.link_poses().len() {
            return Err(CentersError::InvalidLink {
                sphere_index,
                source: InvalidLinkIndex {
                    index: self.link_index,
                    link_count: fk.link_poses().len(),
                },
            });
        }
        if !self.position.iter().all(|value| value.is_finite()) {
            return Err(CentersError::NonFinitePosition { sphere_index });
        }
        Ok(())
    }
}

pub(crate) fn center_expressions(
    fk: &FkDag,
    spheres: &[SphereCenter],
    precision: Precision,
    optimization: Optimization,
) -> Result<(Dag, Vec<NodeId>), CentersError> {
    let mut counts = vec![0; fk.link_poses().len()];
    for (sphere_index, sphere) in spheres.iter().enumerate() {
        sphere.validate(fk, sphere_index)?;
        counts[sphere.link_index] += 1;
    }
    let mut dag = fk.dag().clone();
    let mut rotations = vec![None; fk.link_poses().len()];
    let shared_rotations = precision == Precision::F32 && optimization == Optimization::Algebraic;
    let mut outputs = Vec::with_capacity(3 * spheres.len());
    for sphere in spheres {
        let pose = &fk.link_poses()[sphere.link_index];
        let local = sphere.position.map(|value| dag.constant(value));
        let transform = PoseNodes {
            translation: pose.translation,
            rotation: pose.rotation,
        };
        // Short groups do not amortize the matrix setup and live coefficients.
        if shared_rotations && counts[sphere.link_index] >= 4 {
            let rotation = *rotations[sphere.link_index]
                .get_or_insert_with(|| transform.rotation_matrix(&mut dag));
            outputs.extend(transform.transform_point_by_matrix(&mut dag, local, rotation));
        } else {
            outputs.extend(transform.transform_point(&mut dag, local));
        }
    }
    Ok(dag.optimize(&outputs, precision, optimization))
}
