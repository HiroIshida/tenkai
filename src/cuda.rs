//! CUDA C++ generation for batched FK with one thread per configuration.

use std::fmt;

use crate::dag::{Dag, NodeId};
use crate::fk::FkDag;
use crate::scheduling::{self, Schedule};
use crate::transform::PoseNodes;

mod kernel;
use kernel::Kernel;

/// Math and scheduling choices for either CUDA kernel.
/// Defaults to FP32 with SFU trigonometry and DAG-order evaluation (scheduling OFF).
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    pub math: MathMode,
    pub scheduling: scheduling::Mode,
}

/// Scalar type used for kernel buffers, intermediates, and trigonometric functions.
/// The source FK DAG retains its f64 constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    F32,
    F64,
}

impl Precision {
    fn cuda_spelling(self) -> (&'static str, &'static str) {
        match self {
            Self::F32 => ("float", "f"),
            Self::F64 => ("double", ""),
        }
    }
}

/// Scalar precision and runtime trigonometry used by generated kernels.
/// Defaults to float arithmetic with SFU intrinsics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MathMode {
    F32,
    F64,
    /// Float arithmetic with CUDA's `__sinf` / `__cosf` intrinsics.
    /// Accuracy degrades for large angles and subnormal results are flushed to zero.
    /// This selects fast intrinsics; the standard modes may also use SFU instructions internally.
    #[default]
    F32Sfu,
}

impl From<Precision> for MathMode {
    fn from(precision: Precision) -> Self {
        match precision {
            Precision::F32 => Self::F32Sfu,
            Precision::F64 => Self::F64,
        }
    }
}

impl MathMode {
    fn precision(self) -> Precision {
        match self {
            Self::F32 | Self::F32Sfu => Precision::F32,
            Self::F64 => Precision::F64,
        }
    }

    fn trig_functions(self) -> (&'static str, &'static str) {
        match self {
            Self::F32 => ("sinf", "cosf"),
            Self::F64 => ("sin", "cos"),
            Self::F32Sfu => ("__sinf", "__cosf"),
        }
    }
}

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

/// Generates `rkern_fk(configurations, poses, count)` for `link_poses()[link_index]`.
///
/// Buffer elements use the scalar type selected by `options.math` (f32 by default).
/// Inputs are `[count][joint_names().len()]`
/// and outputs are `[count][7]` in `[x, y, z, qx, qy, qz, qw]` order. Input joints
/// follow `joint_names()`. Use separate input/output buffers and a one-dimensional
/// CUDA launch with at least `count` threads. Excess threads do no work.
pub fn generate_fk_kernel(
    fk: &FkDag,
    link_index: usize,
    options: Options,
) -> Result<String, InvalidLinkIndex> {
    let pose = fk.link_poses().get(link_index).ok_or(InvalidLinkIndex {
        index: link_index,
        link_count: fk.link_poses().len(),
    })?;
    let outputs = [
        pose.translation[0],
        pose.translation[1],
        pose.translation[2],
        pose.rotation[0],
        pose.rotation[1],
        pose.rotation[2],
        pose.rotation[3],
    ];
    Ok(Kernel {
        name: "rkern_fk",
        dag: fk.dag(),
        dof: fk.joint_names().len(),
        schedule: Schedule::with_mode(fk.dag(), &outputs, options.scheduling)
            .expect("FK outputs belong to the FK DAG"),
        math: options.math,
    }
    .source())
}

/// Generates `rkern_centers(configurations, centers, count)` for debugging sphere positions.
///
/// Buffer elements use the scalar type selected by `options.math` (f32 by default).
/// Inputs are `[count][joint_names().len()]`
/// and outputs are `[count][spheres.len()][3]`, in input sphere order and xyz order.
/// Centers are relative to the robot root frame, with no additional base transform.
/// Each center is `link_translation + link_rotation * sphere.position`.
/// Use separate buffers and a one-dimensional launch with at least `count` threads.
/// An empty sphere slice produces a kernel that writes nothing.
pub fn generate_centers_kernel(
    fk: &FkDag,
    spheres: &[SphereCenter],
    options: Options,
) -> Result<String, CentersError> {
    let (dag, outputs) = center_expressions(fk, spheres)?;
    Ok(Kernel {
        name: "rkern_centers",
        dag: &dag,
        dof: fk.joint_names().len(),
        schedule: Schedule::with_mode(&dag, &outputs, options.scheduling)
            .expect("center outputs belong to the extended FK DAG"),
        math: options.math,
    }
    .source())
}

fn center_expressions(
    fk: &FkDag,
    spheres: &[SphereCenter],
) -> Result<(Dag, Vec<NodeId>), CentersError> {
    for (sphere_index, sphere) in spheres.iter().enumerate() {
        if sphere.link_index >= fk.link_poses().len() {
            return Err(CentersError::InvalidLink {
                sphere_index,
                source: InvalidLinkIndex {
                    index: sphere.link_index,
                    link_count: fk.link_poses().len(),
                },
            });
        }
        if !sphere.position.iter().all(|value| value.is_finite()) {
            return Err(CentersError::NonFinitePosition { sphere_index });
        }
    }
    let mut dag = fk.dag().clone();
    let mut outputs = Vec::with_capacity(3 * spheres.len());
    for sphere in spheres {
        let pose = &fk.link_poses()[sphere.link_index];
        let local = sphere.position.map(|value| dag.constant(value));
        let transform = PoseNodes {
            translation: pose.translation,
            rotation: pose.rotation,
        };
        outputs.extend(transform.transform_point(&mut dag, local));
    }
    Ok((dag, outputs))
}
