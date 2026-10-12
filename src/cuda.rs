//! CUDA C++ generation for FK, environment collision checks, and batched RRT.

pub use crate::dag::{Optimization, Precision};
pub use crate::fk::{CentersError, InvalidLinkIndex, SphereCenter};
use crate::fk::{FkDag, center_expressions};
use crate::scheduling::{self, Schedule};

mod collision;
mod kernel;
pub use collision::{generate_collision_function, generate_collision_kernel};
mod rrt;
use kernel::Kernel;
pub use rrt::{RrtError, generate_rrt_kernel};

/// Math, optimization and scheduling choices for CUDA kernels.
/// Defaults to FP32 with SFU trigonometry, algebraic CSE and DAG-order evaluation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    pub math: MathMode,
    pub optimization: Optimization,
    pub scheduling: scheduling::Mode,
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
    let (dag, outputs) =
        fk.dag()
            .optimize(&outputs, options.math.precision(), options.optimization);
    Ok(Kernel {
        name: "rkern_fk",
        dag: &dag,
        dof: fk.joint_names().len(),
        schedule: Schedule::with_mode(&dag, &outputs, options.scheduling)
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
/// FP32 algebraic mode shares rotation-matrix coefficients on links with at least
/// four spheres; other modes retain per-point quaternion rotation.
/// Use separate buffers and a one-dimensional launch with at least `count` threads.
/// An empty sphere slice produces a kernel that writes nothing.
pub fn generate_centers_kernel(
    fk: &FkDag,
    spheres: &[SphereCenter],
    options: Options,
) -> Result<String, CentersError> {
    let (dag, outputs) =
        center_expressions(fk, spheres, options.math.precision(), options.optimization)?;
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
