use std::fmt;

use super::{Options, Precision, collision};
use crate::{collision::Sphere, fk::FkDag};

#[derive(Debug)]
pub enum RrtError {
    EmptyControls,
    InvalidControl { index: usize, joint_count: usize },
    DuplicateControl(usize),
    Collision(crate::collision::Error),
}

impl fmt::Display for RrtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyControls => write!(f, "RRT requires at least one controlled joint"),
            Self::InvalidControl { index, joint_count } => {
                write!(
                    f,
                    "RRT control index {index} exceeds FK joint count {joint_count}"
                )
            }
            Self::DuplicateControl(index) => write!(f, "duplicate RRT control index {index}"),
            Self::Collision(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for RrtError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Collision(error) => Some(error),
            _ => None,
        }
    }
}

/// Generates batched RRT and RRT-Connect kernels with their collision predicate.
///
/// `controls` indexes the FK inputs in planning-coordinate order. Other joints
/// take their values from `RrtSpace::reference`. Spheres independent of controlled
/// inputs are checked with the start and omitted from subsequent collision checks.
/// The scene and reference configuration must remain fixed during a solve.
/// The generated namespace contains
/// `RrtSpace`, `RrtProblem`, `RrtCandidate`, and `validate_rrt`. Backend-independent
/// config/result types live in `rkern`. See `src/cuda/rrt.cuh` for buffer layouts.
///
/// Launch `rkern_rrt` with one block per problem and `32 * warps_per_problem`
/// threads. Each round samples against a snapshot of the tree, checks all edge
/// points in parallel, then attempts direct goal connections from valid endpoints.
/// The final goal edge can exceed `step_size` but uses the same collision resolution.
/// Warps take pending edges as they become available. Valid endpoints are inserted
/// in candidate order, independent of the collision work assignment.
///
/// Launch `rkern_rrtc` with the same block layout to alternate two trees and
/// repeatedly extend the opposite tree toward accepted endpoints. Every edge is
/// bounded by `step_size`. With one candidate in FP32, each warp checks points
/// spread across its edge first, covering the same full grid without separate probes.
/// Both kernels batch candidates against tree snapshots;
/// they do not implement sequential sampling. Parent chains are returned as node
/// indices, with a native `RrtPlanner` wrapper for allocation, launches and path
/// extraction. Its algorithm defaults to RRT; select `RrtAlgorithm::RrtConnect`
/// for RRTC.
/// In FP32 with one candidate and one warp, the wrapper selects a dedicated
/// kernel sharing one EXTEND/CONNECT loop. For standard FP32 trigonometry it
/// first bounds spheres depending on at most two controlled joints over the
/// entire joint domain. If these bounds exclude every obstacle, subsequent
/// edges omit those spheres. Bounds include floating-point rounding and are
/// recomputed per environment on every solve; uncertain cases check all moving
/// spheres. Start validation and standalone collision queries check every sphere.
/// For a certified environment containing one axis-aligned box and one ground,
/// a streaming predicate consumes each center immediately.
/// Other scenes and numerical exceptions use the general collision predicate.
/// Sampling and collision resolution are unchanged.
///
/// The caller owns device buffers, validates config/space with `validate_rrt`,
/// checks device launch/storage limits, and supplies valid scene ranges and IDs.
/// Like the collision kernel, only environment collisions are checked.
pub fn generate_rrt_kernel(
    fk: &FkDag,
    spheres: &[Sphere],
    controls: &[usize],
    options: Options,
) -> Result<String, RrtError> {
    if controls.is_empty() {
        return Err(RrtError::EmptyControls);
    }
    for (position, &index) in controls.iter().enumerate() {
        if index >= fk.joint_names().len() {
            return Err(RrtError::InvalidControl {
                index,
                joint_count: fk.joint_names().len(),
            });
        }
        if controls[..position].contains(&index) {
            return Err(RrtError::DuplicateControl(index));
        }
    }
    let mut source = collision::generate_kernel(fk, spheres, options, Some(controls))
        .map_err(RrtError::Collision)?;
    source.push_str(include_str!("../planning/rrt.hpp"));
    let indices = controls
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    source.push_str(
        &include_str!("rrt.cuh")
            .replace(
                "@NAMESPACE@",
                collision::namespace(options.math.precision()),
            )
            .replace("@DIMENSION@", &controls.len().to_string())
            .replace("@FK_DOF@", &fk.joint_names().len().to_string())
            .replace("@CONTROLS@", &indices),
    );
    let fp32 = options.math.precision() == Precision::F32;
    let single = if fp32 {
        include_str!("rrtc_single.cuh")
    } else {
        ""
    };
    for implementation in [
        include_str!("rrtc.cuh"),
        single,
        include_str!("rrt_runtime.cuh"),
    ] {
        source.push_str(
            &implementation
                .replace(
                    "@NAMESPACE@",
                    collision::namespace(options.math.precision()),
                )
                .replace("@CONTROLS@", &indices)
                .replace("@FP32@", if fp32 { "1" } else { "0" })
                .replace(
                    "@RRTC_SINGLE@",
                    if fp32 {
                        "rkern_rrtc_single"
                    } else {
                        "rkern_rrtc"
                    },
                ),
        );
    }
    Ok(source)
}
