//! CUDA C++ generation for batched FK with one thread per configuration.

use std::fmt::{self, Write};

use crate::dag::{Dag, Node, NodeId};
use crate::fk::{FkDag, rotate_vector};

/// Scalar type used for kernel buffers, intermediates, and trigonometric functions.
/// The source FK DAG retains its f64 constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    F32,
    F64,
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

/// Generates `rkern_fk(configurations, poses, count)` for `link_poses()[link_index]`.
///
/// Both buffers contain contiguous floats (f32): inputs are `[count][joint_names().len()]`
/// and outputs are `[count][7]` in `[x, y, z, qx, qy, qz, qw]` order. Input joints
/// follow `joint_names()`. Use separate input/output buffers and a one-dimensional
/// CUDA launch with at least `count` threads. Excess threads do no work.
pub fn generate_fk_kernel(fk: &FkDag, link_index: usize) -> Result<String, InvalidLinkIndex> {
    generate_fk_kernel_with_precision(fk, link_index, Precision::default())
}

/// Like [`generate_fk_kernel`], with input/output buffers and arithmetic in `precision`.
pub fn generate_fk_kernel_with_precision(
    fk: &FkDag,
    link_index: usize,
    precision: Precision,
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
    Ok(generate_kernel(
        fk.dag(),
        fk.joint_names().len(),
        &outputs,
        "rkern_fk",
        precision,
    ))
}

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

/// Generates `rkern_centers(configurations, centers, count)` for debugging sphere positions.
///
/// Both buffers contain contiguous floats (f32): inputs are `[count][joint_names().len()]`
/// and outputs are `[count][spheres.len()][3]`, in input sphere order and xyz order.
/// Centers are relative to the robot root frame, with no additional base transform.
/// Each center is `link_translation + link_rotation * sphere.position`.
/// Use separate buffers and a one-dimensional launch with at least `count` threads.
/// An empty sphere slice produces a kernel that writes nothing.
pub fn generate_centers_kernel(
    fk: &FkDag,
    spheres: &[SphereCenter],
) -> Result<String, CentersError> {
    generate_centers_kernel_with_precision(fk, spheres, Precision::default())
}

/// Like [`generate_centers_kernel`], with input/output buffers and arithmetic in `precision`.
pub fn generate_centers_kernel_with_precision(
    fk: &FkDag,
    spheres: &[SphereCenter],
    precision: Precision,
) -> Result<String, CentersError> {
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
        let offset = rotate_vector(&mut dag, pose.rotation, local);
        outputs.extend((0..3).map(|i| dag.add(pose.translation[i], offset[i])));
    }
    Ok(generate_kernel(
        &dag,
        fk.joint_names().len(),
        &outputs,
        "rkern_centers",
        precision,
    ))
}

fn generate_kernel(
    dag: &Dag,
    dof: usize,
    outputs: &[NodeId],
    name: &str,
    precision: Precision,
) -> String {
    let nodes = dag.nodes();
    let mut needed = vec![false; nodes.len()];
    for output in outputs {
        needed[output.index()] = true;
    }
    for (index, node) in nodes.iter().enumerate().rev() {
        if !needed[index] {
            continue;
        }
        match *node {
            Node::Constant(_) | Node::Input(_) => {}
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                needed[a.index()] = true;
                needed[b.index()] = true;
            }
            Node::Sin(a) | Node::Cos(a) => needed[a.index()] = true,
        }
    }

    let (scalar, suffix) = match precision {
        Precision::F32 => ("float", "f"),
        Precision::F64 => ("double", ""),
    };
    let mut source = format!(
        r#"#include <stddef.h>
#include <math.h>

extern "C" __global__ void {name}(
    const {scalar}* configurations, {scalar}* output, size_t count) {{
    const size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch >= count) return;
"#,
    );
    for (index, node) in nodes.iter().enumerate() {
        if !needed[index] {
            continue;
        }
        write!(source, "    const {scalar} v{index} = ").expect("writing to String is infallible");
        match *node {
            Node::Constant(value) => writeln!(source, "{value:.17e}{suffix};"),
            Node::Input(input) => writeln!(source, "configurations[batch * {dof} + {input}];"),
            Node::Add(a, b) => writeln!(source, "v{} + v{};", a.index(), b.index()),
            Node::Sub(a, b) => writeln!(source, "v{} - v{};", a.index(), b.index()),
            Node::Mul(a, b) => writeln!(source, "v{} * v{};", a.index(), b.index()),
            Node::Sin(a) => writeln!(source, "sin{suffix}(v{});", a.index()),
            Node::Cos(a) => writeln!(source, "cos{suffix}(v{});", a.index()),
        }
        .expect("writing to String is infallible");
    }
    let components = outputs.len();
    for (component, output) in outputs.iter().enumerate() {
        writeln!(
            source,
            "    output[batch * {components} + {component}] = v{};",
            output.index()
        )
        .expect("writing to String is infallible");
    }
    source.push_str("}\n");
    source
}
