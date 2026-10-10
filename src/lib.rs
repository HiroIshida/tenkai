//! Parse URDF models, build scalar FK DAGs, and generate batched CUDA kernels.
//! Robot and sphere definitions are fixed during generation; joint values are runtime inputs.

pub mod cuda;
pub mod dag;
pub mod fk;
pub mod scheduling;
pub mod urdf;

mod transform;
