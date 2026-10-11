//! Parse URDF models, build scalar FK DAGs, and generate CUDA collision and RRT code.
//! Robot and sphere definitions are fixed during generation; joint values and environments
//! are runtime inputs.

pub mod collision;
pub mod cuda;
pub mod dag;
pub mod fk;
pub mod scheduling;
pub mod urdf;

mod transform;
