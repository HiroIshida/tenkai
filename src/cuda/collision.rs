use std::fmt::Write;

use super::kernel::Kernel;
use super::{Options, Precision};
use crate::collision::{Error, Sphere};
use crate::fk::{FkDag, center_expressions};
use crate::scheduling::Schedule;

/// Generates `rkern_collision(configurations, environment_ids, scene, output, count)`.
///
/// One thread checks one configuration against its selected environment. Each sphere
/// is checked as its center is computed, reusing shared FK values across spheres.
/// Every sphere/obstacle pair is evaluated, without early exit.
/// Output is `[count]` bytes: 1 for collision (including touching), 0 otherwise.
/// An empty environment or empty robot sphere list produces 0.
///
/// The generated source defines `rkern_f32` / `rkern_f64` scene types, matching
/// `options.math`. `Scene` is passed by value; its pointers and `environment_ids`
/// (uint32_t per query) reference device buffers. Each `Environment` contains
/// offset/count ranges into the scene's Box, Sphere, Cylinder, and Ground arrays.
/// A union is represented by multiple array entries; a point cloud by equal-radius
/// spheres. No spatial acceleration or self-collision checks are performed.
///
/// Inputs follow `FkDag::joint_names()`, as in `generate_centers_kernel`.
/// Environment geometry must be in the robot root frame. `Pose::world_to_local`
/// is a row-major inverse rigid rotation, with `Pose::position` in that root frame.
/// Box extents and cylinder height are full dimensions; Ground distance is
/// `z + height`, matching plainmp (the surface is at z = -height).
///
/// The caller must supply valid IDs/ranges, finite inputs/geometry, orthonormal
/// rotations, and nonnegative dimensions representable in the selected precision.
/// Buffers must not alias outputs. Zero-count ranges may have null array pointers.
/// Launch a one-dimensional grid with at least `count` threads; excess threads do no work.
pub fn generate_collision_kernel(
    fk: &FkDag,
    spheres: &[Sphere],
    options: Options,
) -> Result<String, Error> {
    for (index, sphere) in spheres.iter().enumerate() {
        sphere.validate(fk, index)?;
    }
    let centers: Vec<_> = spheres.iter().map(|sphere| sphere.center).collect();
    let (dag, outputs) = center_expressions(fk, &centers).map_err(Error::Center)?;
    let groups: Vec<_> = outputs.chunks(3).collect();
    let kernel = Kernel {
        name: "rkern_collision",
        dag: &dag,
        dof: fk.joint_names().len(),
        schedule: Schedule::for_output_groups(&dag, &groups, options.scheduling)
            .expect("center outputs belong to the extended FK DAG"),
        math: options.math,
    };
    let (scalar, suffix) = options.math.precision().cuda_spelling();
    let namespace = match options.math.precision() {
        Precision::F32 => "rkern_f32",
        Precision::F64 => "rkern_f64",
    };
    let mut source = include_str!("sdf.cuh")
        .replace("@NAMESPACE@", namespace)
        .replace("@SCALAR@", scalar)
        .replace("@SUFFIX@", suffix);
    write!(
        source,
        r#"
extern "C" __global__ void rkern_collision(
    const {scalar}* configurations, const uint32_t* environment_ids,
    {namespace}::Scene scene, unsigned char* output, size_t count) {{
    const size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch >= count) return;
"#,
    )
    .expect("writing to String is infallible");
    if spheres.is_empty() {
        source.push_str("    output[batch] = 0;\n}\n");
        return Ok(source);
    }
    write!(
        source,
        r#"    const {namespace}::Environment environment = scene.environments[environment_ids[batch]];
    {scalar} clearance = INFINITY;
    {namespace}::Vec3 center;
"#,
    )
    .expect("writing to String is infallible");
    kernel.write_body(&mut source, |source, index, value| {
        let axis = ["x", "y", "z"][index % 3];
        writeln!(source, "    center.{axis} = v{};", value.index())
            .expect("writing to String is infallible");
        if index % 3 == 2 {
            let radius = spheres[index / 3].radius;
            writeln!(source, "    clearance = fmin{suffix}(clearance, {namespace}::distance(scene, environment, center) - {radius:.17e}{suffix});")
                .expect("writing to String is infallible");
        }
    });
    source.push_str("    output[batch] = static_cast<unsigned char>(clearance <= 0);\n}\n");
    Ok(source)
}
