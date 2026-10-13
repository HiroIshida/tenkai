use std::fmt::Write;

use super::kernel::Kernel;
use super::{MathMode, Options, Precision};
use crate::collision::{Error, Sphere};
use crate::dag::{Node, NodeId};
use crate::fk::{FkDag, center_expressions};
use crate::scheduling::Schedule;

const SPHERES_PER_GROUP: usize = 16;

#[derive(Clone, Copy)]
enum Inlining {
    Automatic,
    Never,
    Always,
}

/// Generates `rkern_collision(configurations, environment_ids, scene, output, count)`.
///
/// One thread checks one configuration against its selected environment. Centers
/// are computed in groups of up to 16 spheres, reusing shared FK values. Each group
/// traverses the runtime obstacles once. A group AABB excludes separated finite
/// obstacles before checking individual spheres; ground checks go directly to
/// the narrow phase. A colliding group ends the query.
/// Output is `[count]` bytes: 1 for collision (including touching), 0 otherwise.
/// An empty environment or empty robot sphere list produces 0.
///
/// The generated source defines `rkern_f32` / `rkern_f64` scene types, matching
/// `options.math`. `Scene` is passed by value; its pointers and `environment_ids`
/// (uint32_t per query) reference device buffers. Each `Environment` contains
/// offset/count ranges into the scene's Box, Sphere, Cylinder, and Ground arrays.
/// A union is represented by multiple array entries; a point cloud by equal-radius
/// spheres. No scene hierarchy or self-collision checks are performed.
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
    generate_kernel(fk, spheres, options, None)
}

pub(super) fn generate_kernel(
    fk: &FkDag,
    spheres: &[Sphere],
    options: Options,
    controls: Option<&[usize]>,
) -> Result<String, Error> {
    let mut source = generate_function(fk, spheres, options, controls)?;
    let (scalar, _) = options.math.precision().cuda_spelling();
    let namespace = namespace(options.math.precision());
    let dof = fk.joint_names().len();
    let configuration = if dof == 0 {
        "configurations".to_owned()
    } else {
        format!("configurations + batch * {dof}")
    };
    let collision = if spheres.is_empty() {
        "false".to_owned()
    } else {
        format!("rkern_collision_one({configuration}, scene, environment_ids[batch])")
    };
    write!(
        source,
        r#"
extern "C" __global__ void rkern_collision(
    const {scalar}* configurations, const uint32_t* environment_ids,
    {namespace}::Scene scene, unsigned char* output, size_t count) {{
    const size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch >= count) return;
    output[batch] = static_cast<unsigned char>({collision});
}}
"#,
    )
    .expect("writing to String is infallible");
    Ok(source)
}

pub(super) fn namespace(precision: Precision) -> &'static str {
    match precision {
        Precision::F32 => "rkern_f32",
        Precision::F64 => "rkern_f64",
    }
}

/// Generates `bool rkern_collision_one(q, scene, environment_id)` and scene types.
/// The predicate is callable by CUDA device code without launching a kernel.
/// Inputs and collision semantics match [`generate_collision_kernel`].
pub fn generate_collision_function(
    fk: &FkDag,
    spheres: &[Sphere],
    options: Options,
) -> Result<String, Error> {
    generate_function(fk, spheres, options, None)
}

fn generate_function(
    fk: &FkDag,
    spheres: &[Sphere],
    options: Options,
    controls: Option<&[usize]>,
) -> Result<String, Error> {
    for (index, sphere) in spheres.iter().enumerate() {
        sphere.validate(fk, index)?;
    }
    let centers: Vec<_> = spheres.iter().map(|sphere| sphere.center).collect();
    let (dag, outputs) =
        center_expressions(fk, &centers, options.math.precision(), options.optimization)
            .map_err(Error::Center)?;
    let groups: Vec<_> = outputs.chunks(3).collect();
    let kernel = Kernel {
        name: "rkern_collision_one",
        dag: &dag,
        dof: fk.joint_names().len(),
        schedule: Schedule::for_output_groups(&dag, &groups, options.scheduling)
            .expect("center outputs belong to the extended FK DAG"),
        math: options.math,
    };
    let (scalar, suffix) = options.math.precision().cuda_spelling();
    let namespace = namespace(options.math.precision());
    let mut source = include_str!("sdf.cuh")
        .replace("@NAMESPACE@", namespace)
        .replace("@SCALAR@", scalar)
        .replace("@SUFFIX@", suffix);
    let inlining = if controls.is_some() {
        // Planning branches must share the large FK predicate.
        Inlining::Never
    } else {
        Inlining::Automatic
    };
    write_function(&mut source, &kernel, spheres, inlining);
    if let Some(controls) = controls {
        let dependent = dag.depends_on_inputs(controls);
        let moving: Vec<_> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, center)| center.iter().any(|id| dependent[id.index()]).then_some(i))
            .collect();
        if moving.len() == spheres.len() {
            write!(
                source,
                r#"
__device__ inline bool rkern_collision_moving(
    const {scalar}* q, {namespace}::Scene scene, uint32_t environment_id) {{
    return rkern_collision_one(q, scene, environment_id);
}}
"#,
            )
            .expect("writing to String is infallible");
        } else {
            let moving_spheres: Vec<_> = moving.iter().map(|&i| spheres[i]).collect();
            let moving_groups: Vec<_> = moving.iter().map(|&i| groups[i]).collect();
            let moving_kernel = Kernel {
                name: "rkern_collision_moving",
                schedule: Schedule::for_output_groups(&dag, &moving_groups, options.scheduling)
                    .expect("moving centers belong to the extended FK DAG"),
                dag: kernel.dag,
                dof: kernel.dof,
                math: kernel.math,
            };
            write_function(&mut source, &moving_kernel, &moving_spheres, inlining);
        }
        if options.math.precision() == Precision::F32 {
            write_planning_functions(
                &mut source,
                &kernel,
                spheres,
                &groups,
                &moving,
                controls,
                options,
            );
        }
    }
    Ok(source)
}

fn write_function(
    source: &mut String,
    kernel: &Kernel<'_>,
    spheres: &[Sphere],
    inlining: Inlining,
) {
    let (scalar, suffix) = kernel.math.precision().cuda_spelling();
    let namespace = namespace(kernel.math.precision());
    let name = kernel.name;
    let qualifier = match inlining {
        Inlining::Automatic => "inline",
        Inlining::Never => "__noinline__",
        Inlining::Always => "__forceinline__",
    };
    write!(
        source,
        r#"
__device__ {qualifier} bool {name}(
    const {scalar}* q, {namespace}::Scene scene, uint32_t environment_id) {{
"#,
    )
    .expect("writing to String is infallible");
    if spheres.is_empty() {
        source.push_str("    return false;\n}\n");
        return;
    }
    write!(
        source,
        r#"    const {namespace}::Environment environment = scene.environments[environment_id];
    bool collision = false;
    {namespace}::Vec3 center;
"#,
    )
    .expect("writing to String is infallible");
    kernel.write_body(source, "q", |source, index, value| {
        let axis = ["x", "y", "z"][index % 3];
        writeln!(source, "    center.{axis} = v{};", value.index())
            .expect("writing to String is infallible");
        if index % 3 == 2 {
            let sphere_index = index / 3;
            writeln!(
                source,
                "    const {namespace}::Vec3 point{sphere_index} = center;"
            )
            .expect("writing to String is infallible");
            let end = sphere_index + 1;
            if end % SPHERES_PER_GROUP == 0 || end == spheres.len() {
                let start = sphere_index / SPHERES_PER_GROUP * SPHERES_PER_GROUP;
                write_collision_group(source, &spheres[start..end], start, namespace, suffix);
            }
        }
    });
    source.push_str("    return collision;\n}\n");
}

fn write_planning_functions(
    source: &mut String,
    kernel: &Kernel<'_>,
    spheres: &[Sphere],
    groups: &[&[NodeId]],
    moving: &[usize],
    controls: &[usize],
    options: Options,
) {
    let dependencies: Vec<_> = controls
        .iter()
        .map(|&input| kernel.dag.depends_on_inputs(&[input]))
        .collect();
    // Interval bounds grow quickly along articulated chains. Limit the extra
    // predicate to shallow dependencies instead of generating every subset.
    let bounded: Vec<_> = moving
        .iter()
        .copied()
        .filter(|&i| {
            options.math == MathMode::F32
                && dependencies
                    .iter()
                    .filter(|dependent| groups[i].iter().any(|id| dependent[id.index()]))
                    .count()
                    <= 2
        })
        .collect();
    let remaining: Vec<_> = moving
        .iter()
        .copied()
        .filter(|i| !bounded.contains(i))
        .collect();
    let edge_spheres: Vec<_> = remaining.iter().map(|&i| spheres[i]).collect();
    let edge_groups: Vec<_> = remaining.iter().map(|&i| groups[i]).collect();
    let edge_kernel = Kernel {
        name: "rkern_collision_edge",
        schedule: Schedule::for_output_groups(kernel.dag, &edge_groups, options.scheduling)
            .expect("edge centers belong to the extended FK DAG"),
        dag: kernel.dag,
        dof: kernel.dof,
        math: kernel.math,
    };
    write_function(source, &edge_kernel, &edge_spheres, Inlining::Always);
    source.push_str(
        "\n__device__ inline bool rkern_collision_in_workspace(const float* q, const float* lower, const float* upper) {\n",
    );
    for (d, dependent) in dependencies.iter().enumerate() {
        if bounded
            .iter()
            .any(|&i| groups[i].iter().any(|id| dependent[id.index()]))
        {
            writeln!(
                source,
                "    if (!(q[{d}] >= lower[{d}] && q[{d}] <= upper[{d}])) return false;"
            )
            .expect("writing to String is infallible");
        }
    }
    source.push_str("    return true;\n}\n");
    source.push_str(include_str!("workspace.cuh"));
    source.push_str(
        r#"
__device__ __noinline__ bool rkern_collision_workspace_clear(
    const float* lower, const float* upper, const float* reference,
    rkern_f32::Scene scene, uint32_t environment_id) {
    using namespace rkern_f32;
    const Environment environment = scene.environments[environment_id];
"#,
    );
    if bounded.is_empty() {
        source.push_str("    return true;\n}\n");
        return;
    }
    let outputs: Vec<_> = bounded
        .iter()
        .flat_map(|&i| groups[i].iter().copied())
        .collect();
    let needed = kernel.dag.required_nodes(&outputs);
    for (index, &node) in kernel.dag.nodes().iter().enumerate() {
        if !needed[index] {
            continue;
        }
        write!(source, "    const WorkspaceInterval v{index} = ")
            .expect("writing to String is infallible");
        match node {
            Node::Constant(value) => writeln!(source, "{{{value:.17e}f, {value:.17e}f}};"),
            Node::Input(input) => match controls.iter().position(|&i| i == input) {
                Some(d) => writeln!(source, "workspace_input(lower[{d}], upper[{d}]);"),
                None => writeln!(source, "{{reference[{input}], reference[{input}]}};"),
            },
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                let operation = match node {
                    Node::Add(..) => "add",
                    Node::Sub(..) => "sub",
                    _ => "mul",
                };
                writeln!(
                    source,
                    "workspace_{operation}(v{}, v{});",
                    a.index(),
                    b.index()
                )
            }
            Node::Sin(a) | Node::Cos(a) => writeln!(
                source,
                "workspace_trig(v{}, {});",
                a.index(),
                matches!(node, Node::Cos(_)),
            ),
        }
        .expect("writing to String is infallible");
    }
    for i in bounded {
        let [x, y, z] = [
            groups[i][0].index(),
            groups[i][1].index(),
            groups[i][2].index(),
        ];
        let radius = spheres[i].radius;
        writeln!(source, "    if (!workspace_clear(v{x}, v{y}, v{z}, {radius:.17e}f, scene, environment)) return false;")
            .expect("writing to String is infallible");
    }
    source.push_str("    return true;\n}\n");
}

fn write_collision_group(
    source: &mut String,
    spheres: &[Sphere],
    start: usize,
    namespace: &str,
    suffix: &str,
) {
    if spheres.len() > 1 {
        let bounds = ["fmin", "fmax"].map(|operation| {
            let coordinates = ["x", "y", "z"].map(|axis| {
                (start + 1..start + spheres.len())
                    .fold(format!("point{start}.{axis}"), |value, i| {
                        format!("{operation}{suffix}({value}, point{i}.{axis})")
                    })
            });
            format!("{{{}}}", coordinates.join(", "))
        });
        let radius = spheres
            .iter()
            .map(|sphere| sphere.radius)
            .fold(0.0, f64::max);
        writeln!(
            source,
            "    const {namespace}::SphereBounds bounds{start} = {{{}, {radius:.17e}{suffix}}};",
            bounds.join(", "),
        )
        .expect("writing to String is infallible");
    }
    for (primitive, field) in [
        ("Box", "boxes"),
        ("Sphere", "spheres"),
        ("Cylinder", "cylinders"),
        ("Ground", "grounds"),
    ] {
        // Only the small, fixed sphere group is expanded; unrolling runtime obstacle
        // loops duplicates every primitive's arithmetic and bloats the kernel.
        write!(
            source,
            r#"#ifdef __CUDACC__
    #pragma unroll 1
#endif
    for (size_t i = 0; i < environment.{field}.count; ++i) {{
        const {namespace}::{primitive}& obstacle = scene.{field}[environment.{field}.offset + i];
"#,
        )
        .expect("writing to String is infallible");
        let broad_phase = spheres.len() > 1 && primitive != "Ground";
        if broad_phase {
            writeln!(
                source,
                "        if ({namespace}::overlaps(obstacle, bounds{start})) {{"
            )
            .expect("writing to String is infallible");
        }
        for (offset, sphere) in spheres.iter().enumerate() {
            let index = start + offset;
            let radius = sphere.radius;
            writeln!(
                source,
                "        collision |= {namespace}::intersects(obstacle, point{index}, {radius:.17e}{suffix});"
            )
            .expect("writing to String is infallible");
        }
        if broad_phase {
            source.push_str("        }\n");
        }
        source.push_str("        if (collision) return true;\n");
        source.push_str("    }\n");
    }
}
