"""CUDA emission using the packaged device and runtime templates."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

from .collision import Sphere, center_expressions
from .dag import Binary, BinaryOp, Constant, Dag, Input, Optimization, Precision, Trig, TrigOp, f32
from .model import FkDag
from .scheduling import Evaluate, Store, schedule

type MathMode = Literal["f32", "f32-sfu", "f64"]
TEMPLATES = Path(__file__).with_name("templates")


@dataclass(frozen=True)
class Options:
    math: MathMode = "f32-sfu"
    optimization: Optimization = "algebraic"
    live_budget: int | None = None

    def __post_init__(self) -> None:
        if self.math not in ("f32", "f32-sfu", "f64"):
            raise ValueError("invalid math mode")
        if self.optimization not in ("none", "basic", "algebraic"):
            raise ValueError("invalid optimization")
        if self.live_budget is not None and self.live_budget < 0:
            raise ValueError("live budget must be nonnegative")

    @property
    def precision(self) -> Precision:
        if self.math == "f64":
            return "f64"
        return "f32"

    @property
    def scalar(self) -> str:
        if self.math == "f64":
            return "double"
        return "float"

    @property
    def suffix(self) -> str:
        if self.math == "f64":
            return ""
        return "f"

    @property
    def namespace(self) -> str:
        return "rkern_" + self.precision


def number(value: float) -> str:
    # Preserve the Rust reference's exponent spelling for exact source comparisons.
    mantissa, exponent = format(value, ".17e").split("e")
    return mantissa + "e" + str(int(exponent))


def template(name: str, replacements: Sequence[tuple[str, str | int]] = ()) -> str:
    source = (TEMPLATES / name).read_text()
    for key, value in replacements:
        source = source.replace("@" + key + "@", str(value))
    return source


class Kernel:
    def __init__(
        self,
        name: str,
        dag: Dag,
        dof: int,
        outputs: Sequence[int],
        options: Options,
        groups: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.name = name
        self.dag = dag
        self.dof = dof
        self.options = options
        self.schedule = schedule(dag, outputs, options.live_budget, groups)

    def source(self) -> str:
        scalar = self.options.scalar
        source = f"""#include <stddef.h>
#include <math.h>

extern "C" __global__ void {self.name}(
    const {scalar}* configurations, {scalar}* output, size_t count) {{
    const size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch >= count) return;
"""
        count = self.schedule.output_count

        def store(index: int, value: int) -> str:
            return f"    output[batch * {count} + {index}] = v{value};\n"

        source += self.body(f"(configurations + batch * {self.dof})", False, store)
        return source + "}\n"

    def trig_pair(self, index: int, next_id: int) -> tuple[int, int, int] | None:
        node = self.dag.nodes[index]
        following = self.dag.nodes[next_id]
        if not isinstance(node, Trig) or not isinstance(following, Trig):
            return None
        if node.angle != following.angle or node.op == following.op:
            return None
        if node.op == TrigOp.SIN:
            return node.angle, index, next_id
        return node.angle, next_id, index

    def expression(self, index: int, input_name: str) -> str:
        node = self.dag.nodes[index]
        match node:
            case Constant(value):
                return number(value) + self.options.suffix
            case Input(index):
                return f"{input_name}[{index}]"
            case Trig(op, angle):
                function = op.value + self.options.suffix
                if self.options.math == "f32-sfu":
                    function = "__" + function
                return f"{function}(v{angle})"
            case Binary(op, a, b):
                operator = {BinaryOp.ADD: "+", BinaryOp.SUB: "-", BinaryOp.MUL: "*"}[op]
                return f"v{a} {operator} v{b}"
        assert_never(node)

    def body(self, input_name: str, pair_trig: bool, store: Callable[[int, int], str]) -> str:
        source: list[str] = []
        steps = self.schedule.steps
        position = 0
        while position < len(steps):
            step = steps[position]
            position += 1
            if isinstance(step, Store):
                source.append(store(step.index, step.node))
                continue
            if pair_trig and self.options.math == "f32" and position < len(steps):
                following = steps[position]
                if isinstance(following, Evaluate):
                    pair = self.trig_pair(step.node, following.node)
                    if pair is not None:
                        angle, sin, cos = pair
                        source.append(f"    float v{sin}, v{cos};\n")
                        source.append(f"    sincosf(v{angle}, &v{sin}, &v{cos});\n")
                        position += 1
                        continue
            expression = self.expression(step.node, input_name)
            source.append(f"    const {self.options.scalar} v{step.node} = {expression};\n")
        return "".join(source)


def generate_fk_kernel(fk: FkDag, link_index: int, options: Options = Options()) -> str:
    if not 0 <= link_index < len(fk.poses):
        raise ValueError("link index is out of range")
    pose = fk.poses[link_index]
    dag, outputs = fk.dag.optimize(
        [*pose.translation, *pose.rotation], options.precision, options.optimization
    )
    return Kernel("rkern_fk", dag, len(fk.joint_names), outputs, options).source()


def generate_centers_kernel(
    fk: FkDag, spheres: Sequence[Sphere], options: Options = Options()
) -> str:
    dag, outputs = center_expressions(fk, spheres, options.precision, options.optimization)
    return Kernel("rkern_centers", dag, len(fk.joint_names), outputs, options).source()


def collision_group(spheres: Sequence[Sphere], start: int, options: Options) -> str:
    namespace = options.namespace
    suffix = options.suffix
    source = []
    if len(spheres) > 1:
        bounds = []
        for operation in ("fmin", "fmax"):
            coordinates = []
            for axis in "xyz":
                value = f"point{start}.{axis}"
                for i in range(start + 1, start + len(spheres)):
                    value = f"{operation}{suffix}({value}, point{i}.{axis})"
                coordinates.append(value)
            bounds.append("{" + ", ".join(coordinates) + "}")
        radius = number(max(s.radius for s in spheres))
        source.append(
            f"    const {namespace}::SphereBounds bounds{start} = {{{', '.join(bounds)}, {radius}{suffix}}};\n"
        )
    for primitive, field in (
        ("Box", "boxes"),
        ("Sphere", "spheres"),
        ("Cylinder", "cylinders"),
        ("Ground", "grounds"),
    ):
        source.append(f"""#ifdef __CUDACC__
    #pragma unroll 1
#endif
    for (size_t i = 0; i < environment.{field}.count; ++i) {{
        const {namespace}::{primitive}& obstacle = scene.{field}[environment.{field}.offset + i];
""")
        broad_phase = len(spheres) > 1 and primitive != "Ground"
        if broad_phase:
            source.append(f"        if ({namespace}::overlaps(obstacle, bounds{start})) {{\n")
        for offset, sphere in enumerate(spheres):
            source.append(
                f"        collision |= {namespace}::intersects(obstacle, point{start + offset}, {number(sphere.radius)}{suffix});\n"
            )
        if broad_phase:
            source.append("        }\n")
        source.append("        if (collision) return true;\n    }\n")
    return "".join(source)


def collision_function(kernel: Kernel, spheres: Sequence[Sphere], qualifier: str) -> str:
    options = kernel.options
    source = f"""
__device__ {qualifier} bool {kernel.name}(
    const {options.scalar}* q, {options.namespace}::Scene scene, uint32_t environment_id) {{
"""
    if not spheres:
        return source + "    return false;\n}\n"
    source += f"""    const {options.namespace}::Environment environment = scene.environments[environment_id];
    bool collision = false;
    {options.namespace}::Vec3 center;
"""

    def store(index: int, value: int) -> str:
        text = f"    center.{'xyz'[index % 3]} = v{value};\n"
        if index % 3 == 2:
            sphere_index = index // 3
            text += f"    const {options.namespace}::Vec3 point{sphere_index} = center;\n"
            end = sphere_index + 1
            if end % 16 == 0 or end == len(spheres):
                start = sphere_index // 16 * 16
                text += collision_group(spheres[start:end], start, options)
        return text

    return source + kernel.body("q", False, store) + "    return collision;\n}\n"


def streaming_function(kernel: Kernel, spheres: Sequence[Sphere]) -> str:
    source = (
        template("cuda/streaming.cuh")
        + """
__device__ inline bool rkern_collision_streamable(rkern_f32::Scene scene, uint32_t environment_id) {
"""
    )
    if any(not math.isfinite(f32(f32(s.radius) * f32(s.radius))) for s in spheres):
        source += "    return false;\n}\n"
    else:
        source += """    const auto environment = scene.environments[environment_id];
    if (environment.boxes.count != 1 || environment.grounds.count != 1 ||
        environment.spheres.count || environment.cylinders.count) return false;
    const auto& box = scene.boxes[environment.boxes.offset];
    for (unsigned i = 0; i < 9; ++i)
        if (box.pose.world_to_local[i] != (i % 4 == 0 ? 1.0f : 0.0f)) return false;
    return true;
}
"""
    source += """
__device__ __forceinline__ rkern_f32::StreamingCollision rkern_collision_streaming(
    const float* q, rkern_f32::Scene scene, uint32_t environment_id, unsigned mask) {
    using namespace rkern_f32;
    const Environment environment = scene.environments[environment_id];
    const Box& box = scene.boxes[environment.boxes.offset];
    const Ground& ground = scene.grounds[environment.grounds.offset];
    bool collision = false, retry = false;
    Vec3 center;
"""

    def store(index: int, value: int) -> str:
        text = f"    center.{'xyz'[index % 3]} = v{value};\n"
        if index % 3 == 2:
            radius = number(spheres[index // 3].radius)
            text += f"    collision |= streaming_intersects(box, center, {radius}f, retry);\n    collision |= intersects(ground, center, {radius}f);\n"
            if (index // 3 + 1) % 8 == 0:
                text += "    if (__any_sync(mask, collision || retry))\n        return retry ? StreamingCollision::Retry : StreamingCollision::Blocked;\n"
        return text

    return (
        source
        + kernel.body("q", True, store)
        + "    return retry ? StreamingCollision::Retry :\n        (collision ? StreamingCollision::Blocked : StreamingCollision::Clear);\n}\n"
    )


def planning_functions(
    kernel: Kernel,
    spheres: Sequence[Sphere],
    groups: Sequence[Sequence[int]],
    moving: Sequence[int],
    controls: Sequence[int],
) -> str:
    options = kernel.options
    dag = kernel.dag
    dependent = [dag.depends_on_inputs([i]) for i in controls]
    bounded = []
    if options.math == "f32":
        for index in moving:
            dependency_count = 0
            for flags in dependent:
                if any(flags[node] for node in groups[index]):
                    dependency_count += 1
            if dependency_count <= 2:
                bounded.append(index)
    remaining = [i for i in moving if i not in bounded]
    edge_spheres = [spheres[i] for i in remaining]
    edge_groups = [groups[i] for i in remaining]
    edge = Kernel(
        "rkern_collision_edge",
        dag,
        kernel.dof,
        [node for group in edge_groups for node in group],
        options,
        edge_groups,
    )
    source = collision_function(edge, edge_spheres, "__forceinline__") + streaming_function(
        edge, edge_spheres
    )
    source += "\n__device__ inline bool rkern_collision_in_workspace(const float* q, const float* lower, const float* upper) {\n"
    for d, flags in enumerate(dependent):
        if any(any(flags[node] for node in groups[i]) for i in bounded):
            source += f"    if (!(q[{d}] >= lower[{d}] && q[{d}] <= upper[{d}])) return false;\n"
    source += "    return true;\n}\n" + template("cuda/workspace.cuh")
    source += """
__device__ __noinline__ bool rkern_collision_workspace_clear(
    const float* lower, const float* upper, const float* reference,
    rkern_f32::Scene scene, uint32_t environment_id) {
    using namespace rkern_f32;
    const Environment environment = scene.environments[environment_id];
"""
    needed = dag.required_nodes([node for i in bounded for node in groups[i]])
    for index, node in enumerate(dag.nodes):
        if not needed[index]:
            continue
        match node:
            case Constant(value):
                expression = "{" + number(value) + "f, " + number(value) + "f}"
            case Input(input_index):
                if input_index in controls:
                    dimension = controls.index(input_index)
                    expression = f"workspace_input(lower[{dimension}], upper[{dimension}])"
                else:
                    expression = f"{{reference[{input_index}], reference[{input_index}]}}"
            case Trig(op, angle):
                cosine = "true" if op == TrigOp.COS else "false"
                expression = f"workspace_trig(v{angle}, {cosine})"
            case Binary(op, a, b):
                expression = f"workspace_{op.value}(v{a}, v{b})"
        source += f"    const WorkspaceInterval v{index} = {expression};\n"
    for i in bounded:
        x, y, z = groups[i]
        source += f"    if (!workspace_clear(v{x}, v{y}, v{z}, {number(spheres[i].radius)}f, scene, environment)) return false;\n"
    return source + "    return true;\n}\n"


def generate_collision_function(
    fk: FkDag,
    spheres: Sequence[Sphere],
    options: Options = Options(),
    controls: Sequence[int] | None = None,
) -> str:
    for sphere in spheres:
        sphere.validate(fk)
    dag, outputs = center_expressions(fk, spheres, options.precision, options.optimization)
    groups = [outputs[i : i + 3] for i in range(0, len(outputs), 3)]
    kernel = Kernel("rkern_collision_one", dag, len(fk.joint_names), outputs, options, groups)
    source = template(
        "cuda/sdf.cuh",
        (("NAMESPACE", options.namespace), ("SCALAR", options.scalar), ("SUFFIX", options.suffix)),
    )
    qualifier = "__noinline__" if controls is not None else "inline"
    source += collision_function(kernel, spheres, qualifier)
    if controls is not None:
        dependent = dag.depends_on_inputs(controls)
        moving = [i for i, group in enumerate(groups) if any(dependent[node] for node in group)]
        if len(moving) == len(spheres):
            source += f"""
__device__ inline bool rkern_collision_moving(
    const {options.scalar}* q, {options.namespace}::Scene scene, uint32_t environment_id) {{
    return rkern_collision_one(q, scene, environment_id);
}}
"""
        else:
            moving_groups = [groups[i] for i in moving]
            moving_kernel = Kernel(
                "rkern_collision_moving",
                dag,
                kernel.dof,
                [node for group in moving_groups for node in group],
                options,
                moving_groups,
            )
            source += collision_function(moving_kernel, [spheres[i] for i in moving], qualifier)
        if options.precision == "f32":
            source += planning_functions(kernel, spheres, groups, moving, controls)
    return source


def generate_collision_kernel(
    fk: FkDag,
    spheres: Sequence[Sphere],
    options: Options = Options(),
    controls: Sequence[int] | None = None,
) -> str:
    source = generate_collision_function(fk, spheres, options, controls)
    configuration = (
        f"configurations + batch * {len(fk.joint_names)}" if fk.joint_names else "configurations"
    )
    collision = (
        f"rkern_collision_one({configuration}, scene, environment_ids[batch])"
        if spheres
        else "false"
    )
    return (
        source
        + f"""
extern "C" __global__ void rkern_collision(
    const {options.scalar}* configurations, const uint32_t* environment_ids,
    {options.namespace}::Scene scene, unsigned char* output, size_t count) {{
    const size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch >= count) return;
    output[batch] = static_cast<unsigned char>({collision});
}}
"""
    )


def generate_rrt_kernel(
    fk: FkDag,
    spheres: Sequence[Sphere],
    controls: Sequence[int],
    options: Options = Options(),
) -> str:
    if not controls:
        raise ValueError("RRT requires at least one controlled joint")
    if any(not 0 <= i < len(fk.joint_names) for i in controls):
        raise ValueError("RRT control index exceeds FK joint count")
    if len(set(controls)) != len(controls):
        raise ValueError("duplicate RRT control index")
    source = generate_collision_kernel(fk, spheres, options, controls) + template(
        "planning/rrt.hpp"
    )
    indices = ", ".join(map(str, controls))
    source += template(
        "cuda/rrt.cuh",
        (
            ("NAMESPACE", options.namespace),
            ("DIMENSION", len(controls)),
            ("FK_DOF", len(fk.joint_names)),
            ("CONTROLS", indices),
        ),
    )
    fp32 = options.precision == "f32"
    files = ["cuda/rrtc.cuh"]
    if fp32:
        files.append("cuda/rrtc_single.cuh")
    files.append("cuda/rrt_runtime.cuh")
    for name in files:
        source += template(
            name,
            (
                ("NAMESPACE", options.namespace),
                ("CONTROLS", indices),
                ("FP32", int(fp32)),
                ("RRTC_SINGLE", "rkern_rrtc_single" if fp32 else "rkern_rrtc"),
            ),
        )
    return source
