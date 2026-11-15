"""CUDA emission using the packaged device and runtime templates."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from .collision import (
    CollisionDag,
    SelfCollisionRules,
    Sphere,
    center_expressions,
    collision_expressions,
)
from .dag import Binary, BinaryOp, Constant, Dag, Input, Optimization, Trig, TrigOp, f32
from .model import KinematicModel
from .scheduling import Evaluate, Store, schedule

TEMPLATES = Path(__file__).with_name("templates")


@dataclass(frozen=True)
class Options:
    optimization: Optimization = "algebraic"
    live_budget: int | None = None

    def __post_init__(self) -> None:
        if self.optimization not in ("none", "basic", "algebraic"):
            raise ValueError("invalid optimization")
        if self.live_budget is not None and self.live_budget < 0:
            raise ValueError("live budget must be nonnegative")


def number(value: float) -> str:
    mantissa, exponent = format(value, ".17e").split("e")
    return mantissa + "e" + str(int(exponent))


def template(name: str, replacements: Sequence[tuple[str, str | int]] = ()) -> str:
    source = (TEMPLATES / name).read_text()
    for key, value in replacements:
        # clang-format splits @KEY@ tokens; identifiers and block comments stay intact.
        marker = "TPL_" + key
        source = source.replace("/* " + marker + " */", str(value))
        source = source.replace(marker, str(value))

    return source


class DagEmitter:
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
        count = self.schedule.output_count

        def store(index: int, value: int) -> str:
            return f"    output[batch * {count} + {index}] = v{value};\n"

        body = self.body(f"(configurations + batch * {self.dof})", store)
        return template(
            "kinematics_batch_kernel.cuh",
            (("NAME", self.name), ("BODY", body)),
        )

    def expression(self, index: int, input_name: str) -> str:
        node = self.dag.nodes[index]
        match node:
            case Constant(value):
                return number(value) + "f"
            case Input(index):
                return f"{input_name}[{index}]"
            case Trig(op, angle):
                return f"__{op.value}f(v{angle})"
            case Binary(op, a, b):
                operator = {BinaryOp.ADD: "+", BinaryOp.SUB: "-", BinaryOp.MUL: "*"}[op]
                return f"v{a} {operator} v{b}"
        assert_never(node)

    def body(
        self,
        input_name: str,
        store: Callable[[int, int], str],
        cached: set[int] | None = None,
    ) -> str:
        source: list[str] = []

        for step in self.schedule.steps:
            if isinstance(step, Store):
                source.append(store(step.index, step.node))
                continue
            if cached is not None and step.node in cached:
                continue

            expression = self.expression(step.node, input_name)
            source.append(f"    const float v{step.node} = {expression};\n")

        return "".join(source)


def generate_fk_kernel(kin: KinematicModel, link_index: int, options: Options = Options()) -> str:
    if not 0 <= link_index < len(kin.poses):
        raise ValueError("link index is out of range")

    pose = kin.poses[link_index]
    dag, outputs = kin.dag.optimize([*pose.translation, *pose.rotation], options.optimization)
    emitter = DagEmitter("tenkai_fk", dag, len(kin.joints), outputs, options)
    return emitter.source()


def generate_centers_kernel(
    kin: KinematicModel, spheres: Sequence[Sphere], options: Options = Options()
) -> str:
    dag, outputs = center_expressions(kin, spheres, options.optimization)
    emitter = DagEmitter("tenkai_centers", dag, len(kin.joints), outputs, options)
    return emitter.source()


def sphere_collision(radius: float) -> str:
    source: list[str] = []

    for primitive, field in (
        ("Box", "boxes"),
        ("Sphere", "spheres"),
        ("Cylinder", "cylinders"),
        ("Ground", "grounds"),
    ):
        source.append(
            "#ifdef __CUDACC__\n"
            "    #pragma unroll 1\n"
            "#endif\n"
            f"    for (size_t i = 0; i < environment.{field}.count; ++i) {{\n"
            f"        const tenkai_f32::{primitive}& obstacle = "
            f"scene.{field}[environment.{field}.offset + i];\n"
        )

        source.append(
            f"        collision |= tenkai_f32::intersects(obstacle, center, {number(radius)}f);\n"
        )
        source.append("    }\n")

    return "".join(source)


def collision_definition(name: str, qualifier: str, body: str) -> str:
    return (
        f"\n__device__ {qualifier} bool {name}(\n"
        "    const float* q, tenkai_f32::Scene scene, "
        f"uint32_t environment_id) {{\n"
        f"{body}}}\n"
    )


def collision_function(
    emitter: DagEmitter, spheres: Sequence[Sphere], geometry: CollisionDag, qualifier: str
) -> str:
    if spheres or geometry.rules.self_pairs:
        body = (
            "    const tenkai_f32::Environment environment = "
            "scene.environments[environment_id];\n"
            "    bool collision = false;\n"
            "    tenkai_f32::Vec3 center;\n"
        )
        body += link_collision_body(emitter, spheres, geometry, streaming=False)
        body += "    return collision;\n"
    else:
        body = "    return false;\n"

    return collision_definition(emitter.name, qualifier, body)


def link_collision_body(
    emitter: DagEmitter,
    spheres: Sequence[Sphere],
    geometry: CollisionDag,
    *,
    streaming: bool,
) -> str:
    links: dict[int, list[int]] = {}
    for index, sphere in enumerate(spheres):
        if sphere.link_index in geometry.rules.self_only_links:
            continue
        links.setdefault(sphere.link_index, []).append(index)

    pose_outputs: list[int] = []
    for link in links:
        pose = geometry.poses[link]
        pose_outputs.extend((*pose.translation, *pose.rotation))
    pose_emitter = DagEmitter("", emitter.dag, emitter.dof, pose_outputs, emitter.options)
    source = [pose_emitter.body("q", lambda _index, _node: "")]
    cached = {step.node for step in pose_emitter.schedule.steps if isinstance(step, Evaluate)}
    streaming_exit = (
        "    if (__any_sync(mask, collision || retry))\n"
        "        return retry ? StreamingCollision::Retry : StreamingCollision::Blocked;\n"
    )

    outputs = [step.node for step in emitter.schedule.steps if isinstance(step, Store)]
    for link, indices in links.items():
        members = [spheres[index] for index in indices]
        groups = [outputs[index * 3 : index * 3 + 3] for index in indices]
        points = [node for group in groups for node in group]
        link_emitter = DagEmitter("", emitter.dag, emitter.dof, points, emitter.options, groups)
        bound = geometry.bounds.get(link)
        link_cached = cached.copy()
        source.append("    {\n")
        if bound is not None:
            bound_emitter = DagEmitter("", emitter.dag, emitter.dof, bound.center, emitter.options)
            source.append(bound_emitter.body("q", lambda _index, _node: "", cached))
            link_cached.update(
                step.node for step in bound_emitter.schedule.steps if isinstance(step, Evaluate)
            )
            source.append(link_bound_definition(geometry, link, "enclosing"))
            condition = (
                "streaming_may_intersect(box, enclosing) || may_intersect(ground, enclosing)"
                if streaming
                else "tenkai_f32::may_intersect(scene, environment, enclosing)"
            )
            source.append(f"    if ({condition}) {{\n")

        def store(index: int, value: int) -> str:
            axis = "xyz"[index % 3]
            text = f"    center.{axis} = v{value};\n"
            if index % 3 != 2:
                return text

            sphere = members[index // 3]
            if streaming:
                radius = number(sphere.radius)
                text += f"    collision |= streaming_intersects(box, center, {radius}f, retry);\n"
                text += f"    collision |= intersects(ground, center, {radius}f);\n"
            else:
                text += sphere_collision(sphere.radius)

            end = index // 3 + 1
            if end % 8 == 0 or end == len(members):
                if streaming:
                    if bound is None:
                        text += streaming_exit
                else:
                    text += "    if (collision) return true;\n"
            return text

        source.append(link_emitter.body("q", store, link_cached))
        if bound is not None:
            source.append("    }\n")
            if streaming:
                # Skipped lanes must rejoin before a vote using the caller's mask.
                source.append(streaming_exit)
        source.append("    }\n")
    source.append(self_collision_body(emitter, geometry, cached, streaming=streaming))
    return "".join(source)


def point_expression(nodes: Sequence[int]) -> str:
    return "{" + ", ".join(f"v{node}" for node in nodes) + "}"


def link_bound_definition(geometry: CollisionDag, link: int, name: str) -> str:
    bound = geometry.bounds[link]
    pose = geometry.poses[link]
    arguments = [point_expression(bound.center), point_expression(pose.translation)]
    arguments.extend(f"v{node}" for node in pose.rotation)
    arguments.extend((number(bound.sphere.radius) + "f", number(bound.local_scale) + "f"))
    return f"    const auto {name} = tenkai_f32::link_sphere(" + ", ".join(arguments) + ");\n"


def self_collision_body(
    emitter: DagEmitter,
    geometry: CollisionDag,
    cached: set[int],
    *,
    streaming: bool,
) -> str:
    pairs = geometry.rules.self_pairs
    if not pairs:
        return ""

    links = dict.fromkeys(link for pair in pairs for link in pair)
    members = {
        link: [i for i, sphere in enumerate(geometry.spheres) if sphere.link_index == link]
        for link in links
    }
    source: list[str] = []
    shared_outputs: list[int] = []
    for link in links:
        pose = geometry.poses[link]
        shared_outputs.extend((*pose.translation, *pose.rotation))
        bound = geometry.bounds.get(link)
        shared_outputs.extend(bound.center if bound else geometry.centers[members[link][0]])

    shared = DagEmitter("", emitter.dag, emitter.dof, shared_outputs, emitter.options)
    source.append(shared.body("q", lambda _index, _node: "", cached))
    cached = cached | {step.node for step in shared.schedule.steps if isinstance(step, Evaluate)}
    for link in links:
        name = f"self_bound_{link}"
        if link in geometry.bounds:
            source.append(link_bound_definition(geometry, link, name))
        else:
            index = members[link][0]
            center = point_expression(geometry.centers[index])
            radius = number(geometry.spheres[index].radius)
            source.append(f"    const tenkai_f32::Sphere {name} = {{{center}, {radius}f}};\n")

    # Workspace certificates and moving-only filters cover external obstacles.
    # Every self pair remains active, including pairs with a fixed link.
    for a, b in pairs:
        source.append("    {\n")
        source.append(f"    if (tenkai_f32::may_intersect(self_bound_{a}, self_bound_{b})) {{\n")
        indices = members[a] + members[b]
        outputs = [node for i in indices for node in geometry.centers[i]]
        narrow = DagEmitter("", emitter.dag, emitter.dof, outputs, emitter.options)
        source.append(narrow.body("q", lambda _index, _node: "", cached))

        for i in members[a]:
            center = point_expression(geometry.centers[i])
            radius = number(geometry.spheres[i].radius)
            source.append("    {\n")
            source.append(f"    const tenkai_f32::Sphere self_sphere = {{{center}, {radius}f}};\n")
            for j in members[b]:
                center = point_expression(geometry.centers[j])
                radius = number(geometry.spheres[j].radius)
                source.append(
                    f"    collision |= tenkai_f32::intersects(self_sphere, {center}, {radius}f);\n"
                )
            if not streaming:
                source.append("    if (collision) return true;\n")
            source.append("    }\n")
        source.append("    }\n")
        source.append("    }\n")
        if streaming:
            # All lanes rejoin after their broad-phase branch before voting.
            source.append(
                "    if (__any_sync(mask, collision)) return StreamingCollision::Blocked;\n"
            )
    return "".join(source)


def streaming_replacements(
    emitter: DagEmitter, spheres: Sequence[Sphere], geometry: CollisionDag
) -> list[tuple[str, str | int]]:
    finite_squared_radii = True
    for sphere in spheres:
        radius = f32(sphere.radius)
        squared_radius = f32(radius * radius)
        if not math.isfinite(squared_radius):
            finite_squared_radii = False
            break

    if not finite_squared_radii:
        streamable_body = "    return false;\n"
    else:
        streamable_body = (
            "    const auto environment = scene.environments[environment_id];\n"
            "    if (environment.boxes.count != 1 || environment.grounds.count != 1 ||\n"
            "        environment.spheres.count || environment.cylinders.count) return false;\n"
            "    const auto& box = scene.boxes[environment.boxes.offset];\n"
            "    for (unsigned i = 0; i < 9; ++i)\n"
            "        if (box.pose.world_to_local[i] != (i % 4 == 0 ? 1.0f : 0.0f)) return false;\n"
            "    return true;\n"
        )

    body = link_collision_body(emitter, spheres, geometry, streaming=True)
    return [("STREAMABLE_BODY", streamable_body), ("STREAMING_BODY", body)]


def planning_replacements(
    emitter: DagEmitter,
    spheres: Sequence[Sphere],
    groups: Sequence[Sequence[int]],
    moving: Sequence[int],
    controls: Sequence[int],
    geometry: CollisionDag,
) -> list[tuple[str, str | int]]:
    options = emitter.options
    dag = emitter.dag
    dependencies_by_control = [dag.depends_on_inputs([index]) for index in controls]

    bounded_indices: list[int] = []
    for index in moving:
        dependency_count = 0
        for flags in dependencies_by_control:
            if any(flags[node] for node in groups[index]):
                dependency_count += 1
        if dependency_count <= 2:
            bounded_indices.append(index)

    remaining_indices = [index for index in moving if index not in bounded_indices]
    edge_spheres = [spheres[index] for index in remaining_indices]
    edge_groups = [groups[index] for index in remaining_indices]
    edge_outputs = [node for group in edge_groups for node in group]
    edge_emitter = DagEmitter(
        "tenkai_collision_edge",
        dag,
        emitter.dof,
        edge_outputs,
        options,
        edge_groups,
    )
    edge_function = collision_function(edge_emitter, edge_spheres, geometry, "__forceinline__")

    bounds_checks: list[str] = []
    for dimension, flags in enumerate(dependencies_by_control):
        affects_bounds = any(flags[node] for index in bounded_indices for node in groups[index])
        if affects_bounds:
            bounds_checks.append(
                f"    if (!(q[{dimension}] >= lower[{dimension}] && "
                f"q[{dimension}] <= upper[{dimension}])) return false;\n"
            )

    body: list[str] = []
    bounded_outputs = [node for index in bounded_indices for node in groups[index]]
    needed = dag.required_nodes(bounded_outputs)
    for index, node in enumerate(dag.nodes):
        if not needed[index]:
            continue

        match node:
            case Constant(value):
                constant = number(value)
                expression = "{" + constant + "f, " + constant + "f}"
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
        body.append(f"    const WorkspaceInterval v{index} = {expression};\n")

    for index in bounded_indices:
        x, y, z = groups[index]
        radius = number(spheres[index].radius)
        body.append(
            f"    if (!workspace_clear(v{x}, v{y}, v{z}, {radius}f, "
            "scene, environment)) return false;\n"
        )

    return [
        ("EDGE_FUNCTION", edge_function),
        *streaming_replacements(edge_emitter, edge_spheres, geometry),
        ("BOUNDS_CHECKS", "".join(bounds_checks)),
        ("WORKSPACE_BODY", "".join(body)),
    ]


def prepare_collision(
    kin: KinematicModel, spheres: Sequence[Sphere], options: Options, rules: SelfCollisionRules
) -> tuple[DagEmitter, CollisionDag]:
    for sphere in spheres:
        sphere.validate(kin)

    geometry = collision_expressions(kin, spheres, options.optimization, rules)
    outputs = [node for center in geometry.centers for node in center]
    emitter = DagEmitter(
        "tenkai_collision_one",
        geometry.dag,
        len(kin.joints),
        outputs,
        options,
        geometry.centers,
    )
    return emitter, geometry


def generate_collision_function(
    kin: KinematicModel,
    spheres: Sequence[Sphere],
    options: Options = Options(),
    *,
    rules: SelfCollisionRules = SelfCollisionRules(),
) -> str:
    emitter, geometry = prepare_collision(kin, spheres, options, rules)
    source = template("collision/sdf.cuh")
    # Standalone collision source embeds its dependencies.
    source += template("collision/broad.cuh").replace('#include "sdf.cuh"\n', "")
    return source + collision_function(emitter, spheres, geometry, "inline")


def collision_kernel(kin: KinematicModel, spheres: Sequence[Sphere]) -> str:
    configuration = "configurations"
    if kin.joints:
        configuration = f"configurations + batch * {len(kin.joints)}"

    collision = "false"
    if spheres:
        collision = f"tenkai_collision_one({configuration}, scene, environment_ids[batch])"

    return template(
        "collision/batch_kernel.cuh",
        (("COLLISION", collision),),
    )


def generate_collision_kernel(
    kin: KinematicModel,
    spheres: Sequence[Sphere],
    options: Options = Options(),
    *,
    rules: SelfCollisionRules = SelfCollisionRules(),
) -> str:
    source = generate_collision_function(kin, spheres, options, rules=rules)
    return source + collision_kernel(kin, spheres)


def generate_rrt_sources(
    kin: KinematicModel,
    spheres: Sequence[Sphere],
    controls: Sequence[int],
    options: Options = Options(),
    *,
    rules: SelfCollisionRules = SelfCollisionRules(),
) -> dict[str, str]:
    """Return relative paths and source text for a model-specific native planner."""
    if not controls:
        raise ValueError("RRT requires at least one controlled joint")
    if any(not 0 <= i < len(kin.joints) for i in controls):
        raise ValueError("RRT control index exceeds FK joint count")
    if len(set(controls)) != len(controls):
        raise ValueError("duplicate RRT control index")

    emitter, geometry = prepare_collision(kin, spheres, options, rules)
    groups = geometry.centers
    dependent = emitter.dag.depends_on_inputs(controls)
    moving = [index for index, group in enumerate(groups) if any(dependent[node] for node in group)]
    functions = collision_function(emitter, spheres, geometry, "__noinline__")
    if len(moving) == len(spheres):
        functions += collision_definition(
            "tenkai_collision_moving",
            "inline",
            "    return tenkai_collision_one(q, scene, environment_id);\n",
        )
    else:
        moving_groups = [groups[i] for i in moving]
        moving_outputs = [node for group in moving_groups for node in group]
        moving_emitter = DagEmitter(
            "tenkai_collision_moving",
            emitter.dag,
            emitter.dof,
            moving_outputs,
            options,
            moving_groups,
        )
        functions += collision_function(
            moving_emitter, [spheres[i] for i in moving], geometry, "__noinline__"
        )

    replacements: list[tuple[str, str | int]] = [
        ("DIMENSION", len(controls)),
        ("FK_DOF", len(kin.joints)),
        ("CONTROLS", ", ".join(map(str, controls))),
        ("COLLISION_FUNCTIONS", functions),
        ("COLLISION_KERNEL", collision_kernel(kin, spheres)),
    ]
    replacements.extend(planning_replacements(emitter, spheres, groups, moving, controls, geometry))

    files: dict[str, str] = {}
    for path in sorted((TEMPLATES / "rrt").iterdir()):
        name = path.relative_to(TEMPLATES).as_posix()
        files[name] = template(name, replacements)
    for name in ("sdf", "broad", "streaming", "workspace"):
        output_name = f"collision/{name}.cuh"
        files[output_name] = template(output_name, replacements)
    return files
