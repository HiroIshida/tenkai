"""Compile a robot model into CUDA source files."""

import json
import math
import time
from pathlib import Path
from typing import Literal, TypedDict

from .cuda import (
    Options,
    generate_centers_kernel,
    generate_collision_function,
    generate_collision_kernel,
    generate_fk_kernel,
    generate_rrt_sources,
)
from .dag import Optimization
from .modeling import KinematicModel, read_config


class ControlJoint(TypedDict):
    name: str
    index: int
    lower: float
    upper: float


class ModelMetadata(TypedDict):
    input_joint_names: list[str]
    control_joints: list[ControlJoint]
    sphere_count: int
    scalar: str
    math: Literal["f32-sfu"]
    optimization: Optimization
    live_budget: int | None
    self_collision_pair_count: int
    external_sphere_count: int


def control_joints(kin: KinematicModel, names: list[str]) -> list[ControlJoint]:
    bounds: list[ControlJoint] = []
    joint_indices = {joint.name: index for index, joint in enumerate(kin.joints)}
    for name in names:
        if name not in joint_indices:
            raise ValueError("control is not a movable joint: " + name)

        index = joint_indices[name]
        joint = kin.joints[index]
        if joint.kind == "continuous":
            lower, upper = -math.tau, math.tau
        else:
            lower, upper = joint.lower, joint.upper
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("invalid sampling bounds for " + name)

        bounds.append(ControlJoint(name=name, index=index, lower=lower, upper=upper))

    return bounds


def generate(
    urdf_path: Path,
    yaml_path: Path,
    output: Path,
    options: Options = Options(),
    all_kernels: bool = False,
) -> dict[str, float]:
    total = time.perf_counter()
    start = total
    timings: dict[str, float] = {}

    kin = KinematicModel.from_urdf(urdf_path.read_text())
    timings["kinematics"] = time.perf_counter() - start

    start = time.perf_counter()
    source = yaml_path.read_text()
    config = read_config(source)
    spheres = config.resolve(kin)
    rules = config.resolve_rules(kin, spheres)
    bounds = control_joints(kin, config.control_joint_names)
    controls = [joint["index"] for joint in bounds]
    timings["spheres_and_controls"] = time.perf_counter() - start

    files: dict[str, str] = {}
    start = time.perf_counter()
    files["collision.cuh"] = generate_collision_kernel(kin, spheres, options, rules=rules)
    timings["collision_codegen"] = time.perf_counter() - start

    start = time.perf_counter()
    if controls:
        files.update(generate_rrt_sources(kin, spheres, controls, options, rules=rules))
    timings["rrt_codegen"] = time.perf_counter() - start

    if all_kernels:
        start = time.perf_counter()
        files["fk.cuh"] = generate_fk_kernel(kin, len(kin.poses) - 1, options)
        files["centers.cuh"] = generate_centers_kernel(kin, spheres, options)
        files["collision_function.cuh"] = generate_collision_function(
            kin, spheres, options, rules=rules
        )
        timings["auxiliary_codegen"] = time.perf_counter() - start

    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    model = ModelMetadata(
        input_joint_names=kin.joint_names,
        control_joints=bounds,
        sphere_count=len(spheres),
        scalar="float",
        math="f32-sfu",
        optimization=options.optimization,
        live_budget=options.live_budget,
        self_collision_pair_count=len(rules.self_pairs),
        external_sphere_count=sum(s.link_index not in rules.self_only_links for s in spheres),
    )
    (output / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True, allow_nan=False))

    timings["write_outputs"] = time.perf_counter() - start
    timings["total"] = time.perf_counter() - total
    return timings
