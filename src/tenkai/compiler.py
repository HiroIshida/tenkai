"""Compile a robot model into CUDA source files."""

import json
import math
import time
from pathlib import Path
from typing import TypedDict

from .collision import read_config
from .cuda import (
    MathMode,
    Options,
    generate_centers_kernel,
    generate_collision_function,
    generate_collision_kernel,
    generate_fk_kernel,
    generate_rrt_kernel,
)
from .dag import Optimization
from .model import FkDag, Robot


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
    math: MathMode
    optimization: Optimization
    live_budget: int | None


def control_joints(robot: Robot, fk: FkDag, names: list[str]) -> list[ControlJoint]:
    bounds: list[ControlJoint] = []
    joints = {joint.name: joint for joint in robot.joints}
    for name in names:
        if name not in fk.joint_names:
            raise ValueError("control is not a movable joint: " + name)

        index = fk.joint_names.index(name)
        joint = joints[name]
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

    robot = Robot.from_string(urdf_path.read_text())
    source = yaml_path.read_text()
    timings["parse_urdf"] = time.perf_counter() - start

    start = time.perf_counter()
    fk = FkDag.from_robot(robot)
    timings["fk_dag"] = time.perf_counter() - start

    start = time.perf_counter()
    config = read_config(source)
    spheres = config.resolve(fk)
    bounds = control_joints(robot, fk, config.control_joint_names)
    controls = [joint["index"] for joint in bounds]
    timings["spheres_and_controls"] = time.perf_counter() - start

    files: dict[str, str] = {}
    start = time.perf_counter()
    files["collision.cuh"] = generate_collision_kernel(fk, spheres, options)
    timings["collision_codegen"] = time.perf_counter() - start

    start = time.perf_counter()
    if controls:
        files["rrt.cuh"] = generate_rrt_kernel(fk, spheres, controls, options)
    timings["rrt_codegen"] = time.perf_counter() - start

    if all_kernels:
        start = time.perf_counter()
        files["fk.cuh"] = generate_fk_kernel(fk, len(fk.poses) - 1, options)
        files["centers.cuh"] = generate_centers_kernel(fk, spheres, options)
        files["collision_function.cuh"] = generate_collision_function(fk, spheres, options)
        timings["auxiliary_codegen"] = time.perf_counter() - start

    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        (output / name).write_text(contents)

    model = ModelMetadata(
        input_joint_names=fk.joint_names,
        control_joints=bounds,
        sphere_count=len(spheres),
        scalar=options.scalar,
        math=options.math,
        optimization=options.optimization,
        live_budget=options.live_budget,
    )
    (output / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True, allow_nan=False))

    timings["write_outputs"] = time.perf_counter() - start
    timings["total"] = time.perf_counter() - total
    return timings
