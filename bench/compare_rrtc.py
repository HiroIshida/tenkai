#!/usr/bin/env python3
"""Compare plainmp and rkern RRT-Connect on the same Fetch problem through Python."""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from plainmp.constraint import SphereAttachmentSpec, SphereCollisionCst
from plainmp.kinematics import BaseType, KinematicModel
from plainmp.ompl_solver import (
    OMPLSolver,
    OMPLSolverConfig,
    PlainmpRRTCSettings,
    set_log_level_none,
    set_random_seed,
)
from plainmp.problem import Problem
from plainmp.psdf import BoxSDF, GroundSDF, Pose, UnionSDF

from rkern import Options, RRTConfig, build_rrt
from rkern.collision import read_config
from rkern.compiler import ModelMetadata

ASSETS = Path(__file__).resolve().parents[1] / "assets/fetch"
START = [0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]
GOAL = [0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01]
GEOMETRY = [0.95, 0.0, 0.8, 1.0, 2.0, 0.05, 0.0]
RESOLUTION = 0.005


def make_problem(urdf: Path, spheres: Path, model: ModelMetadata) -> Problem:
    config = read_config(spheres.read_text())
    attachments: list[SphereAttachmentSpec] = []
    for link, definitions in config.groups.items():
        values = np.asarray(definitions, dtype=np.float64).reshape(-1, 4)
        centers = np.asfortranarray(values[:, :3].T)
        radii = np.ascontiguousarray(values[:, 3])
        # FetchSpec's default excludes some spheres from environment checks.
        attachments.append(SphereAttachmentSpec(link, centers, radii, False))
    names = [joint["name"] for joint in model["control_joints"]]
    checker = SphereCollisionCst(
        KinematicModel(urdf.read_text()), names, BaseType.FIXED, attachments, [], None, True
    )
    box = BoxSDF(np.array(GEOMETRY[3:6]), Pose(np.array(GEOMETRY[:3]), np.eye(3)))
    checker.set_sdf(UnionSDF([box, GroundSDF(0.0)]))
    lower = np.array([joint["lower"] for joint in model["control_joints"]])
    upper = np.array([joint["upper"] for joint in model["control_joints"]])
    return Problem(
        np.array(START),
        lower,
        upper,
        np.array(GOAL),
        checker,
        None,
        np.full(len(names), RESOLUTION),
    )


def measure(solve: Callable[[], int], batch: int, repeats: int) -> list[float]:
    if solve() != batch:
        raise RuntimeError("warmup did not solve every problem")
    times: list[float] = []
    for _ in range(repeats):
        begin = time.perf_counter()
        successes = solve()
        times.append((time.perf_counter() - begin) * 1000)
        if successes != batch:
            raise RuntimeError(f"solved {successes}/{batch} problems")
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ASSETS / "fetch.urdf")
    parser.add_argument("--spheres", type=Path, default=ASSETS / "collision.yaml")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--math", choices=["f32", "f32-sfu", "f64"], default="f32")
    parser.add_argument("--arch", default="compute_75")
    parser.add_argument("--output", type=Path, default=Path("artifacts/rrtc-bench"))
    args = parser.parse_args()
    if not 0 < args.batch < 2**32 or args.repeats < 1:
        parser.error("batch must fit positive uint32 and repeats must be positive")
    compiled = build_rrt(
        args.urdf, args.spheres, args.output / "build", options=Options(args.math), arch=args.arch
    )
    config = RRTConfig(max_rounds=8192, samples_per_round=1, goal_bias=0.0)
    problem = make_problem(args.urdf, args.spheres, compiled.model)
    settings = PlainmpRRTCSettings()
    settings.max_samples = config.max_nodes
    solver = OMPLSolver(
        OMPLSolverConfig(
            n_max_call=10000000, algorithm_range=config.step_size, plainmp_rrtc_settings=settings
        )
    )
    set_log_level_none()
    set_random_seed(42)

    def solve_cpu() -> int:
        successes = 0
        for _ in range(args.batch):
            successes += solver.solve(problem).success
        return successes

    print("Measuring plainmp...", flush=True)
    cpu_times = measure(solve_cpu, args.batch, args.repeats)
    starts = np.tile(START, (args.batch, 1))
    goals = np.tile(GOAL, (args.batch, 1))
    ids = np.zeros(args.batch, dtype=np.uint32)
    seeds = np.arange(args.batch, dtype=np.uint64) + 42
    with compiled.planner(
        [GEOMETRY],
        config=config,
        warps=1,
        problem_count=args.batch,
        resolution=RESOLUTION,
        algorithm="rrtc",
    ) as planner:

        def solve_gpu() -> int:
            results = planner.solve(starts, goals, ids, seeds)
            return sum(result["status"] == "success" for result in results)

        print("Measuring rkern...", flush=True)
        gpu_times = measure(solve_gpu, args.batch, args.repeats)
    cpu_ms = statistics.median(cpu_times)
    gpu_ms = statistics.median(gpu_times)
    report = {
        "batch": args.batch,
        "math": args.math,
        "arch": args.arch,
        "plainmp_version": version("plainmp"),
        "config": asdict(config),
        "warps": 1,
        "resolution": RESOLUTION,
        "plainmp_ms": cpu_times,
        "rkern_ms": gpu_times,
        "speedup": cpu_ms / gpu_ms,
        "timing": "Python API wall time per batch; one warmup batch, builds/setup excluded. "
        "CPU includes OMPLSolver construction of each native planner and trajectory conversion; "
        "GPU reuses buffers and includes transfers. CPU uses fresh trials; GPU replays fixed seeds.",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, milliseconds in (("plainmp", cpu_ms), ("rkern", gpu_ms)):
        print(
            f"{name}: {milliseconds:.3f} ms/batch, {args.batch * 1000 / milliseconds:.1f} solves/s"
        )
    print(f"rkern speedup: {cpu_ms / gpu_ms:.2f}x")


if __name__ == "__main__":
    main()
