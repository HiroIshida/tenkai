#!/usr/bin/env python3
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import tyro
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

from tenkai import RRTConfig, build_rrt, sdf
from tenkai.build import Backend, target_arch
from tenkai.collision import read_config
from tenkai.compiler import ModelMetadata

ASSETS = Path(__file__).resolve().parents[1] / "assets/fetch"
START = [0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]
GOAL = [0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01]
BOX = sdf.Box(center=(0.95, 0.0, 0.8), size=(1.0, 2.0, 0.05))
GROUND = sdf.Ground(z=0.0)
GEOMETRY: list[sdf.CollisionObject] = [BOX, GROUND]
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
    box = BoxSDF(np.array(BOX.size), Pose(np.array(BOX.center), np.array(BOX.rotation)))
    checker.set_sdf(UnionSDF([box, GroundSDF(-GROUND.z)]))
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


def main(
    urdf: Path = ASSETS / "fetch.urdf",
    spheres: Path = ASSETS / "collision.yaml",
    batch: int = 256,
    repeats: int = 7,
    cpu_ms_per_problem: float | None = None,
    backend: Backend = "cuda",
    arch: str | None = None,
    output: Path = Path("artifacts/rrtc-bench"),
) -> None:
    """Compare plainmp and tenkai RRT-Connect on the same Fetch problem through Python.

    Args:
        cpu_ms_per_problem: Skip CPU solves and extrapolate from this measured time per problem.
    """
    if not 0 < batch < 2**32 or repeats < 1:
        print("batch must fit positive uint32 and repeats must be positive", file=sys.stderr)
        raise SystemExit(2)
    if cpu_ms_per_problem is not None and (
        not math.isfinite(cpu_ms_per_problem) or cpu_ms_per_problem <= 0
    ):
        print("cpu-ms-per-problem must be finite and positive", file=sys.stderr)
        raise SystemExit(2)

    arch = target_arch(backend, arch)

    compiled = build_rrt(
        urdf,
        spheres,
        output / "build",
        backend=backend,
        arch=arch,
    )
    config = RRTConfig(max_rounds=8192, goal_bias=0.0)
    cpu_times: list[float] = []
    if cpu_ms_per_problem is None:
        problem = make_problem(urdf, spheres, compiled.model)
        settings = PlainmpRRTCSettings()
        settings.max_samples = config.max_nodes
        solver = OMPLSolver(
            OMPLSolverConfig(
                n_max_call=10000000,
                algorithm_range=config.step_size,
                plainmp_rrtc_settings=settings,
            )
        )
        set_log_level_none()
        set_random_seed(42)

        def solve_cpu() -> int:
            successes = 0
            for _ in range(batch):
                successes += solver.solve(problem).success
            return successes

        print("Measuring plainmp...", flush=True)
        cpu_times = measure(solve_cpu, batch, repeats)
        cpu_ms = statistics.median(cpu_times)
        cpu_timing = (
            "CPU includes native planner creation and trajectory conversion; uses fresh trials. "
        )
    else:
        cpu_ms = cpu_ms_per_problem * batch
        cpu_timing = (
            "CPU time is linearly extrapolated from the supplied milliseconds per problem. "
        )
    starts = np.tile(START, (batch, 1))
    goals = np.tile(GOAL, (batch, 1))
    ids = np.zeros(batch, dtype=np.uint32)
    seeds = np.arange(batch, dtype=np.uint64) + 42
    with compiled.planner(
        GEOMETRY,
        config=config,
        problem_count=batch,
        resolution=RESOLUTION,
    ) as planner:

        def solve_gpu() -> int:
            results = planner.solve(starts, goals, ids, seeds)
            return sum(result["status"] == "success" for result in results)

        print("Measuring tenkai...", flush=True)
        gpu_times = measure(solve_gpu, batch, repeats)
    gpu_ms = statistics.median(gpu_times)
    report = {
        "batch": batch,
        "math": compiled.model["math"],
        "backend": backend,
        "arch": arch,
        "plainmp_version": version("plainmp"),
        "config": asdict(config),
        "resolution": RESOLUTION,
        "plainmp_ms": cpu_times,
        "plainmp_ms_per_problem": cpu_ms_per_problem,
        "plainmp_estimated_ms": cpu_ms if cpu_ms_per_problem is not None else None,
        "tenkai_ms": gpu_times,
        "speedup": cpu_ms / gpu_ms,
        "timing": "Python API wall time per batch; one warmup batch, builds/setup excluded. "
        + cpu_timing
        + "GPU reuses buffers, includes transfers, and replays fixed seeds.",
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    cpu_label = "plainmp (extrapolated)" if cpu_ms_per_problem is not None else "plainmp"
    for name, milliseconds in ((cpu_label, cpu_ms), ("tenkai", gpu_ms)):
        print(f"{name}: {milliseconds:.3f} ms/batch, {batch * 1000 / milliseconds:.1f} solves/s")
    print(f"tenkai speedup: {cpu_ms / gpu_ms:.2f}x")


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.DisallowNone,))
