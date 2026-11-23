#!/usr/bin/env python3
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import tyro
from plainmp_baseline import make_cpu_solve, make_problems
from problems import (
    CONFIG,
    PROBLEM_SETS,
    RESOLUTION,
    Batch,
    download_problems,
    load_problems,
    sha256,
)

from tenkai import RRTBuild, build_rrt, sdf
from tenkai.build import Backend, target_arch


def main(
    problem_set: str = "fetch",
    problems: Path | None = None,
    urdf: Path | None = None,
    spheres: Path | None = None,
    multiplier: int | None = None,
    batch_size: int = 8192,
    repeats: int = 7,
    cpu_ms_per_problem: float | None = None,
    backend: Backend = "cuda",
    arch: str | None = None,
    output: Path = Path("artifacts/rrtc-bench"),
) -> None:
    """Compare plainmp and tenkai RRT-Connect on a set of different problems.

    Args:
        problem_set: Registered problem set; its YAML is downloaded from a GitHub release with gh.
        problems: Use this YAML instead of the registered set's download.
        urdf: Robot model; defaults to the registered set's.
        spheres: Collision spheres; defaults to the registered set's.
        multiplier: Repeat the entire problem set this many times; defaults to the YAML value.
        batch_size: Maximum problems per GPU call; larger workloads use multiple calls.
        cpu_ms_per_problem: Skip CPU solves and extrapolate from this measured time per problem.
    """
    if problem_set not in PROBLEM_SETS:
        print(f"problem-set must be one of {sorted(PROBLEM_SETS)}", file=sys.stderr)
        raise SystemExit(2)
    if repeats < 1 or not 0 < batch_size < 2**32:
        print("repeats must be positive and batch-size must fit positive uint32", file=sys.stderr)
        raise SystemExit(2)
    if cpu_ms_per_problem is not None and (
        not math.isfinite(cpu_ms_per_problem) or cpu_ms_per_problem <= 0
    ):
        print("cpu-ms-per-problem must be finite and positive", file=sys.stderr)
        raise SystemExit(2)

    spec = PROBLEM_SETS[problem_set]
    urdf = urdf or spec.urdf
    spheres = spheres or spec.spheres
    problems = problems or download_problems(spec)
    arch = target_arch(backend, arch)

    compiled = build_rrt(urdf, spheres, output / "build", backend=backend, arch=arch)
    cases = load_problems(problems, compiled.model, multiplier)
    inputs = cases.batch()
    batch = len(inputs)
    print(
        f"{len(cases.goals)} different problems x {cases.multiplier} = {batch} solves/batch",
        flush=True,
    )

    cpu_times: list[float] = []
    if cpu_ms_per_problem is None:
        solve_cpu = make_cpu_solve(make_problems(urdf, spheres, compiled.model, cases), inputs)
        print("Measuring plainmp...", flush=True)
        cpu_times = measure(solve_cpu, batch, repeats)
        cpu_ms = statistics.median(cpu_times)
        cpu_timing = "CPU includes seed reset, native planner creation, and trajectory conversion. "
    else:
        cpu_ms = cpu_ms_per_problem * batch
        cpu_timing = (
            "CPU time is linearly extrapolated from the supplied milliseconds per problem. "
        )

    chunks = [slice(begin, min(begin + batch_size, batch)) for begin in range(0, batch, batch_size)]
    print("Measuring tenkai...", flush=True)
    gpu_times = measure_gpu(compiled, cases.environments, inputs, chunks, repeats)
    gpu_ms = statistics.median(gpu_times)

    report = {
        "batch": batch,
        "problem_count": len(cases.goals),
        "multiplier": cases.multiplier,
        "gpu_batch_size": min(batch_size, batch),
        "gpu_batches": len(chunks),
        "problems": str(problems.resolve()),
        "problems_sha256": sha256(problems),
        "math": compiled.model["math"],
        "external_sphere_count": compiled.model["external_sphere_count"],
        "self_collision_pair_count": compiled.model["self_collision_pair_count"],
        "backend": backend,
        "arch": arch,
        "plainmp_version": version("plainmp"),
        "config": asdict(CONFIG),
        "resolution": RESOLUTION,
        "plainmp_ms": cpu_times,
        "plainmp_ms_per_problem": cpu_ms_per_problem,
        "plainmp_estimated_ms": cpu_ms if cpu_ms_per_problem is not None else None,
        "tenkai_ms": gpu_times,
        "speedup": cpu_ms / gpu_ms,
        "timing": "Python API wall time per batch; one warmup batch, builds/setup excluded. "
        + cpu_timing
        + "GPU reuses buffers and includes transfers. Both planners replay fixed per-problem seeds.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    cpu_label = "plainmp (extrapolated)" if cpu_ms_per_problem is not None else "plainmp"
    for name, milliseconds in ((cpu_label, cpu_ms), ("tenkai", gpu_ms)):
        print(f"{name}: {milliseconds:.3f} ms/batch, {batch * 1000 / milliseconds:.1f} solves/s")
    print(f"tenkai speedup: {cpu_ms / gpu_ms:.2f}x")


def measure_gpu(
    compiled: RRTBuild,
    environments: list[list[sdf.CollisionObject]],
    inputs: Batch,
    chunks: list[slice],
    repeats: int,
) -> list[float]:
    with ExitStack() as stack:
        planners = {
            size: stack.enter_context(
                compiled.planner(
                    environments, config=CONFIG, problem_count=size, resolution=RESOLUTION
                )
            )
            for size in {chunk.stop - chunk.start for chunk in chunks}
        }

        def solve_gpu() -> int:
            successes = 0
            for chunk in chunks:
                part = inputs[chunk]
                results = planners[len(part)].solve(part.starts, part.goals, part.ids, part.seeds)
                successes += sum(result["status"] == "success" for result in results)
            return successes

        return measure(solve_gpu, len(inputs), repeats)


def measure(solve: Callable[[], int], batch: int, repeats: int) -> list[float]:
    successes = solve()
    if successes != batch:
        raise RuntimeError(f"warmup solved {successes}/{batch} problems")
    times: list[float] = []
    for _ in range(repeats):
        begin = time.perf_counter()
        successes = solve()
        times.append((time.perf_counter() - begin) * 1000)
        if successes != batch:
            raise RuntimeError(f"solved {successes}/{batch} problems")
    return times


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.DisallowNone,))
