#!/usr/bin/env python3
"""Compare the plainmp and tenkai Python planning APIs on the same Fetch scene."""

import argparse
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
from common import sha256
from plainmp.ompl_solver import (
    OMPLSolver,
    OMPLSolverConfig,
    PlainmpRRTCSettings,
    set_log_level_none,
    set_random_seed,
)
from plainmp.problem import Problem
from reference import make_checker, package_info, validate_points
from rrt import BENCH, GOAL, ROOT, START, Planner, RRTConfig, build, interpolate, write_json

from tenkai.compiler import ModelMetadata


def benchmark_plainmp(
    args: argparse.Namespace,
    model: ModelMetadata,
    geometry: npt.NDArray[np.float64],
    core: int,
) -> dict[str, object]:
    controls = model["control_joints"]
    checker = make_checker([joint["name"] for joint in controls], geometry[0])
    problem = Problem(
        np.asarray(START),
        np.array([joint["lower"] for joint in controls]),
        np.array([joint["upper"] for joint in controls]),
        np.asarray(GOAL),
        checker,
        None,
        np.full(len(controls), args.resolution),
    )
    settings = PlainmpRRTCSettings()
    settings.max_samples = args.max_nodes
    solver = OMPLSolver(
        OMPLSolverConfig(
            n_max_call=10000000,
            algorithm_range=1.0,
            plainmp_rrtc_settings=settings,
        )
    )
    set_log_level_none()
    set_random_seed(args.seed)
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {core})
    try:
        begin = time.perf_counter()
        warmups = 0
        while True:
            for _ in range(args.problems):
                solver.solve(problem)
            warmups += 1
            if time.perf_counter() - begin >= args.warmup_seconds:
                break
        times, successes = [], []
        for _ in range(args.repetitions):
            begin = time.perf_counter()
            solved = [solver.solve(problem) for _ in range(args.problems)]
            times.append((time.perf_counter() - begin) * 1000)
            successes.append(sum(result.success for result in solved))
        paths = [
            result.traj.numpy().tolist() if result.traj is not None else [] for result in solved
        ]
        results = [
            {
                "status": "success" if result.success else "limit",
                "calls": result.n_call,
                "path_length": len(path),
            }
            for result, path in zip(solved, paths)
        ]
        return {
            "wall_ms": times,
            "results": results,
            "paths": paths,
            "success_counts": successes,
            "warmup_pools": warmups,
        }
    finally:
        os.sched_setaffinity(0, allowed)


def validate(
    directory: Path,
    paths: list[list[list[float]]],
    model: ModelMetadata,
    planner: Planner,
    resolution: float,
    dtype: type[np.float32] | type[np.float64],
    step_size: float,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    controls = [joint["index"] for joint in model["control_joints"]]
    lower = np.array([joint["lower"] for joint in model["control_joints"]], dtype=dtype)
    upper = np.array([joint["upper"] for joint in model["control_joints"]], dtype=dtype)
    points, owners = [], []
    for i, path in enumerate(paths):
        path = np.asarray(path)
        if not len(path):
            continue
        if not (
            np.array_equal(path[0], np.asarray(START, dtype=dtype).astype(float))
            and np.array_equal(path[-1], np.asarray(GOAL, dtype=dtype).astype(float))
        ):
            raise RuntimeError("incorrect path endpoints")
        if not np.isfinite(path).all() or (path < lower).any() or (path > upper).any():
            raise RuntimeError("path violates bounds")
        tolerance = 8 * np.finfo(dtype).eps * max(1, np.abs(path).max())
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        if (lengths == 0).any() or (lengths > step_size + tolerance).any():
            raise RuntimeError("path has an invalid edge length")
        q = np.concatenate(
            [
                interpolate(path, float(dtype(resolution))),
                interpolate(path, float(dtype(resolution)) / 5),
            ]
        )
        points.append(q)
        owners.extend([i] * len(q))
    if not points:
        return {"points": 0, "gpu_collisions": 0, "plainmp_collisions": 0}
    points = np.concatenate(points)
    full = np.zeros((len(points), len(model["input_joint_names"])))
    full[:, controls] = points
    ids = np.zeros(len(points), dtype=np.uint32)
    flags = planner.collisions(full, ids)
    geometry = np.array([[0.95, 0, 0.8, 1, 2, 0.05, 0]], dtype=np.float64)
    names = [joint["name"] for joint in model["control_joints"]]
    cpu = validate_points(points, ids, geometry, names)
    return {
        "points": len(points),
        "gpu_collisions": int(flags.sum()),
        "gpu_colliding_path_ids": sorted(set(np.asarray(owners)[flags != 0].tolist())),
        "plainmp_collisions": cpu["collisions"],
    }


def stats(times: list[float], count: int, success: int | list[int]) -> dict[str, object]:
    elapsed = np.asarray(times)
    successes = np.full(len(times), success) if isinstance(success, int) else np.asarray(success)
    rates = successes * 1000 / elapsed
    return {
        "pool_wall_ms": times,
        "median_pool_ms": float(np.median(elapsed)),
        "attempts_per_second": float(count * 1000 / np.median(elapsed)),
        "solves_per_second": float(np.median(rates)),
        "min_solves_per_second": float(rates.min()),
        "max_solves_per_second": float(rates.max()),
        "success_count": int(successes.min()),
        "success_counts": successes.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BENCH / "results/rrtc-comparison")
    parser.add_argument("--problems", type=int, default=256)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128, 256])
    parser.add_argument("--samples-per-round", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmup-seconds", type=float, default=2)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--resolution", type=float, default=0.005)
    parser.add_argument("--math", choices=["f32", "f32-sfu", "f64"], default="f32")
    parser.add_argument("--arch", default="compute_75")
    parser.add_argument("--warps-per-problem", type=int, default=4)
    parser.add_argument("--max-registers", type=int)
    parser.add_argument("--max-nodes", type=int, default=8192)
    parser.add_argument("--max-rounds", type=int, default=8192)
    parser.add_argument("--cpu-core", type=int)
    args = parser.parse_args()
    if args.problems <= 0 or any(b <= 0 or args.problems % b for b in args.batches):
        parser.error("batches must be positive divisors of problems")
    if args.repetitions <= 0 or not np.isfinite(args.warmup_seconds) or args.warmup_seconds <= 0:
        parser.error("repetitions and warmup must be positive")
    if not np.isfinite(args.resolution) or args.resolution <= 0:
        parser.error("resolution must be finite and positive")
    if args.seed <= 0 or args.seed + args.problems >= 2**32:
        parser.error("seeds must fit positive uint32 values for OMPL")
    allowed = sorted(os.sched_getaffinity(0))
    core = allowed[0] if args.cpu_core is None else args.cpu_core
    if core not in allowed:
        parser.error("CPU core is not allowed")
    output = args.output.resolve()
    plainmp_package = package_info()
    print("Building the GPU RRTC library.", flush=True)
    library, model, tool_info = build(output, args.math, args.arch, args.max_registers)
    seeds = np.arange(args.problems, dtype=np.uint64) + args.seed
    geometry = np.array([[0.95, 0, 0.8, 1, 2, 0.05, 0]])
    workload = {
        "geometry": geometry[0].tolist(),
        "start": START,
        "goal": GOAL,
        "seeds": seeds.tolist(),
        "repetitions": args.repetitions,
        "warmup_seconds": args.warmup_seconds,
        "resolution": args.resolution,
        "max_nodes": args.max_nodes,
        "step_size": 1.0,
    }
    write_json(output / "workload.json", workload)
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "plainmp_package": plainmp_package,
        "cpu_core": core,
        "model": model,
        "gpu_toolchain": tool_info,
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "gpu": subprocess.check_output(["nvidia-smi"], text=True),
        "tenkai_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "source_hashes": {
            str(p): sha256(p)
            for p in [BENCH / "compare_rrtc.py", BENCH / "reference.py", output / "build/rrt.cuh"]
        },
        "timing": "Python API wall time per pool; CPU OMPLSolver.solve includes planner construction and trajectory conversion; GPU reuses buffers; compilation, warmup and independent validation excluded",
        "randomness": "CPU seeds OMPL once and runs fresh trials, including warmup; GPU repeats a fixed seed pool; CPU and GPU paths are not expected to match",
        "differences": "CPU FP64, GPU selected precision; independent RNG streams; native motion grids differ despite equal per-coordinate resolution; CPU 10M validity-call limit, GPU outer-round limit; both share node capacity; no smoothing",
    }
    write_json(output / "metadata.json", metadata)
    print(f"Measuring plainmp on CPU {core} (single thread).", flush=True)
    cpu = benchmark_plainmp(args, model, geometry, core)
    write_json(output / "plainmp.json", cpu)
    cpu_stats = stats(cpu["wall_ms"], args.problems, cpu["success_counts"])
    summary = {"plainmp": cpu_stats, "gpu": []}
    starts, goals = [np.tile(q, (args.problems, 1)) for q in [START, GOAL]]
    ids = np.zeros(args.problems, dtype=np.uint32)
    dtype = np.float64 if args.math == "f64" else np.float32
    for samples in args.samples_per_round:
        config = RRTConfig(args.max_nodes, args.max_rounds, samples, 1.0, 0.0)
        reference = None
        validation = None
        for batch in args.batches:
            print(f"Measuring GPU: samples/round={samples}, batch={batch}.", flush=True)
            with Planner(
                library,
                model,
                config,
                args.warps_per_problem,
                batch,
                geometry,
                args.resolution,
                "rrtc",
            ) as planner:

                def solve_pool(save_paths=False):
                    results, paths = [], []
                    for begin in range(0, args.problems, batch):
                        end = begin + batch
                        results.extend(
                            planner.solve(
                                starts[begin:end],
                                goals[begin:end],
                                ids[begin:end],
                                seeds[begin:end],
                            )
                        )
                        if save_paths:
                            paths.extend(planner.path(i).tolist() for i in range(batch))
                    return results, paths

                warmup_start = time.perf_counter()
                while True:
                    results, _ = solve_pool()
                    if time.perf_counter() - warmup_start >= args.warmup_seconds:
                        break
                if reference is None:
                    reference = results
                if results != reference:
                    raise RuntimeError("batch size changed planning results")
                times = []
                for _ in range(args.repetitions):
                    begin = time.perf_counter()
                    results, _ = solve_pool()
                    times.append((time.perf_counter() - begin) * 1000)
                    if results != reference:
                        raise RuntimeError("GPU replay changed results")
                if validation is None:
                    replay, paths = solve_pool(save_paths=True)
                    if replay != reference:
                        raise RuntimeError("path replay changed results")
                    directory = output / f"gpu-samples-{samples}"
                    directory.mkdir(exist_ok=True)
                    write_json(directory / "paths.json", paths)
                    write_json(directory / "results.json", results)
                    validation = validate(
                        directory, paths, model, planner, args.resolution, dtype, config.step_size
                    )
                    if "validation" not in cpu_stats:
                        cpu_stats["validation"] = validate(
                            output / "cpu-validation",
                            cpu["paths"],
                            model,
                            planner,
                            args.resolution,
                            np.float64,
                            config.step_size,
                        )
            row = {
                "batch": batch,
                "config": asdict(config),
                "validation": validation,
                **stats(times, args.problems, sum(r["status"] == "success" for r in results)),
            }
            row["speedup"] = (
                row["solves_per_second"] / cpu_stats["solves_per_second"]
                if cpu_stats["solves_per_second"]
                else None
            )
            summary["gpu"].append(row)
            write_json(output / "summary.json", summary)
            print(
                f"  {row['solves_per_second']:.1f} solves/s; {row['success_count']}/{args.problems} success; "
                f"validation={validation}",
                flush=True,
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(
        cpu_stats["solves_per_second"],
        color="black",
        linestyle="--",
        label="plainmp Python API, 1 CPU thread (FP64)",
    )
    ax.axhspan(
        cpu_stats["min_solves_per_second"],
        cpu_stats["max_solves_per_second"],
        color="black",
        alpha=0.08,
    )
    for samples in args.samples_per_round:
        rows = sorted(
            (r for r in summary["gpu"] if r["config"]["samples_per_round"] == samples),
            key=lambda r: r["batch"],
        )
        x = [r["batch"] for r in rows]
        (line,) = ax.plot(
            x,
            [r["solves_per_second"] for r in rows],
            "o-",
            label=f"tenkai {args.math}, {samples} sample(s)/round",
        )
        ax.fill_between(
            x,
            [r["min_solves_per_second"] for r in rows],
            [r["max_solves_per_second"] for r in rows],
            color=line.get_color(),
            alpha=0.15,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="GPU problems per batch",
        ylabel="Successful solves / second",
        title=f"Fetch RRT-Connect, {args.problems} fixed seeds, resolution {args.resolution}",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "throughput.png", dpi=160)
    print(f"plainmp: {cpu_stats['solves_per_second']:.1f} solves/s; results: {output}", flush=True)
    validations = [cpu_stats["validation"], *(row["validation"] for row in summary["gpu"])]
    if any(v["gpu_collisions"] or v["plainmp_collisions"] for v in validations):
        raise RuntimeError(
            "path validation found collisions; timings and counts retained in summary.json"
        )
    if any(row["success_count"] != args.problems for row in [cpu_stats, *summary["gpu"]]):
        raise RuntimeError("some planning problems failed; see summary.json")


if __name__ == "__main__":
    main()
