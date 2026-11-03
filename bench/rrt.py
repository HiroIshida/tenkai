#!/usr/bin/env python3
"""Build and run batched CUDA RRT / RRT-Connect on Fetch with ground and a table."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import time

import numpy as np

from tenkai import Options, Planner, RRTConfig, build_rrt

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
START = [0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]
GOAL = [0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def build(output, math_mode, arch, max_registers=None, live_budget=None):
    compiled = build_rrt(
        ROOT / "assets/fetch/fetch.urdf",
        ROOT / "assets/fetch/collision.yaml",
        output / "build",
        options=Options(math_mode, live_budget=live_budget),
        arch=arch,
        max_registers=max_registers,
    )
    return compiled.library, compiled.model, compiled.toolchain


def interpolate(path, resolution):
    points = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(b - a) / resolution))))
        ratios = np.arange(1, count + 1, dtype=np.float64)[:, None] / count
        edge = a + ratios * (b - a)
        edge[-1] = b
        points.append(edge)
    return np.concatenate(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["rrt", "rrtc"], default="rrt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--math", choices=["f32", "f32-sfu", "f64"], default="f32")
    parser.add_argument("--arch", default="compute_75")
    parser.add_argument("--max-registers", type=int,
                        help="NVCC register limit per thread (32..255; 0 uses compiler default); "
                             "default: 128 for f32, compiler default for f32-sfu/f64")
    parser.add_argument("--live-budget", type=int,
                        help="enable liveness scheduling with this nonnegative scalar-value budget")
    parser.add_argument("--problems", type=int, default=2048,
                        help="planning problems per kernel launch (default: %(default)s)")
    parser.add_argument("--environments", type=int, default=1)
    parser.add_argument("--warps-per-problem", type=int, default=4)
    parser.add_argument("--samples-per-round", type=int, default=32)
    parser.add_argument("--max-nodes", type=int, default=8192)
    parser.add_argument("--max-rounds", type=int, default=1024)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--goal-bias", type=float, default=0.1)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--validate-plainmp", action="store_true",
                        help="also validate paths with the installed plainmp Python API")
    args = parser.parse_args()
    if any(not 0 < getattr(args, name) < 2**32 for name in [
            "problems", "environments", "samples_per_round", "max_nodes", "max_rounds", "repetitions"]):
        parser.error("counts must fit positive uint32 values")
    if not 1 <= args.warps_per_problem <= 32 or not 0 <= args.seed < 2**64 - args.problems:
        parser.error("invalid warp count or seed")
    if not np.isfinite(args.resolution) or args.resolution <= 0:
        parser.error("resolution must be positive and finite")
    if re.fullmatch(r"compute_[0-9]+", args.arch) is None:
        parser.error("arch must be a virtual architecture such as compute_75")
    if args.max_registers not in (None, 0) and not 32 <= args.max_registers <= 255:
        parser.error("max-registers must be 0 or between 32 and 255")
    if args.live_budget is not None and args.live_budget < 0:
        parser.error("live-budget must be nonnegative")
    output = (args.output or BENCH / "results" / args.algorithm).resolve()
    library, model, tool_info = build(output, args.math, args.arch, args.max_registers, args.live_budget)
    config = RRTConfig(args.max_nodes, args.max_rounds, args.samples_per_round, args.step_size, args.goal_bias)
    geometry = np.tile([0.95, 0, 0.8, 1, 2, 0.05, 0], (args.environments, 1))
    geometry[:, 1] = np.arange(args.environments) * 0.02
    ids = np.arange(args.problems, dtype=np.uint32) % args.environments
    seeds = np.arange(args.problems, dtype=np.uint64) + args.seed
    starts, goals = [np.tile(q, (args.problems, 1)) for q in [START, GOAL]]
    dtype = np.float64 if args.math == "f64" else np.float32
    controls = [joint["index"] for joint in model["control_joints"]]
    times = []
    with Planner(library, model, config, args.warps_per_problem, args.problems, geometry,
                 args.resolution, args.algorithm) as planner:
        previous = planner.solve(starts, goals, ids, seeds)  # Includes JIT/warmup; excluded from timings.
        for _ in range(args.repetitions):
            begin = time.perf_counter()
            results = planner.solve(starts, goals, ids, seeds)
            times.append((time.perf_counter() - begin) * 1000)
            if results != previous:
                raise RuntimeError("repeated solve changed deterministic results")
        paths, validation_points, validation_ids = [], [], []
        for problem_id, result in enumerate(results):
            path = planner.path(problem_id)
            paths.append(path.tolist())
            if result["status"] != "success":
                continue
            if not (np.array_equal(path[0], starts[problem_id].astype(dtype).astype(float))
                    and np.array_equal(path[-1], goals[problem_id].astype(dtype).astype(float))):
                raise RuntimeError("path endpoints differ from the input endpoints")
            lower = np.array([j["lower"] for j in model["control_joints"]], dtype=dtype)
            upper = np.array([j["upper"] for j in model["control_joints"]], dtype=dtype)
            if not np.isfinite(path).all() or (path < lower).any() or (path > upper).any():
                raise RuntimeError("path violates joint bounds")
            if args.algorithm == "rrtc":
                lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
                tolerance = 8 * np.finfo(dtype).eps * max(1, np.abs(path).max())
                if (lengths == 0).any() or (lengths > config.step_size + tolerance).any():
                    raise RuntimeError("RRT-Connect edge violates step size or repeats a configuration")
            # Keep the planning grid and an additional finer grid; their points differ.
            points = np.concatenate([interpolate(path, float(dtype(args.resolution))),
                                     interpolate(path, float(dtype(args.resolution)) / 5)])
            full = np.zeros((len(points), planner.fk_dof), dtype=np.float64)
            full[:, controls] = points
            validation_points.append(full)
            validation_ids.append(np.full(len(points), ids[problem_id], dtype=np.uint32))
        all_points = np.concatenate(validation_points) if validation_points else np.empty((0, planner.fk_dof))
        all_ids = np.concatenate(validation_ids) if validation_ids else np.empty(0, dtype=np.uint32)
        collisions = int(planner.collisions(all_points, all_ids).sum())
    success = sum(r["status"] == "success" for r in results)
    algorithm = {"rrt": "single-tree batched RRT", "rrtc": "batched RRT-Connect"}[args.algorithm]
    summary = {"algorithm": algorithm, "config": asdict(config),
               "max_registers": tool_info["max_registers"],
               "warps_per_problem": args.warps_per_problem, "model": model, "toolchain": tool_info,
               "problem_count": args.problems, "environment_count": args.environments,
               "environment_ids": ids.tolist(), "geometry": geometry.tolist(), "seeds": seeds.tolist(),
               "resolution": args.resolution, "success_count": success, "results": results,
               "solve_wall_ms": times, "median_solve_wall_ms": float(np.median(times)),
               "timing": "native solve including input/result copies; excludes setup, JIT, path extraction and validation",
               "validation_points": len(all_points), "collision_count": collisions,
               "self_collision": False, "uncontrolled_joint_positions": 0}
    write_json(output / "paths.json", {"joint_names": [j["name"] for j in model["control_joints"]], "paths": paths})
    write_json(output / "summary.json", summary)
    validation_failed = False
    if args.validate_plainmp:
        from reference import package_info, validate_points

        names = [joint["name"] for joint in model["control_joints"]]
        validation = validate_points(all_points[:, controls], all_ids, geometry, names)
        summary["plainmp_validation"] = {**validation, "package": package_info()}
        validation_failed = bool(validation["collisions"])
        write_json(output / "summary.json", summary)
    print(f"Solved {success}/{args.problems}; batch median {np.median(times):.3f} ms; "
          f"checked {len(all_points)} path points, {collisions} collisions.")
    print(f"Results: {output}")
    if args.validate_plainmp:
        print(f"Independent plainmp validation: {summary['plainmp_validation']['collisions']} collisions.")
    if success != args.problems or collisions or validation_failed:
        raise RuntimeError("planning or path validation failed; see summary.json")


if __name__ == "__main__":
    main()
