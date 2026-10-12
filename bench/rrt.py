#!/usr/bin/env python3
"""Build and run batched CUDA RRT / RRT-Connect on Fetch with ground and a table."""

import argparse
import ctypes as ct
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from common import toolchain

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
START = [0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]
GOAL = [0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01]
STATUSES = ["success", "round_limit", "node_limit", "invalid_start", "invalid_goal"]


@dataclass(frozen=True)
class RRTConfig:
    max_nodes: int = 8192
    max_rounds: int = 1024
    samples_per_round: int = 32
    step_size: float = 1.0
    goal_bias: float = 0.1


class NativeConfig(ct.Structure):
    _fields_ = [("max_nodes", ct.c_uint32), ("max_rounds", ct.c_uint32),
                ("samples_per_round", ct.c_uint32), ("step_size", ct.c_double),
                ("goal_bias", ct.c_double)]


class NativeResult(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ["status", "node_count", "rounds", "path_length"]]


def pointer(array, scalar=ct.c_double):
    return array.ctypes.data_as(ct.POINTER(scalar))


class Planner:
    """Benchmark binding; the native handle owns search storage and scene buffers."""

    def __init__(self, library, model, config, warps, problem_count, geometry, resolution, algorithm="rrt"):
        if algorithm not in ("rrt", "rrtc"):
            raise ValueError("algorithm must be rrt or rrtc")
        self.library = ct.CDLL(str(library))
        self.handle = ct.c_void_p()
        self.problem_count = problem_count
        self.dimension = len(model["control_joints"])
        self.fk_dof = len(model["input_joint_names"])
        self.max_nodes = config.max_nodes
        p_double, p_u32, p_u64 = ct.POINTER(ct.c_double), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint64)
        self.library.rrt_error.restype = ct.c_char_p
        create = getattr(self.library, algorithm + "_create")
        create.argtypes = [ct.POINTER(NativeConfig), ct.c_uint32, ct.c_uint32,
                           ct.c_uint32, p_double, p_double, p_double, p_double,
                           ct.POINTER(ct.c_void_p)]
        create.restype = ct.c_int
        self.library.rrt_destroy.argtypes = [ct.c_void_p]
        self.library.rrt_destroy.restype = None
        self.library.rrt_solve.argtypes = [ct.c_void_p, p_double, p_double, p_u32, p_u64, ct.POINTER(NativeResult)]
        self.library.rrt_solve.restype = ct.c_int
        self.library.rrt_path.argtypes = [ct.c_void_p, ct.c_uint32, p_double, ct.c_uint32, p_u32]
        self.library.rrt_path.restype = ct.c_int
        self.library.rrt_check.argtypes = [ct.c_void_p, p_double, p_u32, ct.c_uint32, ct.POINTER(ct.c_ubyte)]
        self.library.rrt_check.restype = ct.c_int
        lower = np.array([joint["lower"] for joint in model["control_joints"]], dtype=np.float64)
        upper = np.array([joint["upper"] for joint in model["control_joints"]], dtype=np.float64)
        widths = np.full(self.dimension, resolution, dtype=np.float64)
        geometry = np.ascontiguousarray(geometry, dtype=np.float64)
        if geometry.ndim != 2 or geometry.shape[1] != 7:
            raise ValueError("geometry must have shape (environments, 7)")
        for name in ["max_nodes", "max_rounds", "samples_per_round"]:
            if not 0 < getattr(config, name) < 2**32:
                raise ValueError(f"{name} must fit a positive uint32")
        if not 1 <= warps <= 32 or not 0 < problem_count < 2**32:
            raise ValueError("invalid warp or problem count")
        native = NativeConfig(**asdict(config))
        self.check(create(ct.byref(native), warps, problem_count, len(geometry),
                          pointer(geometry), pointer(lower), pointer(upper),
                          pointer(widths), ct.byref(self.handle)))

    def check(self, status):
        if status:
            raise RuntimeError(self.library.rrt_error().decode())

    def close(self):
        if self.handle.value:
            self.library.rrt_destroy(self.handle)
            self.handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def solve(self, starts, goals, environment_ids, seeds):
        starts, goals = [np.ascontiguousarray(q, dtype=np.float64) for q in [starts, goals]]
        ids = np.ascontiguousarray(environment_ids, dtype=np.uint32)
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        if (starts.shape != (self.problem_count, self.dimension) or goals.shape != starts.shape
                or ids.shape != (self.problem_count,) or seeds.shape != ids.shape):
            raise ValueError("invalid problem array shapes")
        results = (NativeResult * self.problem_count)()
        self.check(self.library.rrt_solve(self.handle, pointer(starts), pointer(goals),
                                        pointer(ids, ct.c_uint32), pointer(seeds, ct.c_uint64), results))
        return [{"status": STATUSES[r.status], "nodes": r.node_count, "rounds": r.rounds,
                 "path_length": r.path_length} for r in results]

    def path(self, problem_id):
        path = np.empty((self.max_nodes, self.dimension), dtype=np.float64)
        length = ct.c_uint32()
        self.check(self.library.rrt_path(self.handle, problem_id, pointer(path), self.max_nodes, ct.byref(length)))
        return path[:length.value].copy()

    def collisions(self, configurations, environment_ids):
        q = np.ascontiguousarray(configurations, dtype=np.float64)
        ids = np.ascontiguousarray(environment_ids, dtype=np.uint32)
        if q.ndim != 2 or q.shape[1] != self.fk_dof or ids.shape != (len(q),) or len(q) >= 2**32:
            raise ValueError("invalid collision array shapes")
        flags = np.empty(len(q), dtype=np.uint8)
        self.check(self.library.rrt_check(self.handle, pointer(q), pointer(ids, ct.c_uint32),
                                        len(q), pointer(flags, ct.c_ubyte)))
        return flags


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def build(output, math_mode, arch, max_registers=None, live_budget=None):
    if max_registers is None:
        max_registers = 128 if math_mode == "f32" else 0
    if max_registers != 0 and not 32 <= max_registers <= 255:
        raise ValueError("max_registers must be 0 or between 32 and 255")
    if live_budget is not None and live_budget < 0:
        raise ValueError("live_budget must be nonnegative")
    directory = output / "build"
    directory.mkdir(parents=True, exist_ok=True)
    nvcc, compiler, version = toolchain()
    options = {"f32": ["--float", "--no-sfu"], "f32-sfu": ["--float", "--sfu"], "f64": ["--double"]}[math_mode]
    if live_budget is not None:
        options += ["--live-budget", str(live_budget)]
    subprocess.run(["cargo", "run", "--quiet", "--locked", "--release", "--example", "bench_generate",
                    "--", *options, str(directory)], cwd=ROOT, check=True)
    model = json.loads((directory / "model.json").read_text())
    namespace = "rkern_f64" if model["scalar"] == "double" else "rkern_f32"
    library = directory / "librrt.so"
    command = [nvcc, "-ccbin", compiler, "-std=c++14", "-O3", "-arch=" + arch, "-code=" + arch,
               "-shared", "-Xcompiler", "-fPIC", "-DRKERN_MODEL_NAMESPACE=" + namespace,
               "-I", str(directory), str(BENCH / "rrt.cu"), "-o", str(library)]
    if max_registers:
        command.append("--maxrregcount=" + str(max_registers))
    with (directory / "compile.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=log)
    if completed.returncode:
        raise RuntimeError((directory / "compile.log").read_text())
    return library, model, {"nvcc": version, "compile_command": command,
                            "max_registers": max_registers or None}


def interpolate(path, resolution):
    points = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(b - a) / resolution))))
        ratios = np.arange(1, count + 1, dtype=np.float64)[:, None] / count
        edge = a + ratios * (b - a)
        edge[-1] = b
        points.append(edge)
    return np.concatenate(points)


def validate_plainmp(output):
    directory = output / "build/plainmp"
    commands = [
        ["cmake", "-S", str(BENCH), "-B", str(directory), "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CXX_COMPILER=" + os.environ.get("CPU_CXX", "c++"),
         "-DWORKLOAD_DIR=" + str(output / "build")],
        ["cmake", "--build", str(directory), "--target", "rrt_validate", "--parallel", "4"],
    ]
    with (output / "build/plainmp.log").open("w") as log:
        for command in commands:
            if subprocess.run(command, stdout=log, stderr=log).returncode:
                raise RuntimeError((output / "build/plainmp.log").read_text())
    result = subprocess.run([str(directory / "rrt_validate"), str(ROOT), str(output)],
                            capture_output=True, text=True)
    if not result.stdout.strip():
        raise RuntimeError("plainmp validation failed: " + result.stderr)
    validation = json.loads(result.stdout)
    validation["returncode"] = result.returncode
    validation["revision"] = subprocess.check_output(
        ["git", "-C", str(BENCH / "plainmp"), "rev-parse", "HEAD"], text=True).strip()
    return validation


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
    np.ascontiguousarray(all_points[:, controls], dtype=np.float64).tofile(output / "validation_q.bin")
    all_ids.tofile(output / "validation_env_ids.bin")
    geometry.tofile(output / "geometry.bin")
    write_json(output / "summary.json", summary)
    summary["plainmp_validation"] = validate_plainmp(output)
    write_json(output / "summary.json", summary)
    print(f"Solved {success}/{args.problems}; batch median {np.median(times):.3f} ms; "
          f"checked {len(all_points)} path points, {collisions} collisions.")
    print(f"Results: {output}")
    print(f"Independent plainmp validation: {summary['plainmp_validation']['collisions']} collisions.")
    if success != args.problems or collisions or summary["plainmp_validation"]["returncode"]:
        raise RuntimeError("planning or path validation failed; see summary.json")


if __name__ == "__main__":
    main()
