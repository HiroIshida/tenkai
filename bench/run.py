#!/usr/bin/env python3
"""Generate, compile, validate, and benchmark Fetch environment collision checks."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
PLAINMP = BENCH / "plainmp"
URDF = ROOT / "assets/fetch/fetch.urdf"
BATCHES = [1, 16, 64, 256, 1024, 4096, 8192, 16384, 65536, 262144, 1048576]
THROUGHPUT_PLOTS = {"throughput.png": "log", "throughput_linear.png": "linear"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BENCH / "results")
    parser.add_argument("--math", choices=["f32-sfu", "f32", "f64"], default="f32-sfu")
    parser.add_argument("--live-budget", type=int, help="enable liveness scheduling with this budget")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--pool", type=int, default=2**20)
    parser.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--queries-per-round", type=int, default=2**20)
    parser.add_argument("--max-repetitions", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--cpu-core", type=int, help="logical CPU for plainmp (default: first allowed CPU)")
    parser.add_argument("--arch", default="compute_75", help="PTX target supported by both nvcc and the GPU")
    args = parser.parse_args()
    for name in ["pool", "rounds", "queries_per_round", "block_size"]:
        if getattr(args, name) <= 0:
            parser.error(name.replace("_", "-") + " must be positive")
    if args.seed < 0 or args.max_repetitions < 2:
        parser.error("seed must be nonnegative and max-repetitions must be at least 2")
    if args.live_budget is not None and args.live_budget <= 0:
        parser.error("live-budget must be positive")
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds <= 0:
        parser.error("warmup-seconds must be finite and positive")
    if any(batch <= 0 or batch > args.pool for batch in args.batches):
        parser.error("every batch must be between 1 and pool")
    if len(set(args.batches)) != len(args.batches):
        parser.error("batches must be distinct")
    if re.fullmatch(r"compute_[0-9]+", args.arch) is None:
        parser.error("arch must be a virtual architecture such as compute_75")
    args.output = args.output.resolve()
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def toolchain():
    nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
    if nvcc is None:
        raise RuntimeError("nvcc not found; install the CUDA toolkit or set NVCC")
    version = subprocess.check_output([nvcc, "--version"], text=True)
    release = re.search(r"release (\d+)\.", version)
    compiler = os.environ.get("CXX")
    # CUDA 10 rejects the newer default GCC shipped on this benchmark's original host.
    if compiler is None:
        compiler = "g++-8" if release and int(release[1]) == 10 else "c++"
    compiler = shutil.which(compiler)
    if compiler is None:
        raise RuntimeError("CUDA host compiler not found; set CXX to a compiler supported by nvcc")
    return nvcc, compiler, version


def plainmp_revision():
    if not (PLAINMP / ".git").exists():
        subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--", "bench/plainmp"],
                       cwd=ROOT, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PLAINMP, text=True).strip()
    entry = subprocess.check_output(["git", "ls-files", "--stage", "--", "bench/plainmp"], cwd=ROOT, text=True).split()
    if len(entry) < 2 or entry[0] != "160000" or entry[1] != revision:
        raise RuntimeError("plainmp HEAD differs from the pinned submodule; run git submodule update --init bench/plainmp")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PLAINMP, text=True)
    if status:
        raise RuntimeError("plainmp has tracked changes; the benchmark requires an unmodified submodule")
    return revision


def prepare_plainmp(args, revision):
    available = sorted(os.sched_getaffinity(0))
    core = available[0] if args.cpu_core is None else args.cpu_core
    if core not in available:
        raise ValueError(f"CPU {core} is not in this process's allowed affinity: {available}")
    compiler = shutil.which(os.environ.get("CPU_CXX", "c++"))
    if compiler is None:
        raise RuntimeError("plainmp compiler not found; set CPU_CXX to a C++17 compiler")
    build = args.output / "build/plainmp"
    commands = [
        ["cmake", "-S", str(BENCH), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CXX_COMPILER=" + compiler, "-DWORKLOAD_DIR=" + str(args.output / "build")],
        ["cmake", "--build", str(build), "--parallel", str(min(4, len(available)))],
    ]
    print(f"Building plainmp {revision[:7]} (CPU {core}, one thread, double precision).", flush=True)
    with (args.output / "build/plainmp.log").open("w") as log:
        try:
            for command in commands:
                subprocess.run(command, stdout=log, stderr=log, check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("plainmp build failed:\n" + (args.output / "build/plainmp.log").read_text()) from None
    return {
        "revision": revision, "threads": 1, "pinned_logical_cpu": core,
        "cpu_info": subprocess.check_output(["lscpu"], text=True),
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True),
        "build_commands": commands,
        "command": ["taskset", "-c", str(core), str(build / "plainmp_bench"), str(ROOT), str(args.output)],
        "timed_call": "SphereCollisionCst::is_valid(const Eigen::VectorXd&)",
        "precision": "double", "trigonometry": "native polynomial approximation (high_accuracy=false)",
        "optimizations_preserved": ["AABB", "broad/narrow phase", "lazy FK cache", "early exit", "sphere reordering"],
        "build": "Release -O3 -DNDEBUG -march=native LTO; EIGEN_NO_DEBUG EIGEN_DONT_VECTORIZE",
        "math_header": "preinclude math.h to preserve floating-point abs overloads normally supplied by Python.h",
        "excluded": ["Python", "setup", "file IO", "float-to-double conversion", "Eigen allocation", "validation"],
    }


def prepare(args):
    output = args.output
    build = output / "build"
    build.mkdir(parents=True, exist_ok=True)
    for filename in ["summary.csv", "summary.json", "validation.json", *THROUGHPUT_PLOTS]:
        (output / filename).unlink(missing_ok=True)
    nvcc, compiler, nvcc_version = toolchain()
    codegen_options = {
        "f32-sfu": ["--float", "--sfu"],
        "f32": ["--float", "--no-sfu"],
        "f64": ["--double"],
    }[args.math]
    if args.live_budget is not None:
        codegen_options += ["--live-budget", str(args.live_budget)]
    command = ["cargo", "run", "--locked", "--quiet", "--release", "--example", "bench_generate",
               "--", *codegen_options, str(build)]
    subprocess.run(command, cwd=ROOT, check=True)
    model = json.loads((build / "model.json").read_text())
    dtype = np.float32 if model["scalar"] == "float" else np.float64
    controls = model["control_joints"]
    start = time.perf_counter()
    configurations = np.zeros((args.pool, len(model["input_joint_names"])), dtype=dtype)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    configurations[:, [joint["index"] for joint in controls]] = rng.uniform(
        [joint["lower"] for joint in controls], [joint["upper"] for joint in controls],
        (args.pool, len(controls)),
    ).astype(dtype)
    sampling_seconds = time.perf_counter() - start
    configurations.tofile(output / "configurations.bin")
    scene = np.array([0.95, 0, 0.8, 1, 2, 0.05, 0], dtype=dtype)
    scene.tofile(output / "scene.bin")

    namespace = "rkern_f32" if dtype == np.float32 else "rkern_f64"
    config = f"using Scalar = {model['scalar']};\n#ifdef __CUDACC__\nnamespace sdf = {namespace};\n#endif\n"
    constants = {
        "kPool": args.pool, "kDof": configurations.shape[1], "kMaxBatch": max(args.batches),
        "kRounds": args.rounds, "kQueriesPerRound": args.queries_per_round,
        "kMaxRepetitions": args.max_repetitions,
    }
    config += "".join(f"constexpr size_t {name} = {value};\n" for name, value in constants.items())
    config += f"constexpr int kBlock = {args.block_size};\n"
    config += f"constexpr double kWarmupSeconds = {args.warmup_seconds};\n"
    config += "constexpr size_t kBatches[] = {" + ",".join(map(str, args.batches)) + "};\n"
    config_path = build / "config.h"
    if not config_path.exists() or config_path.read_text() != config:
        config_path.write_text(config)
    compile_command = [nvcc, "-ccbin", compiler, "-std=c++14", "-O3", "-arch=" + args.arch,
                       "-code=" + args.arch, "-I", str(build), str(BENCH / "collision.cu"),
                       "-o", str(build / "collision_bench")]
    metadata = {
        "arguments": {**vars(args), "output": str(output)},
        "model": model, "rng": "NumPy PCG64", "numpy_version": np.__version__,
        "python_version": sys.version, "byte_order": sys.byteorder,
        "scene": {"box_center": scene[:3].tolist(), "box_extents": scene[3:6].tolist(),
                  "ground_height": float(scene[6])},
        "self_collision": False, "uncontrolled_joint_positions": 0,
        "sampling_seconds_excluded": sampling_seconds,
        "input_sha256": sha256(output / "configurations.bin"),
        "scene_sha256": sha256(output / "scene.bin"),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [
            URDF, ROOT / "assets/fetch/collision.yaml", BENCH / "generate.rs",
            BENCH / "collision.cu", BENCH / "run.py",
            BENCH / "plainmp_bench.cpp", BENCH / "CMakeLists.txt", BENCH / "workload.hpp",
        ]},
        "generated_kernel_sha256": sha256(build / "collision.cuh"),
        "codegen_command": command, "compile_command": compile_command,
        "nvcc_version": nvcc_version,
        "host_compiler_version": subprocess.check_output([compiler, "--version"], text=True),
        "timing": {
            "kernel": "CUDA events around repeated launches; configurations and environment on device",
            "transfer": "wall time / repetitions; pinned H2D configurations + kernel + D2H flags, one stream",
            "plainmp": "C++ wall time / repetitions; loop over original is_valid calls and byte flag stores",
            "environment_and_ids": "uploaded once, excluded from both timings",
            "repetitions": "max(2, min(max_repetitions, ceil(queries_per_round / batch)))",
            "job_order": "GPU: batch order, kernel then transfer; CPU: batch order; each rotated by round * 5 jobs",
            "input_offset": "((round * 104729 + iteration) % (pool / batch)) * batch, integer division",
            "excluded": ["sampling", "codegen", "compilation/JIT", "allocation", "warmup", "validation", "file IO"],
        },
    }
    write_json(output / "metadata.json", metadata)
    print(f"Compiling {model['math']} collision kernel; {model['sphere_count']} spheres, "
          f"{args.pool:,} configurations (sampling {sampling_seconds:.3f} s, excluded).", flush=True)
    with (build / "compile.log").open("w") as log:
        try:
            subprocess.run(compile_command, stdout=log, stderr=log, check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("CUDA compilation failed:\n" + (build / "compile.log").read_text()) from None
    return metadata


def validate(args):
    gpu = np.fromfile(args.output / "collision.u8", dtype=np.uint8)
    cpu = np.fromfile(args.output / "plainmp_collision.u8", dtype=np.uint8)
    for name, values in [("GPU", gpu), ("plainmp", cpu)]:
        if len(values) != args.pool or np.any(values > 1):
            raise RuntimeError(f"{name} output has the wrong length or contains values other than 0/1")
    different = np.flatnonzero(cpu != gpu)
    agreement = 1 - len(different) / args.pool
    required_agreement = 0.999
    cpu_only = int(np.count_nonzero(cpu[different]))
    gpu_only = len(different) - cpu_only
    result = {
        "samples": args.pool, "mismatches": len(different),
        "agreement": agreement, "required_agreement": required_agreement,
        "passed": agreement >= required_agreement,
        "plainmp_collision_rkern_free": cpu_only,
        "plainmp_free_rkern_collision": gpu_only,
        "rkern": {
            "collision_count": int(gpu.sum()), "collision_fraction": float(gpu.mean()),
            "output_sha256": sha256(args.output / "collision.u8"),
        },
        "plainmp": {
            "collision_count": int(cpu.sum()), "collision_fraction": float(cpu.mean()),
            "output_sha256": sha256(args.output / "plainmp_collision.u8"),
        },
    }
    with (args.output / "plainmp_differences.csv").open("w", newline="") as dest:
        writer = csv.writer(dest)
        writer.writerow(["index", "plainmp_collision", "rkern_collision"])
        writer.writerows((i, cpu[i], gpu[i]) for i in different)
    write_json(args.output / "validation.json", result)
    print(f"plainmp/GPU agreement: {agreement:.6%} (required {required_agreement:.1%}); "
          f"{len(different):,}/{args.pool:,} differing flags.")
    print(f"plainmp collision / rkern free: {cpu_only:,}; "
          f"plainmp free / rkern collision: {gpu_only:,} (details in validation.json).")
    if not result["passed"]:
        raise RuntimeError(f"plainmp/GPU agreement below {required_agreement:.1%}; "
                           f"see {args.output / 'validation.json'}")


def plot_throughput(args, summaries, device):
    figure = Figure(figsize=(9, 5.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    for mode, label in [
        ("plainmp", "plainmp (CPU, 1 thread, FP64)"),
        ("kernel", "rkern (GPU kernel)"),
        ("transfer", "rkern (H2D + kernel + D2H)"),
    ]:
        rows = sorted((row for row in summaries if row["mode"] == mode), key=lambda row: row["batch"])
        batches = [row["batch"] for row in rows]
        line, = axes.plot(batches, [row["median_mqueries_per_s"] for row in rows],
                          marker="o", markersize=4, label=label)
        axes.fill_between(batches, [row["min_mqueries_per_s"] for row in rows],
                          [row["max_mqueries_per_s"] for row in rows],
                          color=line.get_color(), alpha=0.15)
        last = rows[-1]["median_mqueries_per_s"]
        axes.annotate(f"{last:.2f}", (batches[-1], last), xytext=(8, 0),
                      textcoords="offset points", va="center", color=line.get_color(),
                      fontsize=10, fontweight="bold")
    axes.set_xscale("log")
    axes.margins(x=0.1)
    axes.set_xlabel("Configurations per batch")
    axes.set_ylabel("Throughput (million queries/s)")
    axes.grid(which="major", alpha=0.3)
    axes.legend(loc="best", title="Lines: median; shading: min-max")
    figure.suptitle("Fetch collision throughput")
    axes.set_title(f"{device['name']} | GPU {args.math} | {args.rounds} rounds", fontsize=10)
    for filename, scale in THROUGHPUT_PLOTS.items():
        axes.set_yscale(scale)
        if scale == "linear":
            axes.set_ylim(bottom=0)
        figure.savefig(args.output / filename, dpi=180)


def summarize(args):
    with (args.output / "raw.csv").open() as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 3 * len(args.batches) * args.rounds:
        raise RuntimeError("incomplete timing results")
    summaries = []
    for mode in ["plainmp", "kernel", "transfer"]:
        for batch in args.batches:
            samples = [row for row in rows if row["mode"] == mode and int(row["batch"]) == batch]
            if sorted(int(row["round"]) for row in samples) != list(range(args.rounds)):
                raise RuntimeError("missing or duplicate timing rounds")
            repetitions = max(2, min(args.max_repetitions, math.ceil(args.queries_per_round / batch)))
            if any(int(row["repetitions"]) != repetitions for row in samples):
                raise RuntimeError("incorrect repetition counts")
            field = "gpu_ms" if mode == "kernel" else "wall_ms"
            ms = np.array([float(row[field]) for row in samples])
            if not np.all(np.isfinite(ms) & (ms > 0)):
                raise RuntimeError("invalid timing sample")
            rates = batch / ms / 1000
            summaries.append({
                "mode": mode, "batch": batch, "rounds": len(samples),
                "repetitions": int(samples[0]["repetitions"]),
                "median_batch_us": float(np.median(ms) * 1000),
                "median_amortized_us_per_query": float(np.median(ms) * 1000 / batch),
                "median_mqueries_per_s": float(np.median(rates)),
                "min_mqueries_per_s": float(rates.min()), "max_mqueries_per_s": float(rates.max()),
            })
    cpu_times = {row["batch"]: row["median_batch_us"] for row in summaries if row["mode"] == "plainmp"}
    for row in summaries:
        row["speedup_vs_plainmp"] = cpu_times[row["batch"]] / row["median_batch_us"]
    device = json.loads((args.output / "device.json").read_text())
    metadata = json.loads((args.output / "metadata.json").read_text())
    write_json(args.output / "summary.json", {"device": device, "plainmp": metadata["plainmp"], "measurements": summaries})
    with (args.output / "summary.csv").open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    plot_throughput(args, summaries, device)
    print(f"\n{device['name']} | {args.math} | block {args.block_size} | {args.rounds} rounds")
    print(f"plainmp: FP64, one thread, CPU {metadata['plainmp']['pinned_logical_cpu']}, native trig approximation")
    print("mode          batch   batch us      us/query       Mquery/s       [min, max]       vs CPU")
    for row in summaries:
        print("{mode:9s} {batch:9,d} {median_batch_us:10.3f} {median_amortized_us_per_query:13.6f} "
              "{median_mqueries_per_s:12.3f} [{min_mqueries_per_s:8.3f}, {max_mqueries_per_s:8.3f}] "
              "{speedup_vs_plainmp:9.3f}x".format(**row))
    print("\nMedian timings; us/query is amortized across the batch.")
    print("kernel: GPU-resident; transfer: H2D + kernel + D2H, no overlap.")
    print(f"Plots: {', '.join(THROUGHPUT_PLOTS)}")
    print(f"Results: {args.output}")


def main():
    args = arguments()
    revision = plainmp_revision()
    metadata = prepare(args)
    metadata["plainmp"] = prepare_plainmp(args, revision)
    write_json(args.output / "metadata.json", metadata)
    for name, command in [
        ("gpu", [str(args.output / "build/collision_bench"), str(args.output)]),
        ("plainmp", metadata["plainmp"]["command"]),
    ]:
        with (args.output / f"{name}_raw.csv").open("w") as raw:
            subprocess.run(command, stdout=raw, check=True)
    if plainmp_revision() != revision:
        raise RuntimeError("plainmp revision changed during the benchmark")
    metadata["plainmp"]["source_unchanged"] = True
    write_json(args.output / "metadata.json", metadata)
    with (args.output / "raw.csv").open("w") as raw:
        writer = None
        for name in ["gpu", "plainmp"]:
            with (args.output / f"{name}_raw.csv").open() as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(raw, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)
    validate(args)
    summarize(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
