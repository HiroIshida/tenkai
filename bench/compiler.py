#!/usr/bin/env python3
"""Measure Python code generation and nvcc compilation separately."""

import argparse
import cProfile
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

from common import codegen_command, compiler_sources, copy_rrt_bindings, toolchain

from rkern import Options
from rkern.compiler import generate

ROOT = Path(__file__).resolve().parents[1]


def timed_run(command: list[str], log: TextIO | None = None) -> tuple[float, str]:
    start = time.perf_counter()
    result = subprocess.run(
        command, cwd=ROOT, stdout=log or subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"command failed: {command}\n{result.stdout or 'see compiler log'}")
    return elapsed, result.stdout or ""


def summary(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "minimum": min(values), "maximum": max(values)}


def profile_python(output: Path, options: Options) -> dict[str, float]:
    profiler = cProfile.Profile()
    profiler.runcall(
        generate,
        ROOT / "assets/fetch/fetch.urdf",
        ROOT / "assets/fetch/collision.yaml",
        output,
        options,
    )
    profiler.dump_stats(str(output / "codegen.prof"))
    entries = profiler.getstats()
    result = {"total": sum(entry.inlinetime for entry in entries)}
    selected = {
        ("dag.py", "optimize"): "cse",
        ("scheduling.py", "schedule"): "scheduling",
        ("cuda.py", "body"): "emit_dag",
    }
    for entry in entries:
        code = entry.code
        if isinstance(code, str):
            continue
        key = selected.get((Path(code.co_filename).name, code.co_name))
        if key is not None:
            result[key] = result.get(key, 0.0) + entry.totaltime
    return result


def compiler_commands(
    directory: Path, options: Options, nvcc: str, compiler: str, arch: str
) -> dict[str, list[str]]:
    # Match the existing collision benchmark's default compile-time workload.
    config = f"using Scalar = {options.scalar};\n#ifdef __CUDACC__\nnamespace sdf = {options.namespace};\n#endif\n"
    model = json.loads((directory / "model.json").read_text())
    config += f"constexpr size_t kPool = 1048576, kDof = {len(model['input_joint_names'])}, kMaxBatch = 1048576;\n"
    config += "constexpr size_t kRounds = 7, kQueriesPerRound = 1048576, kMaxRepetitions = 1024;\n"
    config += "constexpr int kBlock = 128;\nconstexpr double kWarmupSeconds = 2.0;\n"
    config += "constexpr size_t kBatches[] = {1, 16, 64, 256, 1024, 4096, 8192, 16384, 65536, 262144, 1048576};\n"
    (directory / "config.h").write_text(config)
    common = [
        nvcc,
        "-ccbin",
        compiler,
        "-std=c++14",
        "-O3",
        "-arch=" + arch,
        "-code=" + arch,
        "-I",
        str(directory),
    ]
    collision = common + [
        str(ROOT / "bench/cuda/collision.cu"),
        "-o",
        str(directory / "collision_bench"),
    ]
    rrt = common + [
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-DRKERN_MODEL_NAMESPACE=" + options.namespace,
        str(copy_rrt_bindings(directory)),
        "-o",
        str(directory / "librrt.so"),
    ]
    if options.math == "f32":
        rrt.append("--maxrregcount=128")
    return {"collision": collision, "rrt": rrt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/compiler-bench")
    parser.add_argument(
        "--modes", nargs="+", choices=("f32", "f32-sfu", "f64"), default=["f32", "f32-sfu", "f64"]
    )
    parser.add_argument(
        "--optimization", choices=("none", "basic", "algebraic"), default="algebraic"
    )
    parser.add_argument("--live-budget", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nvcc-repeats", type=int, default=3)
    parser.add_argument("--arch", default="compute_75")
    args = parser.parse_args()
    if args.repeats < 1 or args.nvcc_repeats < 1:
        parser.error("repeat counts must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nvcc, compiler, nvcc_version = toolchain()
    modes: dict[str, object] = {}
    report = {
        "arguments": {**vars(args), "output": str(output)},
        "python": sys.version,
        "platform": platform.platform(),
        "nvcc": nvcc_version,
        "host_compiler": subprocess.check_output([compiler, "--version"], text=True),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in compiler_sources()
        },
        "methodology": {
            "codegen": "one warmup; fresh Python processes; no generated-code cache",
            "nvcc": "sequential builds; first sample retained; normal OS filesystem cache",
            "profile": "separate instrumented cProfile run, excluded from benchmark samples",
            "excluded": "GPU execution and driver PTX JIT",
        },
        "modes": modes,
    }
    for mode in args.modes:
        options = Options(mode, args.optimization, args.live_budget)
        directory = output / mode
        command = codegen_command(directory, options)
        timed_run(command)
        samples = []
        for _ in range(args.repeats):
            elapsed, stdout = timed_run(command)
            sample: dict[str, float] = json.loads(stdout)
            sample["process_seconds"] = elapsed
            samples.append(sample)
        stages = {stage: summary([sample[stage] for sample in samples]) for stage in samples[0]}
        compile_commands = compiler_commands(directory, options, nvcc, compiler, args.arch)
        compile_samples: dict[str, list[float]] = {name: [] for name in compile_commands}
        for repetition in range(args.nvcc_repeats):
            for name, compile_command in compile_commands.items():
                print(f"{mode}: nvcc {name}, {repetition + 1}/{args.nvcc_repeats}", flush=True)
                with (directory / f"nvcc-{name}-{repetition}.log").open("w") as log:
                    elapsed, _ = timed_run(compile_command, log)
                compile_samples[name].append(elapsed)
        compile_times = {name: summary(values) for name, values in compile_samples.items()}
        modes[mode] = {
            "codegen_command": command,
            "codegen_samples": samples,
            "codegen_seconds": stages,
            "nvcc_commands": compile_commands,
            "nvcc_samples": compile_samples,
            "nvcc_seconds": compile_times,
            "profile_seconds": profile_python(directory, options),
        }
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            f"{mode}: codegen {stages['total']['median']:.4f}s, "
            f"nvcc collision {compile_times['collision']['median']:.3f}s, "
            f"nvcc RRT {compile_times['rrt']['median']:.3f}s",
            flush=True,
        )
    print(f"Results: {output / 'results.json'}")


if __name__ == "__main__":
    main()
