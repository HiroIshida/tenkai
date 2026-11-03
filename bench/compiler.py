#!/usr/bin/env python3
"""Measure code generation and nvcc time for the packaged RRT/RRT-Connect library."""

import argparse
import json
import statistics
from pathlib import Path

from rkern import Options, build_rrt

ASSETS = Path(__file__).resolve().parents[1] / "assets/fetch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ASSETS / "fetch.urdf")
    parser.add_argument("--spheres", type=Path, default=ASSETS / "collision.yaml")
    parser.add_argument("--math", choices=["f32", "f32-sfu", "f64"], default="f32")
    parser.add_argument("--arch", default="compute_75")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/compiler-bench"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    samples: list[dict[str, float]] = []
    for repetition in range(args.repeats):
        compiled = build_rrt(
            args.urdf,
            args.spheres,
            args.output / "build",
            options=Options(args.math),
            arch=args.arch,
        )
        samples.append(compiled.timings)
        print(f"Build {repetition + 1}: {compiled.timings}", flush=True)
    medians = {
        stage: statistics.median(sample[stage] for sample in samples)
        for stage in ("codegen", "nvcc", "total")
    }
    report = {
        "math": args.math,
        "arch": args.arch,
        "samples_seconds": samples,
        "median_seconds": medians,
        "toolchain": compiled.toolchain,
        "timing": "Uncached build_rrt calls in one Python process; first sample retained. "
        "Includes source/file generation and nvcc; excludes Python imports and GPU driver JIT.",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Median: codegen {medians['codegen']:.3f}s, nvcc {medians['nvcc']:.3f}s, total {medians['total']:.3f}s"
    )


if __name__ == "__main__":
    main()
