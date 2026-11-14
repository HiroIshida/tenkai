#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path

import tyro

from tenkai import build_rrt
from tenkai.build import Backend, target_arch

ASSETS = Path(__file__).resolve().parents[1] / "assets/fetch"


def main(
    urdf: Path = ASSETS / "fetch.urdf",
    spheres: Path = ASSETS / "collision.yaml",
    backend: Backend = "cuda",
    arch: str | None = None,
    repeats: int = 3,
    output: Path = Path("artifacts/compiler-bench"),
) -> None:
    """Measure code generation and GPU compilation time for the planner library."""
    if repeats < 1:
        print("repeats must be positive", file=sys.stderr)
        raise SystemExit(2)

    arch = target_arch(backend, arch)
    compiler_name = "hipcc" if backend == "hip" else "nvcc"

    samples: list[dict[str, float]] = []
    for repetition in range(repeats):
        compiled = build_rrt(
            urdf,
            spheres,
            output / "build",
            backend=backend,
            arch=arch,
        )
        samples.append(compiled.timings)
        print(f"Build {repetition + 1}: {compiled.timings}", flush=True)
    medians = {
        stage: statistics.median(sample[stage] for sample in samples)
        for stage in ("codegen", compiler_name, "total")
    }
    report = {
        "math": compiled.model["math"],
        "backend": backend,
        "arch": arch,
        "samples_seconds": samples,
        "median_seconds": medians,
        "toolchain": compiled.toolchain,
        "timing": "Uncached build_rrt calls in one Python process; first sample retained. "
        "Includes source/file generation and GPU compilation; excludes Python imports and GPU driver JIT.",
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Median: codegen {medians['codegen']:.3f}s, "
        f"{compiler_name} {medians[compiler_name]:.3f}s, total {medians['total']:.3f}s"
    )


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.DisallowNone,))
