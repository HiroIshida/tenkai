# rkern

Generate robot-specific CUDA kernels for forward kinematics, collision checks,
RRT, and RRT-Connect. The compiler is Python 3.12; CUDA C++ implements the GPU
kernels and native planner runtime.

## Setup

```sh
uv sync --all-extras
```

`src/rkern` contains the Python package, CUDA templates, and planner bindings.
The package depends on PyYAML and NumPy; the `bench` extra adds Matplotlib.

## Generate kernels

```sh
uv run rkern-generate \
  --urdf assets/fetch/fetch.urdf \
  --spheres assets/fetch/collision.yaml \
  --float --no-sfu artifacts/generated
```

This writes `collision.cuh`, `rrt.cuh`, and `model.json`. Use `--all-kernels` to
also emit FK, sphere-center, and collision-predicate sources. `--double`,
`--optimization none|basic|algebraic`, and `--live-budget N` select the numerical
and scheduling behavior. The CLI prints stage durations as JSON.

```python
from pathlib import Path

from rkern import FkDag, Options, Robot, generate_collision_kernel, read_spheres

robot = Robot.from_string(Path("assets/fetch/fetch.urdf").read_text())
fk = FkDag.from_robot(robot)
spheres = read_spheres(Path("assets/fetch/collision.yaml").read_text(), fk)
source = generate_collision_kernel(fk, spheres, Options(math="f32"))
```

## Run and time a planner

RRT and RRT-Connect can be used from one script with the installed package and
your model files. No code from `bench/` or `examples/` is needed. For the included
Fetch model, this is a complete example:

```python
from time import perf_counter

import numpy as np

from rkern import RRTConfig, build_rrt

compiled = build_rrt(
    "assets/fetch/fetch.urdf",
    "assets/fetch/collision.yaml",
    "build/fetch",
)
starts = [[0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]]
goals = [[0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01]]
geometry = [[0.95, 0, 0.8, 1, 2, 0.05, 0]]

with compiled.planner(
    geometry,
    algorithm="rrtc",
    config=RRTConfig(max_rounds=8192),
    problem_count=len(starts),
    resolution=0.01,
) as planner:
    planner.solve(starts, goals, [0], [42])  # Warm up the CUDA driver and kernels.
    times = []
    for _ in range(10):
        begin = perf_counter()
        results = planner.solve(starts, goals, [0], [42])
        times.append((perf_counter() - begin) * 1000)
    print(f"Median solve: {np.median(times):.3f} ms", results)
    path = planner.path(0)
```

`build_rrt` generates CUDA, runs `nvcc` via `subprocess`, and writes `librrt.so`,
`model.json`, and a compiler log under the output directory. Each call rebuilds;
use a separate directory for each distinct build. Import `Options` from `rkern`
and pass, for example, `options=Options(math="f64")` to select precision
(`f32`, `f32-sfu`, or `f64`). Set the target with `arch="compute_75"`.
The build needs a CUDA toolkit and compatible C++ compiler (`NVCC` and `CXX` may
override discovery). Creating or running a planner also needs a CUDA GPU/driver.

`compiled.planner(...)` loads the shared library through `ctypes.CDLL` and owns
the native scene and search buffers until `close()` or context exit. Use
`algorithm="rrt"` for single-tree RRT. Solves reuse these buffers and finish copying
results to the host before returning, so the timer above includes Python and
host/device transfers, and excludes compilation, setup, and first-use driver JIT.

Starts, goals, and returned paths use `compiled.model["control_joints"]` order.
Pass one environment ID and uint64 seed per problem; repeated solves with the
same inputs are deterministic. `path(i)` retrieves the last solve's path.
`collisions(q, ids)` accepts full configurations in
`compiled.model["input_joint_names"]` order.

The current binding accepts one axis-aligned box and one ground plane per
environment: each geometry row is `[cx, cy, cz, hx, hy, hz, ground_z]`, with box
half extents `hx/hy/hz`. Uncontrolled joints stay at zero; self-collision is not
checked. Model inputs and scene geometry are supplied by the caller.

## Check

```sh
uv run pytest
uv run mypy
uv run ruff check src tests bench/common.py bench/generate.py bench/compiler.py bench/verify_migration.py
uv run ruff format --check src tests bench/common.py bench/generate.py bench/compiler.py bench/verify_migration.py
uv build
```

Tests include frozen Rust source hashes for 108 configurations and CPU execution
of generated kernels. `uv run pytest --run-cuda` also compiles with nvcc.
`--run-gpu` enables the planner execution tests and requires a CUDA GPU. Set
`NVCC` and `CXX` when the toolkit requires a specific compiler.

## Benchmarks and migration reference

```sh
uv run python bench/compiler.py
uv run python bench/run.py
uv run python bench/rrt.py --algorithm rrtc
```

See [benchmark details](bench/README.md) for workloads and measurement boundaries.
The existing benchmarks now generate kernels with the installed Python package.
The RRT benchmark uses the public planner API. Add `--validate-plainmp` to also
build and run the independent plainmp path checker; it is optional.

The pre-migration Rust generators, source snapshot, generated CUDA, and compiled
binaries are saved locally in `artifacts/rust-reference/`. `source-manifest.json`
records their checksums. They are outside `target/` and ignored by Git.

```sh
uv run python bench/verify_migration.py
```

This uses the saved build commands and paths to compare whole executable/shared
library files byte-for-byte, for f32, f32/SFU, and f64. The reference builds use
fixed intermediate paths and a fixed host compiler seed to make the comparison
reproducible. No binary fields are stripped or normalized. The recorded toolkit
is required for this check; the portable source parity tests need no Rust compiler.
The original Rust source also remains available in the local source archive and
Git history.
