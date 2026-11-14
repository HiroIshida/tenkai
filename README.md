# tenkai

Generate robot-specific GPU kernels for forward kinematics, collision checks, and
single-candidate RRT-Connect. The compiler is Python 3.12; shared C++ kernels and the
native planner runtime compile with CUDA or AMD HIP.

## Setup

```sh
uv sync --all-extras
```

`src/tenkai` contains the Python package, CUDA templates, and planner bindings.
The package depends on PyYAML and NumPy; the `bench` extra adds plainmp.

## Generate kernels

```sh
uv run tenkai-generate \
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

from tenkai import FkDag, Options, Robot, generate_collision_kernel, read_spheres

robot = Robot.from_string(Path("assets/fetch/fetch.urdf").read_text())
fk = FkDag.from_robot(robot)
spheres = read_spheres(Path("assets/fetch/collision.yaml").read_text(), fk)
source = generate_collision_kernel(fk, spheres, Options(math="f32"))
```

## Run and time a planner

RRT-Connect can be used from one script with the installed package and
your model files. No code from `bench/` or `examples/` is needed. For the included
Fetch model, this is a complete example:

```python
from time import perf_counter

import numpy as np

from tenkai import RRTConfig, build_rrt

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
    config=RRTConfig(max_rounds=8192),
    problem_count=len(starts),
    resolution=0.01,
) as planner:
    planner.solve(starts, goals, [0], [42])  # Warm up the GPU driver and kernels.
    times = []
    for _ in range(10):
        begin = perf_counter()
        results = planner.solve(starts, goals, [0], [42])
        times.append((perf_counter() - begin) * 1000)
    print(f"Median solve: {np.median(times):.3f} ms", results)
    path = planner.path(0)
```

`build_rrt` generates CUDA, runs `nvcc` via `subprocess`, and writes `librrt.so`,
`model.json`, and a compiler log under the output directory. `compiled.timings`
reports `codegen`, `nvcc`, and `total` wall times in seconds. Each call rebuilds;
use a separate directory for each distinct build. Import `Options` from `tenkai`
and pass, for example, `options=Options(math="f64")` to select precision
(`f32`, `f32-sfu`, or `f64`). Set the target with `arch="compute_75"`.
The build needs a CUDA toolkit and compatible C++ compiler (`NVCC` and `CXX` may
override discovery). Creating or running a planner also needs a CUDA GPU/driver.

For AMD Radeon RX 9070 XT, select HIP:

```python
compiled = build_rrt(
    "assets/fetch/fetch.urdf",
    "assets/fetch/collision.yaml",
    "artifacts/fetch-hip",
    backend="hip",
    arch="gfx1201",
)
```

The same single-candidate RRT-Connect API supports `f32`, `f32-sfu`, and `f64`.
Each problem uses one warp; the f32 modes also use streaming collision checks
when the scene permits them.
HIP defaults to `gfx1201`; CUDA remains the
default backend with `compute_75`. Set `HIPCC` to override `hipcc` discovery.
HIP builds report `hipcc` instead of `nvcc` in `compiled.timings` and
`compiled.toolchain`. The CUDA-only `max_registers` option must be omitted or zero.

This initial HIP port targets wave32 GPUs and was tested on RX 9070 XT with
HIP 7.16 / AMD Clang 24. Wave64 devices are rejected when creating a planner.
If several GPUs are installed, use `HIP_VISIBLE_DEVICES=0` (or the appropriate
device index) to select the Radeon. Generated source is shared with CUDA; the
build supplies a compatibility header and enables OCML directed rounding for
workspace bounds. Numerical results need not be bit-identical across backends.

For custom native builds, `tenkai.build.compiler_command("hip", "gfx1201")`
returns `(flags, version)` with the required includes and compiler definitions;
append your source files and output flags before invoking it with `subprocess`.

`compiled.planner(...)` loads the shared library through `ctypes.CDLL` and owns
the native scene and search buffers until `close()` or context exit. Solves reuse
these buffers and finish copying results to the host before returning, so the
timer above includes Python and
host/device transfers, and excludes compilation, setup, and first-use driver JIT.

Starts, goals, and returned paths use `compiled.model["control_joints"]` order.
Pass one environment ID and uint64 seed per problem; repeated solves with the
same inputs are deterministic. `path(i)` retrieves the last solve's path.
`collisions(q, ids)` accepts full configurations in
`compiled.model["input_joint_names"]` order.

The current binding accepts one axis-aligned box and one ground plane per
environment: each geometry row is `[cx, cy, cz, sx, sy, sz, ground_offset]`, with
full box side lengths `sx/sy/sz` and the ground plane at `z = -ground_offset`.
Uncontrolled joints stay at zero; self-collision is not checked. Model inputs and
scene geometry are supplied by the caller.

## Check

Run `./format.sh` to format CUDA templates with clang-format (Chromium style),
format Python with Ruff, and run Ruff lint checks. All tools are managed by uv.

```sh
uv run pytest
uv run mypy
uv run ruff check src tests bench tools
uv run ruff format --check src tests bench tools
uv build
```

Tests include source hashes for 108 configurations, normalizing the tenkai/TENKAI
names back to rkern/RKERN, and CPU execution of generated kernels. Source hashes
track the current formatted templates.
`uv run pytest --run-cuda` also compiles with nvcc.
`--run-gpu` enables GPU execution tests. Set
`NVCC` and `CXX` when the toolkit requires a specific compiler.

```sh
uv run pytest --run-hip --run-gpu
```

`--run-hip` selects HIP and enables compilation tests; `--run-gpu` additionally
executes them on the GPU. `--gpu-arch gfx1201` overrides the default target.
CUDA and HIP runs reuse the same arithmetic, collision, planner, and binding tests.

## Benchmarks and migration reference

```sh
uv run python bench/compiler.py
uv run --extra bench python bench/compare_rrtc.py
```

These are independent, single-file benchmarks: compilation time and Python API
RRT-Connect throughput against plainmp. Results and builds go under `artifacts/`.
See [benchmark details](bench/README.md) for workloads and measurement boundaries.

The pre-migration Rust generators, source snapshot, generated CUDA, and compiled
binaries are saved locally in `artifacts/rust-reference/`. `source-manifest.json`
records their checksums. They are outside `target/` and ignored by Git.

```sh
uv run python tools/verify_migration.py
```

This compares generated CUDA byte-for-byte with the saved Rust sources for f32,
f32/SFU, and f64 after replacing rkern/RKERN with tenkai/TENKAI in the reference.
It also checks model metadata and the saved reference checksums. It requires
neither a Rust compiler nor a CUDA toolkit and does not rebuild binaries.
The original Rust source also remains available in the local source archive and
Git history. The historical reports and native comparison fixtures are kept in
`tests/reference/migration/`.
