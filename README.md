# rkern

Generate robot-specific CUDA kernels for forward kinematics, collision checks,
RRT, and RRT-Connect. The compiler is Python 3.12; CUDA C++ implements the GPU
kernels and native planner runtime.

## Setup

```sh
uv sync --all-extras
```

`src/rkern` contains the Python package and its CUDA templates. The `bench` extra
installs NumPy and Matplotlib. Code generation itself only requires PyYAML.

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
