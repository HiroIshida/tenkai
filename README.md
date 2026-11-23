# tenkai [![CI](https://github.com/HiroIshida/tenkai/actions/workflows/ci.yml/badge.svg)](https://github.com/HiroIshida/tenkai/actions/workflows/ci.yml)

tenkai (展開 in Japanese) is a GPU-accelerated batched motion planner (rrtc only for now).
It generates robot-specific kernels from URDF models and plans paths from Python
on NVIDIA CUDA or AMD HIP GPUs.

tenkai is heavily inspired by [VAMP](https://github.com/KavrakiLab/vamp)'s
robot-specific code generation and vectorized collision checking, adapted here
for GPUs. It also builds on my earlier projects,
[_tenkai](https://github.com/HiroIshida/_tenkai) (expression compilation) and
[plainmp](https://github.com/HiroIshida/plainmp) (SDFs and collision checking).

See [PORTING.md](PORTING.md) for porting credits (whether ported manually or with AI assistance).

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA or HIP (wave32)
GPU with its toolkit.

```sh
uv sync
uv run python example/rrtc.py
```

## Usage

```python
from tenkai import RRTConfig, ShortcutConfig, build_rrt, sdf

compiled = build_rrt("assets/fetch/fetch.urdf", "assets/fetch/collision.yaml", "build/fetch")
obstacles = [
    sdf.Box(center=(0.95, 0, 0.8), size=(1, 2, 0.05)),
    sdf.Ground(z=0),
]
start = [0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]
goal = [0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01]

with compiled.planner(obstacles, config=RRTConfig(max_rounds=8192), resolution=0.01) as planner:
    result = planner.solve(
        [start], [goal], environment_ids=[0], seeds=[42], shortcut=ShortcutConfig()
    )[0]
    if result["status"] == "success":
        print(planner.path(0))
```

With visualization:

```sh
uv run --extra visualize python example/rrtc.py --visualize
```

For more usage details, see the [example directory](example/).
To add your own robot, use the [Fetch robot configuration](assets/fetch/collision.yaml)
as a reference.

[Benchmarks](bench/README.md)
