# tenkai

tenkai (展開 in Japanese) generates robot-specific GPU kernels for forward
kinematics, collision checking, and RRT-Connect motion planning from URDF models.
Plan paths from Python on NVIDIA CUDA or AMD HIP GPUs.

tenkai is heavily inspired by [VAMP](https://github.com/KavrakiLab/vamp)'s
robot-specific code generation and vectorized collision checking, adapted here
for GPUs. It also builds on my earlier projects,
[_tenkai](https://github.com/HiroIshida/_tenkai) (expression compilation) and
[plainmp](https://github.com/HiroIshida/plainmp) (SDFs and collision checking).

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

Planning requires a supported GPU, its driver, and a CUDA or HIP toolkit with a
compatible C++ compiler. Set `NVCC`, `HIPCC`, or `CXX` to select your compilers.

## Plan your first path

This example uses the included Fetch robot. Save it as a Python file and run it
with `uv run python your_script.py` from the repository root.

```python
from tenkai import RRTConfig, ShortcutConfig, build_rrt, sdf

compiled = build_rrt(
    "assets/fetch/fetch.urdf",
    "assets/fetch/collision.yaml",
    "build/fetch",
)

start = [0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]
goal = [0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01]
obstacles = [
    sdf.Box(center=(0.95, 0, 0.8), size=(1, 2, 0.05)),
    sdf.Ground(z=0),
]

with compiled.planner(
    obstacles,
    config=RRTConfig(max_rounds=8192),
    problem_count=1,
    resolution=0.01,
) as planner:
    result = planner.solve([start], [goal], [0], [42], shortcut=ShortcutConfig())[0]
    print(result)
    if result["status"] == "success":
        path = planner.path(0)
        print(path)
```

`build_rrt` compiles the robot once. Reuse `compiled` with different obstacles,
and call `solve` repeatedly on a planner with new starts, goals, and seeds.
Each call to `build_rrt` rebuilds the library, so keep it outside your planning loop.

For a known set of obstacle types, pass `specialize_types={sdf.Box, sdf.Ground}`
to `build_rrt`. This builds one additional search kernel for that union, with
runtime obstacle counts and poses. A batch can mix Box-only, Ground-only,
combined, and empty environments. Each `solve` selects the specialized kernel
when all referenced environments fit; if any contains a Sphere or Cylinder,
the entire batch uses the generic kernel. No compilation occurs during `solve`.
The default `None`, or a set of all four types, builds only the generic kernel.
An empty set specializes environments without external obstacles; self-collision
checking remains enabled.

Starts, goals, and returned paths follow
`compiled.model["control_joints"]` order. The last two positional arguments to `solve` are
one environment ID and one unsigned 64-bit random seed per problem. Repeated
solves with the same inputs on the same backend are deterministic.
`shortcut=ShortcutConfig()` enables path shortening.
Use `path(i)` to retrieve one path, or `paths()` to retrieve all paths.

## Define your environments

Pass a list of `tenkai.sdf` objects to `compiled.planner`. Positions and lengths
are in meters, in the robot's root frame. An empty list creates an empty
environment; there is no implicit ground plane.

| Object | Parameters |
| --- | --- |
| `sdf.Box` | `center=(x, y, z)`, `size=(sx, sy, sz)` with full side lengths |
| `sdf.Sphere` | `center=(x, y, z)`, `radius` |
| `sdf.Cylinder` | `center=(x, y, z)`, `radius`, `height` with full height along local Z |
| `sdf.Ground` | `z`, the height of a horizontal surface with everything below it occupied |

Boxes and cylinders also accept `rotation`, a 3×3 local-to-world rotation matrix;
the default is the identity. You can mix primitive types and include any number
of each in an environment.

For multiple environments, pass a list of object lists. Environment IDs are
their zero-based indices, and multiple problems can share an environment:

```python
environments = [obstacles, []]

with compiled.planner(environments, problem_count=2, resolution=0.01) as planner:
    results = planner.solve([start, start], [goal, goal], [0, 1], [42, 43])
    for i, result in enumerate(results):
        if result["status"] == "success":
            print(i, planner.path(i))
```

Changing obstacle counts, types, or poses does not require recompiling the robot.
`planner.collisions(q, environment_ids)` checks configurations without planning;
these full configurations follow `compiled.model["input_joint_names"]` order.

## Use your own robot

Pass your URDF and collision-sphere YAML to `build_rrt`. The YAML lists the
controlled joint names and the collision spheres attached to each robot link;
see [the Fetch configuration](assets/fetch/collision.yaml) for the format.
Obstacle spheres (`sdf.Sphere`) describe the environment separately.

Uncontrolled joints stay at zero during planning. Collision checking first tests
the robot against the environment, then checks the link pairs listed in
`self_collision_pairs` if no external collision was found. Links listed in
`only_self_collision_links` participate only in self collision. An omitted pair
list disables self collision. The bundled Fetch configuration uses VAMP's spheres
for the base, torso, head, and arm, without separate body collision primitives;
see [its provenance and adaptations](assets/fetch/NOTICE.md).

Self collision always uses a conservative enclosing sphere per link to reject
separated link pairs, then tests their individual sphere pairs. This also applies
during RRT edge checks and when the environment is empty.

## Choose a GPU backend

CUDA is the default, targeting `compute_75`. Override the target with
`arch="compute_XX"` as appropriate for your GPU. For AMD HIP, for example on an
RX 9070 XT, use:

```python
compiled = build_rrt(
    "assets/fetch/fetch.urdf",
    "assets/fetch/collision.yaml",
    "build/fetch-hip",
    backend="hip",
    arch="gfx1201",
)
```

HIP supports wave32 GPUs; wave64 devices are rejected. Select a device with
`HIP_VISIBLE_DEVICES` if needed. Both backends always use FP32 with SFU
trigonometric intrinsics. Results can differ between backends.

## Generate kernels without running a planner

```sh
uv run tenkai-generate \
  --urdf assets/fetch/fetch.urdf \
  --spheres assets/fetch/collision.yaml \
  --all-kernels artifacts/generated
```

This generates forward-kinematics, sphere-center, collision, and planner sources
with model metadata. Keep the generated directory tree together when using
the sources in your application.

## Benchmarks

```sh
uv run python bench/compiler.py
uv run --extra bench python bench/compare_rrtc.py
```

These measure compilation time and planning throughput against plainmp.
See [benchmark details](bench/README.md) for workloads and timing boundaries.

## Development

```sh
./format.sh
uv run mypy
uv run pytest
```

Use `uv run pytest --run-cuda --run-gpu` for CUDA tests, or
`uv run pytest --run-hip --run-gpu` for HIP tests.
