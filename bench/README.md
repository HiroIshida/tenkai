# Fetch collision benchmark

From the repository root:

```sh
uv run --extra bench python bench/run.py
```

Run `uv sync --extra bench` first. Requires Linux, Python 3.12, an NVIDIA GPU, the CUDA
toolkit (`nvcc`), and a host C++ compiler supported by that toolkit. `NVCC` and `CXX` override the
compiler paths. CUDA 10 uses `g++-8` by default; newer toolkits use `c++`.
The default PTX target is `compute_75` (GPU compute capability 7.5 or newer).
Use `--arch compute_86`, for example, when both the toolkit and GPU support it.

Plainmp is installed by uv with the `bench` extra, at the version recorded in
`uv.lock`. There is no submodule checkout or benchmark-specific CMake build.
The runner compiles the GPU timing harness, prepares shared inputs, measures
GPU and CPU sequentially, compares collision flags, and saves throughput plots.
CPU metadata records the installed plainmp version and native module hash.

## Workload and timing

- Bundled Fetch URDF and all 69 collision spheres from `assets/fetch/collision.yaml`.
- Table: center `(0.95, 0, 0.8)`, full extents `(1, 2, 0.05)`; floor: `z = 0`.
  Environment collision only; self-collision and self-body primitives are excluded.
  The plainmp reference requires ground offset zero because its ground AABB is
  fixed at `z = 0`; other offsets are rejected rather than silently misvalidated.
- 1,048,576 configurations sampled uniformly with NumPy PCG64, seed `20261011`.
  The eight control joints use URDF bounds; continuous joints use `[-2π, 2π)`.
  Other joint inputs are zero. All modes use the same pool in the FK input order.
- Default codegen: float, SFU on, algebraic CSE, scheduler off; block size 128.
  Use `--optimization none|basic|algebraic` to select DAG optimization.
  Collision kernels compute up to 16 sphere centers at a time, then traverse the
  runtime obstacles once for that group. An AABB of those computed centers,
  expanded by the largest sphere radius, excludes separated boxes, spheres and
  cylinders before individual sphere checks. Rotated obstacles use conservative
  world AABBs; ground checks run directly. Bounds include a floating-point margin,
  and only a narrow-phase collision ends a query. Shared FK values are reused;
  box and sphere checks use squared distances. Environment changes require no
  recompilation.
- Two-second warmup for each backend, plus two untimed repetitions before each measurement.
  Seven rounds with fixed job order and input offsets. Repetitions per measurement
  are `max(2, min(1024, ceil(1048576 / batch)))`; there is no timing-based calibration.
- `kernel`: CUDA events around repeated launches with inputs already on the GPU.
- `transfer`: wall time for repeated pinned-memory H2D configurations, kernel,
  and D2H one-byte flags. One stream, no transfer/compute overlap. Environment and
  environment IDs stay on the GPU in both modes.
- `plainmp`: single-threaded Python wall time around the loop calling
  `SphereCollisionCst.is_valid()`, including FK, Python dispatch, row access, and
  flag stores. These timings differ from the earlier C++-only baseline. All input
  vectors, conversions from the input precision to double, and model/environment
  setup precede timing.
  AABB, broad/narrow phase, early exit, lazy FK caches, sphere reordering, and
  native polynomial sin/cos remain enabled. The implementation is unmodified.
  The process is pinned to the first allowed logical CPU, or `--cpu-core N`.
- Both backends use the same batch sizes, input offsets, and repetition counts
  (the CUDA schedule is in `cuda/workload.hpp`, with the same formulas in Python).
  CPU input values match GPU inputs, but plainmp always computes in double;
  `--math` only controls rkern. GPU/CPU runs do not overlap.
- Sampling, compilation/JIT, allocation, initialization, validation, and file I/O
  are outside timing. Batch times are per-launch averages within each round;
  the table reports the median over rounds and the throughput range. `us/query`
  is amortized throughput cost, not an individual query's latency in a large batch.
  `vs CPU` compares median batch time with plainmp at the same batch size.

Inputs, measured repetition counts, and job order are deterministic for the same
arguments and software versions. Actual timings vary with clocks, temperature,
other workloads, and hardware. Inputs and generated source have SHA-256 hashes
in `metadata.json`; the pool is rotated through using a fixed per-round offset.
Small batches can visit only part of the pool within a measurement.

Before reporting timings, all configurations are compared with plainmp. At least
99.9% agreement is required; approximation differences count toward the allowed
0.1%. Neither backend is treated as exact ground truth. Disagreements are counted
separately for plainmp-collision/rkern-free and plainmp-free/rkern-collision, and
every differing index and pair of flags is recorded in `plainmp_differences.csv`.
`validation.json` records the agreement rate, directional counts, and pass/fail
result even if the threshold is not met. Existing analytical unit tests retain
their strict checks. NumPy supplies input arrays, sampling, and statistics.

## Options and results

```sh
uv run --extra bench python bench/run.py --math f64 --output bench/results/double
uv run --extra bench python bench/run.py --math f32 --live-budget 64 --output bench/results/scheduled
uv run --extra bench python bench/run.py --cpu-core 8
uv run --extra bench python bench/run.py --pool 4096 --batches 1 256 4096 --rounds 3 \
  --warmup-seconds 0.2 --queries-per-round 4096 --max-repetitions 16
uv run --extra bench python bench/run.py --help
```

Outputs are in the git-ignored `bench/results/` by default. A run replaces files
with the same names; use a separate `--output` directory to retain comparisons.

- `summary.csv`, `summary.json`: median batch time, amortized time/query,
  median/min/max throughput, and speedup versus plainmp by mode and batch size.
- `throughput.png`, `throughput_linear.png`: plainmp, GPU kernel, and
  transfer-inclusive throughput with logarithmic and linear Y axes, respectively.
  Batch size uses a logarithmic X axis in both plots. Lines show medians, shaded
  bands show the minimum/maximum over rounds, and rightmost points are labeled
  in million queries/s. Plotting works without a display.
- `raw.csv`: all measured rounds, including repetition counts and timers;
  `gpu_raw.csv` / `plainmp_raw.csv` retain each executable's output.
- `metadata.json`, `device.json`, `validation.json`: workload, toolchain, hashes,
  GPU/kernel information, and correctness results.
- `configurations.bin`, `scene.bin`: shared inputs;
  `collision.u8` / `plainmp_collision.u8`: GPU/CPU outputs.
- `plainmp_differences.csv`: differing query indices and both collision flags.
- `build/`: generated CUDA source, model metadata, configuration header,
  compiler log, and executable.

`run.py` coordinates inputs, measurements, validation, and plotting.
`reference.py` constructs the matching plainmp scene through its Python API.
`cuda/collision.cu` and `cuda/workload.hpp` are the remaining native benchmark
files: they measure GPU-resident kernel time and pinned-memory transfers with
CUDA events. They are not planner implementations.

## Batched CUDA RRT and RRT-Connect

```sh
uv run python bench/rrt.py
uv run python bench/rrt.py --problems 32 --environments 2 --warps-per-problem 4
uv run python bench/rrt.py --math f64 --output bench/results/rrt-f64
uv run python bench/rrt.py --algorithm rrtc --problems 32 --environments 2 --resolution 0.01
uv run --extra bench python bench/rrt.py --algorithm rrtc --validate-plainmp
```

Uses the installed `rkern` package and the same CUDA/C++ toolchains as the collision
benchmark. The `bench` extra is needed only with `--validate-plainmp`.
`NVCC`, `CXX`, and `--arch` have the same meaning. The package owns
code generation, compilation, and the Python/native planner binding; the benchmark
supplies the Fetch inputs and measures solves. Python constructs an `RRTConfig`
dataclass and calls the planner through `build/librrt.so`; sampling, nearest-neighbor
search, interpolation, collision checking, tree updates, and path tracing execute
inside CUDA.

One problem owns one block for the entire solve. `--warps-per-problem` fixes its
thread count to `32 * warps` at launch; the default is four. Environments and
problems have separate IDs, so many problems can share a scene. Environment zero
is the ground/table case above; additional environments shift the table by 0.02 m
along Y per environment. The planner keeps its allocated device buffers across
solves. Its scene buffers are owned by the library's native handle.

FP32 without SFU defaults to `--max-registers 128`, passing `--maxrregcount=128`
to NVCC. This reduced solve time by 8–16% on the RTX A500 with 2,048 plans;
FP32 with SFU and FP64 keep the compiler default because the benefit did not
generalize across math modes. Use `--max-registers 0` for the compiler default,
or a value from 32 through 255 for an explicit limit. The summary records the
effective limit. This is a compile-time setting, while `--warps-per-problem` is
fixed when constructing the planner. More warps can accelerate one problem but
reduce how many problem blocks fit on an SM. A register limit can allow more
resident blocks at the cost of spills. Tune the two settings together for the
intended batch size; neither changes the reserved tree capacity or sampling.
The runtime rejects warp counts exceeding the compiled kernel/device limits.

`--live-budget 64` enables the CUDA generator's liveness scheduler; omitting it
keeps DAG order. The budget counts live scalar DAG values, whereas
`--max-registers` constrains the compiler's registers per thread. These are
independent inputs to code generation and compilation; tune their combination.
A budget of zero always prioritizes the next output's dependencies.

The start/goal come from plainmp's Fetch planning example, using the
eight control joints and default per-joint resolution 0.05. Bounds and collision geometry
match the collision benchmark: all 69 spheres, uncontrolled joints at zero,
environment collisions only, no self-collision. The full collision setup therefore
differs from `FetchSpec.create_collision_const()` in that example.

`--algorithm rrt` (the default) selects single-tree batched RRT with direct goal connections.
Each round samples 32 targets against the same tree snapshot, finds Euclidean
nearest neighbors by linear scan, steers by at most `step_size` (default 1.0), and
distributes the edge's configuration checks across the block. An edge uses
`ceil(max(abs(delta) / resolution))` equal intervals, including the destination;
its source is already known valid. After validating candidate edges, the block
also checks direct connections from valid endpoints to the goal. This final goal
edge may exceed `step_size`, and is checked at the same resolution. Candidates
are inserted in fixed order after validation. No intermediate nodes or path
smoothing are added. A collision rejects the entire edge.

EXTEND and CONNECT edges first probe four points near the quarter intervals,
including the new endpoint, on the original checking grid. A collision rejects
the edge immediately; surviving edges receive the full resolution check.
Edges with at most four intervals go straight to the full check.
One warp checks each surviving edge in groups of 32 points and stops on the first
group containing a collision. Edges and insertion order are independent of warp count.

`--algorithm rrtc` selects batched RRT-Connect, following the EXTEND/CONNECT
structure in `plainmp/third/ompl_rrtc`. Each round alternates the active tree
between start and goal. It samples and validates a batch of extensions against
that tree's snapshot, then repeatedly extends the opposite tree toward those
accepted nodes until each connection is blocked or one reaches its target.
Each CONNECT target caches its nearest node and scanned prefix. Later waves
compare that cached node with the newly appended nodes in the opposite tree,
preserving the full scan's nearest node and tie order. Candidates share nodes
inserted by earlier waves. All edges, including the connection,
respect `step_size`. Goal-tree edges are checked toward their parents, including
an explicit check of the new endpoint. The two parent chains are joined without
duplicating the connection point. Both trees and their two roots share the
`max_nodes` budget; `max_rounds` counts outer EXTEND rounds, not CONNECT waves.
Setting `samples_per_round=1` gives one EXTEND/CONNECT attempt per round; larger
batches differ from sequential RRTConnect. `goal_bias` samples the opposite root
(set it to zero for uniform sampling as in plainmp). There is no path smoothing.

The RRTC command above uses resolution 0.01. At 0.05, this 32-problem case can
pass the planning grid yet collide between samples in the finer validation grid.
Resolution is a per-coordinate (Box) limit; nearest-neighbor distances and
`step_size` use the Euclidean norm. Changing algorithms does not change the
default resolution of 0.05.

Distance accumulation and interpolation use the selected scalar precision. FP32
squared distances fall back to double when they overflow or underflow; interval
counts, random sampling, and the distance/step-size comparison remain in double.
FP32 paths can differ from versions that used double throughout the planner.

Defaults are 8,192 nodes, 1,024 rounds, goal sampling probability 0.1, and 2,048
problems with different seeds. The seed, round, and candidate index determine
sampling independently of warp count. Failure results distinguish invalid start,
invalid goal, node capacity, and round limit. `--help` lists all settings.
The default output directory is `bench/results/<algorithm>`.

`--problems` sets the number of independent planning problems submitted in one
kernel launch. The larger default lets queued problems fill available GPU capacity
while slower problems finish, avoiding repeated drains of smaller batches.
`--samples-per-round` sets the candidate count within each problem separately.

`summary.json` reports per-problem results and whole-batch wall times for repeated
solves, including input/result copies. Construction, compilation/JIT, warmup,
path extraction, and validation are excluded. The repetitions use identical seeds
to check replay; they are timing repetitions, not additional independent trials.
`paths.json` contains control-joint waypoints and their joint names. The benchmark
checks bounds/endpoints and rechecks both the planning grid and a five-times finer
grid with CUDA, and independently with plainmp when `--validate-plainmp` is set.
These remain discrete checks, not a continuous collision certificate.
Validation counts and build logs are saved
alongside the results. The command fails if any problem fails or any checked path
point collides.

CUDA integration tests (FP32, FP64, and FP32 SFU) cover reachable/blocked problems,
invalid endpoints/configs, shared and distinct environments, control ordering and
fixed joints, repeated solves, and identical paths with one/three/four/eight warps for
both algorithms. RRTC additionally checks repeated CONNECT steps, node capacity
across both trees, and joining an existing opposite root:

```sh
CXX=g++-8 uv run pytest tests/test_cuda.py --run-cuda --run-gpu
```

Use the host compiler supported by the installed CUDA toolkit; `g++-8` above is
for the original CUDA 10 host. `rkern.generate_collision_function` exposes the
single-configuration device predicate. `rkern.generate_rrt_kernel` emits
both specialized kernels and native `RrtPlanner`; pass
`rkern::RrtAlgorithm::RrtConnect` as its final constructor argument to select RRTC.
The Python `rkern.Planner` accepts `algorithm="rrtc"`. Common planner config/result types are
in `src/rkern/templates/planning/rrt.hpp`, separate from CUDA execution configuration and storage.

## RRT-Connect comparison

```sh
uv run --extra bench python bench/compare_rrtc.py
```

Both sides use their installed Python APIs. The CPU side calls
`OMPLSolver.solve(Problem(...))`, with plainmp's RRT-Connect implementation,
range 1, no smoothing, the same joint bounds, all 69 spheres, and the same scene.
The GPU side calls `rkern.Planner.solve` in batches. Python wall time includes
CPU planner construction and trajectory conversion; GPU buffers are reused.
Compilation, warmup, and independent path checking are outside the measurements.
This replaces the old native CPU loop, so old and new CPU timings are different
measurement boundaries.

CPU sampling uses one seeded OMPL stream and fresh trials, including warmup;
GPU repetitions replay a fixed seed pool. CPU and GPU paths need not match.
`plainmp.json` retains per-round success counts and the last pool's paths.
Each side's paths are checked with both collision implementations on the planning
grid and a five-times finer grid. `summary.json` and `throughput.png` report
successful solves per second; failures are retained and make the command fail.

## Compile-time comparison

The Python 3.12 compiler and CUDA templates are installed from `src/rkern` by uv.
All benchmark entry points use this compiler.

```sh
uv run python bench/generate.py --float --no-sfu artifacts/generated
uv run python bench/compiler.py --output artifacts/compiler-bench
uv run python bench/verify_migration.py
```

The compile benchmark warms up the generator, measures five fresh Python
processes, and separately measures three nvcc builds per target and math mode.
Stage timings exclude Python startup/imports; `process_seconds` includes them.
A separate cProfile run records CSE/scheduling/emission with instrumentation
overhead. Compilation uses the collision executable and RRT shared-library
builds described above. Driver PTX JIT and GPU execution are excluded.

`results.json` contains raw samples, medians/ranges, stage breakdowns, commands,
source hashes and toolchain versions. Generated source, compiler logs, binaries,
and `codegen.prof` remain in each mode's subdirectory. Use `--modes`,
`--optimization`, `--live-budget`, `--repeats`, `--nvcc-repeats`, and `--arch`
to change the experiment.

[The original experiment](compiler_results.json) records Rust versus Python 3.8
before the migration, on a Ryzen 7 7840HS with CUDA 10.1 and GCC 8.4.
[Migration verification](migration_results.json) records whole-file binary
identity against the Rust baseline for all three math modes. The saved binaries
and Rust source archive live in `artifacts/rust-reference/`; they are required
only for the local binary replay. Ordinary tests compare against the 108 frozen
Rust output records in `tests/reference/compiler.json` without invoking Rust.
