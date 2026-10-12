# Fetch collision benchmark

From the repository root:

```sh
python3 bench/run.py
```

Requires Linux, Git, Rust/Cargo, Python 3.8+ with NumPy and Matplotlib, an NVIDIA GPU, the CUDA
toolkit (`nvcc`), and a host C++ compiler supported by that toolkit. `NVCC` and `CXX` override the
compiler paths. CUDA 10 uses `g++-8` by default; newer toolkits use `c++`.
The default PTX target is `compute_75` (GPU compute capability 7.5 or newer).
Use `--arch compute_86`, for example, when both the toolkit and GPU support it.

The CPU comparison additionally requires CMake 3.15+, a C++17 compiler with LTO,
Eigen3, yaml-cpp, and `taskset`/`lscpu` (util-linux). On Ubuntu, the development
packages are `libeigen3-dev` and `libyaml-cpp-dev`. `CPU_CXX` overrides the CPU
compiler independently of the CUDA host compiler; it defaults to `c++`.

The command initializes the pinned `bench/plainmp` submodule when needed, builds
both executables, prepares the shared inputs, warms up and measures the GPU and
CPU sequentially, validates outputs, prints the comparison table, and saves
throughput plots. The SSH submodule URL requires GitHub SSH access on first
initialization. No Python
plainmp installation, OMPL, nanobind, or files from `tmp/` are needed.

The submodule is pinned to `f1151eb16c282934a1b2b8d1cd599829eb294247`.
An existing checkout must match the recorded gitlink and have no tracked changes;
the runner will not reset edits. No patch is applied to plainmp. The standalone
CMake target compiles its original kinematics/collision sources and bundled URDF
parser. It preincludes `math.h` to reproduce the floating-point `abs` overloads
normally made available by `Python.h` in the bindings.

## Workload and timing

- Bundled Fetch URDF and all 69 collision spheres from `assets/fetch/collision.yaml`.
- Table: center `(0.95, 0, 0.8)`, full extents `(1, 2, 0.05)`; floor: `z = 0`.
  Environment collision only; self-collision and self-body primitives are excluded.
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
- `plainmp`: single-threaded C++ wall time around the loop calling
  `SphereCollisionCst::is_valid()`, including FK. All input vectors, conversions
  from the input precision to double, and model/environment setup precede timing.
  AABB, broad/narrow phase, early exit, lazy FK caches, sphere reordering, and
  native polynomial sin/cos remain enabled. The implementation is unmodified.
  The process is pinned to the first allowed logical CPU, or `--cpu-core N`.
- Both backends use the same batch sizes, input offsets, and repetition counts
  from `workload.hpp`. CPU input values match GPU inputs, but plainmp always
  computes in double; `--math` only controls rkern. GPU/CPU runs do not overlap.
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
their strict checks. NumPy is used for sampling and statistics only.

## Options and results

```sh
python3 bench/run.py --math f64 --output bench/results/double
python3 bench/run.py --math f32 --live-budget 64 --output bench/results/scheduled
python3 bench/run.py --cpu-core 8
python3 bench/run.py --pool 4096 --batches 1 256 4096 --rounds 3 \
  --warmup-seconds 0.2 --queries-per-round 4096 --max-repetitions 16
python3 bench/run.py --help
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

`generate.rs` prepares the code and model metadata; `collision.cu` and
`plainmp_bench.cpp` own GPU/CPU execution and timers; `workload.hpp` shares the
sampling schedule and binary input handling; `run.py` coordinates preparation,
execution, validation, and aggregation.

## Batched CUDA RRT and RRT-Connect

```sh
python3 bench/rrt.py
python3 bench/rrt.py --problems 32 --environments 2 --warps-per-problem 4
python3 bench/rrt.py --math f64 --output bench/results/rrt-f64
python3 bench/rrt.py --algorithm rrtc --problems 32 --environments 2 --resolution 0.01
```

Uses the same CUDA/C++ toolchains and plainmp build dependencies as the collision
benchmark, plus NumPy. `NVCC`, `CXX`, `CPU_CXX`, and `--arch` have the same meaning.
Python constructs an `RRTConfig` dataclass and calls an opaque native planner
through `build/librrt.so`; sampling, nearest-neighbor search, interpolation,
collision checking, tree updates, and path tracing execute inside CUDA.

One problem owns one block for the entire solve. `--warps-per-problem` fixes its
thread count to `32 * warps` at launch; the default is four. Environments and
problems have separate IDs, so many problems can share a scene. Environment zero
is the ground/table case above; additional environments shift the table by 0.02 m
along Y per environment. The planner keeps its allocated device buffers across
solves. Its scene buffers are owned by the benchmark's native handle.

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

The start/goal are copied from `plainmp/example/bench/fetch_plan.py`, using the
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
grid with CUDA and independently with plainmp. These remain discrete checks,
not a continuous collision certificate. Validation arrays and build logs are saved
alongside the results. The command fails if any problem fails or any checked path
point collides.

CUDA integration tests (FP32, FP64, and FP32 SFU) cover reachable/blocked problems,
invalid endpoints/configs, shared and distinct environments, control ordering and
fixed joints, repeated solves, and identical paths with one/three/four/eight warps for
both algorithms. RRTC additionally checks repeated CONNECT steps, node capacity
across both trees, and joining an existing opposite root:

```sh
CXX=g++-8 cargo test --test rrt -- --include-ignored
```

Use the host compiler supported by the installed CUDA toolkit; `g++-8` above is
for the original CUDA 10 host. `cuda::generate_collision_function` exposes the
single-configuration device predicate. `cuda::generate_rrt_kernel` emits
both specialized kernels and native `RrtPlanner`; pass
`rkern::RrtAlgorithm::RrtConnect` as its final constructor argument to select RRTC.
The Python `Planner` accepts `algorithm="rrtc"`. Common planner config/result types are
in `src/planning/rrt.hpp`, separate from CUDA execution configuration and storage.
