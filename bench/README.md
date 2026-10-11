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
- Default codegen: float, SFU on, scheduler off; block size 128.
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
