# Benchmarks

Scripts in this directory use the installed `tenkai` package and do not build native
code outside the library. `compiler.py` is standalone. The RRT-Connect comparison is split by
role: `problems.py` (problem file format, registered sets, release download; no plainmp),
`plainmp_baseline.py` (all plainmp code), `compare_rrtc.py` (CLI, tenkai GPU timing, report).


```sh
uv run --extra bench python bench/compare_rrtc.py --multiplier 1 --repeats 7
uv run python bench/compiler.py --repeats 3
```

`compare_rrtc.py` compares plainmp CPU and tenkai GPU RRT-Connect on the 1,024
different Fetch problems in `fetch_problems.yaml`, kept out of the repository
(see [Problem set](#problem-set)).
The start is shared; every problem has a different goal and environment. There are
128 environments for each obstacle count from 1 to 8, mixing boxes, spheres, and
cylinders with different sizes and poses, plus a ground plane in every environment.
Both planners use the same geometry, VAMP's 111 collision spheres,
joint bounds, step size, and motion-check resolution (0.02). The collision YAML selects 97 spheres
for environment checks and 48 self collision link pairs (2,586 sphere pairs).
The base spheres participate only in self collision because they overlap the floor.
tenkai uses one sample and one warp per problem. Failed solves stop the benchmark.

The problem YAML contains `start`, `multiplier`, and a `problems` list; each entry
contains `goal` and `obstacles`. Joint vectors follow `collision.yaml`'s
`control_joint_names`. Obstacles use the `tenkai.sdf` class names and fields:
`Box`, `Sphere`, `Cylinder`, and `Ground`. `--problems PATH` selects another set.
Set `multiplier: 50` in the YAML or pass `--multiplier 50` to run 51,200 solves.
Each copy reuses the environment and the verified per-problem seed (`42 + index`).
`--batch-size` caps the number of problems per GPU call (default 8,192), so large
multipliers run in chunks with reusable buffers. Reported batch time covers the
entire multiplied set, including all chunks; `gpu_batches` records the call count.

## Problem set

Problem sets are release assets, not tracked files. Each is registered in `PROBLEM_SETS` in
`problems.py` with its release tag, sha256, and robot files; select one with
`--problem-set NAME` (default `fetch`). `fetch` is
<https://github.com/HiroIshida/tenkai/releases/download/bench-fetch-data-v1/fetch_problems.yaml>

`compare_rrtc.py` downloads the set to `artifacts/rrtc-problems/` on first use, with `gh`,
and checks it against the pinned sha256. To fetch it manually:

```sh
mkdir -p artifacts/rrtc-problems
curl -L -o artifacts/rrtc-problems/fetch_problems.yaml \
  https://github.com/HiroIshida/tenkai/releases/download/bench-fetch-data-v1/fetch_problems.yaml
sha256sum artifacts/rrtc-problems/fetch_problems.yaml
```

A regenerated or new set is published under a new tag (`bench-fetch-data-v2`, ...) and its
`PROBLEM_SETS` entry is updated together; published tags are not overwritten.

### Provenance

The `fetch` set holds 1,024 problems (seed 42) kept only if both plainmp and tenkai solve
them and both paths pass endpoint, joint-bound, self-collision, and environment-collision
checks along every edge at 0.005 resolution. It was verified on HIP. Feasibility is
established for the supplied Fetch model and collision settings.

Times cover Python API calls for a batch after one warmup batch. plainmp solves
sequentially and includes native planner creation and trajectory conversion;
tenkai solves the batch in parallel, reuses GPU buffers, and assembles paths on
the GPU. Its timing includes input/result-metadata transfers but not path
retrieval to the CPU. Compilation and setup are excluded. Both planners replay
fixed per-problem seeds for copies and timing repetitions. This measures the two Python
interfaces, including their different overheads. A GPU for the selected backend is required.

`compiler.py` reports code generation, nvcc/hipcc, and total `build_rrt()` wall time.
Every repetition rebuilds; the first sample is retained. Python imports and GPU
driver JIT are excluded. Only the selected toolkit and a compatible C++ compiler are
required; no GPU is needed.

Both scripts use FP32 with SFU trigonometric intrinsics and accept
`--backend cuda|hip` (default `cuda`) and `--arch`
(default `compute_75` for CUDA, `gfx1201` for HIP). For RX 9070 XT:

```sh
uv run python bench/compiler.py --backend hip --repeats 1
uv run --extra bench python bench/compare_rrtc.py --backend hip --multiplier 50
```

For larger batches, `--cpu-ms-per-problem MS` skips CPU solves and extrapolates
linearly from a prior measured CPU median divided by its batch size. Use a
baseline from the same machine, problem set, model, and planning settings. For
example, measure `--multiplier 1`, divide the CPU median by 1,024, and supply that
value with `--multiplier 50 --cpu-ms-per-problem MS`.
Historical single-problem timings are not comparable to this varied set.
The report keeps `plainmp_ms` empty and stores the estimate separately as
`plainmp_estimated_ms`, alongside the supplied `plainmp_ms_per_problem`.
It also records the problem file's SHA-256, distinct problem count, and multiplier.

`--urdf` and `--spheres` override the model file paths;
the planning benchmark's start, goal, and scene are specific to Fetch.
JSON results and generated builds go to `artifacts/rrtc-bench/` or
`artifacts/compiler-bench/`, configurable with `--output`.
Each build contains `rrt/bindings.cu`, the separate planner headers under `rrt/`,
and shared collision headers under `collision/`. The compiler timing includes
writing this source tree and compiling its normal C++ includes.

The historical self collision comparison compiled variants with and without the
link broad phase, leaving external collision identical. It compared 32,768 random
configurations in empty and box/ground environments, then RRT batches of 512,
2,048 and 8,192. Each allocation received three warmup calls; nine measured pairs
alternated the variant order. Reports include all samples, medians and quartiles.
Collision flags, search results and first-trial path hashes matched between
variants; every planning problem succeeded. Times include Python API overhead,
transfers and per-call collision scratch allocation. They exclude compilation,
planner allocation, warmups and validation.

On 2026-11-15, the previous Fetch configuration (69 spheres, 14 self-collision
link pairs, without lower-body primitives) gave these median RRT times
(nine measured pairs, milliseconds per batch):

| GPU | Batch | Direct narrow | Broad → narrow | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA RTX A500 Laptop | 512 | 156.719 | 63.591 | 59.4% |
| NVIDIA RTX A500 Laptop | 2,048 | 559.159 | 192.731 | 65.5% |
| NVIDIA RTX A500 Laptop | 8,192 | 2,277.111 | 769.708 | 66.2% |
| AMD Radeon RX 9070 XT | 512 | 58.047 | 35.135 | 39.5% |
| AMD Radeon RX 9070 XT | 2,048 | 64.132 | 38.108 | 40.6% |
| AMD Radeon RX 9070 XT | 8,192 | 153.090 | 83.718 | 45.3% |

For the 32,768-configuration collision API batch, direct/broad times were
0.905/0.880 ms (empty scene) and 1.912/1.664 ms (box/ground) on NVIDIA;
0.509/0.440 ms and 0.527/0.485 ms on AMD. These short API timings include
allocation and transfers, so they do not isolate kernel execution time.
All flags, search results and paths matched between variants on each backend;
cross-backend paths differ. Broad phase is now always enabled based on the RRT
results. This comparison covers one robot and planning problem.

Raw samples and generated CUDA sources are under
`bench/artifacts/self-collision-20261115/cuda/`; the AMD report is under the
adjacent `hip/` directory. The compiler, assets and comparison script from before
the option was removed are preserved in `source.tar.gz` in the same artifact
directory. To reproduce the comparison, extract that archive and run its
`bench/self_collision.py` with `PYTHONPATH=src`, adding `--backend hip` on AMD.
