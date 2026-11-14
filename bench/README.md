# Benchmarks

Two independent scripts use the installed `tenkai` package. Neither imports
another bench file or builds native code outside the library.

```sh
uv run --extra bench python bench/compare_rrtc.py --batch 256 --repeats 7
uv run python bench/compiler.py --repeats 3
```

`compare_rrtc.py` compares plainmp CPU and tenkai GPU RRT-Connect on one fixed
Fetch problem. Both use all 69 collision spheres, the same box and ground plane,
joint bounds, step size, and motion-check resolution; self-collision is disabled.
tenkai uses one sample and one warp per problem. Failed solves stop the benchmark.

Times cover Python API calls for a batch after one warmup batch. plainmp solves
sequentially and includes native planner creation and trajectory conversion;
tenkai solves the batch in parallel, reuses GPU buffers, and includes transfers
but not path retrieval. Compilation and setup are excluded. CPU trials advance
the random generator; GPU trials repeat fixed seeds. This measures the two Python
interfaces, including their different overheads. A GPU for the selected backend is required.

`compiler.py` reports code generation, nvcc/hipcc, and total `build_rrt()` wall time.
Every repetition rebuilds; the first sample is retained. Python imports and GPU
driver JIT are excluded. Only the selected toolkit and a compatible C++ compiler are
required; no GPU is needed.

Both scripts accept `--backend cuda|hip` (default `cuda`),
`--math f32|f32-sfu|f64` (default `f32`) and `--arch`
(default `compute_75` for CUDA, `gfx1201` for HIP). For RX 9070 XT:

```sh
uv run python bench/compiler.py --backend hip --repeats 1
uv run --extra bench python bench/compare_rrtc.py --backend hip --batch 256
```

For larger batches, `--cpu-ms-per-problem MS` skips CPU solves and extrapolates
linearly from a prior measured CPU median divided by its batch size. Use a
baseline from the same machine, model, and planning settings. For example,
`--batch 8192 --cpu-ms-per-problem 3.281841224609394` estimates CPU time from
the 4096-problem median of 13442.422 ms while measuring the GPU normally.
The report keeps `plainmp_ms` empty and stores the estimate separately as
`plainmp_estimated_ms`, alongside the supplied `plainmp_ms_per_problem`.

`--urdf` and `--spheres` override the model file paths;
the planning benchmark's start, goal, and scene are specific to Fetch.
JSON results and generated builds go to `artifacts/rrtc-bench/` or
`artifacts/compiler-bench/`, configurable with `--output`.

Rust migration checks live in `tools/verify_migration.py`, with historical
reports and native fixtures under `tests/reference/migration/`.
