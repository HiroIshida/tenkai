# Benchmarks

Two independent scripts use the installed `rkern` package. Neither imports
another bench file or builds native code outside the library.

```sh
uv run --extra bench python bench/compare_rrtc.py --batch 256 --repeats 7
uv run python bench/compiler.py --repeats 3
```

`compare_rrtc.py` compares plainmp CPU and rkern GPU RRT-Connect on one fixed
Fetch problem. Both use all 69 collision spheres, the same box and ground plane,
joint bounds, step size, and motion-check resolution; self-collision is disabled.
rkern uses one sample and one warp per problem. Failed solves stop the benchmark.

Times cover Python API calls for a batch after one warmup batch. plainmp solves
sequentially and includes native planner creation and trajectory conversion;
rkern solves the batch in parallel, reuses GPU buffers, and includes transfers
but not path retrieval. Compilation and setup are excluded. CPU trials advance
the random generator; GPU trials repeat fixed seeds. This measures the two Python
interfaces, including their different overheads. A CUDA GPU is required.

`compiler.py` reports code generation, nvcc, and total `build_rrt()` wall time.
Every repetition rebuilds; the first sample is retained. Python imports and GPU
driver JIT are excluded. Only the CUDA toolkit and a compatible C++ compiler are
required; no GPU is needed.

Both scripts accept `--math f32|f32-sfu|f64` (default `f32`) and `--arch`
(default `compute_75`). `--urdf` and `--spheres` override the model file paths;
the planning benchmark's start, goal, and scene are specific to Fetch.
JSON results and generated builds go to `artifacts/rrtc-bench/` or
`artifacts/compiler-bench/`, configurable with `--output`.

Rust migration checks live in `tools/verify_migration.py`, with historical
reports and native fixtures under `tests/reference/migration/`.
