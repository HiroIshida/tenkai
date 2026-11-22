# Writing guidelines

Code should explain how. Tests should specify what. Commit messages should explain why. Code comments should explain why not.
Unless explicitly requested, avoid unnecessary explanations or extra content.

Markdown documentation is for users. Do not use Markdown files to record internal technical details; put technical details relevant to a change in its commit message.

When copying or porting external code, reimplementing an external algorithm, or carrying over parameters/defaults, add or update a concise row in [PORTING.md](PORTING.md). Use the existing columns: upstream repository, license, local files, and porting method. Keep the upstream repository first so its authors can quickly recognize reuse.

Do not commit benchmark code unless the user explicitly asks to include it in a commit. Running or writing a benchmark does not imply permission to commit it.

Preserve significant decisions from discussions with the user in commit messages, especially nontrivial decisions involving trade-offs. Record the alternatives considered, the trade-offs, and the rationale for the chosen approach.

Include the evidence used to make those decisions in the commit log, such as benchmark results. Record the measured results, comparison baseline, and relevant measurement conditions so the reasoning remains verifiable.

Treat questions about reasons (e.g., "Why did you do X?") as requests for explanation only. Do not modify code or files in response unless the user explicitly requests a change.

# Python guidelines

- Use Python 3.12, the standard `src/tenkai` package layout, and uv with a committed lockfile.
- Use absolute imports for package references (e.g. `from tenkai.dag import Dag`), including in `__init__.py`.
- Prefer simple, explicit control flow. Give intermediate results meaningful names; avoid nested expressions and dense one-liners.
- Use ordinary dataclasses for records and concrete types. Add abstractions only when they serve an existing need.
- Annotate functions and data structures. Keep the compiler and its tooling passing strict mypy checks.
- Validate external input at the boundary and raise useful exceptions. Do not hide errors with defaults or broad exception handling.
- Preserve arithmetic order, precision, and deterministic generated output when refactoring the compiler.
- Add dependencies only when they meaningfully simplify the solution; manage them through uv.
- After relevant changes, run applicable pytest tests, Ruff formatting/lint checks, and mypy. Fix issues at their cause.
- CUDA compilation tests use `--run-cuda`; GPU execution tests use `--run-gpu`. Generated-source regression tests run without a GPU toolkit.

# Structure and testing

- Organize code around cohesive responsibilities. Make high-level control flow, data flow, and dependencies clear without requiring every implementation detail to be read.
- Use blank lines to separate meaningful groups within functions, such as validation, preparation, computation, and result handling. Flag uninterrupted blocks that obscure these boundaries during review; keep closely related statements together rather than separating every statement.
- Use modules, types, methods, and free functions where they naturally fit. Prefer clarity over brevity; avoid excessive fragmentation, indirection, and object-oriented abstractions without a concrete need.
- As features evolve, proactively consolidate related functions, options, and duplicated logic instead of accumulating variants; do not wait for a separate refactoring request.
- When refactoring, preserve behavior and public APIs unless a change is requested. Add concise comments for non-obvious design choices and invariants.
- Cover each kernel and supported precision. Reuse test cases and execution harnesses instead of duplicating setup, compilation, and comparison logic; keep test intent separate from execution mechanics.
