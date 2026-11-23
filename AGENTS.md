# Writing guidelines

Code should explain how. Tests should specify what. Commit messages should explain why. Code comments should explain why not.
Unless explicitly requested, avoid unnecessary explanations or extra content.

Markdown documentation is for users. Do not use Markdown files to record internal technical details; put technical details relevant to a change in its commit message.

When copying or porting external code, reimplementing an external algorithm, or carrying over parameters/defaults, add or update a concise row in [PORTING.md](PORTING.md). Use the existing columns: upstream repository, license, local files, and porting method. Keep the upstream repository first so its authors can quickly recognize reuse.

Do not commit benchmark code unless the user explicitly asks to include it in a commit. Running or writing a benchmark does not imply permission to commit it.

Keep commit messages concise and explain why the change is needed. A subject line is sufficient when it conveys that reason; otherwise, add a short body.

Add a short body only for significant reasons or trade-offs that are not apparent from the diff. Preserve relevant decisions from user discussions without recounting the discussion.

Summarize routine validation in at most one line. Include measured results, the comparison baseline, and measurement conditions when they justify a design decision.

Do not include implementation walkthroughs, work logs, exhaustive checklists, or lists of unchanged behavior.

Treat questions about reasons (e.g., "Why did you do X?") as requests for explanation only. Do not modify code or files in response unless the user explicitly requests a change.

# Python guidelines

- Use Python 3.12, the standard `src/tenkai` package layout, and uv with a committed lockfile.
- Use absolute imports for package references (e.g. `from tenkai.dag import Dag`), including in `__init__.py`.
- Prefer simple, explicit control flow. Name an intermediate result only when the name explains something the expression does not; do not name every step.
- Use ordinary dataclasses for records and concrete types. Add abstractions only when they serve an existing need.
- Annotate functions and data structures. Keep the compiler and its tooling passing strict mypy checks.
- Give distinct kinds of integer IDs (node, link, joint, sphere) distinct types (`NewType`) instead of bare `int`.
- Validate external input once, at the outermost boundary where it enters, and raise useful exceptions. Inner layers trust validated values; do not repeat the same check in each layer. Do not hide errors with defaults or broad exception handling.
- Do not add a check for an input or state that cannot occur through the public API. Do not add a guard for a failure you have not reproduced; if one is needed, put the reproducer in a test and keep the comment to one line of why.
- Return results as values. Do not write a file only for another part of the same process to read it back.
- Keep timing and logging out of core logic; collect them in one place at the call boundary or drop them.
- Preserve arithmetic order, precision, and deterministic generated output when refactoring the compiler.
- Add dependencies only when they meaningfully simplify the solution; manage them through uv.
- After relevant changes, run applicable pytest tests, Ruff formatting/lint checks, and mypy. Fix issues at their cause.
- CUDA compilation tests use `--run-cuda`; GPU execution tests use `--run-gpu`. Generated-source regression tests run without a GPU toolkit.

# Structure and testing

- Organize code around cohesive responsibilities. Make high-level control flow, data flow, and dependencies clear without requiring every implementation detail to be read.
- Write for a reader who needs the main idea first. Put the core algorithm and public entry points at the top of a module, with edge-case handling and helpers below; keep the core dense and plain, and keep boilerplate short. Not every part deserves equal weight.
- Use blank lines to separate meaningful groups within functions, not every statement.
- Use modules, types, methods, and free functions where they naturally fit. Prefer clarity over brevity; avoid excessive fragmentation, indirection, and object-oriented abstractions without a concrete need.
- Deletion is part of every change. Before adding an argument, flag, option, or check, try to remove or merge an existing one. Do not thread a new `bool` through several functions; split the function or introduce a type for the variants.
- When a feature adds a variant to an existing function, consolidate the variants in the same change rather than accumulating them.
- Keep public APIs small. Export only what users need; internal and test-only helpers are not exported from `__init__.py`.
- Keep code generation in one place. Do not split one generated artifact between string concatenation in Python and a template; prefer templates for fixed text and fill only the variable parts.
- Comments describe the current design, not its history. Do not record past bugs, reviews, or fixes in comments; that belongs in commit messages and tests.
- After a refactor, reread the changed modules from the top as a new reader and fix anything that needs an explanation to follow.
- When refactoring, preserve behavior and public APIs unless a change is requested. Add concise comments for non-obvious design choices and invariants.
- Cover each kernel and supported precision. Reuse test cases and execution harnesses instead of duplicating setup, compilation, and comparison logic; keep test intent separate from execution mechanics.
