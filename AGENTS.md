# Writing guidelines

Code should explain how. Tests should specify what. Commit messages should explain why. Code comments should explain why not.
Unless explicitly requested, avoid unnecessary explanations or extra content.

Treat questions about reasons (e.g., "Why did you do X?") as requests for explanation only. Do not modify code or files in response unless the user explicitly requests a change.

# Rust guidelines

- Prefer straightforward, idiomatic Rust and concrete types. Add traits, generics, macros, or wrapper types only when they serve a concrete need; avoid speculative abstractions.
- Design ownership and borrowing deliberately. Do not add `clone()`, `Rc`, `Arc`, or interior mutability just to silence the borrow checker. Use them when their semantics are needed.
- Accept borrowed inputs such as `&str` and `&[T]` when ownership is unnecessary. Avoid needless allocations and conversions, but keep optimization driven by actual requirements or measurements.
- Use `Result` and `?` for recoverable failures. Do not silently discard errors or substitute defaults that hide failures. Add useful context at error boundaries.
- Avoid `unwrap()`, `expect()`, and panics for expected failures in production code. Reserve them for tests or established invariants; make the invariant clear when it is not obvious.
- Use enums, `Option`, and appropriate types to express valid states. Avoid sentinel values, stringly typed states, and combinations of booleans that permit invalid states.
- Prefer safe Rust. Introduce `unsafe` only for a demonstrated need, keep it small, and document its safety invariants and why safe alternatives are insufficient.
- Keep visibility and mutable state as narrow as practical. Do not introduce async execution, shared state, or locks without a requirement that justifies them.
- Add dependencies only when they meaningfully simplify the solution. Follow the project's Rust edition, minimum supported Rust version, and existing conventions.
- After relevant changes, run `cargo fmt`, the applicable tests, and `cargo clippy` with the project's supported features and targets. Fix warnings at their cause; do not add broad lint suppressions to make checks pass.

# Structure and testing

- Organize code around cohesive responsibilities. Make high-level control flow, data flow, and dependencies clear without requiring every implementation detail to be read.
- Use modules, types, methods, and free functions where they naturally fit. Prefer clarity over brevity; avoid excessive fragmentation, indirection, and object-oriented abstractions without a concrete need.
- When refactoring, preserve behavior and public APIs unless a change is requested. Add concise comments for non-obvious design choices and invariants.
- Cover each kernel and supported precision. Reuse test cases and execution harnesses instead of duplicating setup, compilation, and comparison logic; keep test intent separate from execution mechanics.
