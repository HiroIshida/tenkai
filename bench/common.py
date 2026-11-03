"""Compiler selection shared by the CUDA benchmarks."""

import hashlib
import sys
from pathlib import Path

import rkern
from rkern import Options
from rkern.build import copy_rrt_bindings as copy_rrt_bindings
from rkern.build import toolchain as toolchain


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def codegen_command(output: Path, options: Options) -> list[str]:
    command = [sys.executable, str(Path(__file__).with_name("generate.py"))]
    if options.math == "f64":
        command.append("--double")
    elif options.math == "f32":
        command.extend(["--float", "--no-sfu"])
    else:
        command.extend(["--float", "--sfu"])
    command.extend(["--optimization", options.optimization])
    if options.live_budget is not None:
        command.extend(["--live-budget", str(options.live_budget)])
    command.append(str(output))
    return command


def compiler_sources() -> list[Path]:
    package = Path(rkern.__file__).parent
    return sorted(
        path for path in package.rglob("*") if path.suffix in (".py", ".cu", ".cuh", ".hpp")
    )
