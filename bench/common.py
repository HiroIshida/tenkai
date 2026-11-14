"""Compiler selection shared by the CUDA benchmarks."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import tenkai
from tenkai import Options


def toolchain() -> tuple[str, str, str]:
    nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
    if nvcc is None:
        raise RuntimeError("nvcc not found; install the CUDA toolkit or set NVCC")
    version = subprocess.check_output([nvcc, "--version"], text=True)
    release = re.search(r"release (\d+)\.", version)
    compiler = os.environ.get("CXX")
    # CUDA 10 rejects the newer default GCC shipped on the original benchmark host.
    if compiler is None:
        compiler = "g++-8" if release and int(release[1]) == 10 else "c++"
    compiler = shutil.which(compiler)
    if compiler is None:
        raise RuntimeError("CUDA host compiler not found; set CXX to a compiler supported by nvcc")
    return nvcc, compiler, version


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
    package = Path(tenkai.__file__).parent
    return sorted(path for path in package.rglob("*") if path.suffix in (".py", ".cuh", ".hpp"))
