"""CUDA compiler discovery and packaged build sources."""

import os
import re
import shutil
import subprocess
from pathlib import Path

from .cuda import TEMPLATES


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


def copy_rrt_bindings(directory: Path) -> Path:
    # Compile beside the generated rrt.cuh, not the adjacent source template.
    source = directory / "rrt.cu"
    shutil.copyfile(TEMPLATES / "cuda/rrt_bindings.cu", source)
    return source
