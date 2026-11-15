"""GPU compiler discovery and packaged build sources."""

import os
import re
import shutil
import subprocess
from typing import Literal

from tenkai.cuda import TEMPLATES

type Backend = Literal["cuda", "hip"]


def target_arch(backend: Backend, arch: str | None) -> str:
    if backend not in ("cuda", "hip"):
        raise ValueError("backend must be cuda or hip")
    if arch is None:
        return "gfx1201" if backend == "hip" else "compute_75"

    pattern = r"gfx[0-9a-f]+" if backend == "hip" else r"compute_[0-9]+"
    if re.fullmatch(pattern, arch) is None:
        example = "gfx1201" if backend == "hip" else "compute_75"
        raise ValueError(f"invalid {backend} arch; expected an architecture such as {example}")

    return arch


def compiler_command(backend: Backend = "cuda", arch: str | None = None) -> tuple[list[str], str]:
    arch = target_arch(backend, arch)
    if backend == "cuda":
        nvcc, host, version = toolchain()
        return [nvcc, "-ccbin", host, "-std=c++14", "-arch=" + arch, "-code=" + arch], version

    hipcc = shutil.which(os.environ.get("HIPCC", "hipcc"))
    if hipcc is None:
        raise RuntimeError("hipcc not found; install ROCm or set HIPCC")

    version = subprocess.check_output([hipcc, "--version"], text=True)
    command = [
        hipcc,
        "-x",
        "hip",
        "-std=c++14",
        "--offload-arch=" + arch,
        # Workspace bounds need directed rounding, not the default nearest-only intrinsics.
        "-DOCML_BASIC_ROUNDED_OPERATIONS",
        "-I",
        str(TEMPLATES),
        "-include",
        str(TEMPLATES / "amd_hip_compat.h"),
    ]
    return command, version


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
