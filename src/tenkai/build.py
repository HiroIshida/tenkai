"""GPU compiler discovery and shared-library builds."""

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from tenkai.cuda import TEMPLATES
from tenkai.validation import check_count

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


@dataclass(frozen=True)
class CompileOptions:
    backend: Backend = "cuda"
    arch: str | None = None
    max_registers: int | None = None

    def __post_init__(self) -> None:
        target_arch(self.backend, self.arch)
        registers = 0 if self.max_registers is None else self.max_registers
        check_count(registers, "max_registers", 0)
        if registers != 0 and not 32 <= registers <= 255:
            raise ValueError("max_registers must be 0 or between 32 and 255")
        if self.backend == "hip" and registers:
            raise ValueError("max_registers is only supported by the CUDA backend")


class BuildInfo(TypedDict):
    nvcc: NotRequired[str]
    hipcc: NotRequired[str]
    backend: NotRequired[Backend]
    compile_command: list[str]
    max_registers: int | None


@dataclass
class Compilation:
    library: Path
    toolchain: BuildInfo
    compiler_name: str
    seconds: float


def compile_library(
    directory: Path,
    source: str | Path,
    library: Path,
    *,
    options: CompileOptions = CompileOptions(),
) -> Compilation:
    """Compile source relative to directory into library on every call."""
    directory = directory.resolve()
    source = directory / source
    command, version = compiler_command(options.backend, options.arch)
    compiler_name = "hipcc" if options.backend == "hip" else "nvcc"
    command.extend(["-O3", "-shared"])
    if options.backend == "cuda":
        command.append("-Xcompiler")
    command.extend(["-fPIC", "-I", str(directory), str(source), "-o", str(library)])
    if options.max_registers:
        command.append("--maxrregcount=" + str(options.max_registers))

    log_path = library.parent / "compile.log"
    with log_path.open("w") as log:
        begin = time.perf_counter()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        seconds = time.perf_counter() - begin
    if result.returncode:
        raise RuntimeError(f"{compiler_name} failed; see {log_path}\n{log_path.read_text()}")

    info = BuildInfo(
        backend=options.backend,
        compile_command=command,
        max_registers=options.max_registers or None,
    )
    if options.backend == "hip":
        info["hipcc"] = version
    else:
        info["nvcc"] = version
    return Compilation(library, info, compiler_name, seconds)


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
