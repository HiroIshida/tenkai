"""GPU compiler discovery and caching of compiled libraries."""

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, NotRequired, TypedDict

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


@dataclass(frozen=True)
class CompileOptions:
    backend: Backend = "cuda"
    arch: str | None = None
    max_registers: int | None = None

    def __post_init__(self) -> None:
        target_arch(self.backend, self.arch)
        registers = self.max_registers or 0
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
    cache_hit: bool


def _source_digest(directory: Path, backend: Backend) -> str:
    files = [
        (path.relative_to(directory).as_posix(), path)
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".cu", ".cuh", ".h", ".hpp"}
    ]
    if backend == "hip":
        for name in ("amd_hip_compat.h", "cuda_runtime.h"):
            files.append(("hip/" + name, TEMPLATES / name))

    contents = [
        (name, hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in sorted(files)
    ]
    return hashlib.sha256(json.dumps(contents).encode()).hexdigest()


def _library_cache_path(sources: str, command: list[str], version: str, directory: Path) -> Path:
    output_index = command.index("-o") + 1
    normalized_command: list[str] = []
    for index, argument in enumerate(command):
        path = Path(argument)
        if index == output_index:
            argument = "<output>"
        elif path.is_relative_to(TEMPLATES):
            argument = (Path("<templates>") / path.relative_to(TEMPLATES)).as_posix()
        elif path.is_relative_to(directory):
            argument = path.relative_to(directory).as_posix()
        normalized_command.append(argument)

    host_version = ""
    if "-ccbin" in command:
        host = command[command.index("-ccbin") + 1]
        host_version = subprocess.check_output([host, "--version"], text=True)

    environment = {
        name: os.environ.get(name)
        for name in (
            "NVCC_PREPEND_FLAGS",
            "NVCC_APPEND_FLAGS",
            "HIPCC_COMPILE_FLAGS_APPEND",
            "HIPCC_LINK_FLAGS_APPEND",
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "LIBRARY_PATH",
        )
    }
    inputs = [
        sources,
        normalized_command,
        version,
        host_version,
        environment,
    ]
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "tenkai" / (key + ".so")


def _export_sources(directory: Path, library: Path, command: list[str]) -> list[str]:
    output = library.parent
    if directory != output:
        for path in list(directory.rglob("*")):
            if not path.is_file():
                continue
            if output.is_relative_to(directory) and path.is_relative_to(output):
                continue
            destination = output / path.relative_to(directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)

    exported_command = command.copy()
    output_index = command.index("-o") + 1
    for index, argument in enumerate(command):
        path = Path(argument)
        if index == output_index:
            exported_command[index] = str(library)
        elif path.is_relative_to(directory):
            exported_command[index] = str(output / path.relative_to(directory))
    return exported_command


def compile_library(
    directory: Path,
    source: str | Path,
    library: Path,
    *,
    options: CompileOptions = CompileOptions(),
    cache: bool = True,
) -> Compilation:
    """Compile sources and preserve them beside the output library.

    source is relative to directory; library is the requested output path.
    Only native source/header contents and compiler settings determine reuse;
    model metadata is excluded.
    The returned command can be rerun from the output directory after temporary
    sources are removed.
    """
    directory = directory.resolve()
    source = (directory / source).resolve()
    if not source.is_relative_to(directory):
        raise ValueError("source must be within the generated source directory")
    library = library.absolute()
    command, version = compiler_command(options.backend, options.arch)
    compiler_name = "hipcc" if options.backend == "hip" else "nvcc"
    command.extend(["-O3", "-shared"])
    if options.backend == "cuda":
        command.append("-Xcompiler")
    command.extend(["-fPIC", "-I", str(directory), str(source), "-o", str(library)])
    if options.max_registers:
        command.append("--maxrregcount=" + str(options.max_registers))

    library.parent.mkdir(parents=True, exist_ok=True)
    cached = None
    if cache:
        sources = _source_digest(directory, options.backend)
        cached = _library_cache_path(sources, command, version, directory)
    exported_command = _export_sources(directory, library, command)
    seconds, hit = _compile_or_restore(directory, library, command, compiler_name, cached)

    info = BuildInfo(
        backend=options.backend,
        compile_command=exported_command,
        max_registers=options.max_registers or None,
    )
    if options.backend == "hip":
        info["hipcc"] = version
    else:
        info["nvcc"] = version
    return Compilation(library, info, compiler_name, seconds, hit)


def _compile_or_restore(
    directory: Path, library: Path, command: list[str], compiler_name: str, cached: Path | None
) -> tuple[float, bool]:
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
    lock_context = cached.with_suffix(".lock").open("a") if cached else nullcontext()
    with lock_context as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)

        log_path = library.parent / "compile.log"
        target = library.resolve()
        with TemporaryDirectory(prefix=".tenkai-", dir=target.parent) as temporary:
            built = Path(temporary) / library.name
            hit = cached is not None and cached.is_file()
            seconds = 0.0
            if hit:
                assert cached is not None
                shutil.copyfile(cached, built)
                log_path.write_text("Compilation cache hit.\n")
            else:
                invocation = command.copy()
                invocation[invocation.index("-o") + 1] = str(built)
                with log_path.open("w") as log:
                    begin = time.perf_counter()
                    result = subprocess.run(
                        invocation, cwd=directory, stdout=log, stderr=subprocess.STDOUT
                    )
                    seconds = time.perf_counter() - begin
                if result.returncode:
                    raise RuntimeError(
                        f"{compiler_name} failed; see {log_path}\n{log_path.read_text()}"
                    )

                if cached is not None:
                    with TemporaryDirectory(prefix=".tenkai-", dir=cached.parent) as staging:
                        stored = Path(staging) / library.name
                        shutil.copyfile(built, stored)
                        stored.replace(cached)

            built.replace(target)
    return seconds, hit


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
