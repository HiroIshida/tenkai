"""GPU compiler discovery and caching of compiled libraries."""

import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, NotRequired, TypedDict

from tenkai.cuda import TEMPLATES

logger = logging.getLogger(__name__)

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


@dataclass
class _CompileCommand:
    compiler: list[str]
    directory: Path
    source: Path
    output: Path
    max_registers: int | None

    def arguments(
        self,
        *,
        output: str | Path | None = None,
        path_roots: tuple[tuple[Path, Path], ...] = (),
    ) -> list[str]:
        arguments = [*self.compiler, "-I", str(self.directory), str(self.source)]
        for index, argument in enumerate(arguments):
            path = Path(argument)
            for root, replacement in path_roots:
                if path.is_relative_to(root):
                    arguments[index] = (replacement / path.relative_to(root)).as_posix()
                    break

        arguments.extend(["-o", str(self.output if output is None else output)])
        if self.max_registers:
            arguments.append("--maxrregcount=" + str(self.max_registers))
        return arguments


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


def _library_cache_path(sources: str, command: _CompileCommand, version: str) -> Path:
    normalized_command = command.arguments(
        output="<output>",
        path_roots=((TEMPLATES, Path("<templates>")), (command.directory, Path("."))),
    )

    host_version = ""
    if "-ccbin" in command.compiler:
        host = command.compiler[command.compiler.index("-ccbin") + 1]
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


def _export_sources(directory: Path, output: Path) -> None:
    if directory != output:
        for path in list(directory.rglob("*")):
            if not path.is_file():
                continue
            # A nested output directory must not be copied into itself.
            if output.is_relative_to(directory) and path.is_relative_to(output):
                continue
            destination = output / path.relative_to(directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)


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
    compiler, version = compiler_command(options.backend, options.arch)
    compiler_name = "hipcc" if options.backend == "hip" else "nvcc"
    compiler.extend(["-O3", "-shared"])
    if options.backend == "cuda":
        compiler.append("-Xcompiler")
    compiler.append("-fPIC")
    command = _CompileCommand(compiler, directory, source, library, options.max_registers)

    library.parent.mkdir(parents=True, exist_ok=True)
    cached = None
    if cache:
        sources = _source_digest(directory, options.backend)
        cached = _library_cache_path(sources, command, version)
    _export_sources(directory, library.parent)
    exported_command = command.arguments(path_roots=((directory, library.parent),))
    seconds, hit = _compile_or_restore(command, compiler_name, cached)

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
    command: _CompileCommand, compiler_name: str, cached: Path | None
) -> tuple[float, bool]:
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        with cached.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return _build_library(command, compiler_name, cached)
    return _build_library(command, compiler_name, cached)


def _build_library(
    command: _CompileCommand, compiler_name: str, cached: Path | None
) -> tuple[float, bool]:
    log_path = command.output.parent / "compile.log"
    target = command.output.resolve()
    with TemporaryDirectory(prefix=".tenkai-", dir=target.parent) as temporary:
        built = Path(temporary) / command.output.name
        if cached is not None and cached.is_file():
            shutil.copyfile(cached, built)
            log_path.write_text("Compilation cache hit.\n")
            seconds, hit = 0.0, True
            logger.info("Reusing cached GPU kernel: %s", cached)
        else:
            logger.info("Compiling GPU kernel with %s (cache disabled).", compiler_name)
            seconds = _run_compiler(command, compiler_name, built, log_path)
            hit = False
            if cached is not None:
                with TemporaryDirectory(prefix=".tenkai-", dir=cached.parent) as staging:
                    stored = Path(staging) / command.output.name
                    shutil.copyfile(built, stored)
                    stored.replace(cached)
            logger.info("Compilation finished in %.2f s.", seconds)

        built.replace(target)
    return seconds, hit


def _run_compiler(
    command: _CompileCommand, compiler_name: str, output: Path, log_path: Path
) -> float:
    invocation = command.arguments(output=output)
    with log_path.open("w") as log:
        begin = time.perf_counter()
        result = subprocess.run(
            invocation, cwd=command.directory, stdout=log, stderr=subprocess.STDOUT
        )
        seconds = time.perf_counter() - begin
    if result.returncode:
        raise RuntimeError(f"{compiler_name} failed; see {log_path}\n{log_path.read_text()}")
    return seconds


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
