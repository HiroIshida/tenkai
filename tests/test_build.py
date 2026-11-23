"""Compiled libraries are reused only when their build inputs match."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from helpers import FakeCompiler

from tenkai.build import Backend, CompileOptions, compile_library

pytestmark = pytest.mark.parametrize("backend", ["cuda", "hip"])


def kernel_sources(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "kernel.cu").write_text('#include "kernel.cuh"\n')
    (directory / "kernel.cuh").write_text("__global__ void kernel() {}\n")


def test_cache_reuses_library_across_output_directories(
    tmp_path: Path, fake_compiler: FakeCompiler, backend: Backend
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    kernel_sources(first)
    kernel_sources(second)
    options = CompileOptions(backend=backend)
    first_build = compile_library(first, "kernel.cu", first / "libkernel.so", options=options)
    second_build = compile_library(second, "kernel.cu", second / "libkernel.so", options=options)

    assert not first_build.cache_hit and second_build.cache_hit
    assert first_build.seconds > 0 and second_build.seconds == 0
    assert len(fake_compiler.calls) == 1
    assert (first / "libkernel.so").read_bytes() == (second / "libkernel.so").read_bytes()


def test_build_preserves_sources_and_command(
    tmp_path: Path, fake_compiler: FakeCompiler, backend: Backend
) -> None:
    output = tmp_path / "output"
    options = CompileOptions(backend=backend)
    with TemporaryDirectory(dir=tmp_path) as temporary:
        directory = Path(temporary) / "sources"
        kernel_sources(directory)
        source = (directory / "kernel.cu").read_bytes()
        header = (directory / "kernel.cuh").read_bytes()
        compiled = compile_library(directory, "kernel.cu", output / "libkernel.so", options=options)

    assert compiled.library == output / "libkernel.so"
    assert (output / "kernel.cu").read_bytes() == source
    assert (output / "kernel.cuh").read_bytes() == header
    command = compiled.toolchain["compile_command"]
    assert str(output / "kernel.cu") in command
    assert command[command.index("-I") + 1] == str(output)

    replayed = compile_library(output, "kernel.cu", output / "libkernel.so", options=options)
    assert replayed.cache_hit
    assert len(fake_compiler.calls) == 1


def test_build_preserves_library_symlink(
    tmp_path: Path, fake_compiler: FakeCompiler, backend: Backend
) -> None:
    directory = tmp_path / "build"
    kernel_sources(directory)
    target = tmp_path / "target.so"
    target.write_bytes(b"previous library")
    library = directory / "libkernel.so"
    library.symlink_to(target)
    options = CompileOptions(backend=backend)

    for expected_hit in (False, True):
        compiled = compile_library(directory, "kernel.cu", library, options=options)
        assert compiled.cache_hit == expected_hit
        assert compiled.library == library
        assert library.is_symlink()
        assert target.read_bytes() == b"compiled library"

    command = compiled.toolchain["compile_command"]
    assert command[command.index("-o") + 1] == str(library)
    assert (directory / "compile.log").is_file()
    assert len(fake_compiler.calls) == 1


@pytest.mark.parametrize("changed", ["source", "header", "options", "version"])
def test_cache_invalidates_changed_inputs(
    tmp_path: Path, fake_compiler: FakeCompiler, backend: Backend, changed: str
) -> None:
    directory = tmp_path / "build"
    kernel_sources(directory)
    options = CompileOptions(backend=backend)
    library = directory / "libkernel.so"
    compile_library(directory, "kernel.cu", library, options=options)

    if changed == "source":
        (directory / "kernel.cu").write_text('__global__ void other() {}\n#include "kernel.cuh"\n')
    elif changed == "header":
        (directory / "kernel.cuh").write_text("__global__ void renamed() {}\n")
    elif changed == "options":
        arch = "compute_80" if backend == "cuda" else "gfx1100"
        options = CompileOptions(backend=backend, arch=arch)
    else:
        fake_compiler.version = "version 2"
    compiled = compile_library(directory, "kernel.cu", library, options=options)

    assert not compiled.cache_hit
    assert len(fake_compiler.calls) == 2


def test_cache_can_be_bypassed(
    tmp_path: Path, fake_compiler: FakeCompiler, backend: Backend
) -> None:
    directory = tmp_path / "build"
    kernel_sources(directory)
    options = CompileOptions(backend=backend)
    hits: list[bool] = []
    for enabled in (False, True, True, False):
        compiled = compile_library(
            directory, "kernel.cu", directory / "libkernel.so", options=options, cache=enabled
        )
        hits.append(compiled.cache_hit)

    assert hits == [False, False, True, False]
    assert len(fake_compiler.calls) == 3


def test_failed_compilation_is_retried(
    tmp_path: Path, fake_compiler: FakeCompiler, backend: Backend
) -> None:
    directory = tmp_path / "build"
    kernel_sources(directory)
    options = CompileOptions(backend=backend)
    library = directory / "libkernel.so"
    library.write_bytes(b"previous library")

    fake_compiler.fail = True
    with pytest.raises(RuntimeError, match="fake compiler error"):
        compile_library(directory, "kernel.cu", library, options=options)
    assert library.read_bytes() == b"previous library"

    fake_compiler.fail = False
    compiled = compile_library(directory, "kernel.cu", library, options=options)
    assert not compiled.cache_hit
    assert len(fake_compiler.calls) == 2
    assert library.read_bytes() == b"compiled library"


def test_build_records_compiler_settings(
    tmp_path: Path, fake_compiler: FakeCompiler, backend: Backend
) -> None:
    directory = tmp_path / "build"
    kernel_sources(directory)
    arch = "compute_80" if backend == "cuda" else "gfx1100"
    max_registers = 128 if backend == "cuda" else None
    options = CompileOptions(backend=backend, arch=arch, max_registers=max_registers)
    compiled = compile_library(directory, "kernel.cu", directory / "libkernel.so", options=options)

    info = compiled.toolchain
    command = info["compile_command"]
    assert info["backend"] == backend
    assert info["max_registers"] == max_registers
    assert "-arch=" + arch in command
    assert "-O3" in command and "-shared" in command and "-fPIC" in command
    if backend == "cuda":
        assert compiled.compiler_name == "nvcc"
        assert info["nvcc"] == fake_compiler.version
        assert "-Xcompiler" in command and "--maxrregcount=128" in command
    else:
        assert compiled.compiler_name == "hipcc"
        assert info["hipcc"] == fake_compiler.version
        assert "-Xcompiler" not in command
