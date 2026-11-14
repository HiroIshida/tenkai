"""Generated kernels compile and preserve the existing numeric/planner checks."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tenkai import (
    FkDag,
    Options,
    Robot,
    Sphere,
    generate_centers_kernel,
    generate_collision_kernel,
    generate_fk_kernel,
    generate_rrt_kernel,
)
from tenkai.build import Backend, compiler_command
from tenkai.cuda import MathMode
from tenkai.dag import Optimization

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def chain_sources(directory: Path, options: Options) -> None:
    fk = FkDag.from_robot(Robot.from_string((FIXTURES / "fk_chain.urdf").read_text()))
    tool = fk.link_names.index("tool")
    spheres = [Sphere(tool, (0, 0, 0), 0.25)]
    (directory / "fk.cuh").write_text(generate_fk_kernel(fk, tool, options))
    (directory / "centers.cuh").write_text(generate_centers_kernel(fk, spheres, options))
    (directory / "collision.cuh").write_text(generate_collision_kernel(fk, spheres, options))
    tolerance = "1e-12" if options.math == "f64" else "1e-5"
    (directory / "config.h").write_text(
        f"using Scalar = {options.scalar};\nnamespace sdf = {options.namespace};\n"
        f"constexpr double tolerance = {tolerance};\n"
    )


def planner_source(directory: Path, options: Options) -> None:
    robot = '<robot name="cartesian"><link name="root"/>'
    parent = "root"
    for name, axis in [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]:
        robot += f'<link name="{name}"/><joint name="{name}" type="prismatic">'
        robot += f'<parent link="{parent}"/><child link="{name}"/><axis xyz="{axis}"/></joint>'
        parent = name
    robot += "</robot>"
    fk = FkDag.from_robot(Robot.from_string(robot))
    spheres = [Sphere(3, (0, 0, 0), 0.05), Sphere(3, (0, 0, 0.025), 0.025)]
    (directory / "rrt.cuh").write_text(generate_rrt_kernel(fk, spheres, [1, 0], options))


@pytest.mark.parametrize("math_mode", ["f32", "f64"])
@pytest.mark.parametrize("optimization", ["none", "basic", "algebraic"])
@pytest.mark.parametrize("budget", [None, 64])
def test_host_arithmetic(
    tmp_path: Path, math_mode: MathMode, optimization: Optimization, budget: int | None
) -> None:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("host arithmetic requires a C++ compiler")
    options = Options(math_mode, optimization, budget)
    chain_sources(tmp_path, options)
    binary = tmp_path / "arithmetic"
    run(
        [
            compiler,
            "-std=c++14",
            "-O2",
            "-I",
            str(tmp_path),
            "-I",
            str(FIXTURES),
            str(FIXTURES / "python_host.cpp"),
            "-o",
            str(binary),
        ]
    )
    run([str(binary)])


@pytest.mark.gpu_compile
@pytest.mark.parametrize("math_mode", ["f32", "f32-sfu", "f64"])
def test_all_kernels_compile(
    tmp_path: Path, math_mode: MathMode, gpu_backend: Backend, gpu_arch: str
) -> None:
    options = Options(math_mode)
    chain_sources(tmp_path, options)
    planner_source(tmp_path, options)
    arithmetic = tmp_path / "arithmetic.cu"
    arithmetic.write_text('#include "fk.cuh"\n#include "centers.cuh"\n#include "collision.cuh"\n')
    planner = tmp_path / "planner.cu"
    planner.write_text('#include "rrt.cuh"\n')
    command, _ = compiler_command(gpu_backend, gpu_arch)
    compile_only = "-c" if gpu_backend == "hip" else "--ptx"
    for source in (arithmetic, planner):
        run([*command, compile_only, str(source), "-o", str(source.with_suffix(".o"))])


@pytest.mark.gpu
@pytest.mark.parametrize("math_mode", ["f32", "f32-sfu", "f64"])
def test_arithmetic_on_gpu(
    tmp_path: Path, math_mode: MathMode, gpu_backend: Backend, gpu_arch: str
) -> None:
    chain_sources(tmp_path, Options(math_mode))
    command, _ = compiler_command(gpu_backend, gpu_arch)
    if gpu_backend == "cuda":
        command.extend(["-x", "cu"])
    binary = tmp_path / "arithmetic"
    run(
        [
            *command,
            "-I",
            str(tmp_path),
            "-I",
            str(FIXTURES),
            str(FIXTURES / "python_host.cpp"),
            "-o",
            str(binary),
        ]
    )
    run([str(binary)])


@pytest.mark.gpu
@pytest.mark.parametrize("math_mode", ["f32", "f32-sfu", "f64"])
def test_planner_on_gpu(
    tmp_path: Path, math_mode: MathMode, gpu_backend: Backend, gpu_arch: str
) -> None:
    options = Options(math_mode)
    planner_source(tmp_path, options)
    binary = tmp_path / "rrt-test"
    command, _ = compiler_command(gpu_backend, gpu_arch)
    if math_mode == "f32":
        command.append("-DTEST_STANDARD_FP32")
    run(
        [
            *command,
            "-DTENKAI_MODEL_NAMESPACE=" + options.namespace,
            "-I",
            str(tmp_path),
            "-I",
            str(FIXTURES),
            str(FIXTURES / "rrt.cu"),
            "-o",
            str(binary),
        ]
    )
    run([str(binary)])
