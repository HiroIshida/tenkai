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
    generate_rrt_sources,
)
from tenkai.build import Backend, compiler_command
from tenkai.cuda import MathMode
from tenkai.dag import Optimization

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def write_sources(directory: Path, files: dict[str, str]) -> None:
    for name, contents in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


def chain_sources(directory: Path, options: Options) -> None:
    fk = FkDag.from_robot(Robot.from_string((FIXTURES / "fk_chain.urdf").read_text()))
    tool = fk.link_names.index("tool")
    spheres = [Sphere(tool, (0, 0, 0), 0.25)]
    (directory / "fk.cuh").write_text(generate_fk_kernel(fk, tool, options))
    (directory / "centers.cuh").write_text(generate_centers_kernel(fk, spheres, options))
    (directory / "collision.cuh").write_text(generate_collision_kernel(fk, spheres, options))
    (directory / "config.h").write_text(
        "using Scalar = float;\nnamespace sdf = tenkai_f32;\nconstexpr double tolerance = 1e-5;\n"
    )


def planner_source(directory: Path, options: Options, controls: tuple[int, ...] = (1, 0)) -> None:
    robot = '<robot name="cartesian"><link name="root"/>'
    parent = "root"
    for name, axis in [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]:
        robot += f'<link name="{name}"/><joint name="{name}" type="prismatic">'
        robot += f'<parent link="{parent}"/><child link="{name}"/><axis xyz="{axis}"/></joint>'
        parent = name
    robot += "</robot>"
    fk = FkDag.from_robot(Robot.from_string(robot))
    spheres = [Sphere(3, (0, 0, 0), 0.05), Sphere(3, (0, 0, 0.025), 0.025)]
    write_sources(directory, generate_rrt_sources(fk, spheres, controls, options))


def static_planner_source(directory: Path, options: Options, sphere_count: int) -> None:
    robot = Robot.from_string("""<robot name="fixed_spheres">
      <link name="root"/><link name="arm"/><link name="slide"/><link name="lift"/>
      <joint name="spin" type="revolute"><parent link="lift"/><child link="arm"/>
        <axis xyz="0 0 1"/></joint>
      <joint name="slide" type="prismatic"><parent link="root"/><child link="slide"/>
        <origin xyz="5 0 0"/><axis xyz="1 0 0"/></joint>
      <joint name="lift" type="prismatic"><parent link="root"/><child link="lift"/>
        <axis xyz="0 0 1"/></joint>
    </robot>""")
    fk = FkDag.from_robot(robot)
    spheres = [
        Sphere(fk.link_names.index("root"), (-5, 0, 0), 0.05),
        Sphere(fk.link_names.index("slide"), (0, 0, 0), 0.05),
        Sphere(fk.link_names.index("lift"), (0, 0, 0.5), 0.05),
        Sphere(fk.link_names.index("arm"), (2, 0, 0), 0.05),
    ]
    write_sources(directory, generate_rrt_sources(fk, spheres[:sphere_count], [0], options))


def run_planner_fixture(
    directory: Path,
    options: Options,
    backend: Backend,
    arch: str,
    fixture: str,
    defines: tuple[str, ...] = (),
) -> None:
    binary = directory / "planner-test"
    command, _ = compiler_command(backend, arch)
    if options.math == "f32":
        command.append("-DTEST_STANDARD_FP32")
    command.extend("-D" + define for define in defines)
    run(
        [
            *command,
            "-DTENKAI_MODEL_NAMESPACE=tenkai_f32",
            "-I",
            str(directory),
            "-I",
            str(FIXTURES),
            str(FIXTURES / fixture),
            "-o",
            str(binary),
        ]
    )
    run([str(binary)])


@pytest.mark.parametrize("optimization", ["none", "basic", "algebraic"])
@pytest.mark.parametrize("budget", [None, 64])
def test_host_arithmetic(tmp_path: Path, optimization: Optimization, budget: int | None) -> None:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("host arithmetic requires a C++ compiler")
    options = Options("f32", optimization, budget)
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
            str(FIXTURES / "arithmetic.cpp"),
            "-o",
            str(binary),
        ]
    )
    run([str(binary)])


@pytest.mark.gpu_compile
@pytest.mark.parametrize("math_mode", ["f32", "f32-sfu"])
def test_all_kernels_compile(
    tmp_path: Path, math_mode: MathMode, gpu_backend: Backend, gpu_arch: str
) -> None:
    options = Options(math_mode)
    chain_sources(tmp_path, options)
    planner_source(tmp_path, options)
    arithmetic = tmp_path / "arithmetic.cu"
    arithmetic.write_text('#include "fk.cuh"\n#include "centers.cuh"\n#include "collision.cuh"\n')
    command, _ = compiler_command(gpu_backend, gpu_arch)
    compile_only = "-c" if gpu_backend == "hip" else "--ptx"
    for source in (arithmetic, tmp_path / "rrt/bindings.cu"):
        run([*command, compile_only, str(source), "-o", str(source.with_suffix(".o"))])


@pytest.mark.gpu_compile
@pytest.mark.parametrize("math_mode", ["f32", "f32-sfu"])
def test_planner_headers_are_self_contained(
    tmp_path: Path, math_mode: MathMode, gpu_backend: Backend, gpu_arch: str
) -> None:
    planner_source(tmp_path, Options(math_mode))
    command, _ = compiler_command(gpu_backend, gpu_arch)
    compile_only = "-c" if gpu_backend == "hip" else "--ptx"
    headers = sorted(path for path in tmp_path.rglob("*") if path.suffix in (".cuh", ".hpp"))
    for header in headers:
        name = header.relative_to(tmp_path).as_posix()
        source = tmp_path / "header.cu"
        source.write_text(f'#include "{name}"\n#include "rrt/planner.cuh"\n#include "{name}"\n')
        run([*command, compile_only, str(source), "-o", str(tmp_path / "header.o")])


@pytest.mark.gpu
@pytest.mark.parametrize("math_mode", ["f32", "f32-sfu"])
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
            str(FIXTURES / "arithmetic.cpp"),
            "-o",
            str(binary),
        ]
    )
    run([str(binary)])


@pytest.mark.gpu
@pytest.mark.parametrize("math_mode", ["f32", "f32-sfu"])
def test_planner_on_gpu(
    tmp_path: Path, math_mode: MathMode, gpu_backend: Backend, gpu_arch: str
) -> None:
    options = Options(math_mode)
    planner_source(tmp_path, options)
    run_planner_fixture(tmp_path, options, gpu_backend, gpu_arch, "rrt.cu")


@pytest.mark.gpu
@pytest.mark.parametrize("math_mode", ["f32", "f32-sfu"])
def test_streaming_and_fallback_paths_on_gpu(
    tmp_path: Path, math_mode: MathMode, gpu_backend: Backend, gpu_arch: str
) -> None:
    options = Options(math_mode)
    planner_source(tmp_path, options, (2, 0, 1))
    run_planner_fixture(tmp_path, options, gpu_backend, gpu_arch, "rrtc_streaming.cu")


@pytest.mark.gpu
@pytest.mark.parametrize("math_mode", ["f32", "f32-sfu"])
@pytest.mark.parametrize("sphere_count", [0, 3, 4])
def test_fixed_spheres_and_reference_joints_on_gpu(
    tmp_path: Path,
    math_mode: MathMode,
    sphere_count: int,
    gpu_backend: Backend,
    gpu_arch: str,
) -> None:
    options = Options(math_mode)
    static_planner_source(tmp_path, options, sphere_count)
    run_planner_fixture(
        tmp_path,
        options,
        gpu_backend,
        gpu_arch,
        "rrt_static.cu",
        (f"TEST_SPHERE_COUNT={sphere_count}",),
    )
