"""Generated kernels compile and preserve the existing numeric/planner checks."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tenkai import (
    KinematicModel,
    Options,
    SelfCollisionRules,
    Sphere,
    generate_centers_kernel,
    generate_collision_kernel,
    generate_fk_kernel,
    generate_rrt_sources,
)
from tenkai.build import Backend, compiler_command
from tenkai.dag import Optimization
from tenkai.modeling import read_config

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
    kin = KinematicModel.from_urdf((FIXTURES / "fk_chain.urdf").read_text())
    tool = kin.link_names.index("tool")
    spheres = [Sphere(tool, (0, 0, 0), 0.25)]
    spheres.extend(Sphere(tool, (i, 0, 0), (i + 1) / 64) for i in range(1, 9))
    spheres.insert(1, Sphere(kin.link_names.index("sensor"), (0, 0, 0), 0.375))
    (directory / "fk.cuh").write_text(generate_fk_kernel(kin, tool, options))
    (directory / "centers.cuh").write_text(generate_centers_kernel(kin, spheres, options))
    (directory / "collision.cuh").write_text(generate_collision_kernel(kin, spheres, options))
    radii = ", ".join(f"{sphere.radius}f" for sphere in spheres)
    (directory / "config.h").write_text(
        "using Scalar = float;\nnamespace sdf = tenkai_f32;\nconstexpr double tolerance = 1e-5;\n"
        f"constexpr size_t sphere_count = {len(spheres)};\n"
        f"__device__ const Scalar radii[] = {{{radii}}};\n"
    )


def planner_source(directory: Path, options: Options, controls: tuple[int, ...] = (1, 0)) -> None:
    robot = '<robot name="cartesian"><link name="root"/>'
    parent = "root"
    for name, axis in [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]:
        robot += f'<link name="{name}"/><joint name="{name}" type="prismatic">'
        robot += f'<parent link="{parent}"/><child link="{name}"/><axis xyz="{axis}"/></joint>'
        parent = name
    robot += "</robot>"
    kin = KinematicModel.from_urdf(robot)
    spheres = [Sphere(3, (0, 0, 0), 0.05), Sphere(3, (0, 0, 0.025), 0.025)]
    write_sources(directory, generate_rrt_sources(kin, spheres, controls, options))


def static_planner_source(directory: Path, options: Options, sphere_count: int) -> None:
    kin = KinematicModel.from_urdf("""<robot name="fixed_spheres">
      <link name="root"/><link name="arm"/><link name="slide"/><link name="lift"/>
      <joint name="spin" type="revolute"><parent link="lift"/><child link="arm"/>
        <axis xyz="0 0 1"/></joint>
      <joint name="slide" type="prismatic"><parent link="root"/><child link="slide"/>
        <origin xyz="5 0 0"/><axis xyz="1 0 0"/></joint>
      <joint name="lift" type="prismatic"><parent link="root"/><child link="lift"/>
        <axis xyz="0 0 1"/></joint>
    </robot>""")
    spheres = [
        Sphere(kin.link_names.index("root"), (-5, 0, 0), 0.05),
        Sphere(kin.link_names.index("slide"), (0, 0, 0), 0.05),
        Sphere(kin.link_names.index("lift"), (0, 0, 0.5), 0.05),
        Sphere(kin.link_names.index("arm"), (2, 0, 0), 0.05),
    ]
    write_sources(directory, generate_rrt_sources(kin, spheres[:sphere_count], [0], options))


def run_planner_fixture(
    directory: Path,
    backend: Backend,
    arch: str,
    fixture: str,
    defines: tuple[str, ...] = (),
) -> None:
    binary = directory / "planner-test"
    command, _ = compiler_command(backend, arch)
    # Generated headers must precede HIP's template include directory.
    command[1:1] = ["-I", str(directory)]
    command.extend("-D" + define for define in defines)
    run(
        [
            *command,
            "-DTENKAI_MODEL_NAMESPACE=tenkai_f32",
            "-I",
            str(FIXTURES),
            str(FIXTURES / fixture),
            "-o",
            str(binary),
        ]
    )
    run([str(binary)])


def self_collision_sources(directory: Path, options: Options, members: int, controls: int) -> None:
    robot = '<robot name="self"><link name="root"/>'
    parent = "root"
    for name, axis in [("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")]:
        robot += f'<link name="{name}"/><joint name="{name}" type="prismatic">'
        robot += f'<parent link="{parent}"/><child link="{name}"/><axis xyz="{axis}"/></joint>'
        parent = name
    kin = KinematicModel.from_urdf(robot + "</robot>")
    spheres = [Sphere(link, (0, i * 0.5, 0), 0.125) for link in (0, 3) for i in range(members)]
    rules = SelfCollisionRules(((0, 3),), (0,))
    source = generate_collision_kernel(kin, spheres, options, rules=rules)
    (directory / "collision.cuh").write_text(source)
    write_sources(
        directory, generate_rrt_sources(kin, spheres, list(range(controls)), options, rules=rules)
    )


@pytest.mark.parametrize("optimization", ["none", "basic", "algebraic"])
@pytest.mark.parametrize("members", [1, 2])
def test_self_collision_on_host(tmp_path: Path, optimization: Optimization, members: int) -> None:
    options = Options(optimization=optimization)
    self_collision_sources(tmp_path, options, members, 3)
    binary = tmp_path / "self-collision"
    run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++14",
            "-O2",
            "-x",
            "c++",
            "-I",
            str(tmp_path),
            f"-DTEST_MEMBERS={members}",
            str(FIXTURES / "self_collision.cu"),
            "-o",
            str(binary),
        ]
    )
    run([str(binary)])


@pytest.mark.gpu
@pytest.mark.parametrize("controls", [1, 3])
@pytest.mark.parametrize("members", [1, 2])
def test_self_collision_on_gpu(
    tmp_path: Path,
    controls: int,
    members: int,
    gpu_backend: Backend,
    gpu_arch: str,
) -> None:
    options = Options()
    self_collision_sources(tmp_path, options, members, controls)
    run_planner_fixture(
        tmp_path, gpu_backend, gpu_arch, "self_collision.cu", (f"TEST_MEMBERS={members}",)
    )


@pytest.mark.parametrize("optimization", ["none", "basic", "algebraic"])
def test_fetch_collision_against_exhaustive_pairs(
    tmp_path: Path, optimization: Optimization
) -> None:
    assets = ROOT / "assets/fetch"
    kin = KinematicModel.from_urdf((assets / "fetch.urdf").read_text())
    config = read_config((assets / "collision.yaml").read_text())
    spheres = config.resolve(kin)
    rules = config.resolve_rules(kin, spheres)
    options = Options(optimization=optimization, live_budget=64)
    (tmp_path / "centers.cuh").write_text(generate_centers_kernel(kin, spheres, options))
    (tmp_path / "collision.cuh").write_text(
        generate_collision_kernel(kin, spheres, options, rules=rules)
    )
    pairs = {frozenset(pair) for pair in rules.self_pairs}
    rows = []
    for a in spheres:
        row = [str(int(frozenset((a.link_index, b.link_index)) in pairs)) for b in spheres]
        rows.append("{" + ",".join(row) + "}")
    radii = ",".join(f"{sphere.radius}f" for sphere in spheres)
    external = ",".join(str(int(s.link_index not in rules.self_only_links)) for s in spheres)
    (tmp_path / "reference.h").write_text(
        f"constexpr unsigned sphere_count = {len(spheres)};\n"
        f"constexpr float radii[] = {{{radii}}};\n"
        f"constexpr bool external[] = {{{external}}};\n"
        "constexpr bool self_pairs[sphere_count][sphere_count] = {" + ",".join(rows) + "};\n"
    )
    binary = tmp_path / "self-reference"
    run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++14",
            "-O2",
            "-I",
            str(tmp_path),
            str(FIXTURES / "self_collision_reference.cpp"),
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
    options = Options(optimization=optimization, live_budget=budget)
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
def test_all_kernels_compile(tmp_path: Path, gpu_backend: Backend, gpu_arch: str) -> None:
    options = Options()
    chain_sources(tmp_path, options)
    planner_source(tmp_path, options)
    arithmetic = tmp_path / "arithmetic.cu"
    arithmetic.write_text('#include "fk.cuh"\n#include "centers.cuh"\n#include "collision.cuh"\n')
    command, _ = compiler_command(gpu_backend, gpu_arch)
    compile_only = "-c" if gpu_backend == "hip" else "--ptx"
    for source in (arithmetic, tmp_path / "rrt/bindings.cu"):
        run([*command, compile_only, str(source), "-o", str(source.with_suffix(".o"))])


@pytest.mark.gpu_compile
def test_planner_headers_are_self_contained(
    tmp_path: Path, gpu_backend: Backend, gpu_arch: str
) -> None:
    planner_source(tmp_path, Options())
    command, _ = compiler_command(gpu_backend, gpu_arch)
    compile_only = "-c" if gpu_backend == "hip" else "--ptx"
    headers = sorted(path for path in tmp_path.rglob("*") if path.suffix in (".cuh", ".hpp"))
    for header in headers:
        name = header.relative_to(tmp_path).as_posix()
        source = tmp_path / "header.cu"
        source.write_text(f'#include "{name}"\n#include "rrt/planner.cuh"\n#include "{name}"\n')
        run([*command, compile_only, str(source), "-o", str(tmp_path / "header.o")])


@pytest.mark.gpu
def test_arithmetic_on_gpu(tmp_path: Path, gpu_backend: Backend, gpu_arch: str) -> None:
    chain_sources(tmp_path, Options())
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
def test_planner_on_gpu(tmp_path: Path, gpu_backend: Backend, gpu_arch: str) -> None:
    planner_source(tmp_path, Options())
    run_planner_fixture(tmp_path, gpu_backend, gpu_arch, "rrt.cu")


@pytest.mark.gpu
def test_cooperative_planning_on_gpu(tmp_path: Path, gpu_backend: Backend, gpu_arch: str) -> None:
    planner_source(tmp_path, Options(), (2, 0, 1))
    run_planner_fixture(tmp_path, gpu_backend, gpu_arch, "rrtc_cooperative.cu")


@pytest.mark.gpu
@pytest.mark.parametrize("sphere_count", [0, 3, 4])
def test_fixed_spheres_and_reference_joints_on_gpu(
    tmp_path: Path,
    sphere_count: int,
    gpu_backend: Backend,
    gpu_arch: str,
) -> None:
    static_planner_source(tmp_path, Options(), sphere_count)
    run_planner_fixture(
        tmp_path,
        gpu_backend,
        gpu_arch,
        "rrt_static.cu",
        (f"TEST_SPHERE_COUNT={sphere_count}",),
    )
