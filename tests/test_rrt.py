"""The installed RRT API owns builds, native calls, and planner lifetime."""

import ctypes as ct
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tenkai import Options, Planner, RRTBuild, RRTConfig, build_rrt
from tenkai.build import Backend
from tenkai.compiler import ModelMetadata
from tenkai.cuda import MathMode
from tenkai.rrt import BuildInfo

FIXTURES = Path(__file__).with_name("fixtures")


@pytest.fixture(scope="module")
def binding(tmp_path_factory: pytest.TempPathFactory) -> RRTBuild:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("binding tests require a C++ compiler")
    library = tmp_path_factory.mktemp("binding") / "binding.so"
    subprocess.run(
        [
            compiler,
            "-std=c++14",
            "-shared",
            "-fPIC",
            str(FIXTURES / "rrt_binding_stub.cpp"),
            "-o",
            str(library),
        ],
        check=True,
    )
    model = ModelMetadata(
        input_joint_names=["x", "y", "z"],
        control_joints=[
            {"name": "y", "index": 1, "lower": -1.0, "upper": 1.0},
            {"name": "x", "index": 0, "lower": -1.0, "upper": 1.0},
        ],
        sphere_count=1,
        scalar="double",
        math="f64",
        optimization="algebraic",
        live_budget=None,
    )
    info = BuildInfo(nvcc="CPU ABI fixture", compile_command=[], max_registers=None)
    return RRTBuild(library, model, info)


def test_binding_round_trip_and_lifetime(binding: RRTBuild) -> None:
    native = ct.CDLL(str(binding.library))
    assert native.live_contexts() == 0
    geometry = [[10, 10, 10, 1, 1, 1, 10], [20, 20, 20, 1, 1, 1, 20]]
    config = RRTConfig(max_nodes=16, max_rounds=9, step_size=0.5, goal_bias=0.25)
    starts = np.array([[-1.0, 10, -0.5, 20], [0.0, 30, 0.5, 40]])[:, ::2]
    goals = np.array([[0.5, 1], [1, 0]], dtype=np.float32)
    with binding.planner(geometry, config=config, problem_count=2) as planner:
        assert native.live_contexts() == 1
        with pytest.raises(RuntimeError, match="path unavailable"):
            planner.path(0)
        seeds = [0, 2**64 - 2]
        results = planner.solve(starts, goals, [1, 0], seeds)
        assert results == [
            {"status": "success", "nodes": 0, "rounds": 9, "path_length": 3},
            {"status": "success", "nodes": 1, "rounds": 9, "path_length": 3},
        ]
        path = planner.path(1)
        np.testing.assert_array_equal(path[0], starts[1])
        np.testing.assert_array_equal(path[1], starts[1] + 0.125 * (goals[1] - starts[1]))
        np.testing.assert_array_equal(path[-1], goals[1])
        assert path.dtype == np.float64
        np.testing.assert_array_equal(
            planner.collisions([[0, 0, 0.5], [1, 0, 0.5]], [1, 0]), [1, 0]
        )
        assert planner.collisions(np.empty((0, 3)), []).shape == (0,)
        assert planner.solve(starts, goals, [1, 0], seeds) == results
        np.testing.assert_array_equal(planner.path(1), path)
    assert native.live_contexts() == 0
    planner.close()
    with pytest.raises(RuntimeError, match="closed"):
        planner.solve(starts, goals, [1, 0], seeds)
    with pytest.raises(RuntimeError, match="closed"):
        planner.path(0)
    with pytest.raises(RuntimeError, match="closed"):
        planner.collisions([[0, 0, 0]], [0])
    with pytest.raises(RuntimeError, match="closed"):
        planner.__enter__()


def test_context_closes_on_exception(binding: RRTBuild) -> None:
    with pytest.raises(ValueError, match="user exception"):
        with binding.planner([[10, 10, 10, 1, 1, 1, 10]]):
            raise ValueError("user exception")
    assert ct.CDLL(str(binding.library)).live_contexts() == 0


@pytest.mark.parametrize(
    "config",
    [
        RRTConfig(max_nodes=1),
        RRTConfig(max_rounds=2**32),
        RRTConfig(step_size=float("nan")),
        RRTConfig(goal_bias=1.1),
        RRTConfig(max_nodes=True),
    ],
)
def test_invalid_config_before_load(binding: RRTBuild, config: RRTConfig) -> None:
    with pytest.raises(ValueError):
        Planner("missing.so", binding.model, config, 1, [[0] * 7], 0.05)


@pytest.mark.parametrize(
    "geometry", [[], [[0] * 6], [[0, 0, 0, -1, 1, 1, 0]], [[float("inf")] * 7]]
)
def test_invalid_geometry_before_load(binding: RRTBuild, geometry: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        Planner("missing.so", binding.model, RRTConfig(), 1, geometry, 0.05)


@pytest.mark.parametrize(
    "bad_ids,bad_seeds",
    [
        ([-1], [0]),
        ([2**32], [0]),
        ([1], [0]),
        ([0.5], [0]),
        ([0], [-1]),
        ([0], [2**64]),
        ([0], [0.5]),
        ([0], []),
    ],
)
def test_invalid_ids_and_seeds(
    binding: RRTBuild, bad_ids: list[int | float], bad_seeds: list[int | float]
) -> None:
    with binding.planner([[10, 10, 10, 1, 1, 1, 10]]) as planner:
        with pytest.raises(ValueError):
            planner.solve([[0, 0]], [[1, 1]], bad_ids, bad_seeds)


def test_invalid_shapes_and_problem_ids(binding: RRTBuild) -> None:
    with binding.planner([[10, 10, 10, 1, 1, 1, 10]]) as planner:
        with pytest.raises(ValueError, match="shape"):
            planner.solve([[0, 0, 0]], [[1, 1]], [0], [0])
        with pytest.raises(ValueError, match="shape"):
            planner.collisions([[0, 0]], [0])
        for problem in (-1, 1, 2**32):
            with pytest.raises(ValueError, match="problem_id"):
                planner.path(problem)


def robot_inputs(directory: Path) -> tuple[Path, Path]:
    urdf = directory / "robot.urdf"
    urdf.write_text("""<robot name="xy">
      <link name="base"/><link name="x"/><link name="tool"/>
      <joint name="x" type="prismatic"><parent link="base"/><child link="x"/>
        <axis xyz="1 0 0"/><limit lower="-1" upper="1"/></joint>
      <joint name="y" type="prismatic"><parent link="x"/><child link="tool"/>
        <axis xyz="0 1 0"/><limit lower="-1" upper="1"/></joint>
    </robot>""")
    spheres = directory / "spheres.yaml"
    spheres.write_text("""control_joint_names: [y, x]
collision_spheres:
  tool:
    spheres: [[0, 0, 0, 0.05]]
""")
    return urdf, spheres


def test_build_rejects_missing_controls(tmp_path: Path) -> None:
    urdf, spheres = robot_inputs(tmp_path)
    spheres.write_text("collision_spheres: {}\n")
    with pytest.raises(ValueError, match="control_joint_names"):
        build_rrt(urdf, spheres, tmp_path / "build")


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_compiler_failure_includes_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: Backend
) -> None:
    urdf, spheres = robot_inputs(tmp_path)
    monkeypatch.setattr("tenkai.rrt.compiler_command", lambda *_: ([sys.executable], "test"))
    compiler_name = "hipcc" if backend == "hip" else "nvcc"
    with pytest.raises(RuntimeError, match=compiler_name + " failed; see") as failure:
        build_rrt(urdf, spheres, tmp_path / "build", backend=backend)
    log = tmp_path / "build/compile.log"
    assert log.read_text().strip()
    assert log.read_text() in str(failure.value)


@pytest.mark.parametrize(
    "backend,arch,max_registers,message",
    [
        ("other", None, None, "backend"),
        ("hip", "compute_75", None, "arch"),
        ("cuda", "gfx1201", None, "arch"),
        ("hip", "gfx1201; false", None, "arch"),
        ("hip", None, 128, "only supported by the CUDA"),
    ],
)
def test_invalid_build_options_before_generation(
    tmp_path: Path, backend: Backend, arch: str | None, max_registers: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_rrt(
            "missing.urdf",
            "missing.yaml",
            tmp_path / "build",
            backend=backend,
            arch=arch,
            max_registers=max_registers,
        )
    assert not (tmp_path / "build").exists()


@pytest.fixture(scope="module", params=["f32", "f32-sfu", "f64"])
def compiled(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    gpu_backend: Backend,
    gpu_arch: str,
) -> RRTBuild:
    directory = tmp_path_factory.mktemp("rrt " + request.param)
    urdf, spheres = robot_inputs(directory)
    mode: MathMode = request.param
    return build_rrt(
        urdf,
        spheres,
        directory / "build",
        options=Options(mode),
        backend=gpu_backend,
        arch=gpu_arch,
    )


@pytest.mark.gpu_compile
def test_public_build(compiled: RRTBuild, gpu_backend: Backend) -> None:
    compiler_name = "hipcc" if gpu_backend == "hip" else "nvcc"
    assert compiled.library.is_file()
    assert compiled.timings["codegen"] > 0
    assert compiled.timings[compiler_name] > 0
    assert (
        compiled.timings["total"] >= compiled.timings["codegen"] + compiled.timings[compiler_name]
    )
    assert compiled.model == json.loads((compiled.library.parent / "model.json").read_text())
    assert [j["name"] for j in compiled.model["control_joints"]] == ["y", "x"]
    library = ct.CDLL(str(compiled.library))
    for symbol in (
        "rrtc_create",
        "rrt_destroy",
        "rrt_solve",
        "rrt_path",
        "rrt_check",
    ):
        assert getattr(library, symbol)


@pytest.mark.gpu
def test_public_planner_on_gpu(compiled: RRTBuild) -> None:
    config = RRTConfig(max_nodes=256, max_rounds=512, goal_bias=1.0)
    with compiled.planner([[10, 10, 10, 1, 1, 1, 10]], config=config) as planner:
        inputs = ([[-0.5, -0.5]], [[0.5, 0.5]], [0], [42])
        results = planner.solve(*inputs)
        assert results[0]["status"] == "success"
        path = planner.path(0)
        np.testing.assert_array_equal(path[0], inputs[0][0])
        np.testing.assert_array_equal(path[-1], inputs[1][0])
        assert results[0]["path_length"] == len(path)
        assert not planner.collisions(path[:, ::-1], [0] * len(path)).any()
        assert planner.solve(*inputs) == results
        np.testing.assert_array_equal(planner.path(0), path)
