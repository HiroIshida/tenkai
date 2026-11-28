"""The installed RRT API owns builds, native calls, and planner lifetime."""

import ctypes as ct
import json
import shutil
import subprocess
import sys
from collections.abc import Set
from pathlib import Path

import numpy as np
import pytest
from helpers import FakeCompiler, fetch_collision_cases

from tenkai import Options, Planner, RRTBuild, RRTConfig, ShortcutConfig, build_rrt, sdf
from tenkai.build import Backend
from tenkai.compiler import ControlJoint, ModelMetadata
from tenkai.modeling import KinematicModel, read_config
from tenkai.rrt import BuildInfo, sampling_space

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
        scalar="float",
        math="f32-sfu",
        optimization="algebraic",
        live_budget=None,
        sphere_batch_size=8,
        ground=False,
    )
    info = BuildInfo(nvcc="CPU ABI fixture", compile_command=[], max_registers=None)
    return RRTBuild(library, model, info)


def test_binding_round_trip_and_lifetime(binding: RRTBuild) -> None:
    native = ct.CDLL(str(binding.library))
    assert native.live_contexts() == 0
    geometry: sdf.Geometry = [[], []]
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
        with binding.planner([]):
            raise ValueError("user exception")
    assert ct.CDLL(str(binding.library)).live_contexts() == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_nodes": 1},
        {"max_rounds": 2**32},
        {"step_size": float("nan")},
        {"goal_bias": 1.1},
        {"max_nodes": True},
    ],
)
def test_invalid_config(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RRTConfig(**arguments)  # type: ignore[arg-type]


def control(lower: float, upper: float) -> ControlJoint:
    return {"name": "x", "index": 0, "lower": lower, "upper": upper}


@pytest.mark.parametrize(
    "joint,resolution",
    [
        (control(-1, 1), float("nan")),
        (control(-1, 1), 0.0),
        (control(-1, 1), 1e-50),  # rounds to zero in float32
        (control(-1, 1), 1e50),  # overflows float32
        (control(-1, 1), 1e-30),  # more than 2**32 - 2 steps
        (control(-1, 1e39), 0.05),  # overflows float32
        (control(-3e38, 3e38), 1e30),  # span overflows float32
        (control(1.0, 1.0 + 1e-9), 0.05),  # collapses in float32
        (control(1, -1), 0.05),
        (control(float("-inf"), 1), 0.05),
    ],
)
def test_invalid_sampling_space(joint: ControlJoint, resolution: float) -> None:
    with pytest.raises(ValueError):
        sampling_space([joint], resolution)


@pytest.mark.parametrize("geometry", [[object()], [[sdf.Ground()], [object()]], [[0] * 7]])
def test_invalid_geometry_before_load(binding: RRTBuild, geometry: object) -> None:
    with pytest.raises(ValueError, match="collision objects"):
        Planner("missing.so", binding.model, RRTConfig(), 1, geometry, 0.05)  # type: ignore[arg-type]


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
    with binding.planner([]) as planner:
        with pytest.raises(ValueError):
            planner.solve([[0, 0]], [[1, 1]], bad_ids, bad_seeds)


def test_invalid_shapes_and_problem_ids(binding: RRTBuild) -> None:
    with binding.planner([]) as planner:
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
    monkeypatch.setattr("tenkai.build.compiler_command", lambda *_: ([sys.executable], "test"))
    compiler_name = "hipcc" if backend == "hip" else "nvcc"
    with pytest.raises(RuntimeError, match=compiler_name + " failed; see") as failure:
        build_rrt(urdf, spheres, tmp_path / "build", backend=backend, cache=False)
    log = tmp_path / "build/compile.log"
    assert log.read_text().strip()
    assert log.read_text() in str(failure.value)


def test_cached_build_refreshes_model_metadata(tmp_path: Path, fake_compiler: FakeCompiler) -> None:
    urdf, spheres = robot_inputs(tmp_path)
    first = build_rrt(urdf, spheres, tmp_path / "first")

    urdf.write_text(urdf.read_text().replace('lower="-1"', 'lower="-2"'))
    second = build_rrt(urdf, spheres, tmp_path / "second")

    assert not first.cache_hit and second.cache_hit
    assert len(fake_compiler.calls) == 1
    assert first.library.read_bytes() == second.library.read_bytes()
    assert [joint["lower"] for joint in first.model["control_joints"]] == [-1, -1]
    assert [joint["lower"] for joint in second.model["control_joints"]] == [-2, -2]
    assert second.model == json.loads((second.library.parent / "model.json").read_text())


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


@pytest.mark.parametrize(
    "types", [{object}, {sdf.Ground}, {sdf.Ground()}, {"Box"}, sdf.Box, [sdf.Box]]
)
def test_invalid_specialization_before_generation(tmp_path: Path, types: object) -> None:
    with pytest.raises(ValueError, match="specialize_types"):
        build_rrt(
            "missing.urdf",
            "missing.yaml",
            tmp_path / "build",
            specialize_types=types,  # type: ignore[arg-type]
        )
    assert not (tmp_path / "build").exists()


@pytest.fixture(scope="module")
def compiled(
    tmp_path_factory: pytest.TempPathFactory,
    gpu_backend: Backend,
    gpu_arch: str,
) -> RRTBuild:
    directory = tmp_path_factory.mktemp("rrt build")
    urdf, spheres = robot_inputs(directory)
    return build_rrt(
        urdf,
        spheres,
        directory / "build",
        options=Options(ground=False),
        backend=gpu_backend,
        arch=gpu_arch,
        cache=False,
    )


@pytest.fixture(scope="module")
def compiled_ground(
    tmp_path_factory: pytest.TempPathFactory,
    gpu_backend: Backend,
    gpu_arch: str,
) -> RRTBuild:
    directory = tmp_path_factory.mktemp("rrt ground build")
    urdf, spheres = robot_inputs(directory)
    return build_rrt(
        urdf,
        spheres,
        directory / "build",
        backend=gpu_backend,
        arch=gpu_arch,
    )


@pytest.mark.gpu_compile
def test_public_build(compiled: RRTBuild, gpu_backend: Backend) -> None:
    compiler_name = "hipcc" if gpu_backend == "hip" else "nvcc"
    assert compiled.library.is_file()
    assert compiled.toolchain["max_registers"] is None
    assert not any(
        argument.startswith("--maxrregcount") for argument in compiled.toolchain["compile_command"]
    )
    assert compiled.timings["codegen"] > 0
    assert compiled.timings[compiler_name] > 0
    assert (
        compiled.timings["total"] >= compiled.timings["codegen"] + compiled.timings[compiler_name]
    )
    assert compiled.model == json.loads((compiled.library.parent / "model.json").read_text())
    assert [j["name"] for j in compiled.model["control_joints"]] == ["y", "x"]
    library = ct.CDLL(str(compiled.library))
    for symbol in (
        "rrtc_create_scene",
        "rrt_destroy",
        "rrt_solve",
        "rrt_path",
        "rrt_paths",
        "rrt_check",
    ):
        assert getattr(library, symbol)


@pytest.mark.gpu
def test_public_planner_on_gpu(compiled_ground: RRTBuild) -> None:
    config = RRTConfig(max_nodes=256, max_rounds=512, goal_bias=1.0)
    obstacles = [sdf.Box((10, 10, 10), (1, 1, 1)), sdf.Ground(z=-10)]
    with compiled_ground.planner(obstacles, config=config) as planner:
        inputs = ([[-0.5, -0.5]], [[0.5, 0.5]], [0], [42])
        results = planner.solve(*inputs)
        assert results[0]["status"] == "success"
        path = planner.path(0)
        np.testing.assert_array_equal(path[0], inputs[0][0])
        np.testing.assert_array_equal(path[-1], inputs[1][0])
        assert results[0]["path_length"] == len(path)
        assert not planner.collisions(path[:, ::-1], [0] * len(path)).any()
        planner.solve(*inputs, shortcut=ShortcutConfig())
        assert planner.solve(*inputs) == results
        np.testing.assert_array_equal(planner.path(0), path)


@pytest.mark.gpu
def test_fetch_body_collisions_and_planning_on_gpu(
    tmp_path: Path, gpu_backend: Backend, gpu_arch: str
) -> None:
    assets = Path(__file__).resolve().parents[1] / "assets/fetch"
    urdf = assets / "fetch.urdf"
    yaml = assets / "collision.yaml"
    kin = KinematicModel.from_urdf(urdf.read_text())
    spheres = read_config(yaml.read_text()).resolve(kin)
    cases = fetch_collision_cases(kin, spheres)
    compiled = build_rrt(
        urdf, yaml, tmp_path, options=Options(ground=True), backend=gpu_backend, arch=gpu_arch
    )
    controls = [joint["index"] for joint in compiled.model["control_joints"]]
    configurations = np.array([case.joints for case in cases])
    home, goal, base_collision, torso_collision = configurations[:4, controls]
    environments = [[sdf.Ground(z=-100)], [sdf.Box((0.95, 0, 0.8), (1, 2, 0.05)), sdf.Ground()]]

    with compiled.planner(
        environments,
        config=RRTConfig(max_rounds=8192, goal_bias=0.0),
        problem_count=3,
        resolution=0.005,
    ) as planner:
        np.testing.assert_array_equal(
            planner.collisions(configurations, [0] * len(cases)),
            [case.collision for case in cases],
        )
        results = planner.solve(
            [home, base_collision, home],
            [goal, goal, torso_collision],
            [1, 0, 0],
            [42] * 3,
            shortcut=ShortcutConfig(max_steps=50, max_empty_steps=20),
        )
        assert [result["status"] for result in results] == [
            "success",
            "invalid_start",
            "invalid_goal",
        ]

        path = planner.path(0)
        np.testing.assert_array_equal(path[0], home.astype(np.float32))
        np.testing.assert_array_equal(path[-1], goal.astype(np.float32))
        for begin, end in zip(path[:-1], path[1:], strict=True):
            steps = int(np.ceil(np.max(np.abs(end - begin)) / 0.0025))
            samples = np.zeros((steps + 1, len(kin.joints)))
            samples[:, controls] = np.linspace(begin, end, steps + 1)
            assert not planner.collisions(samples, [1] * len(samples)).any()


@pytest.mark.parametrize("geometry", [[], [[], []], [sdf.Sphere((0, 0, 0), 0.1)]])
def test_object_scenes_and_empty_environments(binding: RRTBuild, geometry: sdf.Geometry) -> None:
    with binding.planner(geometry) as planner:
        expected_count = 2 if geometry == [[], []] else 1
        assert planner.environment_count == expected_count
        assert (
            planner.solve([[0, 0]], [[0.5, 0.5]], [expected_count - 1], [42])[0]["status"]
            == "success"
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "obstacles,points,expected",
    [
        ([], [[0, 0], [0.5, 0.5]], [0, 0]),
        ([sdf.Sphere((0.25, 0, 0), 0.1)], [[0.25, 0], [0.375, 0], [0.5, 0]], [1, 1, 0]),
        ([sdf.Cylinder((0, 0, 0), 0.1, 0.4)], [[0, 0], [0.125, 0], [0.25, 0]], [1, 1, 0]),
        ([sdf.Ground(z=0.02)], [[0, 0], [0.5, 0.5]], [1, 1]),
        ([sdf.Ground(z=-0.2)], [[0, 0], [0.5, 0.5]], [0, 0]),
        (
            [sdf.Box((0, 0, 0), (0.8, 0.1, 0.2), ((0.6, -0.8, 0), (0.8, 0.6, 0), (0, 0, 1)))],
            [[0.18, 0.24], [-0.18, 0.24]],
            [1, 0],
        ),
        (
            [sdf.Cylinder((0.2, -0.1, 0), 0.05, 1.0, ((0, -0.8, 0.6), (0, 0.6, 0.8), (-1, 0, 0)))],
            [[0.38, 0.14], [-0.04, 0.08]],
            [1, 0],
        ),
        (
            [sdf.Sphere((-0.5, 0, 0), 0.1), sdf.Sphere((0.5, 0, 0), 0.1)],
            [[-0.5, 0], [0, 0], [0.5, 0]],
            [1, 0, 1],
        ),
        (
            [
                sdf.Box((-0.75, 0, 0), (0.1, 0.1, 0.2)),
                sdf.Box((-0.5, 0, 0), (0.1, 0.1, 0.2)),
                sdf.Sphere((0, 0.5, 0), 0.1),
                sdf.Sphere((0.25, 0.5, 0), 0.1),
                sdf.Cylinder((0.5, 0, 0), 0.05, 0.4),
                sdf.Cylinder((0.75, 0, 0), 0.05, 0.4),
                sdf.Ground(z=-0.2),
            ],
            [[-0.75, 0], [-0.5, 0], [0, 0.5], [0.25, 0.5], [0.5, 0], [0.75, 0], [0, 0]],
            [1, 1, 1, 1, 1, 1, 0],
        ),
    ],
)
def test_primitive_collisions_on_gpu(
    compiled: RRTBuild,
    compiled_ground: RRTBuild,
    obstacles: list[sdf.CollisionObject],
    points: list[list[float]],
    expected: list[int],
) -> None:
    build = compiled_ground if any(isinstance(item, sdf.Ground) for item in obstacles) else compiled
    with build.planner(obstacles) as planner:
        np.testing.assert_array_equal(planner.collisions(points, [0] * len(points)), expected)


@pytest.mark.gpu
def test_mixed_environments_share_one_build_on_gpu(
    compiled_ground: RRTBuild, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_compilation(*args: object, **kwargs: object) -> None:
        pytest.fail("creating or solving scenes must not invoke a compiler")

    monkeypatch.setattr(subprocess, "run", no_compilation)
    obstacles = [
        sdf.Box((0.5, -0.5, 0), (0.2, 0.2, 0.2)),
        sdf.Sphere((-0.5, 0.5, 0), 0.1),
        sdf.Cylinder((0.75, -0.75, 0), 0.1, 0.4),
        sdf.Ground(z=-0.2),
    ]
    environments = [
        obstacles,
        [sdf.Sphere((-0.5, -0.5, 0), 0.1), sdf.Ground(z=-0.2)],
        [sdf.Cylinder((0.5, 0.5, 0), 0.1, 0.4), sdf.Ground(z=-0.3)],
        [sdf.Ground(z=-0.4)],
    ]
    config = RRTConfig(max_nodes=64, max_rounds=32, goal_bias=1.0)
    starts = [[-0.5, -0.5]] * 4
    goals = [[0.5, 0.5]] * 4
    with compiled_ground.planner(environments, config=config, problem_count=4) as planner:
        # Scene storage is owned by the planner after creation.
        environments.clear()
        results = planner.solve(starts, goals, [0, 1, 2, 3], [42] * 4)
        assert [result["status"] for result in results] == [
            "success",
            "invalid_start",
            "invalid_goal",
            "success",
        ]
        np.testing.assert_array_equal(
            planner.collisions([[-0.5, -0.5]] * 4, [0, 1, 2, 3]), [0, 1, 0, 0]
        )
        for index in (0, 3):
            path = planner.path(index)
            np.testing.assert_array_equal(path[0], starts[index])
            np.testing.assert_array_equal(path[-1], goals[index])
            assert not planner.collisions(path[:, ::-1], [index] * len(path)).any()

    with compiled_ground.planner([sdf.Ground(z=0.0)]) as other:
        assert other.solve([[-0.5, -0.5]], [[0.5, 0.5]], [0], [42])[0]["status"] == "invalid_start"


@pytest.mark.gpu
@pytest.mark.parametrize("ground", [False, True])
@pytest.mark.parametrize(
    "types",
    [
        set(),
        {sdf.Box},
        {sdf.Sphere, sdf.Cylinder},
        {sdf.Box, sdf.Sphere, sdf.Cylinder},
    ],
    ids=["empty", "box", "sphere-cylinder", "all"],
)
def test_union_specialization_matches_generic_paths(
    tmp_path: Path,
    compiled: RRTBuild,
    compiled_ground: RRTBuild,
    gpu_backend: Backend,
    gpu_arch: str,
    types: Set[sdf.SpecializationType],
    ground: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urdf, spheres = robot_inputs(tmp_path)
    specialized = build_rrt(
        urdf,
        spheres,
        tmp_path / "specialized",
        backend=gpu_backend,
        arch=gpu_arch,
        specialize_types=types,
        options=Options(ground=ground),
    )

    def no_compilation(*args: object, **kwargs: object) -> None:
        pytest.fail("choosing a kernel must not invoke a compiler")

    monkeypatch.setattr(subprocess, "run", no_compilation)
    box = sdf.Box((0, 0, 0), (0.3, 0.4, 0.2), ((0.6, -0.8, 0), (0.8, 0.6, 0), (0, 0, 1)))
    sphere = sdf.Sphere((0, 0, 0), 0.2)
    cylinder = sdf.Cylinder((0, 0, 0), 0.2, 0.5)
    environments: list[list[sdf.CollisionObject]] = [
        [],
        [box],
        [],
        [box, sdf.Box((0.5, -0.5, 0), (0.2, 0.2, 0.2))],
        [sphere],
        [cylinder],
        [sphere, cylinder, sdf.Sphere((0.5, -0.5, 0), 0.1), sdf.Cylinder((-0.5, 0.5, 0), 0.1, 0.5)],
        [sdf.Sphere((-0.75, -0.75, 0), 0.1)],
        [sdf.Cylinder((0.75, 0.75, 0), 0.1, 0.4)],
    ]
    if ground:
        for index, environment in enumerate(environments):
            environment.append(sdf.Ground(z=-0.2 - 0.1 * index))
    generic_build = compiled_ground if ground else compiled
    count = 37
    starts = np.tile([-0.75, -0.75], (count, 1))
    goals = -starts
    seeds = np.arange(count, dtype=np.uint64) + 42
    config = RRTConfig(max_nodes=1024, max_rounds=512, step_size=0.3)
    with (
        generic_build.planner(environments, config=config, problem_count=count) as generic,
        specialized.planner(environments, config=config, problem_count=count) as actual,
    ):
        # Unreferenced environments must not prevent specialization; later solves
        # can introduce other types and then return to the original union.
        for ids in ([0], [1, 2, 3, 0], [4, 5, 6, 0], list(range(9)), [1, 2, 3, 0]):
            environment_ids = np.resize(np.asarray(ids, dtype=np.uint32), count)
            expected = generic.solve(starts, goals, environment_ids, seeds)
            result = actual.solve(starts, goals, environment_ids, seeds)
            assert result == expected
            for index, item in enumerate(result):
                path = actual.path(index)
                np.testing.assert_array_equal(path, generic.path(index))
                if item["status"] == "success":
                    assert not actual.collisions(
                        path[:, ::-1], [environment_ids[index]] * len(path)
                    ).any()
            assert any(item["status"] == "success" for item in result)
