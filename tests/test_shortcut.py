"""Shortcuts follow arc length, retain endpoints, and respect collision resolution."""

import ctypes as ct
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from test_rrt import binding as binding
from test_rrt import compiled as compiled

from tenkai import Planner, RRTBuild, RRTConfig, ShortcutConfig, sdf
from tenkai.shortcut import NativeShortcutConfig


@pytest.fixture(scope="module")
def kernel_build(compiled: RRTBuild, tmp_path_factory: pytest.TempPathFactory) -> RRTBuild:
    library = tmp_path_factory.mktemp("shortcut kernel") / "kernel.so"
    command = compiled.toolchain["compile_command"].copy()
    command[1:1] = ["-I", str(compiled.library.parent)]
    source = compiled.library.parent / "rrt/bindings.cu"
    fixture = Path(__file__).with_name("fixtures") / "shortcut.cu"
    command[command.index(str(source))] = str(fixture)
    command[command.index(str(compiled.library))] = str(library)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return RRTBuild(library, compiled.model, compiled.toolchain)


def run_shortcut(
    planner: Planner,
    paths: Sequence[npt.ArrayLike],
    environment_ids: npt.ArrayLike,
    seeds: npt.ArrayLike,
    *,
    config: ShortcutConfig = ShortcutConfig(),
    resolution: float = 0.005,
) -> list[npt.NDArray[np.float64]]:
    arrays = [np.asarray(path, dtype=np.float32) for path in paths]
    lengths = np.array([len(path) for path in arrays], dtype=np.uint32)
    steps = lengths if config.max_steps is None else config.max_steps
    capacities = lengths.astype(np.uint64) + steps
    offsets = np.concatenate([np.zeros(1, dtype=np.uint64), np.cumsum(capacities)])
    packed = np.zeros((int(offsets[-1]), planner.dimension), dtype=np.float32)
    for offset, array in zip(offsets, arrays):
        packed[int(offset) : int(offset) + len(array)] = array
    ids = np.asarray(environment_ids, dtype=np.uint32)
    seed = np.asarray(seeds, dtype=np.uint64)
    native_config = NativeShortcutConfig.from_config(config)
    operation = planner.library.test_shortcut
    operation.argtypes = [
        ct.c_void_p,
        ct.POINTER(NativeShortcutConfig),
        ct.c_uint32,
        np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"),
        ct.c_double,
    ]
    operation.restype = ct.c_int
    planner.check(
        operation(
            planner.handle,
            ct.byref(native_config),
            len(paths),
            offsets,
            packed,
            lengths,
            ids,
            seed,
            resolution,
        )
    )
    return [
        packed[int(offset) : int(offset) + int(length)].astype(np.float64)
        for offset, length in zip(offsets, lengths)
    ]


def path_length(path: npt.NDArray[np.float64]) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


@pytest.mark.parametrize(
    "make_config",
    [
        lambda: ShortcutConfig(max_steps=-1),
        lambda: ShortcutConfig(max_steps=2**32 - 1),
        lambda: ShortcutConfig(max_empty_steps=1.5),  # type: ignore[arg-type]
        lambda: ShortcutConfig(max_empty_steps=True),
        lambda: ShortcutConfig(range_ratio=0),
        lambda: ShortcutConfig(range_ratio=1.01),
        lambda: ShortcutConfig(range_ratio=float("nan")),
        lambda: ShortcutConfig(snap_to_vertex=-0.01),
        lambda: ShortcutConfig(snap_to_vertex=1.01),
        lambda: ShortcutConfig(snap_to_vertex=float("inf")),
    ],
)
def test_invalid_shortcut_config_at_construction(make_config: Callable[[], ShortcutConfig]) -> None:
    with pytest.raises(ValueError):
        make_config()


@pytest.mark.gpu
@pytest.mark.parametrize("range_ratio", [0.33, 1.0])
def test_samples_follow_arc_length_and_can_add_a_vertex(
    kernel_build: RRTBuild, range_ratio: float
) -> None:
    path = np.array([[-1, 0], [0, 0], [0, 0.5]])
    # SplitMix64 seed 4: the path's unequal edge lengths distinguish arc-length
    # sampling from uniform edge/vertex sampling.
    first = 1.5 * 0.43145581774497377
    lower = max(0, first - range_ratio * 1.5)
    upper = min(1.5, first + range_ratio * 1.5)
    second = lower + (upper - lower) * 0.8924068459997183
    expected = np.array([path[0], [first - 1, 0], [0, second - 1], path[-1]])
    config = ShortcutConfig(max_steps=1, range_ratio=range_ratio, snap_to_vertex=0)
    with kernel_build.planner([]) as planner:
        result = run_shortcut(planner, [path], [0], [4], config=config)[0]
    np.testing.assert_allclose(result, expected, rtol=0, atol=5e-8)
    assert path_length(result) < path_length(path)


@pytest.mark.gpu
def test_snapping_and_collisions_in_independent_environments(kernel_build: RRTBuild) -> None:
    path = np.array([[-0.75, -0.75], [-0.75, 0.75], [0.75, 0.75]])
    config = ShortcutConfig(max_steps=1, range_ratio=1, snap_to_vertex=0.1)
    with kernel_build.planner([[], [sdf.Sphere((0, 0, 0), 0.2)]], resolution=0.005) as planner:
        clear, blocked = run_shortcut(planner, [path, path], [0, 1], [21, 21], config=config)
        np.testing.assert_array_equal(clear, path[[0, -1]])
        np.testing.assert_array_equal(blocked, path)
        near, far = run_shortcut(
            planner, [path, path], [0, 0], [21, 21], config=replace(config, snap_to_vertex=0)
        )
        assert len(near) == 4
        np.testing.assert_array_equal(near, far)


@pytest.mark.gpu
def test_empty_step_limit_and_default_limits(kernel_build: RRTBuild) -> None:
    path = np.array([[-1, 0], [0, 0], [0, 1]])
    config = ShortcutConfig(max_steps=2, max_empty_steps=1, range_ratio=1, snap_to_vertex=0)
    with kernel_build.planner([]) as planner:
        # Seed 1 first samples the same edge; its second trial crosses the corner.
        stopped = run_shortcut(planner, [path], [0], [1], config=config)[0]
        continued = run_shortcut(
            planner, [path], [0], [1], config=replace(config, max_empty_steps=2)
        )[0]
        np.testing.assert_array_equal(stopped, path)
        assert path_length(continued) < path_length(path)

        # Accepted trials reset the counter: seed 1 rejects trials 1, 6, and 7,
        # but resumes shortening on trial 8 instead of exhausting a limit of 3.
        limited = run_shortcut(
            planner, [path], [0], [1], config=replace(config, max_steps=10, max_empty_steps=3)
        )[0]
        unrestricted = run_shortcut(
            planner, [path], [0], [1], config=replace(config, max_steps=10, max_empty_steps=10)
        )[0]
        prefix = run_shortcut(
            planner, [path], [0], [1], config=replace(config, max_steps=5, max_empty_steps=10)
        )[0]
        np.testing.assert_array_equal(limited, unrestricted)
        assert path_length(limited) < path_length(prefix)

        longer = np.array([[-1, 0], [-0.5, 0], [0, 0], [0, 0.5], [0, 1]])
        defaults = run_shortcut(planner, [path, longer], [0, 0], [4, 8])
        for original, default, seed in zip([path, longer], defaults, [4, 8], strict=True):
            explicit = run_shortcut(
                planner,
                [original],
                [0],
                [seed],
                config=ShortcutConfig(max_steps=len(original), max_empty_steps=len(original)),
            )[0]
            np.testing.assert_array_equal(default, explicit)


@pytest.mark.gpu
def test_degenerate_paths_and_disabled_shortcuts(kernel_build: RRTBuild) -> None:
    paths = [np.zeros((0, 2)), np.full((1, 2), 0.1), np.full((2, 2), 0.1), np.zeros((5, 2))]
    corner = np.array([[-0.75, -0.75], [-0.75, 0.75], [0.75, 0.75]])
    with kernel_build.planner([]) as planner:
        results = run_shortcut(planner, paths, [0] * 4, [0] * 4)
        for original, result in zip(paths, results, strict=True):
            np.testing.assert_array_equal(result, original.astype(np.float32))
        for config in [ShortcutConfig(max_steps=0), ShortcutConfig(max_empty_steps=0)]:
            result = run_shortcut(planner, [corner], [0], [21], config=config)[0]
            np.testing.assert_array_equal(result, corner)
        repeated = np.repeat(corner, 3, axis=0)
        result = run_shortcut(
            planner,
            [repeated],
            [0],
            [21],
            config=ShortcutConfig(max_steps=1, range_ratio=1, snap_to_vertex=0.1),
        )[0]
        np.testing.assert_array_equal(result[[0, -1]], corner[[0, -1]])
        assert np.isfinite(result).all()
        assert path_length(result) < path_length(repeated)


@pytest.mark.gpu
def test_repeated_shortcuts_preserve_collision_free_paths(kernel_build: RRTBuild) -> None:
    corner = np.array([[-0.75, -0.75], [-0.75, 0.75], [0.75, 0.75]])
    paths = [corner.copy() for _ in range(16)]
    seeds = list(range(len(paths)))
    config = ShortcutConfig(max_steps=80, max_empty_steps=30)
    obstacle = sdf.Sphere((0, 0, 0), 0.2)
    with kernel_build.planner([obstacle], resolution=0.002) as planner:
        results = run_shortcut(
            planner, paths, [0] * len(paths), seeds, config=config, resolution=0.002
        )
        again = run_shortcut(
            planner, paths, [0] * len(paths), seeds, config=config, resolution=0.002
        )
        for original, result, repeat in zip(paths, results, again, strict=True):
            np.testing.assert_array_equal(result, repeat)
            np.testing.assert_array_equal(original, corner)
            np.testing.assert_array_equal(result[[0, -1]], original[[0, -1]])
            assert path_length(result) < path_length(original)
            samples = []
            for a, b in zip(result[:-1], result[1:], strict=True):
                steps = int(np.ceil(np.max(np.abs(b - a)) / 0.002))
                samples.extend(np.linspace(a, b, steps + 1))
            full = np.asarray(samples)[:, ::-1]
            assert not planner.collisions(full, [0] * len(full)).any()


def test_solve_shortcut_binding_and_lifetime(binding: RRTBuild) -> None:
    inputs = ([[-1, 0], [0, -1]], [[1, 0], [0, 1]], [1, 0], [0, 2**64 - 2])
    with binding.planner([[], []], problem_count=2, config=RRTConfig(max_nodes=3)) as planner:
        with pytest.raises(RuntimeError, match="unavailable"):
            planner.paths()
        raw_results = planner.solve(*inputs)
        original = planner.paths()
        config = ShortcutConfig(
            max_steps=7, max_empty_steps=2, range_ratio=0.5, snap_to_vertex=0.125
        )
        for _ in range(2):
            results = planner.solve(*inputs, shortcut=config)
            assert [result["path_length"] for result in results] == [4, 4]
            for p, path in enumerate(planner.paths()):
                np.testing.assert_array_equal(path, planner.path(p))
                np.testing.assert_array_equal(path[[0, -1]], original[p][[0, -1]])
                seed = inputs[3][p]
                np.testing.assert_array_equal(
                    path[1], [0.5 * (seed ^ (seed >> 32)), 0.125 * inputs[2][p]]
                )
                assert results[p]["nodes"] == raw_results[p]["nodes"]
                assert results[p]["rounds"] == raw_results[p]["rounds"]
            results[0]["path_length"] = 0
            assert len(planner.path(0)) == 4

        with pytest.raises(ValueError, match="range_ratio"):
            planner.solve(*inputs, shortcut=ShortcutConfig(range_ratio=0))
        assert len(planner.path(0)) == 4
        for shortcut in (None, ShortcutConfig(max_steps=0), ShortcutConfig(max_empty_steps=0)):
            assert planner.solve(*inputs, shortcut=shortcut) == raw_results
            for p, path in enumerate(planner.paths()):
                np.testing.assert_array_equal(path, original[p])
                path[:] = 123
                np.testing.assert_array_equal(planner.path(p), original[p])
    with pytest.raises(RuntimeError, match="closed"):
        planner.solve(*inputs, shortcut=config)
    with pytest.raises(RuntimeError, match="closed"):
        planner.paths()


@pytest.mark.gpu
def test_solve_shortcut_matches_kernel_and_replaces_previous_paths(kernel_build: RRTBuild) -> None:
    environments: sdf.Geometry = [[], [sdf.Sphere((0, 0, 0), 0.2)]]
    starts = np.array([[-0.75, -0.75], [-0.75, -0.75], [0, 0], [-0.75, -0.75], [0.5, -0.5]])
    goals = np.array([[0.75, 0.75], [0.75, 0.75], [0.75, 0.75], [0, 0], [0.5, -0.5]])
    ids = [0, 1, 1, 1, 0]
    seeds = [42, 43, 44, 45, 46]
    with kernel_build.planner(
        environments,
        problem_count=5,
        config=RRTConfig(max_nodes=512, max_rounds=1024, step_size=0.25),
        resolution=0.005,
    ) as planner:
        raw_results = planner.solve(starts, goals, ids, seeds)
        assert [result["status"] for result in raw_results] == [
            "success",
            "success",
            "invalid_start",
            "invalid_goal",
            "success",
        ]
        raw = planner.paths()
        for limit in (None, 1, 80, 200, 0, None):
            config = ShortcutConfig(max_steps=limit, max_empty_steps=30, snap_to_vertex=0)
            expected = run_shortcut(planner, raw, ids, seeds, config=config)
            results = planner.solve(starts, goals, ids, seeds, shortcut=config)
            for p, (actual, wanted) in enumerate(zip(planner.paths(), expected, strict=True)):
                np.testing.assert_array_equal(actual, wanted)
                np.testing.assert_array_equal(planner.path(p), wanted)
                assert results[p] == {**raw_results[p], "path_length": len(actual)}
                assert path_length(actual) <= path_length(raw[p]) + 1e-6

        zeros = np.zeros((5, 2))
        for env, count in ((0, 1), (1, 0)):
            results = planner.solve(zeros, zeros, [env] * 5, seeds, shortcut=ShortcutConfig())
            assert [result["path_length"] for result in results] == [count] * 5
            assert [path.shape for path in planner.paths()] == [(count, 2)] * 5
            if count:
                np.testing.assert_array_equal(planner.path(0), [[0, 0]])
        assert planner.solve(starts, goals, ids, seeds) == raw_results
        for actual, wanted in zip(planner.paths(), raw, strict=True):
            np.testing.assert_array_equal(actual, wanted)


@pytest.mark.gpu
def test_all_failed_paths_before_any_success(compiled: RRTBuild) -> None:
    with compiled.planner([sdf.Sphere((0, 0, 0), 0.2)], problem_count=2) as planner:
        results = planner.solve(
            [[0, 0]] * 2, [[0, 0]] * 2, [0, 0], [0, 1], shortcut=ShortcutConfig()
        )
        assert [result["status"] for result in results] == ["invalid_start"] * 2
        assert [result["path_length"] for result in results] == [0, 0]
        assert [path.shape for path in planner.paths()] == [(0, 2), (0, 2)]
        assert planner.path(0).shape == (0, 2)


@pytest.mark.gpu
def test_shortcut_capacity_overflow_invalidates_paths_and_recovers(compiled: RRTBuild) -> None:
    with compiled.planner([], config=RRTConfig(step_size=0.25)) as planner:
        inputs = ([[-0.75, -0.75]], [[0.75, 0.75]], [0], [42])
        original = planner.solve(*inputs)
        assert original[0]["path_length"] >= 3
        with pytest.raises(RuntimeError, match="capacity"):
            planner.solve(*inputs, shortcut=ShortcutConfig(max_steps=2**32 - 2))
        with pytest.raises(RuntimeError, match="unavailable"):
            planner.paths()
        assert planner.solve(*inputs) == original
