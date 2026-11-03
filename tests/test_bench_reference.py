"""Optional plainmp benchmarks preserve the Fetch collision workload."""

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from rkern import Options
from rkern.compiler import ModelMetadata, generate

pytest.importorskip("plainmp", reason="install the bench extra for plainmp integration tests")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reference(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT / "bench"))
    return importlib.import_module("reference")


@pytest.fixture
def model(tmp_path: Path) -> ModelMetadata:
    generate(
        ROOT / "assets/fetch/fetch.urdf", ROOT / "assets/fetch/collision.yaml", tmp_path, Options()
    )
    metadata: ModelMetadata = json.loads((tmp_path / "model.json").read_text())
    return metadata


def test_checker_keeps_all_spheres_and_full_box_extents(
    reference: ModuleType, model: ModelMetadata
) -> None:
    names = [joint["name"] for joint in model["control_joints"]]
    checker = reference.make_checker(names, [0, 0, 10, 2, 4, 6, 0])
    assert len(checker.get_all_spheres()) == 69
    sdf = checker.get_sdf()
    assert sdf.evaluate(np.array([1.0, 0, 10])) == pytest.approx(0)
    assert sdf.evaluate(np.array([0.0, 2, 10])) == pytest.approx(0)
    assert sdf.evaluate(np.array([0.0, 0, 13])) == pytest.approx(0)
    assert sdf.evaluate(np.array([0.0, 0, 10])) == pytest.approx(-1)


def test_control_order_and_environment_assignment(
    reference: ModuleType, model: ModelMetadata
) -> None:
    names = [joint["name"] for joint in model["control_joints"]]
    q = np.array([0.0, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0])
    geometry = np.array([[0.95, 0, 0.8, 1, 2, 0.05, 0], [0, 0, 0, 20, 20, 20, 0]])
    normal = reference.make_checker(names, geometry[0])
    reversed_order = reference.make_checker(names[::-1], geometry[0])
    assert normal.is_valid(q)
    assert reversed_order.is_valid(q[::-1].copy())
    result = reference.validate_points(
        np.tile(q, (3, 1)), np.array([1, 0, 1], dtype=np.uint32), geometry, names
    )
    assert result == {"points": 3, "collisions": 2}
    assert reference.validate_points(
        np.empty((0, 8)), np.empty(0, dtype=np.uint32), geometry, names
    ) == {"points": 0, "collisions": 0}
    with pytest.raises(ValueError, match="environment IDs"):
        reference.validate_points(q[None], np.array([2], dtype=np.uint32), geometry, names)
    with pytest.raises(ValueError, match="ground_offset=0"):
        reference.make_checker(names, [0.95, 0, 0.8, 1, 2, 0.05, -10])


def test_cpu_collision_benchmark(
    reference: ModuleType, model: ModelMetadata, tmp_path: Path
) -> None:
    run = importlib.import_module("run")
    controls = model["control_joints"]
    low = np.array([joint["lower"] for joint in controls])
    high = np.array([joint["upper"] for joint in controls])
    rng = np.random.default_rng(123)
    q = rng.uniform(low, high, (19, len(controls))).astype(np.float32)
    full = np.zeros((len(q), len(model["input_joint_names"])), dtype=np.float32)
    full[:, [joint["index"] for joint in controls]] = q
    full.tofile(tmp_path / "configurations.bin")
    geometry = np.array([0.95, 0, 0.8, 1, 2, 0.05, 0], dtype=np.float32)
    geometry.tofile(tmp_path / "scene.bin")
    args = argparse.Namespace(
        output=tmp_path,
        cpu_core=None,
        pool=19,
        batches=[1, 7, 19],
        rounds=2,
        queries_per_round=38,
        max_repetitions=4,
        warmup_seconds=0.001,
    )
    affinity = os.sched_getaffinity(0)
    info = run.benchmark_plainmp(args, model)
    assert os.sched_getaffinity(0) == affinity
    assert info["interface"] == "installed plainmp Python API"
    assert "Python" in info["timed_call"]
    checker = reference.make_checker([j["name"] for j in controls], geometry)
    expected = [not checker.is_valid(row.astype(np.float64)) for row in q]
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "plainmp_collision.u8", dtype=np.uint8), expected
    )
    with (tmp_path / "plainmp_raw.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6
    assert {int(row["batch"]) for row in rows} == {1, 7, 19}
    assert all(float(row["wall_ms"]) > 0 and row["mode"] == "plainmp" for row in rows)
    assert [int(row["batch"]) for row in rows[3:]] == [19, 1, 7]


def test_cpu_planning_benchmark(reference: ModuleType, model: ModelMetadata) -> None:
    compare = importlib.import_module("compare_rrtc")
    args = argparse.Namespace(
        problems=2,
        max_nodes=8192,
        resolution=0.005,
        seed=20261012,
        repetitions=2,
        warmup_seconds=0.001,
    )
    geometry = np.array([[0.95, 0, 0.8, 1, 2, 0.05, 0]])
    affinity = os.sched_getaffinity(0)
    result = compare.benchmark_plainmp(args, model, geometry, min(affinity))
    assert os.sched_getaffinity(0) == affinity
    assert result["success_counts"] == [2, 2]
    assert len(result["wall_ms"]) == 2
    names = [joint["name"] for joint in model["control_joints"]]
    for path in result["paths"]:
        path = np.asarray(path)
        np.testing.assert_array_equal(path[0], compare.START)
        np.testing.assert_array_equal(path[-1], compare.GOAL)
        points = compare.interpolate(path, args.resolution / 5)
        validation = reference.validate_points(
            points, np.zeros(len(points), dtype=np.uint32), geometry, names
        )
        assert validation["collisions"] == 0


def test_success_throughput_counts_each_round(reference: ModuleType) -> None:
    compare = importlib.import_module("compare_rrtc")
    stats = compare.stats([10.0, 20.0], 4, [4, 2])
    assert stats["success_count"] == 2
    assert stats["success_counts"] == [4, 2]
    assert stats["solves_per_second"] == 250.0
    assert stats["min_solves_per_second"] == 100.0
    assert stats["max_solves_per_second"] == 400.0
