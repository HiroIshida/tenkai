"""Standalone benchmarks retain the common Fetch planning problem."""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tenkai import Options
from tenkai.build import Backend
from tenkai.compiler import ModelMetadata, generate

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/fetch/fetch.urdf"
SPHERES = ROOT / "assets/fetch/collision.yaml"


@pytest.fixture
def compare(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("plainmp", reason="install the bench extra for plainmp integration tests")
    monkeypatch.syspath_prepend(str(ROOT / "bench"))
    return importlib.import_module("compare_rrtc")


@pytest.fixture
def model(tmp_path: Path) -> ModelMetadata:
    generate(URDF, SPHERES, tmp_path, Options())
    metadata: ModelMetadata = json.loads((tmp_path / "model.json").read_text())
    return metadata


@pytest.mark.parametrize("script", ["compiler.py", "compare_rrtc.py"])
def test_script_runs_without_sibling_files(script: str, tmp_path: Path) -> None:
    if script == "compare_rrtc.py":
        pytest.importorskip("plainmp")
    destination = tmp_path / script
    shutil.copyfile(ROOT / "bench" / script, destination)
    result = subprocess.run(
        [sys.executable, str(destination), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--repeats" in result.stdout


@pytest.mark.parametrize(
    "script,cpu_ms_per_problem",
    [
        pytest.param("compiler.py", None, marks=pytest.mark.gpu_compile),
        pytest.param("compare_rrtc.py", None, marks=pytest.mark.gpu),
        pytest.param("compare_rrtc.py", 3.5, marks=pytest.mark.gpu),
    ],
)
def test_standalone_benchmark_report(
    script: str,
    cpu_ms_per_problem: float | None,
    tmp_path: Path,
    gpu_backend: Backend,
    gpu_arch: str,
) -> None:
    if script == "compare_rrtc.py":
        pytest.importorskip("plainmp")
    destination = tmp_path / script
    shutil.copyfile(ROOT / "bench" / script, destination)
    command = [
        sys.executable,
        str(destination),
        "--urdf",
        str(URDF),
        "--spheres",
        str(SPHERES),
        "--output",
        str(tmp_path / "results"),
        "--repeats",
        "1",
        "--backend",
        gpu_backend,
        "--arch",
        gpu_arch,
    ]
    if script == "compare_rrtc.py":
        command.extend(["--batch", "2"])
    if cpu_ms_per_problem is not None:
        command.extend(["--cpu-ms-per-problem", str(cpu_ms_per_problem)])
    result = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "results/results.json").read_text())
    assert report["math"] == "f32"
    assert report["backend"] == gpu_backend
    if script == "compiler.py":
        compiler_name = "hipcc" if gpu_backend == "hip" else "nvcc"
        sample = report["samples_seconds"][0]
        assert sample["codegen"] > 0
        assert sample[compiler_name] > 0
        assert sample["total"] >= sample["codegen"] + sample[compiler_name]
        assert report["median_seconds"] == sample
    else:
        if cpu_ms_per_problem is None:
            assert report["plainmp_ms"][0] > 0
            assert report["plainmp_estimated_ms"] is None
            cpu_ms = report["plainmp_ms"][0]
        else:
            assert report["plainmp_ms"] == []
            assert report["plainmp_ms_per_problem"] == cpu_ms_per_problem
            assert report["plainmp_estimated_ms"] == 7.0
            assert "Measuring plainmp" not in result.stdout
            assert "plainmp (extrapolated)" in result.stdout
            cpu_ms = report["plainmp_estimated_ms"]
        assert report["tenkai_ms"][0] > 0
        assert report["speedup"] == cpu_ms / report["tenkai_ms"][0]


def test_problem_keeps_all_spheres_and_full_box_extents(
    compare: ModuleType, model: ModelMetadata
) -> None:
    problem = compare.make_problem(URDF, SPHERES, model)
    checker = problem.global_ineq_const
    assert len(checker.get_all_spheres()) == 69
    sdf = checker.get_sdf()
    assert sdf.evaluate(np.array([0.45, 0, 0.8])) == pytest.approx(0)
    assert sdf.evaluate(np.array([0.95, 1, 0.8])) == pytest.approx(0)
    assert sdf.evaluate(np.array([0.95, 0, 0.825])) == pytest.approx(0)
    assert sdf.evaluate(np.array([0.95, 0, 0.8])) == pytest.approx(-0.025)
    assert sdf.evaluate(np.array([3.0, 0, 0.1])) == pytest.approx(0.1)
    assert checker.is_valid(np.array(compare.START))
    assert checker.is_valid(np.array(compare.GOAL))


def test_cpu_planning_benchmark(compare: ModuleType, model: ModelMetadata) -> None:
    problem = compare.make_problem(URDF, SPHERES, model)
    settings = compare.PlainmpRRTCSettings()
    settings.max_samples = 8192
    solver = compare.OMPLSolver(
        compare.OMPLSolverConfig(
            n_max_call=10000000, algorithm_range=1.0, plainmp_rrtc_settings=settings
        )
    )
    compare.set_log_level_none()
    compare.set_random_seed(42)

    def solve() -> int:
        result = solver.solve(problem)
        assert result.success
        path = result.traj.numpy()
        np.testing.assert_array_equal(path[0], compare.START)
        np.testing.assert_array_equal(path[-1], compare.GOAL)
        assert all(problem.global_ineq_const.is_valid(point) for point in path)
        return 1

    times = compare.measure(solve, batch=1, repeats=2)
    assert len(times) == 2
    assert all(milliseconds > 0 for milliseconds in times)


@pytest.mark.parametrize("successes", [[1], [2, 1]])
def test_failed_solve_is_not_reported_as_speedup(compare: ModuleType, successes: list[int]) -> None:
    results = iter(successes)
    with pytest.raises(RuntimeError, match="warmup|solved 1/2"):
        compare.measure(lambda: next(results), batch=2, repeats=1)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_cpu_estimate(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["compare_rrtc.py", "--cpu-ms-per-problem", value])
    with pytest.raises(SystemExit) as failure:
        compare.main()
    assert failure.value.code == 2
