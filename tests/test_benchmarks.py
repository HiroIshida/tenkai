"""Standalone benchmark entry points and compiler timing."""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import tyro

from tenkai.build import Backend

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/fetch/fetch.urdf"
SPHERES = ROOT / "assets/fetch/collision.yaml"


@pytest.fixture
def compare(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("plainmp", reason="install the bench extra for plainmp integration tests")
    monkeypatch.syspath_prepend(str(ROOT / "bench"))
    return importlib.import_module("compare_rrtc")


@pytest.mark.parametrize("script", ["compiler.py", "compare_rrtc.py"])
def test_script_help(script: str, tmp_path: Path) -> None:
    if script == "compare_rrtc.py":
        pytest.importorskip("plainmp")
    result = subprocess.run(
        [sys.executable, str(ROOT / "bench" / script), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--repeats" in result.stdout


@pytest.mark.gpu_compile
def test_standalone_benchmark_report(
    tmp_path: Path,
    gpu_backend: Backend,
    gpu_arch: str,
) -> None:
    destination = tmp_path / "compiler.py"
    shutil.copyfile(ROOT / "bench/compiler.py", destination)
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
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "results/results.json").read_text())
    assert report["math"] == "f32-sfu"
    assert report["backend"] == gpu_backend
    compiler_name = "hipcc" if gpu_backend == "hip" else "nvcc"
    sample = report["samples_seconds"][0]
    assert sample["codegen"] > 0
    assert sample[compiler_name] > 0
    assert sample["total"] >= sample["codegen"] + sample[compiler_name]
    assert report["median_seconds"] == sample


@pytest.mark.parametrize("successes", [[1], [2, 1]])
def test_failed_solve_is_not_reported_as_speedup(compare: ModuleType, successes: list[int]) -> None:
    results = iter(successes)
    with pytest.raises(RuntimeError, match="warmup|solved 1/2"):
        compare.measure(lambda: next(results), batch=2, repeats=1)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_cpu_estimate(compare: ModuleType, value: str) -> None:
    with pytest.raises(SystemExit) as failure:
        tyro.cli(compare.main, args=["--cpu-ms-per-problem", value])
    assert failure.value.code == 2
