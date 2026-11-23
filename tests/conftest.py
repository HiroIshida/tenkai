from pathlib import Path

import pytest
from helpers import FakeCompiler

from tenkai.build import Backend, target_arch


@pytest.fixture
def fake_compiler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCompiler:
    compiler = FakeCompiler()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("tenkai.build.subprocess.run", compiler.run)
    monkeypatch.setattr("tenkai.build.compiler_command", compiler.command)
    return compiler


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-cuda", action="store_true", help="compile generated kernels with nvcc")
    parser.addoption(
        "--run-hip", action="store_true", help="compile with hipcc; select HIP for GPU tests"
    )
    parser.addoption("--run-gpu", action="store_true", help="execute kernels and planners on a GPU")
    parser.addoption("--gpu-arch", help="compiler target (default: compute_75 or gfx1201)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-cuda") and config.getoption("--run-hip"):
        raise pytest.UsageError("select one of --run-cuda or --run-hip per test run")
    for item in items:
        if "gpu_compile" in item.keywords and not (
            config.getoption("--run-cuda") or config.getoption("--run-hip")
        ):
            item.add_marker(
                pytest.mark.skip(reason="pass --run-cuda or --run-hip to compile kernels")
            )
        if "gpu" in item.keywords and not config.getoption("--run-gpu"):
            item.add_marker(pytest.mark.skip(reason="pass --run-gpu to execute on a GPU"))


@pytest.fixture(scope="session")
def gpu_backend(request: pytest.FixtureRequest) -> Backend:
    return "hip" if request.config.getoption("--run-hip") else "cuda"


@pytest.fixture(scope="session")
def gpu_arch(request: pytest.FixtureRequest, gpu_backend: Backend) -> str:
    return target_arch(gpu_backend, request.config.getoption("--gpu-arch"))
