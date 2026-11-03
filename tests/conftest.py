import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-cuda", action="store_true", help="compile generated kernels with nvcc")
    parser.addoption("--run-gpu", action="store_true", help="run CUDA planner tests on a GPU")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "cuda" in item.keywords and not config.getoption("--run-cuda"):
            item.add_marker(pytest.mark.skip(reason="pass --run-cuda to compile CUDA"))
        if "gpu" in item.keywords and not config.getoption("--run-gpu"):
            item.add_marker(pytest.mark.skip(reason="pass --run-gpu to execute on a GPU"))
