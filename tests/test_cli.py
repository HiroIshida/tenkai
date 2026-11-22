import json
from pathlib import Path

import pytest

from tenkai.cli import cli

FIXTURES = Path(__file__).with_name("fixtures")


def arguments(output: Path) -> list[str]:
    return [
        "--urdf",
        str(FIXTURES / "fk_chain.urdf"),
        "--spheres",
        str(FIXTURES / "compiler/chain.yaml"),
        str(output),
    ]


@pytest.mark.parametrize(
    "flags,live_budget,sphere_batch_size,ground",
    [
        ([], None, 8, True),
        (["--scheduling"], 64, 8, True),
        (["--no-scheduling"], None, 8, True),
        (["--live-budget", "16"], 16, 8, True),
        (["--sphere-batch-size", "3"], None, 3, True),
        (["--no-ground"], None, 8, False),
    ],
)
def test_cli_generates_sfu_kernels(
    tmp_path: Path,
    flags: list[str],
    live_budget: int | None,
    sphere_batch_size: int,
    ground: bool,
) -> None:
    cli([*arguments(tmp_path), *flags])
    model = json.loads((tmp_path / "model.json").read_text())
    assert model["math"] == "f32-sfu"
    assert model["scalar"] == "float"
    assert model["live_budget"] == live_budget
    assert model["sphere_batch_size"] == sphere_batch_size
    assert model["ground"] is ground


@pytest.mark.parametrize("size", ["0", "-1"])
def test_cli_rejects_invalid_sphere_batch_size(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], size: str
) -> None:
    output = tmp_path / "generated"
    with pytest.raises(SystemExit) as failure:
        cli([*arguments(output), "--sphere-batch-size", size])
    assert failure.value.code == 2
    assert "sphere batch size must be a positive integer" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    "flag",
    [
        "--float",
        "--double",
        "--sfu",
        "--no-sfu",
        "--self-collision-broad-phase",
        "--no-self-collision-broad-phase",
    ],
)
def test_cli_rejects_removed_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    output = tmp_path / "generated"
    with pytest.raises(SystemExit) as failure:
        cli([*arguments(output), flag])
    assert failure.value.code == 2
    assert flag in capsys.readouterr().err
    assert not output.exists()
