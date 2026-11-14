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
    "flags,live_budget",
    [
        ([], None),
        (["--scheduling"], 64),
        (["--no-scheduling"], None),
        (["--live-budget", "16"], 16),
    ],
)
def test_cli_generates_sfu_kernels(
    tmp_path: Path, flags: list[str], live_budget: int | None
) -> None:
    cli([*arguments(tmp_path), *flags])
    model = json.loads((tmp_path / "model.json").read_text())
    assert model["math"] == "f32-sfu"
    assert model["scalar"] == "float"
    assert model["live_budget"] == live_budget


@pytest.mark.parametrize("flag", ["--float", "--double", "--sfu", "--no-sfu"])
def test_cli_rejects_math_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    output = tmp_path / "generated"
    with pytest.raises(SystemExit) as failure:
        cli([*arguments(output), flag])
    assert failure.value.code == 2
    assert flag in capsys.readouterr().err
    assert not output.exists()
