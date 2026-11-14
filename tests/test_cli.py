import json
from pathlib import Path

import pytest

from tenkai import Options
from tenkai.cli import main

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
    "flags,math_mode",
    [
        ([], "f32-sfu"),
        (["--sfu"], "f32-sfu"),
        (["--no-sfu"], "f32"),
        (["--float", "--no-sfu"], "f32"),
    ],
)
def test_cli_generates_float_kernels(tmp_path: Path, flags: list[str], math_mode: str) -> None:
    main([*arguments(tmp_path), *flags])
    model = json.loads((tmp_path / "model.json").read_text())
    assert model["math"] == math_mode
    assert model["scalar"] == "float"


def test_fp64_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="math mode must be f32 or f32-sfu"):
        Options(math="f64")  # type: ignore[arg-type]


def test_cli_rejects_double(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "generated"
    with pytest.raises(SystemExit) as failure:
        main([*arguments(output), "--double"])
    assert failure.value.code == 2
    assert "unrecognized arguments: --double" in capsys.readouterr().err
    assert not output.exists()
