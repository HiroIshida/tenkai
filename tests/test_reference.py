"""Frozen Rust outputs remain exact across precision, CSE and scheduling modes."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import pytest

from rkern.compiler import ModelMetadata, generate
from rkern.cuda import MathMode, Options
from rkern.dag import Optimization

ROOT = Path(__file__).resolve().parents[1]


class Reference(TypedDict):
    name: str
    urdf: str
    spheres: str
    math: MathMode
    optimization: Optimization
    live_budget: int | None
    sha256: dict[str, str]
    model: ModelMetadata


REFERENCES: list[Reference] = json.loads((ROOT / "tests/reference/compiler.json").read_text())


@pytest.mark.parametrize("reference", REFERENCES, ids=[item["name"] for item in REFERENCES])
def test_generated_sources_match_frozen_rust(reference: Reference, tmp_path: Path) -> None:
    options = Options(reference["math"], reference["optimization"], reference["live_budget"])
    generate(ROOT / reference["urdf"], ROOT / reference["spheres"], tmp_path, options, True)
    actual = {}
    for path in tmp_path.glob("*.cuh"):
        actual[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == reference["sha256"]
    assert json.loads((tmp_path / "model.json").read_text()) == reference["model"]
