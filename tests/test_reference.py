"""Generated source bytes and model metadata match the recorded outputs."""

import hashlib
import json
import re
from pathlib import Path
from typing import TypedDict

import pytest

from tenkai.compiler import ModelMetadata, generate
from tenkai.cuda import Options
from tenkai.dag import Optimization

ROOT = Path(__file__).resolve().parents[1]


class Reference(TypedDict):
    name: str
    urdf: str
    spheres: str
    optimization: Optimization
    live_budget: int | None
    sha256: dict[str, str]
    model: ModelMetadata


REFERENCES: list[Reference] = json.loads((ROOT / "tests/reference/compiler.json").read_text())


@pytest.mark.parametrize("reference", REFERENCES, ids=[item["name"] for item in REFERENCES])
def test_generated_sources_match_reference(reference: Reference, tmp_path: Path) -> None:
    options = Options(optimization=reference["optimization"], live_budget=reference["live_budget"])
    generate(ROOT / reference["urdf"], ROOT / reference["spheres"], tmp_path, options, True)
    actual: dict[str, str] = {}
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.suffix in (".cu", ".cuh", ".hpp"):
            source = path.read_text()
            assert "TPL_" not in source
            for include in re.findall(r'^#include "([^"]+)"', source, re.MULTILINE):
                assert (path.parent / include).is_file(), (path, include)
            actual[path.relative_to(tmp_path).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    assert actual == reference["sha256"]
    assert json.loads((tmp_path / "model.json").read_text()) == reference["model"]
