#!/usr/bin/env python3
"""Rebuild the saved Rust artifacts with Python and compare every binary byte."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import TypedDict

from common import copy_rrt_bindings

from rkern import Options
from rkern.compiler import generate
from rkern.cuda import MathMode

ROOT = Path(__file__).resolve().parents[1]


class ReferenceMode(TypedDict):
    commands: dict[str, list[str]]
    sha256: dict[str, str]


class Manifest(TypedDict):
    build_root: str
    modes: dict[MathMode, ReferenceMode]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(reference: Path, report: Path) -> None:
    manifest: Manifest = json.loads((reference / "manifest.json").read_text())
    original_hashes: dict[str, str] = json.loads((reference / "source-manifest.json").read_text())
    for name, expected_hash in original_hashes.items():
        if digest(reference / name) != expected_hash:
            raise RuntimeError("saved reference changed: " + name)
    build_root = Path(manifest["build_root"])
    results: dict[str, dict[str, str]] = {}
    for mode, record in manifest["modes"].items():
        output = build_root / mode
        output.mkdir(parents=True, exist_ok=True)
        generate(
            ROOT / "assets/fetch/fetch.urdf",
            ROOT / "assets/fetch/collision.yaml",
            output,
            Options(mode),
        )
        matched = {}
        for name in ("collision.cuh", "rrt.cuh"):
            expected = reference / mode / name
            if (output / name).read_bytes() != expected.read_bytes():
                raise RuntimeError(f"generated source differs: {mode}/{name}")
            matched[name] = digest(output / name)
        if json.loads((output / "model.json").read_text()) != json.loads(
            (reference / mode / "model.json").read_text()
        ):
            raise RuntimeError("metadata differs: " + mode)
        shutil.copyfile(reference / mode / "config.h", output / "config.h")
        for target, recorded_command in record["commands"].items():
            command = recorded_command.copy()
            if target == "rrt":
                source = copy_rrt_bindings(output)
                source_index = command.index(str(ROOT / "bench/rrt.cu"))
                command[source_index] = str(source)
            elif target == "collision":
                source_index = command.index(str(ROOT / "bench/collision.cu"))
                command[source_index] = str(ROOT / "bench/cuda/collision.cu")
            keep_dir = Path(command[command.index("--keep-dir") + 1])
            keep_dir.mkdir(parents=True, exist_ok=True)
            filename = "collision_bench" if target == "collision" else "librrt.so"
            print(f"{mode}: compiling {filename}", flush=True)
            with (output / (target + "-verify.log")).open("w") as log:
                subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            actual = output / filename
            expected = reference / mode / filename
            if actual.read_bytes() != expected.read_bytes():
                raise RuntimeError(f"binary differs: {actual} vs {expected}")
            matched[filename] = digest(actual)
            print(f"{mode}: {filename} matches byte-for-byte", flush=True)
        results[mode] = matched
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "reference": str(reference),
                "matched_sha256": results,
                "comparison": "whole-file byte equality; no normalization",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/rust-reference")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "artifacts/migration-verification.json"
    )
    args = parser.parse_args()
    verify(args.reference.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
