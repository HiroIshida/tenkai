#!/usr/bin/env python3
"""Compare collision CUDA and metadata with saved Rust outputs after renaming."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tenkai import Options
from tenkai.compiler import generate
from tenkai.cuda import MathMode

ROOT = Path(__file__).resolve().parents[1]
MATH_MODES: tuple[MathMode, ...] = ("f32", "f32-sfu", "f64")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(reference: Path, report: Path) -> None:
    original_hashes: dict[str, str] = json.loads((reference / "source-manifest.json").read_text())
    for name, expected_hash in original_hashes.items():
        if digest(reference / name) != expected_hash:
            raise RuntimeError("saved reference changed: " + name)
    results: dict[str, dict[str, str]] = {}
    with TemporaryDirectory(prefix="tenkai-migration-") as temporary:
        for mode in MATH_MODES:
            output = Path(temporary) / mode
            generate(
                ROOT / "assets/fetch/fetch.urdf",
                ROOT / "assets/fetch/collision.yaml",
                output,
                Options(mode),
            )
            matched = {}
            for name in ("collision.cuh",):
                saved = (reference / mode / name).read_bytes()
                expected = saved.replace(b"rkern", b"tenkai").replace(b"RKERN", b"TENKAI")
                if (output / name).read_bytes() != expected:
                    raise RuntimeError(f"generated source differs: {mode}/{name}")
                matched[name] = digest(output / name)
            if json.loads((output / "model.json").read_text()) != json.loads(
                (reference / mode / "model.json").read_text()
            ):
                raise RuntimeError("metadata differs: " + mode)
            results[mode] = matched
            print(f"{mode}: collision source matches byte-for-byte after renaming", flush=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "reference": str(reference),
                "matched_sha256": results,
                "comparison": "collision source equality after rkern/RKERN -> tenkai/TENKAI",
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
