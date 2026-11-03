#!/usr/bin/env python3
"""Generate the Fetch benchmark's CUDA kernels."""

import sys
from pathlib import Path

from rkern.cli import main

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(
        [
            "--urdf",
            str(root / "assets/fetch/fetch.urdf"),
            "--spheres",
            str(root / "assets/fetch/collision.yaml"),
            *sys.argv[1:],
        ]
    )
