"""Generate CUDA source from a URDF model and collision sphere configuration."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .compiler import generate
from .cuda import Options
from .dag import Optimization


def optimization(value: str) -> Optimization:
    match value:
        case "none" | "basic" | "algebraic":
            return value
    raise argparse.ArgumentTypeError("optimization must be none, basic or algebraic")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization", type=optimization, default="algebraic")

    scheduling = parser.add_mutually_exclusive_group()
    scheduling.add_argument("--scheduling", dest="live_budget", action="store_const", const=64)
    scheduling.add_argument("--no-scheduling", dest="live_budget", action="store_const", const=None)
    scheduling.add_argument("--live-budget", type=int)

    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--spheres", type=Path, required=True)
    parser.add_argument("--all-kernels", action="store_true")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    if args.live_budget is not None and args.live_budget < 0:
        parser.error("live budget must be nonnegative")

    options = Options(optimization=args.optimization, live_budget=args.live_budget)
    timings = generate(args.urdf, args.spheres, args.output, options, args.all_kernels)
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
