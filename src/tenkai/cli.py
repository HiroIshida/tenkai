import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import tyro

from tenkai.compiler import generate
from tenkai.cuda import Options
from tenkai.dag import Optimization

SCHEDULING = tyro.conf.create_mutex_group(required=False, title="scheduling")


def main(
    urdf: Path,
    spheres: Path,
    output: tyro.conf.Positional[Path],
    optimization: Optimization = "algebraic",
    all_kernels: bool = False,
    scheduling: Annotated[bool, SCHEDULING] = False,
    no_scheduling: Annotated[bool, SCHEDULING] = False,
    live_budget: Annotated[int | None, SCHEDULING] = None,
    sphere_batch_size: int = 8,
) -> None:
    """Generate CUDA source from a URDF model and collision sphere configuration.

    Args:
        scheduling: Schedule with a live budget of 64.
        no_scheduling: Disable scheduling (the default).
        live_budget: Schedule with a nonnegative live-value budget.
        sphere_batch_size: Compile-time maximum robot spheres on one link sharing
            an obstacle traversal. Must be positive.
    """
    if scheduling:
        live_budget = 64
    elif no_scheduling:
        live_budget = None
    try:
        options = Options(
            optimization=optimization, live_budget=live_budget, sphere_batch_size=sphere_batch_size
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error
    timings = generate(urdf, spheres, output, options, all_kernels)
    print(json.dumps(timings))


def cli(argv: Sequence[str] | None = None) -> None:
    tyro.cli(
        main,
        args=argv,
        config=(tyro.conf.DisallowNone, tyro.conf.FlagCreatePairsOff),
    )


if __name__ == "__main__":
    cli()
