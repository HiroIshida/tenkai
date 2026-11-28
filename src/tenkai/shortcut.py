"""Random arc-length shortcuts for collision-free paths on CUDA and HIP."""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tenkai.layout import SHORTCUT_CONFIG, pack_struct
from tenkai.validation import check_count


@dataclass(frozen=True)
class ShortcutConfig:
    max_steps: int | None = None
    max_empty_steps: int | None = None
    range_ratio: float = 0.33
    snap_to_vertex: float = 0.005

    def __post_init__(self) -> None:
        for name, value in (
            ("max_steps", self.max_steps),
            ("max_empty_steps", self.max_empty_steps),
        ):
            if value is not None:
                check_count(value, name, 0)
                if value == 2**32 - 1:
                    raise ValueError(f"{name} must be less than 2**32 - 1")
        if not math.isfinite(self.range_ratio) or not 0 < self.range_ratio <= 1:
            raise ValueError("range_ratio must be finite and in (0, 1]")
        if not math.isfinite(self.snap_to_vertex) or not 0 <= self.snap_to_vertex <= 1:
            raise ValueError("snap_to_vertex must be finite and in [0, 1]")


def native_shortcut(config: ShortcutConfig) -> npt.NDArray[np.void]:
    return pack_struct(
        SHORTCUT_CONFIG,
        max_steps=2**32 - 1 if config.max_steps is None else config.max_steps,
        max_empty_steps=2**32 - 1 if config.max_empty_steps is None else config.max_empty_steps,
        range_ratio=config.range_ratio,
        snap_to_vertex=config.snap_to_vertex,
    )
