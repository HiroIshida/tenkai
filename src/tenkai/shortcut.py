"""Random arc-length shortcuts for collision-free paths on CUDA and HIP."""

import ctypes as ct
import math
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class ShortcutConfig:
    max_steps: int | None = None
    max_empty_steps: int | None = None
    range_ratio: float = 0.33
    snap_to_vertex: float = 0.005

    def validate(self) -> None:
        from tenkai.rrt import check_count

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


class NativeShortcutConfig(ct.Structure):
    _fields_ = [
        ("max_steps", ct.c_uint32),
        ("max_empty_steps", ct.c_uint32),
        ("range_ratio", ct.c_double),
        ("snap_to_vertex", ct.c_double),
    ]

    @classmethod
    def from_config(cls, config: ShortcutConfig) -> Self:
        config.validate()
        return cls(
            2**32 - 1 if config.max_steps is None else config.max_steps,
            2**32 - 1 if config.max_empty_steps is None else config.max_empty_steps,
            config.range_ratio,
            config.snap_to_vertex,
        )
