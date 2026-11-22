"""Shared validation for native runtime arguments."""

import numpy as np


def check_count(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value < 2**32:
        raise ValueError(f"{name} must be between {minimum} and 2**32 - 1")
