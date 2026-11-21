"""Environment detection for programs that execute generated GPU kernels."""

import os
import shutil

from tenkai.build import Backend


def detect_backend() -> Backend:
    """Select by compiler executable availability, preferring CUDA over HIP."""
    if shutil.which(os.environ.get("NVCC", "nvcc")) is not None:
        return "cuda"
    if shutil.which(os.environ.get("HIPCC", "hipcc")) is not None:
        return "hip"
    raise RuntimeError("Neither nvcc nor hipcc was found; install CUDA/ROCm or set NVCC/HIPCC")
