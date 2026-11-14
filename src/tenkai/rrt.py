"""Build and run robot-specific single-candidate GPU RRT-Connect planners."""

import ctypes as ct
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Literal, NotRequired, Self, TypedDict

import numpy as np
import numpy.typing as npt

from .build import Backend, compiler_command, copy_rrt_bindings, target_arch
from .compiler import ModelMetadata, generate
from .cuda import Options

type Status = Literal["success", "round_limit", "node_limit", "invalid_start", "invalid_goal"]

STATUSES: tuple[Status, ...] = (
    "success",
    "round_limit",
    "node_limit",
    "invalid_start",
    "invalid_goal",
)


def check_count(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value < 2**32:
        raise ValueError(f"{name} must be between {minimum} and 2**32 - 1")


@dataclass(frozen=True)
class RRTConfig:
    max_nodes: int = 8192
    max_rounds: int = 1024
    step_size: float = 1.0
    goal_bias: float = 0.1

    def validate(self) -> None:
        check_count(self.max_nodes, "max_nodes", 2)
        check_count(self.max_rounds, "max_rounds")
        if not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be finite and positive")
        if not math.isfinite(self.goal_bias) or not 0 <= self.goal_bias <= 1:
            raise ValueError("goal_bias must be finite and between 0 and 1")


class RRTResult(TypedDict):
    status: Status
    nodes: int
    rounds: int
    path_length: int


class BuildInfo(TypedDict):
    nvcc: NotRequired[str]
    hipcc: NotRequired[str]
    backend: NotRequired[Backend]
    compile_command: list[str]
    max_registers: int | None


@dataclass
class RRTBuild:
    library: Path
    model: ModelMetadata
    toolchain: BuildInfo
    timings: dict[str, float] = field(default_factory=dict)

    def planner(
        self,
        geometry: npt.ArrayLike,
        *,
        config: RRTConfig = RRTConfig(),
        problem_count: int = 1,
        resolution: float = 0.05,
    ) -> "Planner":
        """Allocate a planner; close it explicitly or use it as a context manager."""
        return Planner(self.library, self.model, config, problem_count, geometry, resolution)


def build_rrt(
    urdf: str | Path,
    spheres: str | Path,
    output: str | Path,
    *,
    options: Options = Options(),
    backend: Backend = "cuda",
    arch: str | None = None,
    max_registers: int | None = None,
) -> RRTBuild:
    """Generate and compile RRT-Connect into output/librrt.so using nvcc or hipcc.

    Requires the selected GPU toolkit, but no GPU. Each call
    rebuilds the library. Use separate output directories for distinct builds.
    HIP defaults to gfx1201 (RX 9070 XT) and requires wave32 at runtime.
    The result includes codegen, nvcc/hipcc, and total wall times in seconds.
    """
    begin = time.perf_counter()
    arch = target_arch(backend, arch)
    if max_registers is None:
        max_registers = 128 if backend == "cuda" and options.math == "f32" else 0
    check_count(max_registers, "max_registers", 0)
    if max_registers != 0 and not 32 <= max_registers <= 255:
        raise ValueError("max_registers must be 0 or between 32 and 255")
    if backend == "hip" and max_registers:
        raise ValueError("max_registers is only supported by the CUDA backend")
    directory = Path(output).resolve()
    stages = generate(Path(urdf), Path(spheres), directory, options)
    model: ModelMetadata = json.loads((directory / "model.json").read_text())
    if not model["control_joints"]:
        raise ValueError("RRT requires at least one control_joint_names entry")
    command, version = compiler_command(backend, arch)
    compiler_name = "hipcc" if backend == "hip" else "nvcc"
    source = copy_rrt_bindings(directory)
    library = directory / "librrt.so"
    command.extend(["-O3", "-shared"])
    if backend == "cuda":
        command.append("-Xcompiler")
    command.extend(
        [
            "-fPIC",
            "-DTENKAI_MODEL_NAMESPACE=" + options.namespace,
            "-I",
            str(directory),
            str(source),
            "-o",
            str(library),
        ]
    )
    if max_registers:
        command.append("--maxrregcount=" + str(max_registers))
    log_path = directory / "compile.log"
    with log_path.open("w") as log:
        compile_begin = time.perf_counter()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        compile_seconds = time.perf_counter() - compile_begin
    if result.returncode:
        raise RuntimeError(f"{compiler_name} failed; see {log_path}\n{log_path.read_text()}")
    info = BuildInfo(backend=backend, compile_command=command, max_registers=max_registers or None)
    if backend == "hip":
        info["hipcc"] = version
    else:
        info["nvcc"] = version
    timings = {
        "codegen": stages["total"],
        compiler_name: compile_seconds,
        "total": time.perf_counter() - begin,
    }
    return RRTBuild(library, model, info, timings)


class NativeConfig(ct.Structure):
    _fields_ = [
        ("max_nodes", ct.c_uint32),
        ("max_rounds", ct.c_uint32),
        ("step_size", ct.c_double),
        ("goal_bias", ct.c_double),
    ]


class NativeResult(ct.Structure):
    _fields_ = [
        ("status", ct.c_uint32),
        ("node_count", ct.c_uint32),
        ("rounds", ct.c_uint32),
        ("path_length", ct.c_uint32),
    ]


def unsigned_array[T: (np.uint32, np.uint64)](
    values: npt.ArrayLike, dtype: type[T], name: str
) -> npt.NDArray[T]:
    array = np.asarray(values)
    if array.size:
        if array.dtype.kind not in ("i", "u"):
            # NumPy can infer float64 for a list mixing small ints and large uint64 seeds.
            array = np.asarray(values, dtype=object)
            for value in array.flat:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                    raise ValueError(f"{name} must contain integers")
        if int(array.min()) < 0 or int(array.max()) > np.iinfo(dtype).max:
            raise ValueError(f"{name} values must fit {np.dtype(dtype).name}")
    return np.ascontiguousarray(array, dtype=dtype)


class Planner:
    """A reusable batch of RRT-Connect problems with owned GPU search and scene buffers.

    Geometry has shape (environments, 7): box center x/y/z, box full extents
    x/y/z, and ground offset. The ground plane is z = -offset. Each environment
    contains one axis-aligned box and one horizontal ground plane. Uncontrolled
    joints remain at zero.
    """

    def __init__(
        self,
        library: str | Path,
        model: ModelMetadata,
        config: RRTConfig,
        problem_count: int,
        geometry: npt.ArrayLike,
        resolution: float,
    ) -> None:
        config.validate()
        check_count(problem_count, "problem_count")
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        scene = np.ascontiguousarray(geometry, dtype=np.float64)
        if scene.ndim != 2 or scene.shape[1] != 7:
            raise ValueError("geometry must have shape (environments, 7)")
        check_count(len(scene), "environment_count")
        if not np.isfinite(scene).all():
            raise ValueError("geometry must be finite")
        if (scene[:, 3:6] < 0).any():
            raise ValueError("box full extents must be nonnegative")
        self.problem_count = problem_count
        self.environment_count = len(scene)
        self.dimension = len(model["control_joints"])
        self.fk_dof = len(model["input_joint_names"])
        if not 0 < self.dimension <= self.fk_dof:
            raise ValueError("model must have movable control joints")
        self.max_nodes = config.max_nodes
        lower = np.array([joint["lower"] for joint in model["control_joints"]], dtype=np.float64)
        upper = np.array([joint["upper"] for joint in model["control_joints"]], dtype=np.float64)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower >= upper).any():
            raise ValueError("control joint bounds must be finite and increasing")
        widths = np.full(self.dimension, resolution, dtype=np.float64)
        self.handle = ct.c_void_p()
        self.library = ct.CDLL(str(Path(library).resolve()))
        floats = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ids = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
        seeds = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
        flags = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        self.library.rrt_error.argtypes = []
        self.library.rrt_error.restype = ct.c_char_p
        create = self.library.rrtc_create
        create.argtypes = [
            ct.POINTER(NativeConfig),
            ct.c_uint32,
            ct.c_uint32,
            floats,
            floats,
            floats,
            floats,
            ct.POINTER(ct.c_void_p),
        ]
        create.restype = ct.c_int
        self.library.rrt_destroy.argtypes = [ct.c_void_p]
        self.library.rrt_destroy.restype = None
        self.library.rrt_solve.argtypes = [
            ct.c_void_p,
            floats,
            floats,
            ids,
            seeds,
            ct.POINTER(NativeResult),
        ]
        self.library.rrt_solve.restype = ct.c_int
        self.library.rrt_path.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
            floats,
            ct.c_uint32,
            ct.POINTER(ct.c_uint32),
        ]
        self.library.rrt_path.restype = ct.c_int
        self.library.rrt_check.argtypes = [ct.c_void_p, floats, ids, ct.c_uint32, flags]
        self.library.rrt_check.restype = ct.c_int
        native = NativeConfig(**asdict(config))
        self.check(
            create(
                ct.byref(native),
                problem_count,
                len(scene),
                scene,
                lower,
                upper,
                widths,
                ct.byref(self.handle),
            )
        )

    def check(self, status: int) -> None:
        if status:
            raise RuntimeError(self.library.rrt_error().decode())

    def _require_open(self) -> None:
        if not self.handle.value:
            raise RuntimeError("planner is closed")

    def close(self) -> None:
        if self.handle.value:
            self.library.rrt_destroy(self.handle)
            self.handle = ct.c_void_p()

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def solve(
        self,
        starts: npt.ArrayLike,
        goals: npt.ArrayLike,
        environment_ids: npt.ArrayLike,
        seeds: npt.ArrayLike,
    ) -> list[RRTResult]:
        """Solve a batch; results and paths replace those from the previous solve."""
        self._require_open()
        start = np.ascontiguousarray(starts, dtype=np.float64)
        goal = np.ascontiguousarray(goals, dtype=np.float64)
        ids = unsigned_array(environment_ids, np.uint32, "environment_ids")
        seed = unsigned_array(seeds, np.uint64, "seeds")
        if start.shape != (self.problem_count, self.dimension) or goal.shape != start.shape:
            raise ValueError("starts and goals must have shape (problem_count, dimension)")
        if ids.shape != (self.problem_count,) or seed.shape != ids.shape:
            raise ValueError("environment_ids and seeds must have shape (problem_count,)")
        if (ids >= self.environment_count).any():
            raise ValueError("environment ID out of range")
        results = (NativeResult * self.problem_count)()
        self.check(self.library.rrt_solve(self.handle, start, goal, ids, seed, results))
        return [
            RRTResult(
                status=STATUSES[r.status],
                nodes=r.node_count,
                rounds=r.rounds,
                path_length=r.path_length,
            )
            for r in results
        ]

    def path(self, problem_id: int) -> npt.NDArray[np.float64]:
        """Return a copy of the last path, with one column per control joint."""
        self._require_open()
        check_count(problem_id, "problem_id", 0)
        if problem_id >= self.problem_count:
            raise ValueError("problem_id out of range")
        path = np.empty((self.max_nodes, self.dimension), dtype=np.float64)
        length = ct.c_uint32()
        self.check(
            self.library.rrt_path(self.handle, problem_id, path, self.max_nodes, ct.byref(length))
        )
        return path[: length.value].copy()

    def collisions(
        self, configurations: npt.ArrayLike, environment_ids: npt.ArrayLike
    ) -> npt.NDArray[np.uint8]:
        """Check full FK configurations, in model['input_joint_names'] order."""
        self._require_open()
        q = np.ascontiguousarray(configurations, dtype=np.float64)
        ids = unsigned_array(environment_ids, np.uint32, "environment_ids")
        if q.ndim != 2 or q.shape[1] != self.fk_dof or ids.shape != (len(q),):
            raise ValueError("invalid collision array shapes")
        check_count(len(q), "configuration_count", 0)
        if not np.isfinite(q).all():
            raise ValueError("configurations must be finite")
        if (ids >= self.environment_count).any():
            raise ValueError("environment ID out of range")
        flags = np.empty(len(q), dtype=np.uint8)
        self.check(self.library.rrt_check(self.handle, q, ids, len(q), flags))
        return flags
