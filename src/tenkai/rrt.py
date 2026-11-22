"""Build and run robot-specific single-candidate GPU RRT-Connect planners."""

import ctypes as ct
import json
import math
import subprocess
import time
from collections.abc import Set
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Literal, NotRequired, Self, TypedDict

import numpy as np
import numpy.typing as npt

from tenkai.build import Backend, compiler_command, target_arch
from tenkai.compiler import ModelMetadata, generate
from tenkai.cuda import Options
from tenkai.sdf import CollisionObjectType, Geometry, SceneData, pack_geometry
from tenkai.shortcut import NativeShortcutConfig, ShortcutConfig
from tenkai.validation import check_count

type Status = Literal["success", "round_limit", "node_limit", "invalid_start", "invalid_goal"]

STATUSES: tuple[Status, ...] = (
    "success",
    "round_limit",
    "node_limit",
    "invalid_start",
    "invalid_goal",
)


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
        geometry: Geometry,
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
    specialize_types: Set[CollisionObjectType] | None = None,
) -> RRTBuild:
    """Generate and compile RRT-Connect into output/librrt.so using nvcc or hipcc.

    Requires the selected GPU toolkit, but no GPU. Each call
    rebuilds the library. Use separate output directories for distinct builds.
    HIP defaults to gfx1201 (RX 9070 XT) and requires wave32 at runtime.
    The result includes codegen, nvcc/hipcc, and total wall times in seconds.

    specialize_types adds one search kernel for that union of sdf obstacle
    classes. Counts and poses stay dynamic. Each solve uses it if every
    referenced environment fits; otherwise the whole batch uses the generic
    kernel. None (the default) or all four classes builds only the generic
    kernel. An empty set specializes obstacle-free environments.
    """
    begin = time.perf_counter()
    arch = target_arch(backend, arch)
    if max_registers is None:
        max_registers = 0
    check_count(max_registers, "max_registers", 0)
    if max_registers != 0 and not 32 <= max_registers <= 255:
        raise ValueError("max_registers must be 0 or between 32 and 255")
    if backend == "hip" and max_registers:
        raise ValueError("max_registers is only supported by the CUDA backend")

    directory = Path(output).resolve()
    stages = generate(
        Path(urdf), Path(spheres), directory, options, specialize_types=specialize_types
    )
    model: ModelMetadata = json.loads((directory / "model.json").read_text())
    if not model["control_joints"]:
        raise ValueError("RRT requires at least one control_joint_names entry")

    command, version = compiler_command(backend, arch)
    compiler_name = "hipcc" if backend == "hip" else "nvcc"
    source = directory / "rrt/bindings.cu"
    library = directory / "librrt.so"
    command.extend(["-O3", "-shared"])
    if backend == "cuda":
        command.append("-Xcompiler")
    command.extend(
        [
            "-fPIC",
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


class NativeScene(ct.Structure):
    _fields_ = [
        ("ranges", ct.POINTER(ct.c_uint32)),
        ("boxes", ct.POINTER(ct.c_double)),
        ("spheres", ct.POINTER(ct.c_double)),
        ("cylinders", ct.POINTER(ct.c_double)),
        ("grounds", ct.POINTER(ct.c_double)),
        ("box_count", ct.c_uint32),
        ("sphere_count", ct.c_uint32),
        ("cylinder_count", ct.c_uint32),
        ("ground_count", ct.c_uint32),
    ]

    @classmethod
    def from_scene(cls, scene: SceneData) -> Self:
        return cls(
            scene.ranges.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            scene.boxes.ctypes.data_as(ct.POINTER(ct.c_double)),
            scene.spheres.ctypes.data_as(ct.POINTER(ct.c_double)),
            scene.cylinders.ctypes.data_as(ct.POINTER(ct.c_double)),
            scene.grounds.ctypes.data_as(ct.POINTER(ct.c_double)),
            len(scene.boxes),
            len(scene.spheres),
            len(scene.cylinders),
            len(scene.grounds),
        )


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

    Geometry is a list of sdf collision objects for one environment, or a list
    of such lists for multiple environments. An empty list is an empty scene.
    Uncontrolled joints remain at zero.
    """

    def __init__(
        self,
        library: str | Path,
        model: ModelMetadata,
        config: RRTConfig,
        problem_count: int,
        geometry: Geometry,
        resolution: float,
    ) -> None:
        config.validate()
        check_count(problem_count, "problem_count")
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")

        scene = pack_geometry(geometry)

        self.problem_count = problem_count
        self.environment_count = len(scene.ranges)
        self.dimension = len(model["control_joints"])
        self.fk_dof = len(model["input_joint_names"])
        if not 0 < self.dimension <= self.fk_dof:
            raise ValueError("model must have movable control joints")

        self.max_nodes = config.max_nodes
        self._path_lengths: list[int] | None = None
        lower = np.array([joint["lower"] for joint in model["control_joints"]], dtype=np.float64)
        upper = np.array([joint["upper"] for joint in model["control_joints"]], dtype=np.float64)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower >= upper).any():
            raise ValueError("control joint bounds must be finite and increasing")
        resolutions = np.full(self.dimension, resolution, dtype=np.float64)

        self.handle = ct.c_void_p()
        self.library = ct.CDLL(str(Path(library).resolve()))
        float_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        id_array = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
        seed_array = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
        flag_array = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")

        self.library.rrt_error.argtypes = []
        self.library.rrt_error.restype = ct.c_char_p

        create = self.library.rrtc_create_scene
        create.argtypes = [
            ct.POINTER(NativeConfig),
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(NativeScene),
            float_array,
            float_array,
            float_array,
            ct.POINTER(ct.c_void_p),
        ]
        create.restype = ct.c_int

        self.library.rrt_destroy.argtypes = [ct.c_void_p]
        self.library.rrt_destroy.restype = None

        self.library.rrt_solve.argtypes = [
            ct.c_void_p,
            float_array,
            float_array,
            id_array,
            seed_array,
            ct.POINTER(NativeResult),
            ct.POINTER(NativeShortcutConfig),
        ]
        self.library.rrt_solve.restype = ct.c_int

        self.library.rrt_path.argtypes = [
            ct.c_void_p,
            ct.c_uint32,
            float_array,
            ct.c_uint32,
            ct.POINTER(ct.c_uint32),
        ]
        self.library.rrt_path.restype = ct.c_int

        self.library.rrt_check.argtypes = [
            ct.c_void_p,
            float_array,
            id_array,
            ct.c_uint32,
            flag_array,
        ]
        self.library.rrt_check.restype = ct.c_int

        native_config = NativeConfig(**asdict(config))
        native_scene = NativeScene.from_scene(scene)
        status = create(
            ct.byref(native_config),
            problem_count,
            self.environment_count,
            ct.byref(native_scene),
            lower,
            upper,
            resolutions,
            ct.byref(self.handle),
        )
        self.check(status)

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
        *,
        shortcut: ShortcutConfig | None = None,
    ) -> list[RRTResult]:
        """Solve, optionally shorten, and replace the stored paths.

        Shortcutting uses the same per-problem seeds and environments as planning.
        path_length reports the final vertex count; nodes and rounds describe the search.
        """
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

        native_shortcut = None if shortcut is None else NativeShortcutConfig.from_config(shortcut)
        results = (NativeResult * self.problem_count)()
        self._path_lengths = None
        status = self.library.rrt_solve(
            self.handle,
            start,
            goal,
            ids,
            seed,
            results,
            None if native_shortcut is None else ct.byref(native_shortcut),
        )
        self.check(status)
        self._path_lengths = [result.path_length for result in results]

        return [
            RRTResult(
                status=STATUSES[result.status],
                nodes=result.node_count,
                rounds=result.rounds,
                path_length=result.path_length,
            )
            for result in results
        ]

    def path(self, problem_id: int) -> npt.NDArray[np.float64]:
        """Return a copy of the last path, with one column per control joint."""
        self._require_open()
        check_count(problem_id, "problem_id", 0)
        if problem_id >= self.problem_count:
            raise ValueError("problem_id out of range")

        lengths = self._require_paths()
        path = np.empty((lengths[problem_id], self.dimension), dtype=np.float64)
        length = ct.c_uint32()
        status = self.library.rrt_path(self.handle, problem_id, path, len(path), ct.byref(length))
        self.check(status)

        return path[: length.value].copy()

    def _require_paths(self) -> list[int]:
        self._require_open()
        if self._path_lengths is None:
            raise RuntimeError("RRT path unavailable; call solve first")
        return self._path_lengths

    def paths(self) -> list[npt.NDArray[np.float64]]:
        """Download all stored paths in one transfer, including empty failed paths."""
        lengths = self._require_paths()
        packed = np.empty((sum(lengths), self.dimension), dtype=np.float64)
        operation = self.library.rrt_paths
        operation.argtypes = [
            ct.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            ct.c_uint64,
        ]
        operation.restype = ct.c_int
        self.check(operation(self.handle, packed, len(packed)))

        paths: list[npt.NDArray[np.float64]] = []
        offset = 0
        for length in lengths:
            paths.append(packed[offset : offset + length].copy())
            offset += length
        return paths

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
        status = self.library.rrt_check(self.handle, q, ids, len(q), flags)
        self.check(status)

        return flags
