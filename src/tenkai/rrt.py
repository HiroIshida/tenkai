"""Build and run robot-specific single-candidate GPU RRT-Connect planners."""

import ctypes as ct
import math
import time
from collections.abc import Sequence, Set
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Literal, Self, TypedDict

import numpy as np
import numpy.typing as npt

from tenkai.build import Backend, CompileOptions, compile_library
from tenkai.build import BuildInfo as BuildInfo
from tenkai.compiler import ControlJoint, ModelMetadata, generate
from tenkai.cuda import Options
from tenkai.sdf import Geometry, SceneData, SpecializationType, pack_geometry
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

    def __post_init__(self) -> None:
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


@dataclass
class RRTBuild:
    library: Path
    model: ModelMetadata
    toolchain: BuildInfo
    timings: dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False

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
    specialize_types: Set[SpecializationType] | None = None,
    cache: bool = True,
) -> RRTBuild:
    """Generate and compile RRT-Connect into output/librrt.so using nvcc or hipcc.

    Requires the selected GPU toolkit, but no GPU. Each call generates sources
    and model metadata; matching sources and compiler settings reuse a cached
    library. Pass cache=False to always compile. Use separate output directories
    for distinct builds.
    HIP defaults to gfx1201 (RX 9070 XT) and requires wave32 at runtime.
    The result includes codegen, nvcc/hipcc, and total wall times in seconds.
    Compilation time is zero on a cache hit.

    specialize_types adds one search kernel for that union of sdf obstacle
    classes. Counts and poses stay dynamic. Each solve uses it if every
    referenced environment fits; otherwise the whole batch uses the generic
    kernel. None (the default) or all three classes builds only the generic
    kernel. An empty set specializes environments without Box, Sphere, or Cylinder.
    options.ground fixes Ground presence for every environment at compile time.
    """
    begin = time.perf_counter()
    compile_options = CompileOptions(backend=backend, arch=arch, max_registers=max_registers)

    directory = Path(output).resolve()
    with TemporaryDirectory(prefix="tenkai-sources-") as temporary:
        generated = Path(temporary)
        generation = generate(
            Path(urdf), Path(spheres), generated, options, specialize_types=specialize_types
        )
        model = generation.model
        if not model["control_joints"]:
            raise ValueError("RRT requires at least one control_joint_names entry")

        compiled = compile_library(
            generated,
            "rrt/bindings.cu",
            directory / "librrt.so",
            options=compile_options,
            cache=cache,
        )

    timings = {
        "codegen": generation.timings["total"],
        compiled.compiler_name: compiled.seconds,
        "total": time.perf_counter() - begin,
    }
    return RRTBuild(
        compiled.library, model, compiled.toolchain, timings, cache_hit=compiled.cache_hit
    )


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
    ]

    @classmethod
    def from_scene(cls, scene: SceneData) -> Self:
        return cls(
            scene.ranges.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            scene.boxes.ctypes.data_as(ct.POINTER(ct.c_double)),
            scene.spheres.ctypes.data_as(ct.POINTER(ct.c_double)),
            scene.cylinders.ctypes.data_as(ct.POINTER(ct.c_double)),
            scene.grounds.ctypes.data_as(ct.POINTER(ct.c_double)) if scene.grounds.size else None,
            len(scene.boxes),
            len(scene.spheres),
            len(scene.cylinders),
        )


def _integers(values: npt.ArrayLike, name: str) -> npt.NDArray[np.generic]:
    array = np.asarray(values)
    if array.dtype.kind in ("i", "u") or not array.size:
        return array

    # NumPy can infer float64 for a list mixing small ints and large uint64 seeds.
    exact = np.asarray(values, dtype=object)
    for value in exact.flat:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must contain integers")

    return exact


def unsigned_array[T: (np.uint32, np.uint64)](
    values: npt.ArrayLike, dtype: type[T], name: str
) -> npt.NDArray[T]:
    array = _integers(values, name)
    if array.size and (int(array.min()) < 0 or int(array.max()) > np.iinfo(dtype).max):
        raise ValueError(f"{name} values must fit {np.dtype(dtype).name}")

    return np.ascontiguousarray(array, dtype=dtype)


def sampling_space(
    joints: Sequence[ControlJoint], resolution: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return control bounds and per-joint resolution after checking them as the planner stores them.

    The native planner narrows these to float32, which can overflow, collapse
    an interval, or round a resolution to zero.
    """
    lower = np.array([joint["lower"] for joint in joints], dtype=np.float64)
    upper = np.array([joint["upper"] for joint in joints], dtype=np.float64)
    resolutions = np.full(len(joints), resolution, dtype=np.float64)

    with np.errstate(over="ignore"):
        lower32, upper32, resolution32 = (
            array.astype(np.float32) for array in (lower, upper, resolutions)
        )
    span = upper32.astype(np.float64) - lower32.astype(np.float64)
    fits = (span > 0) & (span <= np.finfo(np.float32).max)
    if not (np.isfinite(lower32).all() and np.isfinite(upper32).all() and fits.all()):
        raise ValueError("control joint bounds must be finite and increasing in float32")
    if not (np.isfinite(resolution32).all() and (resolution32 > 0).all()):
        raise ValueError("resolution must be finite and positive in float32")
    steps = np.ceil(span / resolution32.astype(np.float64))
    if (steps > np.iinfo(np.uint32).max - 1).any():
        raise ValueError("resolution is too small for the control joint bounds")

    return lower, upper, resolutions


def _configure_library(library: ct.CDLL) -> None:
    float_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    id_array = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
    seed_array = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    flag_array = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")

    library.rrt_error.argtypes = []
    library.rrt_error.restype = ct.c_char_p

    library.rrtc_create_scene.argtypes = [
        ct.POINTER(NativeConfig),
        ct.c_uint32,
        ct.c_uint32,
        ct.POINTER(NativeScene),
        float_array,
        float_array,
        float_array,
        ct.POINTER(ct.c_void_p),
    ]
    library.rrtc_create_scene.restype = ct.c_int

    library.rrt_destroy.argtypes = [ct.c_void_p]
    library.rrt_destroy.restype = None

    library.rrt_solve.argtypes = [
        ct.c_void_p,
        float_array,
        float_array,
        id_array,
        seed_array,
        ct.POINTER(NativeResult),
        ct.POINTER(NativeShortcutConfig),
    ]
    library.rrt_solve.restype = ct.c_int

    library.rrt_path.argtypes = [
        ct.c_void_p,
        ct.c_uint32,
        float_array,
        ct.c_uint32,
        ct.POINTER(ct.c_uint32),
    ]
    library.rrt_path.restype = ct.c_int

    library.rrt_paths.argtypes = [ct.c_void_p, float_array, ct.c_uint64]
    library.rrt_paths.restype = ct.c_int

    library.rrt_check.argtypes = [
        ct.c_void_p,
        float_array,
        id_array,
        ct.c_uint32,
        flag_array,
    ]
    library.rrt_check.restype = ct.c_int


class Planner:
    """A reusable batch of RRT-Connect problems with owned GPU search and scene buffers.

    Geometry is a list of sdf collision objects for one environment, or a list
    of such lists for multiple environments.
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
        check_count(problem_count, "problem_count")
        scene = pack_geometry(geometry, ground=model["ground"])

        self.problem_count = problem_count
        self.environment_count = len(scene.ranges)
        self.dimension = len(model["control_joints"])
        self.fk_dof = len(model["input_joint_names"])
        if not 0 < self.dimension <= self.fk_dof:
            raise ValueError("model must have movable control joints")

        self.max_nodes = config.max_nodes
        self._path_lengths: list[int] | None = None
        lower, upper, resolutions = sampling_space(model["control_joints"], resolution)

        self.handle = ct.c_void_p()
        self.library = ct.CDLL(str(Path(library).resolve()))
        _configure_library(self.library)

        native_config = NativeConfig(**asdict(config))
        native_scene = NativeScene.from_scene(scene)
        status = self.library.rrtc_create_scene(
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
        self.check(self.library.rrt_paths(self.handle, packed, len(packed)))

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
        if q.ndim != 2 or q.shape[1] != self.fk_dof:
            raise ValueError(f"configurations must have shape (count, {self.fk_dof})")
        if ids.shape != (len(q),):
            raise ValueError("environment_ids must have shape (configuration count,)")
        if not np.isfinite(q).all():
            raise ValueError("configurations must be finite")
        if (ids >= self.environment_count).any():
            raise ValueError("environment ID out of range")

        flags = np.empty(len(q), dtype=np.uint8)
        status = self.library.rrt_check(self.handle, q, ids, len(q), flags)
        self.check(status)

        return flags
