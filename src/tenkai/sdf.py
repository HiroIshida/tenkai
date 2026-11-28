"""Runtime collision objects, in meters in the robot's root frame."""

import json
from collections.abc import Sequence, Set
from dataclasses import asdict, dataclass
from typing import TypeGuard

import numpy as np
import numpy.typing as npt

from tenkai.layout import BOX, CYLINDER, ENVIRONMENT, GROUND, SPHERE

type Vector3 = tuple[float, float, float]
type Rotation = tuple[Vector3, Vector3, Vector3]

IDENTITY: Rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _finite_array(
    value: npt.ArrayLike, shape: tuple[int, ...], name: str
) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all() or (np.abs(array) > np.finfo(np.float32).max).any():
        raise ValueError(f"{name} must contain finite FP32-representable values")
    return array


def _vector(value: Vector3, name: str, *, nonnegative: bool = False) -> Vector3:
    array = _finite_array(value, (3,), name)
    if nonnegative and (array < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    return float(array[0]), float(array[1]), float(array[2])


def _rotation(value: Rotation) -> Rotation:
    array = _finite_array(value, (3, 3), "rotation")
    if not np.allclose(array.T @ array, np.eye(3), rtol=0, atol=1e-6) or np.linalg.det(array) < 0:
        raise ValueError("rotation must be an orthonormal matrix with determinant +1")
    return tuple(array[0]), tuple(array[1]), tuple(array[2])


def _scalar(value: float, name: str, *, nonnegative: bool = False) -> float:
    result = float(_finite_array(value, (), name))
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class Box:
    """Box with full side lengths and a local-to-world rotation matrix."""

    center: Vector3
    size: Vector3
    rotation: Rotation = IDENTITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector(self.center, "center"))
        object.__setattr__(self, "size", _vector(self.size, "size", nonnegative=True))
        object.__setattr__(self, "rotation", _rotation(self.rotation))


@dataclass(frozen=True)
class Sphere:
    """Sphere obstacle, distinct from a robot's link-attached collision sphere."""

    center: Vector3
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector(self.center, "center"))
        object.__setattr__(self, "radius", _scalar(self.radius, "radius", nonnegative=True))


@dataclass(frozen=True)
class Cylinder:
    """Finite cylinder along local Z, with full height and local-to-world rotation."""

    center: Vector3
    radius: float
    height: float
    rotation: Rotation = IDENTITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vector(self.center, "center"))
        object.__setattr__(self, "radius", _scalar(self.radius, "radius", nonnegative=True))
        object.__setattr__(self, "height", _scalar(self.height, "height", nonnegative=True))
        object.__setattr__(self, "rotation", _rotation(self.rotation))


@dataclass(frozen=True)
class Ground:
    """Horizontal surface at z, with the half-space below it occupied."""

    z: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "z", _scalar(self.z, "z"))


type CollisionObject = Box | Sphere | Cylinder | Ground
type CollisionObjectType = type[Box] | type[Sphere] | type[Cylinder] | type[Ground]
# Ground presence is fixed separately from the dynamic obstacle types.
type SpecializationType = type[Box] | type[Sphere] | type[Cylinder]
type Geometry = Sequence[CollisionObject] | Sequence[Sequence[CollisionObject]]


def geometry_to_json(obstacles: Sequence[CollisionObject]) -> str:
    """Serialize one environment without adding implicit obstacles."""
    records: list[dict[str, object]] = []
    for obstacle in obstacles:
        records.append({"type": type(obstacle).__name__, **asdict(obstacle)})
    return json.dumps(records, allow_nan=False)


def geometry_from_json(source: str) -> list[CollisionObject]:
    """Restore validated collision objects in their original order."""
    records = json.loads(source)
    if not isinstance(records, list):
        raise ValueError("obstacles must be a JSON array")

    types: dict[str, CollisionObjectType] = {
        "Box": Box,
        "Sphere": Sphere,
        "Cylinder": Cylinder,
        "Ground": Ground,
    }
    obstacles: list[CollisionObject] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"obstacles[{index}] must be an object")
        parameters = dict(record)
        kind = parameters.pop("type", None)
        if not isinstance(kind, str) or kind not in types:
            raise ValueError(f"obstacles[{index}] has an unknown type: {kind!r}")
        obstacles.append(types[kind](**parameters))
    return obstacles


# Order shared by SceneData's arrays and ranges, the type mask, and the native Environment.
PRIMITIVES = (Box, Sphere, Cylinder)
_DTYPES = (BOX, SPHERE, CYLINDER)
_RANGE_FIELDS = ("boxes", "spheres", "cylinders")


def geometry_type_mask(types: Set[SpecializationType] | None) -> int:
    """Encode specialized primitives as one bit each, in PRIMITIVES order."""
    if types is None:
        return (1 << len(PRIMITIVES)) - 1
    if not isinstance(types, Set) or any(kind not in PRIMITIVES for kind in types):
        raise ValueError(
            "specialize_types must be a set of sdf.Box, Sphere, or Cylinder; "
            "use Options(ground=True) for Ground"
        )
    return sum(1 << index for index, kind in enumerate(PRIMITIVES) if kind in types)


@dataclass
class SceneData:
    """Structured arrays in the native layout of the tenkai.layout scene dtypes."""

    environments: npt.NDArray[np.void]
    boxes: npt.NDArray[np.void]
    spheres: npt.NDArray[np.void]
    cylinders: npt.NDArray[np.void]
    grounds: npt.NDArray[np.void]


def _is_environment(value: object) -> TypeGuard[Sequence[CollisionObject]]:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, (Box, Sphere, Cylinder, Ground)) for item in value)
    )


def _primitive_index(obstacle: Box | Sphere | Cylinder) -> int:
    return PRIMITIVES.index(type(obstacle))


def _pack_environments(
    environments: Sequence[Sequence[CollisionObject]], *, ground: bool
) -> SceneData:
    if not environments:
        raise ValueError("at least one environment is required")

    ranges = np.zeros(len(environments), dtype=ENVIRONMENT)
    grounds = np.zeros(len(environments) if ground else 0, dtype=GROUND)
    totals = [0] * len(PRIMITIVES)
    for i, obstacles in enumerate(environments):
        ground_count = 0
        counts = [0] * len(PRIMITIVES)
        for obstacle in obstacles:
            if isinstance(obstacle, Ground):
                ground_count += 1
                if ground:
                    grounds[i] = -obstacle.z
            else:
                counts[_primitive_index(obstacle)] += 1
        if ground_count != int(ground):
            raise ValueError(f"environment {i} must contain exactly {int(ground)} Ground")
        for kind, name in enumerate(_RANGE_FIELDS):
            ranges[i][name] = totals[kind], counts[kind]
            totals[kind] += counts[kind]

    # Allocate once instead of retaining a second Python object per packed primitive.
    arrays = [np.zeros(total, dtype=kind) for total, kind in zip(totals, _DTYPES)]
    cursors = [0] * len(PRIMITIVES)
    for obstacles in environments:
        for obstacle in obstacles:
            if isinstance(obstacle, Ground):
                continue
            kind = _primitive_index(obstacle)
            arrays[kind][cursors[kind]] = _native_obstacle(obstacle)
            cursors[kind] += 1

    boxes, spheres, cylinders = arrays
    return SceneData(ranges, boxes, spheres, cylinders, grounds)


def _native_obstacle(obstacle: Box | Sphere | Cylinder) -> tuple[object, ...]:
    match obstacle:
        case Box():
            return _native_pose(obstacle.center, obstacle.rotation), obstacle.size
        case Sphere():
            return obstacle.center, obstacle.radius
        case Cylinder():
            return (
                _native_pose(obstacle.center, obstacle.rotation),
                obstacle.radius,
                obstacle.height,
            )


def _native_pose(center: Vector3, rotation: Rotation) -> tuple[Vector3, npt.NDArray[np.float64]]:
    # Native poses need the row-major inverse of the local-to-world rotation.
    return center, np.asarray(rotation).T.reshape(9)


def pack_geometry(geometry: Geometry, *, ground: bool = True) -> SceneData:
    """Pack one collision-object sequence or a sequence of environments."""
    if _is_environment(geometry):
        return _pack_environments([geometry], ground=ground)
    if not isinstance(geometry, Sequence) or isinstance(geometry, (str, bytes)):
        raise ValueError("geometry must contain collision objects or lists of collision objects")

    environments: list[Sequence[CollisionObject]] = []
    for index, environment in enumerate(geometry):
        if not _is_environment(environment):
            raise ValueError(f"geometry[{index}] must be a sequence of collision objects")
        environments.append(environment)
    return _pack_environments(environments, ground=ground)
