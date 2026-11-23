"""Runtime collision objects, in meters in the robot's root frame."""

import json
from collections.abc import Sequence, Set
from dataclasses import asdict, dataclass
from typing import TypeGuard

import numpy as np
import numpy.typing as npt

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


def geometry_type_mask(types: Set[SpecializationType] | None) -> int:
    """Encode the primitive order shared by SceneData and the native Environment."""
    if types is None:
        return 7
    primitives = (Box, Sphere, Cylinder)
    if not isinstance(types, Set) or any(kind not in primitives for kind in types):
        raise ValueError(
            "specialize_types must be a set of sdf.Box, Sphere, or Cylinder; "
            "use Options(ground=True) for Ground"
        )
    return sum(1 << index for index, kind in enumerate(primitives) if kind in types)


@dataclass
class SceneData:
    ranges: npt.NDArray[np.uint32]
    boxes: npt.NDArray[np.float64]
    spheres: npt.NDArray[np.float64]
    cylinders: npt.NDArray[np.float64]
    grounds: npt.NDArray[np.float64]


# Packed row layout consumed by templates/rrt/bindings.cu.
_BOX_WIDTH = 15
_SPHERE_WIDTH = 4
_CYLINDER_WIDTH = 14
_CENTER_COLUMNS = slice(0, 3)
_WORLD_TO_LOCAL_COLUMNS = slice(3, 12)
_BOX_SIZE_COLUMNS = slice(12, 15)
_SPHERE_RADIUS_COLUMN = 3
_CYLINDER_SIZE_COLUMNS = slice(12, 14)


def _is_environment(value: object) -> TypeGuard[Sequence[CollisionObject]]:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, (Box, Sphere, Cylinder, Ground)) for item in value)
    )


def _primitive_index(obstacle: Box | Sphere | Cylinder) -> int:
    match obstacle:
        case Box():
            return 0
        case Sphere():
            return 1
        case Cylinder():
            return 2


def _pack_environments(
    environments: Sequence[Sequence[CollisionObject]], *, ground: bool
) -> SceneData:
    if not environments:
        raise ValueError("at least one environment is required")

    ranges = np.zeros((len(environments), 3, 2), dtype=np.uint32)
    grounds = np.empty(len(environments) if ground else 0, dtype=np.float64)
    totals = [0, 0, 0]
    for i, obstacles in enumerate(environments):
        ranges[i, :, 0] = totals
        ground_count = 0
        for obstacle in obstacles:
            if isinstance(obstacle, Ground):
                ground_count += 1
                if ground:
                    grounds[i] = -obstacle.z
            else:
                totals[_primitive_index(obstacle)] += 1
        if ground_count != int(ground):
            raise ValueError(f"environment {i} must contain exactly {int(ground)} Ground")
        ranges[i, :, 1] = np.asarray(totals, dtype=np.uint32) - ranges[i, :, 0]

    # Allocate once instead of retaining a second Python object per packed primitive.
    boxes = np.empty((totals[0], _BOX_WIDTH), dtype=np.float64)
    spheres = np.empty((totals[1], _SPHERE_WIDTH), dtype=np.float64)
    cylinders = np.empty((totals[2], _CYLINDER_WIDTH), dtype=np.float64)
    arrays = (boxes, spheres, cylinders)
    cursors = [0, 0, 0]
    for obstacles in environments:
        for obstacle in obstacles:
            if isinstance(obstacle, Ground):
                continue
            kind = _primitive_index(obstacle)
            row = arrays[kind][cursors[kind]]
            row[_CENTER_COLUMNS] = obstacle.center
            if isinstance(obstacle, (Box, Cylinder)):
                # Native poses need the row-major inverse of the local-to-world rotation.
                row[_WORLD_TO_LOCAL_COLUMNS] = np.asarray(obstacle.rotation).T.reshape(9)
            match obstacle:
                case Box():
                    row[_BOX_SIZE_COLUMNS] = obstacle.size
                case Sphere():
                    row[_SPHERE_RADIUS_COLUMN] = obstacle.radius
                case Cylinder():
                    row[_CYLINDER_SIZE_COLUMNS] = obstacle.radius, obstacle.height
            cursors[kind] += 1

    return SceneData(ranges, boxes, spheres, cylinders, grounds)


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
