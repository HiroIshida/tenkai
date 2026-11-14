"""Runtime collision objects, in meters in the robot's root frame."""

from collections.abc import Sequence
from dataclasses import dataclass
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
type Geometry = Sequence[CollisionObject] | Sequence[Sequence[CollisionObject]]


@dataclass
class SceneData:
    ranges: npt.NDArray[np.uint32]
    boxes: npt.NDArray[np.float64]
    spheres: npt.NDArray[np.float64]
    cylinders: npt.NDArray[np.float64]
    grounds: npt.NDArray[np.float64]


def _is_environment(value: object) -> TypeGuard[Sequence[CollisionObject]]:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, (Box, Sphere, Cylinder, Ground)) for item in value)
    )


def _primitive_index(obstacle: CollisionObject) -> int:
    match obstacle:
        case Box():
            return 0
        case Sphere():
            return 1
        case Cylinder():
            return 2
        case Ground():
            return 3


def _pack_environments(environments: Sequence[Sequence[CollisionObject]]) -> SceneData:
    if not 0 < len(environments) < 2**32:
        raise ValueError("environment count must be between 1 and 2**32 - 1")

    ranges = np.zeros((len(environments), 4, 2), dtype=np.uint32)
    totals = [0, 0, 0, 0]
    for i, obstacles in enumerate(environments):
        ranges[i, :, 0] = totals
        for obstacle in obstacles:
            totals[_primitive_index(obstacle)] += 1
        if max(totals) >= 2**32:
            raise ValueError("too many collision objects")
        ranges[i, :, 1] = np.asarray(totals, dtype=np.uint32) - ranges[i, :, 0]

    # Allocate once instead of retaining a second Python object per packed primitive.
    boxes = np.empty((totals[0], 15), dtype=np.float64)
    spheres = np.empty((totals[1], 4), dtype=np.float64)
    cylinders = np.empty((totals[2], 14), dtype=np.float64)
    grounds = np.empty((totals[3], 1), dtype=np.float64)
    arrays = (boxes, spheres, cylinders, grounds)
    cursors = [0, 0, 0, 0]
    for obstacles in environments:
        for obstacle in obstacles:
            kind = _primitive_index(obstacle)
            row = arrays[kind][cursors[kind]]
            if isinstance(obstacle, (Box, Sphere, Cylinder)):
                row[:3] = obstacle.center
            if isinstance(obstacle, (Box, Cylinder)):
                row[3:12] = np.asarray(obstacle.rotation).T.reshape(9)
            match obstacle:
                case Box():
                    row[12:15] = obstacle.size
                case Sphere():
                    row[3] = obstacle.radius
                case Cylinder():
                    row[12:14] = obstacle.radius, obstacle.height
                case Ground():
                    row[0] = -obstacle.z
            cursors[kind] += 1

    return SceneData(ranges, boxes, spheres, cylinders, grounds)


def pack_geometry(geometry: Geometry) -> SceneData:
    """Pack one collision-object sequence or a sequence of environments."""
    if _is_environment(geometry):
        return _pack_environments([geometry])
    if not isinstance(geometry, Sequence) or isinstance(geometry, (str, bytes)):
        raise ValueError("geometry must contain collision objects or lists of collision objects")

    environments: list[Sequence[CollisionObject]] = []
    for index, environment in enumerate(geometry):
        if not _is_environment(environment):
            raise ValueError(f"geometry[{index}] must be a sequence of collision objects")
        environments.append(environment)
    return _pack_environments(environments)
