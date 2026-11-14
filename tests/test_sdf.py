"""Collision objects validate geometry and preserve environment membership."""

from collections.abc import Callable

import numpy as np
import pytest

from tenkai import sdf
from tenkai.sdf import pack_geometry


@pytest.mark.parametrize(
    "make_object",
    [
        lambda: sdf.Sphere((float("nan"), 0, 0), 1),
        lambda: sdf.Sphere((1e40, 0, 0), 1),
        lambda: sdf.Sphere((0, 0, 0), -1),
        lambda: sdf.Sphere((0, 0, 0), float("inf")),
        lambda: sdf.Box((0, 0, 0), (-1, 1, 1)),
        lambda: sdf.Box((0, 0, 0), (1, 1, 1), ((1, 0, 0), (0, 2, 0), (0, 0, 1))),
        lambda: sdf.Box((0, 0, 0), (1, 1, 1), ((-1, 0, 0), (0, 1, 0), (0, 0, 1))),
        lambda: sdf.Cylinder((0, 0, 0), -1, 1),
        lambda: sdf.Cylinder((0, 0, 0), 1, -1),
        lambda: sdf.Cylinder((0, 0, 0), 1, 1e40),
        lambda: sdf.Ground(z=float("nan")),
    ],
)
def test_invalid_collision_objects(make_object: Callable[[], sdf.CollisionObject]) -> None:
    with pytest.raises(ValueError):
        make_object()


def test_collision_objects_snapshot_input_arrays() -> None:
    center = np.array([1.0, 2.0, 3.0])
    rotation = np.eye(3)
    box = sdf.Box(center, (1, 2, 3), rotation)  # type: ignore[arg-type]
    center[:] = 0
    rotation[:] = 0
    assert box.center == (1, 2, 3)
    assert box.rotation == sdf.IDENTITY


def test_scene_packing_preserves_objects_rotations_and_environment_ids() -> None:
    rotation: sdf.Rotation = ((0.6, -0.8, 0), (0.8, 0.6, 0), (0, 0, 1))
    box = sdf.Box((1, 2, 3), (4, 5, 6), rotation)
    cylinder = sdf.Cylinder((7, 8, 9), 0.25, 2, rotation)
    scene = pack_geometry(
        [
            [sdf.Sphere((1, 0, 0), 0.1), box, sdf.Sphere((2, 0, 0), 0.2)],
            [],
            [sdf.Ground(z=2), cylinder, sdf.Ground(z=-3)],
            [box],
        ]
    )
    np.testing.assert_array_equal(
        scene.ranges[:, :, 1], [[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 0, 0, 0]]
    )
    np.testing.assert_array_equal(scene.ranges[3, :, 0], [1, 2, 1, 2])
    np.testing.assert_array_equal(scene.boxes[:, :3], [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(scene.boxes[0, 3:12].reshape(3, 3), np.array(rotation).T)
    np.testing.assert_array_equal(scene.boxes[:, 12:], [[4, 5, 6], [4, 5, 6]])
    np.testing.assert_array_equal(scene.spheres, [[1, 0, 0, 0.1], [2, 0, 0, 0.2]])
    np.testing.assert_array_equal(scene.cylinders[0, :3], [7, 8, 9])
    np.testing.assert_array_equal(scene.cylinders[0, 3:12].reshape(3, 3), np.array(rotation).T)
    np.testing.assert_array_equal(scene.cylinders[0, 12:], [0.25, 2])
    np.testing.assert_array_equal(scene.grounds[:, 0], [-2, 3])


@pytest.mark.parametrize("geometry", [[], [[]], [[], []]])
def test_empty_scenes_have_no_implicit_obstacles(geometry: sdf.Geometry) -> None:
    scene = pack_geometry(geometry)
    assert not scene.ranges.any()
    assert len(scene.boxes) == len(scene.spheres) == len(scene.cylinders) == len(scene.grounds) == 0


@pytest.mark.parametrize(
    "geometry",
    [
        [object()],
        [[sdf.Ground()], [object()]],
        [sdf.Ground(), [sdf.Ground()]],
        sdf.Ground(),
        "ground",
        [0, 0, 0, 1, 1, 1, 0],
        [[0, 0, 0, 1, 1, 1, 0]],
        np.array([[0, 0, 0, 1, 1, 1, 0]]),
        np.empty((0, 7)),
    ],
)
def test_scene_requires_collision_object_sequences(geometry: object) -> None:
    with pytest.raises(ValueError, match="collision objects"):
        pack_geometry(geometry)  # type: ignore[arg-type]
