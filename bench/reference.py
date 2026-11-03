"""Plainmp reference checks using the same Fetch model and scene as tenkai."""

import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from plainmp import _plainmp
from plainmp.constraint import SphereAttachmentSpec, SphereCollisionCst
from plainmp.kinematics import BaseType, KinematicModel
from plainmp.psdf import BoxSDF, GroundSDF, Pose, UnionSDF

from tenkai.collision import read_config

ROOT = Path(__file__).resolve().parents[1]


class Validation(TypedDict):
    points: int
    collisions: int


def package_info() -> dict[str, str]:
    native = Path(_plainmp.__file__)
    return {
        "version": version("plainmp"),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "interface": "installed plainmp Python API",
    }


def make_checker(names: list[str], geometry: npt.ArrayLike) -> SphereCollisionCst:
    row = np.asarray(geometry, dtype=np.float64)
    if row.shape != (7,) or not np.isfinite(row).all() or (row[3:6] < 0).any():
        raise ValueError(
            "geometry must contain a finite box center, full extents, and ground offset"
        )
    # Plainmp's GroundSDF AABB is fixed at z=0 even when its distance offset changes.
    if row[6] != 0:
        raise ValueError("plainmp reference checks require ground_offset=0")
    config = read_config((ROOT / "assets/fetch/collision.yaml").read_text())
    attachments = []
    for link, definitions in config.groups.items():
        spheres = np.asarray(definitions, dtype=np.float64).reshape(-1, 4)
        # FetchSpec's default excludes some spheres from environment checks.
        centers = np.asfortranarray(spheres[:, :3].T)
        radii = np.ascontiguousarray(spheres[:, 3])
        attachments.append(SphereAttachmentSpec(link, centers, radii, False))
    model = KinematicModel((ROOT / "assets/fetch/fetch.urdf").read_text())
    checker = SphereCollisionCst(model, names, BaseType.FIXED, attachments, [], None, True)
    box = BoxSDF(row[3:6], Pose(row[:3], np.eye(3)))
    checker.set_sdf(UnionSDF([box, GroundSDF(float(row[6]))]))
    return checker


def validate_points(
    configurations: npt.NDArray[np.float64],
    environment_ids: npt.NDArray[np.uint32],
    geometry: npt.NDArray[np.float64],
    names: list[str],
) -> Validation:
    if configurations.shape != (len(environment_ids), len(names)):
        raise ValueError("invalid validation configuration shape")
    if geometry.ndim != 2 or geometry.shape[1] != 7 or not len(geometry):
        raise ValueError("geometry must have shape (environments, 7)")
    if not np.isfinite(configurations).all() or (environment_ids >= len(geometry)).any():
        raise ValueError("invalid validation configurations or environment IDs")
    checkers = [make_checker(names, row) for row in geometry]
    collisions = 0
    for q, environment in zip(configurations, environment_ids, strict=True):
        if not checkers[int(environment)].is_valid(q):
            collisions += 1
    return Validation(points=len(configurations), collisions=collisions)
