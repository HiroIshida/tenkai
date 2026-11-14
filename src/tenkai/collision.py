"""Collision spheres and their FK expressions."""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from .dag import Dag, Optimization, Precision
from .model import FkDag
from .transform import MatrixNodes, Vector3


@dataclass(frozen=True)
class Sphere:
    link_index: int
    position: Vector3
    radius: float = 0.0

    def validate(self, fk: FkDag) -> None:
        if not 0 <= self.link_index < len(fk.poses):
            raise ValueError("sphere link index is out of range")
        if len(self.position) != 3 or not all(math.isfinite(value) for value in self.position):
            raise ValueError("sphere position must have three finite components")
        if not math.isfinite(self.radius) or self.radius < 0:
            raise ValueError("sphere radius must be finite and nonnegative")


@dataclass
class CollisionConfig:
    groups: dict[str, list[tuple[float, float, float, float]]]
    control_joint_names: list[str]

    def resolve(self, fk: FkDag) -> list[Sphere]:
        unknown_links = self.groups.keys() - set(fk.link_names)
        if unknown_links:
            raise ValueError(
                "collision spheres reference unknown link: " + sorted(unknown_links)[0]
            )

        spheres: list[Sphere] = []
        for index, name in enumerate(fk.link_names):
            for x, y, z, radius in self.groups.get(name, []):
                sphere = Sphere(index, (x, y, z), radius)
                sphere.validate(fk)
                spheres.append(sphere)

        return spheres


def check_unique_keys(node: Node, visited: set[int]) -> None:
    if id(node) in visited:
        return
    visited.add(id(node))

    if isinstance(node, MappingNode):
        keys: set[str] = set()
        for key, value in node.value:
            if not isinstance(key, ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                raise ValueError("configuration keys must be strings")
            if key.value in keys:
                raise ValueError("duplicate YAML key: " + key.value)

            keys.add(key.value)
            check_unique_keys(value, visited)
    elif isinstance(node, SequenceNode):
        for item in node.value:
            check_unique_keys(item, visited)


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")

    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("configuration keys must be strings")
        result[key] = item

    return result


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected a sequence")
    return list(value)


def scalar(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("expected a number")
    return float(value)


def read_config(source: str) -> CollisionConfig:
    node = yaml.compose(source, Loader=yaml.SafeLoader)
    if node is None:
        raise ValueError("empty collision configuration")

    # safe_load otherwise silently overwrites duplicate sphere groups.
    check_unique_keys(node, set())
    config = mapping(yaml.safe_load(source))
    if "collision_spheres" not in config:
        raise ValueError("missing collision_spheres")

    groups: dict[str, list[tuple[float, float, float, float]]] = {}
    for name, value in mapping(config["collision_spheres"]).items():
        group = mapping(value)
        if set(group) != {"spheres"}:
            raise ValueError("expected only spheres in collision group")

        spheres: list[tuple[float, float, float, float]] = []
        for item in sequence(group["spheres"]):
            coordinates = sequence(item)
            if len(coordinates) != 4:
                raise ValueError("expected x, y, z and radius")

            x, y, z, radius = [scalar(value) for value in coordinates]
            spheres.append((x, y, z, radius))
        groups[name] = spheres

    controls: list[str] = []
    for control in sequence(config.get("control_joint_names", [])):
        if not isinstance(control, str):
            raise ValueError("control joint names must be strings")
        controls.append(control)

    return CollisionConfig(groups, controls)


def read_spheres(source: str, fk: FkDag) -> list[Sphere]:
    return read_config(source).resolve(fk)


def center_expressions(
    fk: FkDag, spheres: Sequence[Sphere], precision: Precision, optimization: Optimization
) -> tuple[Dag, list[int]]:
    for sphere in spheres:
        sphere.validate(fk)

    counts = Counter(sphere.link_index for sphere in spheres)
    dag = Dag(fk.dag.nodes.copy())
    rotations: dict[int, MatrixNodes] = {}
    outputs: list[int] = []
    share_rotations = precision == "f32" and optimization == "algebraic"

    for sphere in spheres:
        pose = fk.poses[sphere.link_index]
        x, y, z = [dag.constant(value) for value in sphere.position]
        local = (x, y, z)
        if share_rotations and counts[sphere.link_index] >= 4:
            if sphere.link_index not in rotations:
                rotations[sphere.link_index] = pose.rotation_matrix(dag)
            center = pose.transform_point_by_matrix(dag, local, rotations[sphere.link_index])
        else:
            center = pose.transform_point(dag, local)
        outputs.extend(center)

    return dag.optimize(outputs, precision, optimization)
