"""Collision spheres and their FK expressions."""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from .dag import Dag, Optimization
from .model import KinematicModel
from .transform import MatrixNodes, PointNodes, Pose, Vector3


@dataclass(frozen=True)
class Sphere:
    link_index: int
    position: Vector3
    radius: float = 0.0

    def validate(self, kin: KinematicModel) -> None:
        if not 0 <= self.link_index < len(kin.poses):
            raise ValueError("sphere link index is out of range")
        if len(self.position) != 3 or not all(math.isfinite(value) for value in self.position):
            raise ValueError("sphere position must have three finite components")
        if not math.isfinite(self.radius) or self.radius < 0:
            raise ValueError("sphere radius must be finite and nonnegative")


@dataclass
class CollisionConfig:
    groups: dict[str, list[tuple[float, float, float, float]]]
    control_joint_names: list[str]
    self_collision_pairs: list[tuple[str, str]] = field(default_factory=list)
    only_self_collision_links: list[str] = field(default_factory=list)

    def resolve(self, kin: KinematicModel) -> list[Sphere]:
        unknown_links = self.groups.keys() - set(kin.link_names)
        if unknown_links:
            raise ValueError(
                "collision spheres reference unknown link: " + sorted(unknown_links)[0]
            )

        spheres: list[Sphere] = []
        for index, name in enumerate(kin.link_names):
            for x, y, z, radius in self.groups.get(name, []):
                sphere = Sphere(index, (x, y, z), radius)
                sphere.validate(kin)
                spheres.append(sphere)

        return spheres

    def resolve_rules(self, kin: KinematicModel, spheres: Sequence[Sphere]) -> "SelfCollisionRules":
        def link_index(name: str) -> int:
            if name not in kin.link_names:
                raise ValueError("collision rules reference unknown link: " + name)
            return kin.link_names.index(name)

        pairs = tuple((link_index(a), link_index(b)) for a, b in self.self_collision_pairs)
        self_only = tuple(link_index(name) for name in self.only_self_collision_links)
        rules = SelfCollisionRules(pairs, self_only)
        rules.validate(spheres)
        return rules


@dataclass(frozen=True)
class SelfCollisionRules:
    self_pairs: tuple[tuple[int, int], ...] = ()
    self_only_links: tuple[int, ...] = ()

    def validate(self, spheres: Sequence[Sphere]) -> None:
        links = {sphere.link_index for sphere in spheres}
        seen: set[tuple[int, int]] = set()
        for a, b in self.self_pairs:
            if a not in links or b not in links:
                raise ValueError("self collision pair requires spheres on both links")
            if a == b:
                raise ValueError("self collision pair requires distinct links")
            pair = (min(a, b), max(a, b))
            if pair in seen:
                raise ValueError("duplicate self collision pair")
            seen.add(pair)
        if len(set(self.self_only_links)) != len(self.self_only_links):
            raise ValueError("duplicate only_self_collision_links entry")
        if any(link not in links for link in self.self_only_links):
            raise ValueError("only_self_collision_links requires spheres on each link")


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

    pairs: list[tuple[str, str]] = []
    for item in sequence(config.get("self_collision_pairs", [])):
        pair = sequence(item)
        if len(pair) != 2:
            raise ValueError("self_collision_pairs entries must contain two link names")
        a, b = pair
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError("self_collision_pairs entries must contain two link names")
        pairs.append((a, b))

    self_only: list[str] = []
    for self_link in sequence(config.get("only_self_collision_links", [])):
        if not isinstance(self_link, str):
            raise ValueError("only_self_collision_links entries must be link names")
        self_only.append(self_link)

    return CollisionConfig(groups, controls, pairs, self_only)


def read_spheres(source: str, kin: KinematicModel) -> list[Sphere]:
    return read_config(source).resolve(kin)


def enclosing_sphere(spheres: Sequence[Sphere]) -> Sphere:
    if not spheres:
        raise ValueError("cannot enclose an empty sphere group")
    link = spheres[0].link_index
    if any(sphere.link_index != link for sphere in spheres):
        raise ValueError("enclosing sphere requires a single link")

    count = len(spheres)
    coordinates = [
        math.fsum(sphere.position[axis] / count for sphere in spheres) for axis in range(3)
    ]
    center = (coordinates[0], coordinates[1], coordinates[2])
    radius = max(math.dist(sphere.position, center) + sphere.radius for sphere in spheres)
    return Sphere(link, center, math.nextafter(radius, math.inf))


def center_expressions(
    kin: KinematicModel, spheres: Sequence[Sphere], optimization: Optimization
) -> tuple[Dag, list[int]]:
    dag, outputs = center_dag(kin, spheres, optimization)
    return dag.optimize(outputs, optimization)


def center_dag(
    kin: KinematicModel, spheres: Sequence[Sphere], optimization: Optimization
) -> tuple[Dag, list[int]]:
    for sphere in spheres:
        sphere.validate(kin)

    counts = Counter(sphere.link_index for sphere in spheres)
    dag = Dag(kin.dag.nodes.copy())
    rotations: dict[int, MatrixNodes] = {}
    outputs: list[int] = []
    share_rotations = optimization == "algebraic"

    for sphere in spheres:
        pose = kin.poses[sphere.link_index]
        x, y, z = [dag.constant(value) for value in sphere.position]
        local = (x, y, z)
        if share_rotations and counts[sphere.link_index] >= 4:
            if sphere.link_index not in rotations:
                rotations[sphere.link_index] = pose.rotation_matrix(dag)
            center = pose.transform_point_by_matrix(dag, local, rotations[sphere.link_index])
        else:
            center = pose.transform_point(dag, local)
        outputs.extend(center)

    return dag, outputs


@dataclass
class LinkBound:
    sphere: Sphere
    center: PointNodes
    local_scale: float


@dataclass
class CollisionDag:
    dag: Dag
    centers: list[list[int]]
    poses: dict[int, Pose]
    bounds: dict[int, LinkBound]
    spheres: list[Sphere]
    rules: SelfCollisionRules


def collision_expressions(
    kin: KinematicModel,
    spheres: Sequence[Sphere],
    optimization: Optimization,
    rules: SelfCollisionRules = SelfCollisionRules(),
) -> CollisionDag:
    rules.validate(spheres)
    dag, outputs = center_dag(kin, spheres, optimization)
    center_count = len(outputs)
    links: dict[int, list[Sphere]] = {}
    for sphere in spheres:
        links.setdefault(sphere.link_index, []).append(sphere)
    for link in links:
        pose = kin.poses[link]
        outputs.extend((*pose.translation, *pose.rotation))

    enclosing: list[tuple[Sphere, float]] = []
    for link, members in links.items():
        if len(members) < 2:
            continue
        bound = enclosing_sphere(members)
        local_scale = max(sum(abs(value) for value in s.position) for s in [*members, bound])
        x, y, z = [dag.constant(value) for value in bound.position]
        outputs.extend(kin.poses[link].transform_point(dag, (x, y, z)))
        enclosing.append((bound, local_scale))

    dag, outputs = dag.optimize(outputs, optimization)
    centers = [outputs[i : i + 3] for i in range(0, center_count, 3)]
    poses: dict[int, Pose] = {}
    for index, link in enumerate(links):
        start = center_count + index * 7
        tx, ty, tz, qx, qy, qz, qw = outputs[start : start + 7]
        poses[link] = Pose((tx, ty, tz), (qx, qy, qz, qw))
    bounds: dict[int, LinkBound] = {}
    for index, (bound, local_scale) in enumerate(enclosing):
        start = center_count + len(links) * 7 + index * 3
        x, y, z = outputs[start : start + 3]
        bounds[bound.link_index] = LinkBound(bound, (x, y, z), local_scale)
    return CollisionDag(dag, centers, poses, bounds, list(spheres), rules)
