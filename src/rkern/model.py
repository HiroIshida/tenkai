"""URDF trees and forward kinematics."""

import math
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Literal, Self

from .dag import Dag
from .transform import Pose, Vector3, quaternion_product

type JointKind = Literal["fixed", "revolute", "continuous", "prismatic"]


def vector(element: ET.Element | None, attribute: str, default: str) -> Vector3:
    text = default
    if element is not None:
        text = element.get(attribute, default)
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise ValueError("expected three vector components")
    return values[0], values[1], values[2]


def joint_kind(value: str) -> JointKind:
    match value:
        case "fixed" | "revolute" | "continuous" | "prismatic":
            return value
    raise ValueError("unsupported joint type: " + value)


@dataclass(frozen=True)
class Joint:
    name: str
    kind: JointKind
    parent: str
    child: str
    xyz: Vector3
    rpy: Vector3
    axis: Vector3
    lower: float
    upper: float


@dataclass
class Robot:
    links: list[str]
    joints: list[Joint]

    @classmethod
    def from_string(cls, source: str) -> Self:
        root = ET.fromstring(source)
        if root.tag != "robot":
            raise ValueError("expected a URDF robot")
        joints = []
        for element in root.findall("joint"):
            origin = element.find("origin")
            limit = element.find("limit")
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError("joint requires parent and child")
            if element.find("mimic") is not None:
                raise ValueError("mimic is not supported")
            lower = upper = 0.0
            if limit is not None:
                lower = float(limit.get("lower", "0"))
                upper = float(limit.get("upper", "0"))
            joint = Joint(
                name=element.attrib["name"],
                kind=joint_kind(element.attrib["type"]),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                xyz=vector(origin, "xyz", "0 0 0"),
                rpy=vector(origin, "rpy", "0 0 0"),
                axis=vector(element.find("axis"), "xyz", "1 0 0"),
                lower=lower,
                upper=upper,
            )
            joints.append(joint)
        links = [element.attrib["name"] for element in root.findall("link")]
        return cls(links, joints)

    def tree(self) -> tuple[int, list[list[tuple[int, int]]]]:
        if any(not name for name in self.links) or len(set(self.links)) != len(self.links):
            raise ValueError("empty or duplicate link name")
        indices = {name: i for i, name in enumerate(self.links)}
        outgoing: list[list[tuple[int, int]]] = [[] for _ in self.links]
        incoming: set[int] = set()
        names: set[str] = set()
        for index, joint in enumerate(self.joints):
            if not joint.name or joint.name in names:
                raise ValueError("empty or duplicate joint name")
            names.add(joint.name)
            if joint.parent not in indices or joint.child not in indices:
                raise ValueError("joint references an unknown link: " + joint.name)
            parent = indices[joint.parent]
            child = indices[joint.child]
            if child in incoming:
                raise ValueError("multiple parent joints: " + joint.child)
            incoming.add(child)
            outgoing[parent].append((index, child))
        roots = set(range(len(self.links))) - incoming
        if len(roots) != 1:
            raise ValueError("expected one root link")
        return roots.pop(), outgoing


@dataclass
class FkDag:
    dag: Dag
    joint_names: list[str]
    link_names: list[str]
    poses: list[Pose]

    @classmethod
    def from_robot(cls, robot: Robot) -> Self:
        root, outgoing = robot.tree()
        dag = Dag()
        names: list[str] = []
        inputs: list[int | None] = []
        for joint in robot.joints:
            if joint.kind == "fixed":
                inputs.append(None)
            else:
                inputs.append(dag.input(len(names)))
                names.append(joint.name)
        pending = deque([(root, Pose.identity(dag))])
        link_names = []
        poses = []
        while pending:
            parent, pose = pending.popleft()
            link_names.append(robot.links[parent])
            poses.append(pose)
            for index, child in outgoing[parent]:
                child_pose = joint_pose(dag, pose, robot.joints[index], inputs[index])
                pending.append((child, child_pose))
        if len(poses) != len(robot.links):
            raise ValueError("links contain a disconnected cycle")
        return cls(dag, names, link_names, poses)


def joint_pose(dag: Dag, parent: Pose, joint: Joint, input_id: int | None) -> Pose:
    if not all(math.isfinite(value) for value in (*joint.xyz, *joint.rpy)):
        raise ValueError("joint origin must be finite: " + joint.name)
    origin = Pose.from_origin(dag, joint.xyz, joint.rpy)
    world = parent.compose(dag, origin)
    if input_id is None:
        return world
    x, y, z = joint.axis
    norm = math.hypot(math.hypot(x, y), z)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("joint axis must be finite and nonzero: " + joint.name)
    axis = [dag.constant(value / norm) for value in joint.axis]
    if joint.kind == "prismatic":
        px, py, pz = [dag.mul(value, input_id) for value in axis]
        translation = world.transform_point(dag, (px, py, pz))
        return Pose(translation, world.rotation)
    half = dag.constant(0.5)
    angle = dag.mul(input_id, half)
    sin = dag.sin(angle)
    cos = dag.cos(angle)
    qx, qy, qz = [dag.mul(value, sin) for value in axis]
    rotation = quaternion_product(dag, world.rotation, (qx, qy, qz, cos))
    return Pose(world.translation, rotation)
