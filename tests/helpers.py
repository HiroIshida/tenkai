"""Execution mechanics shared by the compiler tests."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tenkai.dag import Binary, BinaryOp, Constant, Dag, Input, Node, Trig, TrigOp
from tenkai.modeling import KinematicModel, Sphere
from tenkai.scheduling import Evaluate, Schedule, Store


def evaluate_node(node: Node, inputs: Sequence[float], values: dict[int, float]) -> float:
    match node:
        case Constant(value):
            return value
        case Input(index):
            return inputs[index]
        case Trig(op, angle):
            if op == TrigOp.SIN:
                return math.sin(values[angle])
            return math.cos(values[angle])
        case Binary(op, a, b):
            if op == BinaryOp.ADD:
                return values[a] + values[b]
            if op == BinaryOp.SUB:
                return values[a] - values[b]
            return values[a] * values[b]
    raise AssertionError("unknown node")


def evaluate(dag: Dag, inputs: Sequence[float]) -> dict[int, float]:
    values: dict[int, float] = {}
    for index, node in enumerate(dag.nodes):
        values[index] = evaluate_node(node, inputs, values)
    return values


def execute(dag: Dag, order: Schedule, inputs: Sequence[float]) -> list[float]:
    values: dict[int, float] = {}
    outputs = []
    for step in order.steps:
        if isinstance(step, Evaluate):
            assert step.node not in values
            values[step.node] = evaluate_node(dag.nodes[step.node], inputs, values)
        elif isinstance(step, Store):
            assert step.index == len(outputs)
            outputs.append(values[step.node])
    assert len(outputs) == order.output_count
    return outputs


@dataclass
class FetchCollisionCase:
    name: str
    joints: list[float]
    spheres: list[list[float]]
    collision: bool


def fetch_collision_cases(
    kin: KinematicModel, spheres: Sequence[Sphere]
) -> list[FetchCollisionCase]:
    reference = json.loads((Path(__file__).parent / "fixtures/fetch_vamp.json").read_text())
    indices = [
        i
        for link in dict.fromkeys(reference["sphere_links"])
        for i, sphere in enumerate(spheres)
        if kin.link_names[sphere.link_index] == link
    ]
    assert [kin.link_names[spheres[i].link_index] for i in indices] == reference["sphere_links"]

    cases: list[FetchCollisionCase] = []
    for record in reference["cases"]:
        joints = [0.0] * len(kin.joints)
        positions = dict(zip(reference["joint_names"], record["configuration"], strict=True))
        positions.update(reference["fixed_joint_positions"])
        for name, value in positions.items():
            joints[kin.joint_names.index(name)] = value

        expected = [[0.0] * 4 for _ in spheres]
        for i, sphere in zip(indices, record["spheres"], strict=True):
            expected[i] = sphere
        cases.append(FetchCollisionCase(record["name"], joints, expected, record["self_collision"]))
    return cases
