"""Execution mechanics shared by the compiler tests."""

import math
from collections.abc import Sequence

from rkern.dag import Binary, BinaryOp, Constant, Dag, Input, Node, Trig, TrigOp
from rkern.scheduling import Evaluate, Schedule, Store


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
