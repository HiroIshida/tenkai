# Copyright (C) 2026 Hirokazu Ishida. SPDX-License-Identifier: MPL-2.0
# Adapted from tenkai2; see licenses/tenkai2-MPL-2.0.txt.
"""Topologically ordered scalar expressions and deterministic CSE."""

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, assert_never

type Precision = Literal["f32", "f64"]
type Optimization = Literal["none", "basic", "algebraic"]
type Fingerprint = tuple[int, int, int]


class BinaryOp(Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"


class TrigOp(Enum):
    SIN = "sin"
    COS = "cos"


@dataclass(frozen=True)
class Constant:
    value: float


@dataclass(frozen=True)
class Input:
    index: int


@dataclass(frozen=True)
class Binary:
    op: BinaryOp
    a: int
    b: int


@dataclass(frozen=True)
class Trig:
    op: TrigOp
    angle: int


type Node = Constant | Input | Binary | Trig


def dependencies(node: Node) -> tuple[int, ...]:
    match node:
        case Constant() | Input():
            return ()
        case Trig(angle=angle):
            return (angle,)
        case Binary(a=a, b=b):
            if a == b:
                return (a,)
            return a, b
    assert_never(node)


def f32(value: float) -> float:
    try:
        packed = struct.pack("<f", value)
        rounded: float = struct.unpack("<f", packed)[0]
        return rounded
    except OverflowError:
        return math.copysign(math.inf, value)


def bits(value: float) -> int:
    packed = struct.pack("<d", value)
    encoded: int = struct.unpack("<Q", packed)[0]
    return encoded


def arithmetic(op: BinaryOp, x: float, y: float) -> float:
    match op:
        case BinaryOp.ADD:
            return x + y
        case BinaryOp.SUB:
            return x - y
        case BinaryOp.MUL:
            return x * y
    assert_never(op)


@dataclass
class Dag:
    nodes: list[Node] = field(default_factory=list)

    def push(self, node: Node) -> int:
        index = len(self.nodes)
        self.nodes.append(node)
        return index

    def constant(self, value: float) -> int:
        return self.push(Constant(float(value)))

    def input(self, index: int) -> int:
        return self.push(Input(index))

    def add(self, a: int, b: int) -> int:
        return self.push(Binary(BinaryOp.ADD, a, b))

    def sub(self, a: int, b: int) -> int:
        return self.push(Binary(BinaryOp.SUB, a, b))

    def mul(self, a: int, b: int) -> int:
        return self.push(Binary(BinaryOp.MUL, a, b))

    def sin(self, angle: int) -> int:
        return self.push(Trig(TrigOp.SIN, angle))

    def cos(self, angle: int) -> int:
        return self.push(Trig(TrigOp.COS, angle))

    def required_nodes(self, outputs: Sequence[int]) -> list[bool]:
        needed = [False] * len(self.nodes)
        for index in outputs:
            if not 0 <= index < len(self.nodes):
                raise ValueError("output node is outside the DAG")
            needed[index] = True

        for index in range(len(self.nodes) - 1, -1, -1):
            if needed[index]:
                for dependency in dependencies(self.nodes[index]):
                    needed[dependency] = True

        return needed

    def depends_on_inputs(self, inputs: Sequence[int]) -> list[bool]:
        dependent: list[bool] = []
        for node in self.nodes:
            if isinstance(node, Input):
                varies = node.index in inputs
            else:
                varies = any(dependent[i] for i in dependencies(node))
            dependent.append(varies)

        return dependent

    def optimize(
        self, outputs: Sequence[int], precision: Precision, mode: Optimization
    ) -> tuple["Dag", list[int]]:
        if mode == "none":
            return Dag(self.nodes.copy()), list(outputs)

        optimizer = Optimizer(precision, mode)
        remapped_nodes: dict[int, int] = {}
        for index, needed in enumerate(self.required_nodes(outputs)):
            if not needed:
                continue

            node = self.nodes[index]
            match node:
                case Binary(op, a, b):
                    node = Binary(op, remapped_nodes[a], remapped_nodes[b])
                case Trig(op, angle):
                    node = Trig(op, remapped_nodes[angle])
            remapped_nodes[index] = optimizer.insert(node)

        optimized_outputs = [remapped_nodes[index] for index in outputs]
        return optimizer.dag, optimized_outputs


FIELD = (1 << 61) - 1


def atom(kind: bytes, value: int) -> Fingerprint:
    lanes: list[int] = []
    for lane in range(3):
        # Keep the original salt to preserve CSE fingerprints across the rename.
        payload = b"rkern-cse" + bytes([lane]) + kind + value.to_bytes(8, "little")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        hashed_value = int.from_bytes(digest, "little")
        lanes.append(hashed_value % (FIELD - 1) + 1)

    return lanes[0], lanes[1], lanes[2]


class Optimizer:
    def __init__(self, precision: Precision, mode: Optimization) -> None:
        self.dag = Dag()
        self.precision = precision
        self.structural: dict[Node | tuple[str, int], int] = {}

        self.algebraic = mode == "algebraic"
        self.fingerprints: list[Fingerprint] = []
        self.canonical: dict[Fingerprint, int] = {}

        if self.algebraic:
            self.intern(Constant(0.0))
            self.intern(Constant(1.0))

    def constant(self, index: int) -> float | None:
        node = self.dag.nodes[index]
        if isinstance(node, Constant):
            return node.value
        return None

    def insert(self, node: Node) -> int:
        if isinstance(node, Constant):
            rounded = node.value
            if self.precision == "f32":
                rounded = f32(rounded)
            if math.isfinite(rounded):
                return self.intern(Constant(rounded))
        if not isinstance(node, Binary):
            return self.intern(node)

        x = self.constant(node.a)
        y = self.constant(node.b)
        if x is not None and y is not None:
            if self.precision == "f32":
                x, y = f32(x), f32(y)

            value = arithmetic(node.op, x, y)
            if self.precision == "f32":
                value = f32(value)

            if math.isfinite(value):
                return self.intern(Constant(value))
            return self.intern(node)

        if node.op == BinaryOp.ADD and x == 0.0:
            return node.b
        if node.op in (BinaryOp.ADD, BinaryOp.SUB) and y == 0.0:
            return node.a
        if node.op == BinaryOp.MUL:
            if x == 0.0 or y == 0.0:
                return self.intern(Constant(0.0))
            if x == 1.0:
                return node.b
            if y == 1.0:
                return node.a

        return self.intern(node)

    def intern(self, node: Node) -> int:
        key: Node | tuple[str, int] = node
        if isinstance(node, Constant):
            # Float equality would merge signed zero before the selected CSE mode.
            key = ("constant", bits(node.value))
        if key in self.structural:
            return self.structural[key]

        fingerprint = None
        if self.algebraic:
            fingerprint = self.fingerprint(node)
            if fingerprint in self.canonical:
                index = self.canonical[fingerprint]
                self.structural[key] = index
                return index

        index = self.dag.push(node)
        self.structural[key] = index
        if fingerprint is not None:
            self.fingerprints.append(fingerprint)
            self.canonical[fingerprint] = index

        return index

    def fingerprint(self, node: Node) -> Fingerprint:
        match node:
            case Constant(value):
                if value == 0.0:
                    return 0, 0, 0

                if abs(value) == 1.0:
                    magnitude = (1, 1, 1)
                else:
                    magnitude = atom(b"constant", bits(abs(value)))

                if value < 0:
                    return FIELD - magnitude[0], FIELD - magnitude[1], FIELD - magnitude[2]
                return magnitude
            case Input(index):
                return atom(b"input", index)
            case Trig(op, angle):
                return atom(op.value.encode(), angle)
            case Binary(op, a, b):
                lanes: list[int] = []
                for x, y in zip(self.fingerprints[a], self.fingerprints[b], strict=True):
                    if op == BinaryOp.ADD:
                        value = x + y
                    elif op == BinaryOp.SUB:
                        value = x - y
                    else:
                        value = x * y
                    lanes.append(value % FIELD)

                return lanes[0], lanes[1], lanes[2]
        assert_never(node)
