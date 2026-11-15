"""Pose operations expanded into scalar DAG expressions."""

import math
from dataclasses import dataclass
from typing import Self

from tenkai.dag import Dag

type Vector3 = tuple[float, float, float]
type PointNodes = tuple[int, int, int]
type QuaternionNodes = tuple[int, int, int, int]
type MatrixNodes = tuple[PointNodes, PointNodes, PointNodes]


def cross(dag: Dag, a: PointNodes, b: PointNodes) -> PointNodes:
    values: list[int] = []
    for j, k in ((1, 2), (2, 0), (0, 1)):
        positive = dag.mul(a[j], b[k])
        negative = dag.mul(a[k], b[j])
        values.append(dag.sub(positive, negative))

    return values[0], values[1], values[2]


def quaternion_product(dag: Dag, a: QuaternionNodes, b: QuaternionNodes) -> QuaternionNodes:
    product = cross(dag, (a[0], a[1], a[2]), (b[0], b[1], b[2]))
    xyz: list[int] = []
    for i in range(3):
        left = dag.mul(a[3], b[i])
        right = dag.mul(b[3], a[i])
        total = dag.add(left, right)
        xyz.append(dag.add(total, product[i]))

    products = [dag.mul(a[i], b[i]) for i in range(4)]
    total = dag.add(products[0], products[1])
    dot = dag.add(total, products[2])
    w = dag.sub(products[3], dot)

    return xyz[0], xyz[1], xyz[2], w


def rpy_quaternion(rpy: Vector3) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll * 0.5), math.cos(roll * 0.5)
    sp, cp = math.sin(pitch * 0.5), math.cos(pitch * 0.5)
    sy, cy = math.sin(yaw * 0.5), math.cos(yaw * 0.5)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


@dataclass(frozen=True)
class Pose:
    translation: PointNodes
    rotation: QuaternionNodes

    @classmethod
    def identity(cls, dag: Dag) -> Self:
        zero = dag.constant(0)
        one = dag.constant(1)
        return cls((zero, zero, zero), (zero, zero, zero, one))

    @classmethod
    def from_origin(cls, dag: Dag, xyz: Vector3, rpy: Vector3) -> Self:
        x, y, z = [dag.constant(v) for v in xyz]
        qx, qy, qz, qw = [dag.constant(v) for v in rpy_quaternion(rpy)]
        return cls((x, y, z), (qx, qy, qz, qw))

    def compose(self, dag: Dag, local: "Pose") -> "Pose":
        translation = self.transform_point(dag, local.translation)
        rotation = quaternion_product(dag, self.rotation, local.rotation)
        return Pose(translation, rotation)

    def transform_point(self, dag: Dag, point: PointNodes) -> PointNodes:
        imaginary = (self.rotation[0], self.rotation[1], self.rotation[2])
        two = dag.constant(2)
        product = cross(dag, imaginary, point)
        x, y, z = [dag.mul(two, value) for value in product]
        twice_cross = (x, y, z)

        correction = cross(dag, imaginary, twice_cross)
        offset: list[int] = []
        for i in range(3):
            scaled = dag.mul(self.rotation[3], twice_cross[i])
            total = dag.add(point[i], scaled)
            offset.append(dag.add(total, correction[i]))

        x, y, z = [dag.add(self.translation[i], offset[i]) for i in range(3)]
        return x, y, z

    def rotation_matrix(self, dag: Dag) -> MatrixNodes:
        x, y, z, w = self.rotation
        one = dag.constant(1)
        two = dag.constant(2)

        products: list[int] = []
        for a, b in ((x, x), (y, y), (z, z), (x, y), (x, z), (y, z), (w, x), (w, y), (w, z)):
            product = dag.mul(a, b)
            products.append(dag.mul(two, product))
        xx, yy, zz, xy, xz, yz, wx, wy, wz = products

        diagonal: list[int] = []
        for a, b in ((yy, zz), (xx, zz), (xx, yy)):
            total = dag.add(a, b)
            diagonal.append(dag.sub(one, total))

        return (
            (diagonal[0], dag.sub(xy, wz), dag.add(xz, wy)),
            (dag.add(xy, wz), diagonal[1], dag.sub(yz, wx)),
            (dag.sub(xz, wy), dag.add(yz, wx), diagonal[2]),
        )

    def transform_point_by_matrix(
        self, dag: Dag, point: PointNodes, rotation: MatrixNodes
    ) -> PointNodes:
        result: list[int] = []
        for i in range(3):
            x, y, z = [dag.mul(rotation[i][j], point[j]) for j in range(3)]
            xy = dag.add(x, y)
            offset = dag.add(xy, z)
            result.append(dag.add(self.translation[i], offset))

        return result[0], result[1], result[2]
