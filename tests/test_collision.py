import math
from pathlib import Path

import pytest

from tenkai import FkDag, Robot, Sphere
from tenkai.collision import center_dag, collision_expressions, enclosing_sphere
from tenkai.dag import Optimization


def test_enclosing_sphere_contains_members_with_different_radii() -> None:
    spheres = [Sphere(0, (-2, 1, 0), 0.25), Sphere(0, (1, -1, 3), 2)]
    bound = enclosing_sphere(spheres)
    for sphere in spheres:
        assert math.dist(sphere.position, bound.position) + sphere.radius <= bound.radius


@pytest.mark.parametrize("optimization", ["none", "basic", "algebraic"])
def test_link_bounds_preserve_narrow_phase_arithmetic(optimization: Optimization) -> None:
    source = (Path(__file__).parent / "fixtures/fk_chain.urdf").read_text()
    fk = FkDag.from_robot(Robot.from_string(source))
    spheres = [Sphere(link, (i * 0.13, 0.1, -0.2), 0.01) for i in range(4) for link in (5, 1)]
    dag, outputs = center_dag(fk, spheres, optimization)
    center_count = len(outputs)
    for link in (5, 1):
        pose = fk.poses[link]
        outputs.extend((*pose.translation, *pose.rotation))
    dag, outputs = dag.optimize(outputs, optimization)

    actual = collision_expressions(fk, spheres, optimization)
    assert actual.dag.nodes[: len(dag.nodes)] == dag.nodes
    assert [node for center in actual.centers for node in center] == outputs[:center_count]
