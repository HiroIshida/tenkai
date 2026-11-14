import math
from pathlib import Path

import pytest

from tenkai import FkDag, Robot, Sphere
from tenkai.collision import center_dag, collision_expressions, enclosing_sphere, read_config
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


def test_fetch_self_collision_rules() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets/fetch"
    fk = FkDag.from_robot(Robot.from_string((assets / "fetch.urdf").read_text()))
    config = read_config((assets / "collision.yaml").read_text())
    spheres = config.resolve(fk)
    rules = config.resolve_rules(fk, spheres)
    assert len(rules.self_pairs) == 14
    assert sum(s.link_index not in rules.self_only_links for s in spheres) == 51
    assert {fk.link_names[i] for i in rules.self_only_links} == {"head_pan_link", "torso_lift_link"}


@pytest.mark.parametrize(
    "rules",
    [
        "self_collision_pairs: [[base]]",
        "self_collision_pairs: [[base, 1]]",
        "self_collision_pairs: [[base, missing]]",
        "self_collision_pairs: [[base, base]]",
        "self_collision_pairs: [[base, sensor], [sensor, base]]",
        "self_collision_pairs: [[base, tool]]",
        "only_self_collision_links: [missing]",
        "only_self_collision_links: [tool]",
        "only_self_collision_links: [base, base]",
        "only_self_collision_links: [1]",
    ],
)
def test_invalid_self_collision_rules(rules: str) -> None:
    fk = FkDag.from_robot(
        Robot.from_string((Path(__file__).parent / "fixtures/fk_chain.urdf").read_text())
    )
    source = "collision_spheres:\n  base: {spheres: [[0, 0, 0, 1]]}\n  sensor: {spheres: [[0, 0, 0, 1]]}\n"
    with pytest.raises(ValueError):
        config = read_config(source + rules)
        config.resolve_rules(fk, config.resolve(fk))
