import math
from pathlib import Path

import numpy as np
import pytest
from helpers import evaluate, fetch_collision_cases

from tenkai import KinematicModel, Sphere
from tenkai.dag import Optimization
from tenkai.modeling import collision_expressions, read_config
from tenkai.modeling.collision import center_dag, center_expressions, enclosing_sphere


def test_enclosing_sphere_contains_members_with_different_radii() -> None:
    spheres = [Sphere(0, (-2, 1, 0), 0.25), Sphere(0, (1, -1, 3), 2)]
    bound = enclosing_sphere(spheres)
    for sphere in spheres:
        assert math.dist(sphere.position, bound.position) + sphere.radius <= bound.radius


@pytest.mark.parametrize(
    "spheres,center,radius",
    [
        ([Sphere(0, (-2, 0, 0), 0.25), Sphere(0, (1, 0, 0), 2)], (0.375, 0, 0), 2.625),
        ([Sphere(0, (0, 0, 0), 2), Sphere(0, (1, 0, 0), 0.125)], (0, 0, 0), 2),
        ([Sphere(0, (1, 2, 3), 0), Sphere(0, (1, 2, 3), 0)], (1, 2, 3), 0),
        (
            [
                Sphere(0, position, 0.125)
                for position in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
            ],
            (0, 0, 0),
            math.sqrt(3) + 0.125,
        ),
    ],
)
def test_enclosing_sphere_tight_bounds(
    spheres: list[Sphere], center: tuple[float, float, float], radius: float
) -> None:
    bound = enclosing_sphere(spheres)
    assert bound.position == pytest.approx(center)
    assert bound.radius == pytest.approx(radius)
    assert bound.radius <= math.nextafter(radius, math.inf)
    for sphere in spheres:
        assert math.dist(sphere.position, bound.position) + sphere.radius <= bound.radius


@pytest.mark.parametrize("optimization", ["none", "basic", "algebraic"])
def test_link_bounds_preserve_narrow_phase_arithmetic(optimization: Optimization) -> None:
    source = (Path(__file__).parent / "fixtures/fk_chain.urdf").read_text()
    kin = KinematicModel.from_urdf(source)
    spheres = [Sphere(link, (i * 0.13, 0.1, -0.2), 0.01) for i in range(4) for link in (5, 1)]
    dag, outputs = center_dag(kin, spheres, optimization)
    center_count = len(outputs)
    for link in (5, 1):
        pose = kin.poses[link]
        outputs.extend((*pose.translation, *pose.rotation))
    dag, outputs = dag.optimize(outputs, optimization)

    actual = collision_expressions(kin, spheres, optimization)
    assert actual.dag.nodes[: len(dag.nodes)] == dag.nodes
    assert [node for center in actual.centers for node in center] == outputs[:center_count]


def test_fetch_self_collision_rules() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets/fetch"
    kin = KinematicModel.from_urdf((assets / "fetch.urdf").read_text())
    config = read_config((assets / "collision.yaml").read_text())
    spheres = config.resolve(kin)
    rules = config.resolve_rules(kin, spheres)
    assert len(spheres) == 111
    assert len(rules.self_pairs) == 48
    assert sum(s.link_index not in rules.self_only_links for s in spheres) == 97
    assert {kin.link_names[i] for i in rules.self_only_links} == {"base_link"}
    assert (
        sum(len(config.groups[a]) * len(config.groups[b]) for a, b in config.self_collision_pairs)
        == 2586
    )


@pytest.mark.parametrize("optimization", ["none", "basic", "algebraic"])
def test_fetch_spheres_and_collisions_match_vamp(optimization: Optimization) -> None:
    assets = Path(__file__).resolve().parents[1] / "assets/fetch"
    kin = KinematicModel.from_urdf((assets / "fetch.urdf").read_text())
    config = read_config((assets / "collision.yaml").read_text())
    spheres = config.resolve(kin)
    rules = config.resolve_rules(kin, spheres)
    dag, outputs = center_expressions(kin, spheres, optimization)
    radii = np.array([sphere.radius for sphere in spheres])

    for case in fetch_collision_cases(kin, spheres):
        values = evaluate(dag, case.joints)
        centers = np.array([values[node] for node in outputs]).reshape(-1, 3)
        expected = np.array(case.spheres)
        # VAMP uses approximate SIMD trigonometry and FP32 intermediates.
        np.testing.assert_allclose(centers, expected[:, :3], atol=1e-5, rtol=0)
        np.testing.assert_allclose(radii, expected[:, 3], atol=1e-8, rtol=0)

        collision = False
        for a, b in rules.self_pairs:
            first = [i for i, sphere in enumerate(spheres) if sphere.link_index == a]
            second = [i for i, sphere in enumerate(spheres) if sphere.link_index == b]
            distances = np.linalg.norm(centers[first, None] - centers[second], axis=2)
            collision |= bool(np.any(distances <= radii[first, None] + radii[second]))
        assert collision == case.collision, case.name


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
    kin = KinematicModel.from_urdf((Path(__file__).parent / "fixtures/fk_chain.urdf").read_text())
    source = "collision_spheres:\n  base: {spheres: [[0, 0, 0, 1]]}\n  sensor: {spheres: [[0, 0, 0, 1]]}\n"
    with pytest.raises(ValueError):
        config = read_config(source + rules)
        config.resolve_rules(kin, config.resolve(kin))
