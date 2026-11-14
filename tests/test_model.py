import math
from dataclasses import replace
from pathlib import Path

import pytest
from helpers import evaluate

from tenkai import FkDag, Robot, generate_rrt_sources, read_spheres

ROOT = Path(__file__).resolve().parents[1]
CHAIN = (ROOT / "tests/fixtures/fk_chain.urdf").read_text()


def position(fk: FkDag, inputs: list[float], name: str) -> list[float]:
    values = evaluate(fk.dag, inputs)
    pose = fk.poses[fk.link_names.index(name)]
    return [values[i] for i in pose.translation]


@pytest.mark.parametrize(
    "angles",
    [
        (0, 0, 0, 0.5),
        (math.pi / 2, 0, 0, 0.5),
        (0, math.pi / 2, 0, 0.5),
        (0, 0, math.pi / 2, 0.5),
        (0.31, -0.72, 1.17, 0.24),
        (-1.2, 0.8, -0.4, -0.5),
    ],
)
def test_fk_matches_closed_form(angles: tuple[float, float, float, float]) -> None:
    fk = FkDag.from_robot(Robot.from_string(CHAIN))
    yaw, pitch, roll, slide = angles
    sa, ca = math.sin(yaw), math.cos(yaw)
    sb, cb = math.sin(pitch), math.cos(pitch)
    sc, cc = math.sin(roll), math.cos(roll)
    wrist = [1 + 2 * ca + (3 + slide) * ca * cb, 2 * sa + (3 + slide) * sa * cb, -(3 + slide) * sb]
    offset = [ca * sb * sc - sa * cc, sa * sb * sc + ca * cc, cb * sc]
    tool = [wrist[i] + offset[i] for i in range(3)]
    assert position(fk, list(angles), "slide_link") == pytest.approx(wrist, abs=1e-12)
    assert position(fk, list(angles), "tool") == pytest.approx(tool, abs=1e-12)
    assert position(fk, list(angles), "sensor") == pytest.approx([0, 0, 2], abs=1e-12)


def test_joint_order_and_origin_rotation() -> None:
    robot = Robot.from_string(CHAIN)
    robot.joints[3] = replace(robot.joints[3], xyz=(0, 0, 1), rpy=(0, math.pi / 2, 0))
    fk = FkDag.from_robot(robot)
    assert position(fk, [0, 0, 0, 0.5], "slide_link") == pytest.approx([6, 0, 0.5], abs=1e-12)
    robot.links.reverse()
    robot.joints.reverse()
    reversed_fk = FkDag.from_robot(robot)
    assert reversed_fk.joint_names == ["slide", "roll", "pitch", "yaw"]
    assert position(reversed_fk, [0.5, 0, 0, 0], "slide_link") == pytest.approx(
        [6, 0, 0.5], abs=1e-12
    )


def test_fetch_spheres_follow_fk_order() -> None:
    robot = Robot.from_string((ROOT / "assets/fetch/fetch.urdf").read_text())
    fk = FkDag.from_robot(robot)
    spheres = read_spheres((ROOT / "assets/fetch/collision.yaml").read_text(), fk)
    assert len(fk.joint_names) == 15
    assert len(fk.poses) == 26
    assert len(spheres) == 69
    assert [s.link_index for s in spheres] == sorted(s.link_index for s in spheres)
    for name, count in [("gripper_link", 9), ("head_pan_link", 16), ("torso_lift_link", 2)]:
        assert sum(fk.link_names[s.link_index] == name for s in spheres) == count


@pytest.mark.parametrize(
    "source",
    [
        '<robot name="bad"/>',
        '<robot name="bad"><link name="x"/><link name="x"/></robot>',
        '<robot name="bad"><link name="x"/><link name="y"/></robot>',
        (ROOT / "tests/fixtures/arm.urdf").read_text(),
        CHAIN.replace('type="revolute"', 'type="floating"', 1),
        CHAIN.replace('parent link="base"', 'parent link="missing"', 1),
        CHAIN.replace('parent link="base"', 'parent link="tool"', 1),
        CHAIN.replace('axis xyz="0 0 1"', 'axis xyz="0 0 0"', 1),
    ],
)
def test_invalid_models_are_rejected(source: str) -> None:
    with pytest.raises(ValueError):
        FkDag.from_robot(Robot.from_string(source))


@pytest.mark.parametrize(
    "source",
    [
        "{}",
        "collision_spheres: {missing: {spheres: []}}",
        "collision_spheres: {root: {sphere: []}}",
        "collision_spheres: {root: {spheres: [[0, 0, 1]]}}",
        "collision_spheres: {root: {spheres: [[0, 0, 1, 1, 1]]}}",
        "collision_spheres: {root: {spheres: []}, root: {spheres: []}}",
        "collision_spheres: {root: {spheres: [[.nan, 0, 0, 1]]}}",
        "collision_spheres: {root: {spheres: [[0, .inf, 0, 1]]}}",
        "collision_spheres: {root: {spheres: [[0, 0, 0, -1]]}}",
        "collision_spheres: {root: {spheres: [[0, 0, 0, .inf]]}}",
    ],
)
def test_invalid_spheres_are_rejected(source: str) -> None:
    fk = FkDag.from_robot(Robot.from_string('<robot name="one"><link name="root"/></robot>'))
    with pytest.raises(ValueError):
        read_spheres(source, fk)


@pytest.mark.parametrize("controls", [[], [100], [0, 0]])
def test_invalid_planner_controls(controls: list[int]) -> None:
    fk = FkDag.from_robot(Robot.from_string(CHAIN))
    with pytest.raises(ValueError):
        generate_rrt_sources(fk, [], controls)


def test_empty_spheres_and_root_pose() -> None:
    fk = FkDag.from_robot(Robot.from_string('<robot name="one"><link name="root"/></robot>'))
    assert read_spheres("collision_spheres: {}", fk) == []
    values = evaluate(fk.dag, [])
    assert [values[i] for i in fk.poses[0].rotation] == [0, 0, 0, 1]
    assert position(fk, [], "root") == [0, 0, 0]
