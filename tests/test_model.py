import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from helpers import evaluate

from tenkai import JointInfo, KinematicModel, generate_rrt_sources
from tenkai.modeling import read_config

ROOT = Path(__file__).resolve().parents[1]
CHAIN = (ROOT / "tests/fixtures/fk_chain.urdf").read_text()


def position(kin: KinematicModel, inputs: list[float], name: str) -> list[float]:
    values = evaluate(kin.dag, inputs)
    pose = kin.poses[kin.link_names.index(name)]
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
    kin = KinematicModel.from_urdf(CHAIN)
    yaw, pitch, roll, slide = angles
    sa, ca = math.sin(yaw), math.cos(yaw)
    sb, cb = math.sin(pitch), math.cos(pitch)
    sc, cc = math.sin(roll), math.cos(roll)
    wrist = [1 + 2 * ca + (3 + slide) * ca * cb, 2 * sa + (3 + slide) * sa * cb, -(3 + slide) * sb]
    offset = [ca * sb * sc - sa * cc, sa * sb * sc + ca * cc, cb * sc]
    tool = [wrist[i] + offset[i] for i in range(3)]
    assert position(kin, list(angles), "slide_link") == pytest.approx(wrist, abs=1e-12)
    assert position(kin, list(angles), "tool") == pytest.approx(tool, abs=1e-12)
    assert position(kin, list(angles), "sensor") == pytest.approx([0, 0, 2], abs=1e-12)


def test_joint_order_and_origin_rotation() -> None:
    root = ET.fromstring(CHAIN)
    slide = root.find("joint[@name='slide']")
    assert slide is not None
    ET.SubElement(slide, "origin", xyz="0 0 1", rpy=f"0 {math.pi / 2} 0")
    kin = KinematicModel.from_urdf(ET.tostring(root, encoding="unicode"))
    assert position(kin, [0, 0, 0, 0.5], "slide_link") == pytest.approx([6, 0, 0.5], abs=1e-12)

    root[:] = root[::-1]
    reversed_kin = KinematicModel.from_urdf(ET.tostring(root, encoding="unicode"))
    assert reversed_kin.joint_names == ["slide", "roll", "pitch", "yaw"]
    assert position(reversed_kin, [0.5, 0, 0, 0], "slide_link") == pytest.approx(
        [6, 0, 0.5], abs=1e-12
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_joint_metadata_follows_input_order(reverse: bool) -> None:
    root = ET.fromstring(CHAIN)
    expected = [
        JointInfo("yaw", "revolute", -3.2, 3.2),
        JointInfo("pitch", "revolute", -3.2, 3.2),
        JointInfo("roll", "continuous", 0.0, 0.0),
        JointInfo("slide", "prismatic", -1.0, 1.0),
    ]
    if reverse:
        root[:] = root[::-1]
        expected.reverse()

    kin = KinematicModel.from_urdf(ET.tostring(root, encoding="unicode"))
    assert kin.joints == expected
    assert kin.joint_names == [joint.name for joint in expected]
    inputs = [0.0, 0.0, 0.0, 0.5]
    if reverse:
        inputs.reverse()
    assert position(kin, inputs, "slide_link") == pytest.approx([6.5, 0, 0], abs=1e-12)


def test_fetch_spheres_follow_fk_order() -> None:
    kin = KinematicModel.from_urdf((ROOT / "assets/fetch/fetch.urdf").read_text())
    config = read_config((ROOT / "assets/fetch/collision.yaml").read_text())
    spheres = config.resolve(kin)
    assert len(kin.joint_names) == 15
    assert len(kin.poses) == 26
    assert len(spheres) == 69
    assert [s.link_index for s in spheres] == sorted(s.link_index for s in spheres)
    for name, count in [("gripper_link", 9), ("head_pan_link", 16), ("torso_lift_link", 2)]:
        assert sum(kin.link_names[s.link_index] == name for s in spheres) == count


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
        KinematicModel.from_urdf(source)


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
    kin = KinematicModel.from_urdf('<robot name="one"><link name="root"/></robot>')
    with pytest.raises(ValueError):
        config = read_config(source)
        config.resolve(kin)


@pytest.mark.parametrize("controls", [[], [100], [0, 0]])
def test_invalid_planner_controls(controls: list[int]) -> None:
    kin = KinematicModel.from_urdf(CHAIN)
    with pytest.raises(ValueError):
        generate_rrt_sources(kin, [], controls)


def test_empty_spheres_and_root_pose() -> None:
    kin = KinematicModel.from_urdf('<robot name="one"><link name="root"/></robot>')
    config = read_config("collision_spheres: {}")
    assert kin.joints == []
    assert kin.joint_names == []
    assert config.resolve(kin) == []
    values = evaluate(kin.dag, [])
    assert [values[i] for i in kin.poses[0].rotation] == [0, 0, 0, 1]
    assert position(kin, [], "root") == [0, 0, 0]
