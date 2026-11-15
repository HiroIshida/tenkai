import math
from pathlib import Path

import pytest

from tenkai import KinematicModel
from tenkai.compiler import control_joints

CHAIN = (Path(__file__).parent / "fixtures/fk_chain.urdf").read_text()


def test_controls_keep_requested_order_and_input_indices() -> None:
    kin = KinematicModel.from_urdf(CHAIN)
    bounds = control_joints(kin, ["slide", "roll", "yaw"])
    assert bounds == [
        {"name": "slide", "index": 3, "lower": -1.0, "upper": 1.0},
        {"name": "roll", "index": 2, "lower": -math.tau, "upper": math.tau},
        {"name": "yaw", "index": 0, "lower": -3.2, "upper": 3.2},
    ]
    assert (kin.joints[2].lower, kin.joints[2].upper) == (0.0, 0.0)


@pytest.mark.parametrize("name", ["missing", "tool_mount"])
def test_controls_require_movable_joints(name: str) -> None:
    kin = KinematicModel.from_urdf(CHAIN)
    with pytest.raises(ValueError, match="control is not a movable joint: " + name):
        control_joints(kin, [name])


@pytest.mark.parametrize("lower", ["3.2", "4", "nan", "-inf"])
def test_sampling_bounds_are_validated_only_for_controlled_joints(lower: str) -> None:
    source = CHAIN.replace('lower="-3.2"', f'lower="{lower}"', 1)
    kin = KinematicModel.from_urdf(source)
    assert control_joints(kin, ["slide"]) == [
        {"name": "slide", "index": 3, "lower": -1.0, "upper": 1.0}
    ]
    with pytest.raises(ValueError, match="invalid sampling bounds for yaw"):
        control_joints(kin, ["yaw"])
