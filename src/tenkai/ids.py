"""Distinct integer indices, so node and joint indices cannot be mixed up."""

from typing import NewType

NodeId = NewType("NodeId", int)
JointId = NewType("JointId", int)
