"""NumPy dtypes matching the C++ structs that cross the ctypes boundary.

The C++ structs stay hand-written in the templates; the generated binding
asserts that each one matches its dtype here.
"""

from typing import Any

import numpy as np
import numpy.typing as npt


def _c_struct(*fields: tuple[Any, ...]) -> np.dtype[np.void]:
    return np.dtype(list(fields), align=True)


VEC3 = _c_struct(("x", "<f4"), ("y", "<f4"), ("z", "<f4"))
POSE = _c_struct(("position", VEC3), ("world_to_local", "<f4", (9,)))
BOX = _c_struct(("pose", POSE), ("extents", VEC3))
SPHERE = _c_struct(("position", VEC3), ("radius", "<f4"))
CYLINDER = _c_struct(("pose", POSE), ("radius", "<f4"), ("height", "<f4"))
GROUND = _c_struct(("height", "<f4"))
RANGE = _c_struct(("offset", "<u8"), ("count", "<u8"))
ENVIRONMENT = _c_struct(("boxes", RANGE), ("spheres", RANGE), ("cylinders", RANGE))

RRT_CONFIG = _c_struct(
    ("max_nodes", "<u4"), ("max_rounds", "<u4"), ("step_size", "<f8"), ("goal_bias", "<f8")
)
RRT_RESULT = _c_struct(
    ("status", "<u4"), ("node_count", "<u4"), ("rounds", "<u4"), ("path_length", "<u4")
)
SHORTCUT_CONFIG = _c_struct(
    ("max_steps", "<u4"),
    ("max_empty_steps", "<u4"),
    ("range_ratio", "<f8"),
    ("snap_to_vertex", "<f8"),
)

CPP_STRUCTS = {
    "tenkai_f32::Vec3": VEC3,
    "tenkai_f32::Pose": POSE,
    "tenkai_f32::Box": BOX,
    "tenkai_f32::Sphere": SPHERE,
    "tenkai_f32::Cylinder": CYLINDER,
    "tenkai_f32::Ground": GROUND,
    "tenkai_f32::Range": RANGE,
    "tenkai_f32::Environment": ENVIRONMENT,
    "tenkai::RrtConfig": RRT_CONFIG,
    "tenkai::RrtResult": RRT_RESULT,
    "tenkai::ShortcutConfig": SHORTCUT_CONFIG,
}


def pack_struct(dtype: np.dtype[np.void], **values: object) -> npt.NDArray[np.void]:
    """Return a one-element structured array, passable by pointer, with fields set by name."""
    return np.array([tuple(values[name] for name in dtype.names or ())], dtype=dtype)


def cpp_layout_asserts() -> str:
    lines: list[str] = []
    for name, dtype in CPP_STRUCTS.items():
        lines.append(f'static_assert(sizeof({name}) == {dtype.itemsize}, "{name} layout");')
        for field, item in (dtype.fields or {}).items():
            lines.append(f'static_assert(offsetof({name}, {field}) == {item[1]}, "{name} layout");')
    return "\n".join(lines) + "\n"
