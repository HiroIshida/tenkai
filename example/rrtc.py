"""Solve and shortcut a Fetch RRT-Connect path, optionally visualizing it in viser."""

import math
from pathlib import Path

import numpy as np
import tyro

from tenkai import RRTConfig, ShortcutConfig, build_rrt, sdf
from tenkai.build import Backend, target_arch
from tenkai.utils import detect_backend

ROOT = Path(__file__).resolve().parents[1]


def main(
    urdf: Path | None = None,
    backend: Backend | None = None,
    arch: str | None = None,
    visualize: bool = False,
    host: str = "127.0.0.1",
    port: int = 8080,
    fps: float = 30.0,
) -> None:
    """Plan and shortcut a Fetch path, then optionally display it.

    Args:
        urdf: Override the URDF used for planning and visualization.
        backend: Auto-detect from compiler executables when omitted, preferring CUDA over HIP.
        visualize: Download Fetch meshes if needed, display the solution, and open a browser.
    """
    if backend is None:
        backend = detect_backend()
    arch = target_arch(backend, arch)
    print(f"Backend: {backend}, arch: {arch}", flush=True)

    planning_urdf = ROOT / "assets/fetch/fetch.urdf" if urdf is None else urdf
    start = [0.0, 1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]
    goal = [0.386, 0.205, 1.41, 0.308, -1.82, 0.245, 0.417, 6.01]
    obstacles: list[sdf.CollisionObject] = [
        sdf.Box(center=(0.95, 0.0, 0.8), size=(1.0, 2.0, 0.05)),
        sdf.Ground(),
    ]

    print("Compiling Fetch RRT-Connect...", flush=True)
    compiled = build_rrt(
        planning_urdf,
        ROOT / "assets/fetch/collision.yaml",
        ROOT / "artifacts/rrtc-example" / backend,
        backend=backend,
        arch=arch,
    )
    with compiled.planner(
        obstacles,
        config=RRTConfig(max_rounds=8192, goal_bias=0.0),
        problem_count=1,
        resolution=0.005,
    ) as planner:
        print("Solving and shortcutting Fetch problem...", flush=True)
        result = planner.solve([start], [goal], [0], [42], shortcut=ShortcutConfig())[0]
        if result["status"] != "success":
            raise RuntimeError(f"Fetch planning failed: {result['status']}")
        path = planner.path(0)
    print(f"Solved: {len(path)} waypoints after shortcutting.", flush=True)

    if not visualize:
        return

    # Planning alone needs neither the visualization dependencies nor mesh downloads.
    from tenkai.visualize import Scene

    if urdf is None:
        from robot_descriptions.fetch_description import URDF_PATH

        urdf = Path(URDF_PATH)

    frames = [path[:1]]
    for begin, end in zip(path[:-1], path[1:], strict=True):
        steps = max(1, math.ceil(float(np.max(np.abs(end - begin))) / 0.02))
        frames.append(np.linspace(begin, end, steps + 1)[1:])
    configurations = np.concatenate(frames)
    joint_names = [joint["name"] for joint in compiled.model["control_joints"]]

    with Scene(host=host, port=port) as scene:
        for obstacle in obstacles:
            scene.add_obstacle(obstacle)
        robot = scene.add_robot(urdf)
        robot.set_trajectory(configurations, joint_names=joint_names)
        scene.run(fps=fps, open_browser=visualize)


if __name__ == "__main__":
    tyro.cli(main)
