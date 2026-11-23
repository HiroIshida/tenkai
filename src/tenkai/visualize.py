"""Build interactive robot scenes with optional trajectory playback."""

import logging
import math
import time
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Self

import numpy as np
import numpy.typing as npt
import viser
from viser.extras import ViserUrdf
from viser.transforms import SO3

from tenkai import sdf

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    centers: npt.NDArray[np.float64]
    radii: npt.NDArray[np.float64]
    obstacles: list[sdf.CollisionObject]
    joint_names: tuple[str, ...]
    configurations: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.configurations.ndim != 2 or min(self.configurations.shape) < 1:
            raise ValueError("configurations must have nonempty shape (frames, joints)")
        if not np.isfinite(self.configurations).all():
            raise ValueError("configurations must be finite")
        if len(self.joint_names) != self.configurations.shape[1]:
            raise ValueError("joint_names must have one name per configuration column")
        if any(not isinstance(name, str) or not name for name in self.joint_names):
            raise ValueError("joint_names must contain nonempty strings")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")

        if self.radii.ndim != 1:
            raise ValueError("sphere radii must be a one-dimensional array")
        if self.centers.shape != (len(self.configurations), len(self.radii), 3):
            raise ValueError("sphere centers must have shape (frames, spheres, 3)")
        if not np.isfinite(self.centers).all() or not np.isfinite(self.radii).all():
            raise ValueError("sphere centers and radii must be finite")
        if (self.radii < 0).any():
            raise ValueError("sphere radii must be nonnegative")

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path, allow_pickle=False) as data:
            centers = np.asarray(data["centers"], dtype=np.float64)
            radii = np.asarray(data["radii"], dtype=np.float64)
            obstacles = sdf.geometry_from_json(str(data["obstacles"].item()))
            names = data["joint_names"]
            configurations = np.asarray(data["configurations"], dtype=np.float64)

        if names.ndim != 1 or names.dtype.kind != "U" or len(names) == 0:
            raise ValueError("joint_names must be a nonempty array of strings")
        joint_names = tuple(str(name) for name in names)
        return cls(centers, radii, obstacles, joint_names, configurations)

    def configurations_for(self, actuated_joints: tuple[str, ...]) -> npt.NDArray[np.float64]:
        indices = {name: index for index, name in enumerate(actuated_joints)}
        missing = set(self.joint_names) - indices.keys()
        if missing:
            raise ValueError("visual URDF is missing control joints: " + ", ".join(sorted(missing)))

        # The planner fixes all non-control joints at zero.
        result = np.zeros((len(self.configurations), len(actuated_joints)), dtype=np.float64)
        for column, name in enumerate(self.joint_names):
            result[:, indices[name]] = self.configurations[:, column]
        return result


@dataclass
class _Playback:
    frame: viser.GuiSliderHandle[int]
    playing: viser.GuiCheckboxHandle
    speed: viser.GuiSliderHandle[float]
    frame_count: int
    remainder: float = 0.0

    def advance(self, frames: float) -> None:
        if not self.playing.value or self.frame_count <= 1:
            return
        self.remainder += frames * self.speed.value
        steps = int(self.remainder)
        self.remainder -= steps
        if steps:
            # viser 1.1 declares its setter nonadjacently; mypy misses it.
            self.frame.value = (self.frame.value + steps) % self.frame_count  # type: ignore[misc]


class Scene:
    """A viser scene populated explicitly with obstacles and robots."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        self.server = viser.ViserServer(host=host, port=port, label="tenkai")
        self.server.scene.set_up_direction("+z")
        self._obstacles: list[sdf.CollisionObject] = []
        self._robots: list[Robot] = []
        self._closed = False

        @self.server.on_client_connect
        def set_camera(client: viser.ClientHandle) -> None:
            client.camera.position = (3.0, 2.5, 1.9)
            client.camera.look_at = (0.45, 0.0, 0.85)

    @property
    def obstacles(self) -> tuple[sdf.CollisionObject, ...]:
        """Registered geometry, also usable as a planner's environment."""
        return tuple(self._obstacles)

    def add_obstacle(
        self,
        obstacle: sdf.CollisionObject,
        *,
        color: tuple[int, int, int] = (185, 160, 125),
        opacity: float = 0.65,
    ) -> None:
        """Register and display another obstacle; no ground or box is implicit."""
        self._require_open()
        name = f"/obstacles/{len(self._obstacles)}"
        match obstacle:
            case sdf.Box():
                self.server.scene.add_box(
                    name,
                    dimensions=obstacle.size,
                    position=obstacle.center,
                    wxyz=SO3.from_matrix(np.asarray(obstacle.rotation)).wxyz,
                    color=color,
                    opacity=opacity,
                )
            case sdf.Sphere():
                self.server.scene.add_icosphere(
                    name,
                    radius=obstacle.radius,
                    position=obstacle.center,
                    color=color,
                    opacity=opacity,
                )
            case sdf.Cylinder():
                self.server.scene.add_cylinder(
                    name,
                    radius=obstacle.radius,
                    height=obstacle.height,
                    position=obstacle.center,
                    wxyz=SO3.from_matrix(np.asarray(obstacle.rotation)).wxyz,
                    color=color,
                    opacity=opacity,
                )
            case sdf.Ground():
                self.server.scene.add_grid(
                    name,
                    plane="xy",
                    position=(0.0, 0.0, obstacle.z),
                    section_color=color,
                )
            case _:
                raise TypeError("expected sdf.Box, Sphere, Cylinder, or Ground")
        self._obstacles.append(obstacle)

    def add_robot(self, urdf: Path | None = None) -> "Robot":
        """Register a robot independently of a trajectory.

        A URDF displays its mesh immediately at zero joint positions. Without a
        URDF, the robot displays recorded collision spheres once a trajectory is set.
        """
        self._require_open()
        if urdf is not None and not urdf.is_file():
            raise FileNotFoundError(urdf)
        robot = Robot(self, len(self._robots), urdf)
        self._robots.append(robot)
        return robot

    def run(self, fps: float = 30.0, *, open_browser: bool = False) -> None:
        """Serve until interrupted, optionally opening the scene in a browser."""
        self._require_open()
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        try:
            if open_browser:
                host = self.server.get_host()
                if host == "0.0.0.0":
                    host = "127.0.0.1"
                elif host == "::":
                    host = "::1"
                if ":" in host:
                    host = f"[{host}]"
                url = f"http://{host}:{self.server.get_port()}"
                if not webbrowser.open_new_tab(url):
                    logger.warning("Could not open a browser. Open %s manually.", url)

            previous = time.monotonic()
            while not self._closed:
                time.sleep(1.0 / fps)
                if self._closed:
                    break
                now = time.monotonic()
                frames = (now - previous) * fps
                previous = now
                for robot in self._robots:
                    robot._advance(frames)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("scene is closed")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.server.stop()

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class Robot:
    """A registered robot with optional, replaceable trajectory playback."""

    def __init__(self, scene: Scene, index: int, urdf: Path | None) -> None:
        self._scene = scene
        self._root = f"/robots/{index}"
        self._lock = RLock()
        self._mesh: ViserUrdf | None = None
        self._spheres: list[viser.IcosphereHandle] = []
        self._ghosts: list[viser.IcosphereHandle] = []
        self._playback: _Playback | None = None
        self._playback_folder: viser.GuiFolderHandle | None = None
        server = scene.server

        mesh_root = server.scene.add_frame(f"{self._root}/mesh", show_axes=False)
        if urdf is not None:
            self._mesh = ViserUrdf(server, urdf, root_node_name=f"{self._root}/mesh")
            joints = self._mesh.get_actuated_joint_names()
            self._mesh.update_cfg(np.zeros(len(joints)))
        sphere_root = server.scene.add_frame(
            f"{self._root}/spheres", show_axes=False, visible=self._mesh is None
        )

        self._gui = server.gui.add_folder(f"Robot {index + 1}")
        with self._gui:
            show_mesh = server.gui.add_checkbox(
                "Show robot mesh", initial_value=self._mesh is not None, disabled=self._mesh is None
            )
            self._show_spheres = server.gui.add_checkbox(
                "Show collision spheres", initial_value=self._mesh is None, disabled=True
            )

        @show_mesh.on_update
        def update_mesh_visibility(_: viser.GuiEvent[viser.GuiCheckboxHandle]) -> None:
            mesh_root.visible = show_mesh.value

        @self._show_spheres.on_update
        def update_sphere_visibility(_: viser.GuiEvent[viser.GuiCheckboxHandle]) -> None:
            sphere_root.visible = self._show_spheres.value

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Default column order for configuration arrays: all actuated URDF joints."""
        if self._mesh is None:
            return ()
        return self._mesh.get_actuated_joint_names()

    def set_trajectory(
        self,
        configurations: npt.ArrayLike,
        *,
        joint_names: Sequence[str] | None = None,
        sphere_centers: npt.ArrayLike | None = None,
        sphere_radii: npt.ArrayLike | None = None,
    ) -> None:
        """Set or replace playback from a (frames, joints) array.

        Columns follow robot.joint_names unless joint_names specifies a subset
        or another order. Unspecified joints stay at zero. Optional recorded
        sphere poses enable collision overlays or playback without a URDF.
        Input arrays are copied; later caller mutations do not affect playback.
        """
        self._scene._require_open()
        values = np.asarray(configurations)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError("configurations must have nonempty shape (frames, joints)")
        if values.dtype.kind not in "fiu":
            raise ValueError("configurations must contain real numbers")
        values = np.array(values, dtype=np.float64, copy=True)
        names = self.joint_names if joint_names is None else tuple(joint_names)

        if (sphere_centers is None) != (sphere_radii is None):
            raise ValueError("sphere_centers and sphere_radii must be supplied together")
        if sphere_centers is None:
            centers = np.empty((len(values), 0, 3), dtype=np.float64)
            radii = np.empty(0, dtype=np.float64)
        else:
            centers = np.array(sphere_centers, dtype=np.float64, copy=True)
            radii = np.array(sphere_radii, dtype=np.float64, copy=True)
        motion = Trajectory(centers, radii, [], names, values)
        if self._mesh is None and len(radii) == 0:
            raise ValueError("a robot without a URDF requires recorded sphere poses")

        mesh_configurations: npt.NDArray[np.float64] | None = None
        if self._mesh is not None:
            mesh_configurations = motion.configurations_for(self.joint_names)

        server = self._scene.server
        with self._lock, server.atomic():
            self._playback = None
            if self._playback_folder is not None:
                self._playback_folder.remove()
            for handle in self._spheres + self._ghosts:
                handle.remove()
            self._spheres.clear()
            self._ghosts.clear()

            for index, radius in enumerate(motion.radii):
                self._spheres.append(
                    server.scene.add_icosphere(
                        f"{self._root}/spheres/{index}",
                        radius=float(radius),
                        position=motion.centers[0, index],
                        color=(80, 155, 235),
                        subdivisions=2,
                        opacity=0.4 if self._mesh is not None else 1.0,
                    )
                )
                for name, frame_index, color in (
                    ("start", 0, (75, 200, 130)),
                    ("goal", -1, (245, 155, 65)),
                ):
                    self._ghosts.append(
                        server.scene.add_icosphere(
                            f"{self._root}/{name}/{index}",
                            radius=float(radius),
                            position=motion.centers[frame_index, index],
                            color=color,
                            wireframe=True,
                            subdivisions=1,
                            visible=False,
                        )
                    )
            self._show_spheres.disabled = len(motion.radii) == 0

            with self._gui:
                self._playback_folder = server.gui.add_folder("Playback")
            with self._playback_folder:
                self._create_playback(motion, mesh_configurations)
            self._show_frame(motion, mesh_configurations, 0)

    def _create_playback(
        self, motion: Trajectory, mesh_configurations: npt.NDArray[np.float64] | None
    ) -> None:
        server = self._scene.server
        last_frame = len(motion.centers) - 1
        if len(motion.radii):
            server.gui.add_markdown("Blue: robot collision spheres. Green: start. Orange: goal.")
        server.gui.add_markdown("Playback speed is illustrative; this path has no timestamps.")
        playing = server.gui.add_checkbox("Play", initial_value=True)
        frame = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(1, last_frame),
            step=1,
            initial_value=0,
            disabled=last_frame == 0,
        )
        speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        endpoints = server.gui.add_checkbox(
            "Show start / goal", initial_value=False, disabled=len(motion.radii) == 0
        )
        playback = _Playback(frame, playing, speed, len(motion.centers))
        self._playback = playback

        @frame.on_update
        def update_frame(_: viser.GuiEvent[viser.GuiSliderHandle[int]]) -> None:
            with self._lock:
                if self._playback is playback:
                    index = min(frame.value, last_frame)
                    self._show_frame(motion, mesh_configurations, index)

        @endpoints.on_update
        def update_endpoints(_: viser.GuiEvent[viser.GuiCheckboxHandle]) -> None:
            with self._lock, server.atomic():
                if self._playback is playback:
                    for ghost in self._ghosts:
                        ghost.visible = endpoints.value

    def _show_frame(
        self,
        motion: Trajectory,
        mesh_configurations: npt.NDArray[np.float64] | None,
        index: int,
    ) -> None:
        with self._scene.server.atomic():
            for sphere, position in zip(self._spheres, motion.centers[index], strict=True):
                sphere.position = position
            if self._mesh is not None and mesh_configurations is not None:
                self._mesh.update_cfg(mesh_configurations[index])

    def _advance(self, frames: float) -> None:
        with self._lock:
            if self._playback is not None:
                self._playback.advance(frames)
