"""plainmp CPU side of the benchmarks: problem construction, solving, and path checks."""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from plainmp.constraint import SphereAttachmentSpec, SphereCollisionCst
from plainmp.kinematics import BaseType, KinematicModel
from plainmp.ompl_solver import (
    OMPLSolver,
    OMPLSolverConfig,
    PlainmpRRTCSettings,
    set_log_level_none,
    set_random_seed,
)
from plainmp.problem import Problem
from plainmp.psdf import BoxSDF, CylinderSDF, GroundSDF, Pose, SDFBase, SphereSDF, UnionSDF
from problems import CONFIG, RESOLUTION, Batch, ProblemSet

from tenkai import sdf
from tenkai.compiler import ModelMetadata
from tenkai.modeling import read_config


def make_problems(
    urdf: Path, spheres: Path, model: ModelMetadata, cases: ProblemSet
) -> list[Problem]:
    config = read_config(spheres.read_text())
    attachments: list[SphereAttachmentSpec] = []
    for link, definitions in config.groups.items():
        values = np.asarray(definitions, dtype=np.float64).reshape(-1, 4)
        centers = np.asfortranarray(values[:, :3].T)
        radii = np.ascontiguousarray(values[:, 3])
        self_only = link in config.only_self_collision_links
        attachments.append(SphereAttachmentSpec(link, centers, radii, self_only))
    names = [joint["name"] for joint in model["control_joints"]]
    kinematics = KinematicModel(urdf.read_text())
    lower = np.array([joint["lower"] for joint in model["control_joints"]])
    upper = np.array([joint["upper"] for joint in model["control_joints"]])
    problems = []
    for goal, obstacles in zip(cases.goals, cases.environments, strict=True):
        checker = SphereCollisionCst(
            kinematics, names, BaseType.FIXED, attachments, config.self_collision_pairs, None, True
        )
        checker.set_sdf(plainmp_scene(obstacles))
        problems.append(
            Problem(cases.start, lower, upper, goal, checker, None, np.full(len(names), RESOLUTION))
        )
    return problems


def plainmp_scene(obstacles: list[sdf.CollisionObject]) -> UnionSDF:
    primitives: list[SDFBase] = []
    for obstacle in obstacles:
        match obstacle:
            case sdf.Ground():
                primitives.append(GroundSDF(-obstacle.z))
            case sdf.Box():
                pose = Pose(np.array(obstacle.center), np.array(obstacle.rotation))
                primitives.append(BoxSDF(np.array(obstacle.size), pose))
            case sdf.Sphere():
                primitives.append(
                    SphereSDF(obstacle.radius, Pose(np.array(obstacle.center), np.eye(3)))
                )
            case sdf.Cylinder():
                pose = Pose(np.array(obstacle.center), np.array(obstacle.rotation))
                primitives.append(CylinderSDF(obstacle.radius, obstacle.height, pose))
    return UnionSDF(primitives)


def make_solver() -> OMPLSolver:
    settings = PlainmpRRTCSettings()
    settings.max_samples = CONFIG.max_nodes
    return OMPLSolver(
        OMPLSolverConfig(
            n_max_call=10000000, algorithm_range=CONFIG.step_size, plainmp_rrtc_settings=settings
        )
    )


def make_cpu_solve(problems: list[Problem], batch: Batch) -> Callable[[], int]:
    """Return a function solving every (problem, seed) pair and counting successes."""
    solver = make_solver()
    set_log_level_none()

    def solve() -> int:
        successes = 0
        for index, seed in zip(batch.ids, batch.seeds, strict=True):
            set_random_seed(int(seed))
            successes += solver.solve(problems[index]).success
        return successes

    return solve


def path_is_valid(problem: Problem, path: npt.NDArray[np.float64]) -> bool:
    if not (
        np.allclose(path[0], problem.start, rtol=0, atol=1e-6)
        and np.allclose(path[-1], problem.goal_const, rtol=0, atol=1e-6)
        and np.isfinite(path).all()
        and (path >= problem.lb - 1e-6).all()
        and (path <= problem.ub + 1e-6).all()
    ):
        return False
    for begin, end in zip(path[:-1], path[1:], strict=True):
        steps = max(1, math.ceil(float(np.max(np.abs(end - begin))) / (RESOLUTION / 4)))
        if not all(
            problem.global_ineq_const.is_valid(q) for q in np.linspace(begin, end, steps + 1)
        ):
            return False
    return True
