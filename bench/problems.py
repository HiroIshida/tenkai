"""Benchmark problem sets: file format, release download, and shared settings."""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

from tenkai import RRTConfig, sdf
from tenkai.compiler import ModelMetadata

RESOLUTION = 0.02
CONFIG = RRTConfig(max_rounds=8192, goal_bias=0.0)
CACHE = Path("artifacts/rrtc-problems")
SEED_OFFSET = 42
ASSETS = Path(__file__).resolve().parents[1] / "assets"


@dataclass(frozen=True)
class ProblemSetSpec:
    """A problem set published as an asset of the GitHub release `tag`."""

    name: str
    tag: str
    sha256: str
    urdf: Path
    spheres: Path

    @property
    def path(self) -> Path:
        return CACHE / f"{self.name}_problems.yaml"


PROBLEM_SETS = {
    spec.name: spec
    for spec in [
        ProblemSetSpec(
            "fetch",
            "bench-fetch-data-v1",
            "39c343aa5302cad56e3560d4fe6bfea86844e612251b6f2f44bef405349212de",
            ASSETS / "fetch/fetch.urdf",
            ASSETS / "fetch/collision.yaml",
        ),
    ]
}


@dataclass
class Batch:
    starts: npt.NDArray[np.float64]
    goals: npt.NDArray[np.float64]
    ids: npt.NDArray[np.uint32]
    seeds: npt.NDArray[np.uint64]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, chunk: slice) -> "Batch":
        return Batch(self.starts[chunk], self.goals[chunk], self.ids[chunk], self.seeds[chunk])


@dataclass
class ProblemSet:
    start: npt.NDArray[np.float64]
    goals: npt.NDArray[np.float64]
    environments: list[list[sdf.CollisionObject]]
    multiplier: int

    def batch(self) -> Batch:
        ids = np.tile(np.arange(len(self.goals), dtype=np.uint32), self.multiplier)
        np.random.default_rng(42).shuffle(ids)
        return Batch(np.tile(self.start, (len(ids), 1)), self.goals[ids], ids, problem_seed(ids))


def problem_seed(index: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint64]:
    return index.astype(np.uint64) + SEED_OFFSET


def download_problems(spec: ProblemSetSpec) -> Path:
    if not spec.path.exists() or sha256(spec.path) != spec.sha256:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                spec.tag,
                "--clobber",
                "--dir",
                str(spec.path.parent),
                "--pattern",
                spec.path.name,
            ],
            check=True,
        )
        if sha256(spec.path) != spec.sha256:
            raise RuntimeError(f"{spec.path} does not match the pinned release checksum")
    return spec.path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


OBJECT_TYPES: dict[str, sdf.CollisionObjectType] = {
    "Box": sdf.Box,
    "Sphere": sdf.Sphere,
    "Cylinder": sdf.Cylinder,
    "Ground": sdf.Ground,
}


def load_problems(path: Path, model: ModelMetadata, multiplier: int | None = None) -> ProblemSet:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or set(data) != {"start", "multiplier", "problems"}:
        raise ValueError("problem set must contain start, multiplier, and problems")
    if multiplier is None:
        multiplier = data["multiplier"]
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError("multiplier must be a positive integer")
    records = data["problems"]
    if not isinstance(records, list) or not 0 < len(records) * multiplier < 2**32:
        raise ValueError("problems must be a nonempty list; total batch must fit positive uint32")
    lower = np.array([joint["lower"] for joint in model["control_joints"]])
    upper = np.array([joint["upper"] for joint in model["control_joints"]])

    def configuration(value: object, name: str) -> npt.NDArray[np.float64]:
        q = np.asarray(value, dtype=np.float64)
        if q.shape != lower.shape or not (
            np.isfinite(q).all() and (q >= lower).all() and (q <= upper).all()
        ):
            raise ValueError(f"{name} must contain {len(lower)} finite joint values within bounds")
        return q

    start = configuration(data["start"], "start")
    goals = []
    environments = []
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict) or set(record) != {"goal", "obstacles"}:
                raise ValueError("each problem must contain goal and obstacles")
            goals.append(configuration(record["goal"], "goal"))
            if not isinstance(record["obstacles"], list):
                raise ValueError("obstacles must be a list")
            obstacles = []
            for item in record["obstacles"]:
                parameters = dict(item)
                kind = parameters.pop("type")
                obstacles.append(OBJECT_TYPES[kind](**parameters))
            if sum(isinstance(item, sdf.Ground) for item in obstacles) != 1:
                raise ValueError("each environment must contain exactly one Ground")
            environments.append(obstacles)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"problems[{index}]: {error}") from error
    return ProblemSet(start, np.array(goals), environments, multiplier)
