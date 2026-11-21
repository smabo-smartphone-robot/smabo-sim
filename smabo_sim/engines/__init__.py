"""Engine registry. Add new engines here."""

from __future__ import annotations

from ..world import World
from .base import Engine


def create_engine(name: str, world: World, cfg) -> Engine:
    if name == "kinematic":
        from .kinematic import KinematicEngine
        return KinematicEngine(world, cfg)
    if name == "mujoco":
        try:
            from .mujoco_engine import MujocoEngine
        except ImportError as e:
            raise RuntimeError("the mujoco engine needs MuJoCo: pip install 'smabo-sim[mujoco]'") from e
        return MujocoEngine(world, cfg)
    raise ValueError(f"unknown engine {name!r} (available: {', '.join(ENGINES)})")


ENGINES = ("kinematic", "mujoco")


__all__ = ["Engine", "create_engine"]
