"""Engine interface: the simulated hardware under the firmware layer.

The firmware layer (``smabo_sim.firmware``) decides *what* the robot is told to
do — wheel speed targets and servo angles — exactly as the ESP32 does. An
engine decides *what physically happens*: how the wheels actually move, what
the encoders count, what the lidar sees. Swap the engine to trade speed for
fidelity without touching the protocol or firmware code.

Implementations:
  * ``kinematic`` — ideal 2D kinematics, first-order motor lag, ray-cast lidar.
  * ``mujoco``    — contact physics for the chassis, wheels and arm (optional dependency).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from ..world import Pose2D, World


class Engine(ABC):
    def __init__(self, world: World, cfg) -> None:
        self.world = world
        self.cfg = cfg

    @abstractmethod
    def reset(self, pose: Pose2D) -> None:
        """Place the robot at ``pose`` (map frame) at rest."""

    @abstractmethod
    def set_wheel_targets(self, left: float, right: float) -> None:
        """Wheel surface speeds (m/s) the motors are driven toward."""

    @abstractmethod
    def set_joint_targets(self, angles: dict[str, float]) -> None:
        """Servo angles (rad) the firmware is commanding."""

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds."""

    @abstractmethod
    def pose(self) -> Pose2D:
        """Ground-truth pose in the map frame (not visible to the brain)."""

    @abstractmethod
    def read_wheel_distance(self) -> tuple[float, float]:
        """Encoder distance (m) travelled by each wheel since the last call."""

    def close(self) -> None:
        """Release resources (renderers) before the engine is replaced."""

    # vision targets (World.targets). Only engines with a camera show them;
    # the others just keep their settings.
    def targets(self) -> list[dict]:
        return [{"name": t["name"], "kind": t["kind"], "x": t["x"], "y": t["y"], "z": t["z"],
                 "yaw_deg": math.degrees(t["yaw"]), "size": t["size"], "sway": dict(t["sway"])}
                for t in self.world.targets]

    def set_target(self, name: str, **change) -> None:
        apply_target_change(self.world.target(name), change)

    @abstractmethod
    def scan(self, angles: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
        """Lidar ranges for ``angles`` (rad, laser frame); 0.0 = no return."""


def apply_target_change(t: dict, change: dict) -> None:
    """Apply ``set_target`` keywords to one World target dict: x, y, z (m),
    yaw_deg, sway_on, sway_amplitude (m), sway_period (s)."""
    for k in ("x", "y", "z"):
        if change.get(k) is not None:
            t[k] = float(change[k])
    if change.get("yaw_deg") is not None:
        t["yaw"] = math.radians(float(change["yaw_deg"]))
    for k in ("on", "amplitude", "period"):
        v = change.get(f"sway_{k}")
        if v is not None:
            t["sway"][k] = bool(v) if k == "on" else max(float(v), 0.5 if k == "period" else 0.0)
