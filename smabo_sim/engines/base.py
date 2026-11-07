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

    @abstractmethod
    def scan(self, angles: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
        """Lidar ranges for ``angles`` (rad, laser frame); 0.0 = no return."""
