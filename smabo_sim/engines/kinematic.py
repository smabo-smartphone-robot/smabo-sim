"""Kinematic engine: differential drive without dynamics.

* Each wheel's speed follows its target with a first-order lag
  (``sim.motor_time_constant``), standing in for motor inertia.
* The chassis integrates unicycle kinematics. A step that would bring the
  footprint (``sim.robot_radius``) into a wall is rejected: the robot stays put
  while the wheels keep turning, so the encoders (and hence odometry) drift —
  the same thing a real robot pushing a wall does.
* Encoders report wheel travel with multiplicative noise (``sim.encoder_noise``).
* The lidar ray-casts the world from the laser frame with Gaussian range noise
  (``sim.lidar_noise``); misses read 0.0 like the LD06 driver.
* Servos are ideal: joints sit at the commanded angle (the arm does not
  collide with anything in this engine).
* Only models on the TT-motor chassis (``sim.model``, see ``smabo_sim.models``)
  move, and only those with the LD06 see anything.
"""

from __future__ import annotations

import math

import numpy as np

from ..models import get_model
from ..world import Pose2D, World
from .base import Engine


class KinematicEngine(Engine):
    def __init__(self, world: World, cfg) -> None:
        super().__init__(world, cfg)
        self._rng = np.random.default_rng(cfg.get("sim.seed"))
        self._pose = Pose2D()
        self._target = (0.0, 0.0)
        self._speed = [0.0, 0.0]
        self._travel = [0.0, 0.0]
        self._joints: dict[str, float] = {}
        self.robot_model = get_model(cfg.get("sim.model"))
        self.blocked = False
        self.reset(world.start)

    def reset(self, pose: Pose2D) -> None:
        self._pose = Pose2D(pose.x, pose.y, pose.yaw)
        self._target, self._speed, self._travel = (0.0, 0.0), [0.0, 0.0], [0.0, 0.0]

    def set_wheel_targets(self, left: float, right: float) -> None:
        self._target = (left, right)

    def set_joint_targets(self, angles: dict[str, float]) -> None:
        self._joints.update(angles)

    def step(self, dt: float) -> None:
        if not self.robot_model.mobile:
            return
        tau = max(self.cfg.get("sim.motor_time_constant", 0.08), 1e-6)
        a = 1.0 - math.exp(-dt / tau)
        for i in (0, 1):
            self._speed[i] += (self._target[i] - self._speed[i]) * a

        noise = self.cfg.get("sim.encoder_noise", 0.0)
        for i in (0, 1):
            measured = self._speed[i] * (1.0 + (self._rng.normal(0.0, noise) if noise > 0 else 0.0))
            self._travel[i] += measured * dt

        sep = self.cfg.get("dc.wheel_separation", 0.13)
        v = (self._speed[0] + self._speed[1]) / 2.0
        w = (self._speed[1] - self._speed[0]) / sep
        p = self._pose
        mid = p.yaw + w * dt / 2.0
        nx, ny = p.x + v * math.cos(mid) * dt, p.y + v * math.sin(mid) * dt
        nyaw = math.atan2(math.sin(p.yaw + w * dt), math.cos(p.yaw + w * dt))

        radius = self.cfg.get("sim.robot_radius", 0.09)
        moving_in = self.world.clearance(nx, ny) < radius and \
            self.world.clearance(nx, ny) < self.world.clearance(p.x, p.y)
        self.blocked = moving_in
        if moving_in:
            self._pose = Pose2D(p.x, p.y, nyaw)   # translation blocked, spin allowed
        else:
            self._pose = Pose2D(nx, ny, nyaw)

    def pose(self) -> Pose2D:
        return Pose2D(self._pose.x, self._pose.y, self._pose.yaw)

    def read_wheel_distance(self) -> tuple[float, float]:
        out = (self._travel[0], self._travel[1])
        self._travel = [0.0, 0.0]
        return out

    def scan(self, angles: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
        if not self.robot_model.lidar:
            return np.zeros(len(angles))
        p = self._pose
        lx, ly = (self.cfg.get("sim.laser_offset") or [0.0, 0.0])[:2]
        ox = p.x + math.cos(p.yaw) * lx - math.sin(p.yaw) * ly
        oy = p.y + math.sin(p.yaw) * lx + math.cos(p.yaw) * ly
        r = self.world.raycast(ox, oy, np.asarray(angles) + p.yaw, range_max)
        noise = self.cfg.get("sim.lidar_noise", 0.0)
        if noise > 0:
            r = r + self._rng.normal(0.0, noise, size=r.shape)
        r[~np.isfinite(r) | (r < range_min) | (r > range_max)] = 0.0
        return r
