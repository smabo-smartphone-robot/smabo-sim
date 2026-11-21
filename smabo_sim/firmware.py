"""Ports of the smabo-esp32 firmware logic that shapes what the robot does.

These mirror the real board's behaviour (including its quirks) so the brain
sees the same thing it would see on hardware:

  * ``DriveFirmware``  ← dc_motors.DiffDrive: Twist → per-wheel duty by
    differential kinematics, the ``dc.min_duty`` static-friction floor and the
    cmd_vel dead-man timeout.
  * ``ServoFirmware``  ← servo_controller.JointGroup: time-scheduled trajectory
    points, joint-limit clamping, the per-joint ``max_speed`` follower and the
    single-joint ``set_angle_deg`` (with the per-move speed override that
    random motion uses).

Time is injected (``now`` in seconds) so the logic is deterministic in tests.
"""

from __future__ import annotations

import math


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


_COAST = 0.001  # |duty| at or below this coasts the motor (dc_motors._COAST)


def _with_min_duty(duty: float, min_duty: float) -> float:
    """``dc_motors._with_min_duty``: lift non-zero duties to the static-friction floor."""
    if abs(duty) <= _COAST or min_duty <= 0.0 or abs(duty) >= min_duty:
        return duty
    return min_duty if duty > 0 else -min_duty


class DriveFirmware:
    """``dc_motors.DiffDrive.set_cmd_vel`` + ``watchdog_task``."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._duty = (0.0, 0.0)
        self._last_cmd = -math.inf

    def full_duty_speed(self) -> float:
        """Wheel surface speed (m/s) at duty 1.0.

        The firmware normalises wheel commands by ``max_linear + max_angular *
        sep / 2`` and maps that to full duty, so this is the speed the motors
        are assumed to reach at 100 % — the simulated motor's top speed.
        """
        max_lin = self.cfg.get("dc.max_linear", 0.3)
        max_ang = self.cfg.get("dc.max_angular", 1.5)
        sep = self.cfg.get("dc.wheel_separation", 0.13)
        return max(max_lin + max_ang * sep / 2.0, 1e-6)

    def set_cmd_vel(self, linear_x: float, angular_z: float, now: float) -> None:
        sep = self.cfg.get("dc.wheel_separation", 0.13)
        min_duty = self.cfg.get("dc.min_duty", 0.0)
        self._last_cmd = now
        v_l = linear_x - angular_z * sep / 2.0
        v_r = linear_x + angular_z * sep / 2.0
        norm = self.full_duty_speed()
        self._duty = (_with_min_duty(_clamp(v_l / norm, -1.0, 1.0), min_duty),
                      _with_min_duty(_clamp(v_r / norm, -1.0, 1.0), min_duty))

    def stop(self) -> None:
        self._duty = (0.0, 0.0)

    def duty(self, now: float) -> tuple[float, float]:
        """Current (left, right) duty in [-1, 1], zero after ``cmd_timeout``."""
        if now - self._last_cmd > self.cfg.get("dc.cmd_timeout", 0.5):
            self._duty = (0.0, 0.0)
        return self._duty

    def wheel_targets(self, now: float) -> tuple[float, float]:
        """Target wheel surface speeds (m/s) the motors are driven toward."""
        top = self.full_duty_speed()
        dl, dr = self.duty(now)
        # _Motor.drive coasts at |duty| <= 0.001
        return (0.0 if abs(dl) <= _COAST else dl * top,
                0.0 if abs(dr) <= _COAST else dr * top)


class ServoFirmware:
    """``servo_controller.JointGroup`` (trajectory queue + speed-limited follower).

    Angles are kept in degrees internally like the firmware; ``state()``
    returns radians for ``/joint_states``.
    """

    def __init__(self, joints: dict) -> None:
        self.joints = joints
        self._target: dict[str, float] = {}
        self._current: dict[str, float] = {}
        self._queue: list[tuple[float, dict[str, float]]] = []
        self._idx = 0
        self._speed_override: dict[str, float] = {}
        for name, spec in joints.items():
            init = spec.get("last_angle", spec.get("init_angle", 0.0))
            init = _clamp(init, spec["min_angle"], spec["max_angle"])
            self._target[name] = init
            self._current[name] = init

    def set_angle_deg(self, name: str, angle: float, speed: float | None = None) -> None:
        """Move one joint toward ``angle`` (deg), like ``JointGroup.set_angle_deg``.

        Unknown or disabled joints are ignored. ``speed`` (deg/s) overrides the
        joint's ``max_speed`` for this move (``None`` clears the override); an
        effective speed of 0 applies the angle at once.
        """
        spec = self.joints.get(name)
        if spec is None or not spec.get("enabled", True):
            return
        self._target[name] = _clamp(angle, spec["min_angle"], spec["max_angle"])
        if speed is None:
            self._speed_override.pop(name, None)
        else:
            self._speed_override[name] = speed
        eff = speed if speed is not None else spec.get("max_speed", 0)
        if eff <= 0:
            self._current[name] = self._target[name]

    def load_trajectory(self, joint_names: list[str], points: list[dict], now: float) -> None:
        """Schedule each point at ``now + time_from_start``; replaces any queue."""
        queue = []
        for pt in points or []:
            tfs = pt.get("time_from_start") or {}
            t = float(tfs.get("sec", 0)) + float(tfs.get("nanosec", 0)) * 1e-9
            targets = {n: p * 180.0 / math.pi
                       for n, p in zip(joint_names, pt.get("positions") or []) if n in self.joints}
            for n in targets:
                self._speed_override.pop(n, None)     # trajectories use the config speed
            if targets:
                queue.append((now + t, targets))
        if queue:
            self._queue, self._idx = queue, 0

    def tick(self, now: float, dt: float) -> None:
        while self._idx < len(self._queue) and now >= self._queue[self._idx][0]:
            for name, deg in self._queue[self._idx][1].items():
                spec = self.joints[name]
                if spec.get("enabled", True):
                    clamped = _clamp(deg, spec["min_angle"], spec["max_angle"])
                    self._target[name] = clamped
                    if spec.get("max_speed", 0) <= 0:
                        self._current[name] = clamped
            self._idx += 1
        for name, spec in self.joints.items():
            speed = self._speed_override.get(name, spec.get("max_speed", 0))
            if not spec.get("enabled", True) or speed <= 0:
                continue
            cur, tgt, step = self._current[name], self._target[name], speed * dt
            self._current[name] = tgt if abs(tgt - cur) <= step else cur + (step if tgt > cur else -step)

    def state(self) -> dict[str, float]:
        """Commanded angle of every joint in radians (what the firmware reports)."""
        return {n: a * math.pi / 180.0 for n, a in self._current.items()}
