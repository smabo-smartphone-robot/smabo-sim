"""Ports of the smabo-esp32 firmware logic that shapes what the robot does.

These mirror the real board's behaviour (including its quirks) so the brain
sees the same thing it would see on hardware:

  * ``DriveFirmware``  ← dc_motors.DiffDrive: Twist → per-wheel duty by
    differential kinematics and the cmd_vel dead-man timeout. Open loop
    (``dc_drive``) applies the ``dc.min_duty`` static-friction floor; in
    ``encoder_drive`` a PI loop on the measured wheel speeds sets the duty.
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


_COAST = 0.001      # |duty| at or below this coasts the motor (dc_motors._COAST)
_HOLD_DUTY = 0.01   # smallest speed-control output (dc_motors._HOLD_DUTY)
_STOP_SPEED = 1e-4  # m/s; a wheel target below this means "stop" (dc_motors._STOP_SPEED)


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
        # encoder_drive speed control (dc_motors.DiffDrive.update_speed)
        self.closed_loop = False
        self._target = (0.0, 0.0)
        self._integ = [0.0, 0.0]
        self._learned = [0.0, 0.0]

    def set_closed_loop(self, on: bool) -> None:
        """``DiffDrive.set_closed_loop``: speed control on while encoders run."""
        if bool(on) != self.closed_loop:
            self.closed_loop = bool(on)
            self._target, self._integ = (0.0, 0.0), [0.0, 0.0]

    def _speed_control(self) -> bool:
        return self.closed_loop and self.cfg.get("dc.speed_control", True)

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
        if self._speed_control():
            self._set_target(v_l, v_r)
            return
        self._duty = (_with_min_duty(_clamp(v_l / norm, -1.0, 1.0), min_duty),
                      _with_min_duty(_clamp(v_r / norm, -1.0, 1.0), min_duty))

    def _set_target(self, v_l: float, v_r: float) -> None:
        """``DiffDrive._set_target``: new wheel speed targets (m/s)."""
        old, self._target = self._target, (v_l, v_r)
        ff = 1.0 / self.full_duty_speed()
        duty = list(self._duty)
        for i, v in enumerate((v_l, v_r)):
            if abs(v) < _STOP_SPEED:
                self._remember(i)
                duty[i] = 0.0
            elif old[i] * v <= 0:            # from rest, or reversing
                self._remember(i)
                self._integ[i] = self._learned[i] if v > 0 else -self._learned[i]
                duty[i] = _clamp(v * ff + self._integ[i], -1.0, 1.0)
        self._duty = (duty[0], duty[1])

    def _remember(self, i: int) -> None:
        if self._integ[i] != 0.0:
            self._learned[i] = abs(self._integ[i])
        self._integ[i] = 0.0

    def update_speed(self, meas_l: float, meas_r: float, dt: float) -> None:
        """``DiffDrive.update_speed``: PI step from the measured wheel speeds (m/s)."""
        if not self._speed_control() or dt <= 0:
            return
        kp = self.cfg.get("dc.speed_kp", 2.0)
        ki = self.cfg.get("dc.speed_ki", 15.0)
        i_max = self.cfg.get("dc.speed_i_max", 0.6)
        ff = 1.0 / self.full_duty_speed()
        duty = list(self._duty)
        for i, (v, m) in enumerate(zip(self._target, (meas_l, meas_r))):
            if abs(v) < _STOP_SPEED:
                continue
            err = v - m
            lo, hi = (-v * ff, i_max) if v > 0 else (-i_max, -v * ff)
            self._integ[i] = _clamp(self._integ[i] + ki * err * dt, lo, hi)
            d = v * ff + kp * err + self._integ[i]
            sign = 1.0 if v > 0 else -1.0
            if d * sign < _HOLD_DUTY:           # never against the target, never 0
                d = _HOLD_DUTY * sign
            duty[i] = _clamp(d, -1.0, 1.0)
        self._duty = (duty[0], duty[1])

    def stop(self) -> None:
        self._duty = (0.0, 0.0)
        self._target = (0.0, 0.0)
        self._remember(0)
        self._remember(1)

    def duty(self, now: float) -> tuple[float, float]:
        """Current (left, right) duty in [-1, 1], zero after ``cmd_timeout``."""
        if now - self._last_cmd > self.cfg.get("dc.cmd_timeout", 0.5):
            self.stop()
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
