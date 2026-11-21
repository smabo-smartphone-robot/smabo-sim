import math

import pytest

from smabo_sim.config import Config
from smabo_sim.firmware import DriveFirmware, ServoFirmware


def drive():
    return DriveFirmware(Config())


def test_straight_command_maps_to_equal_wheel_speeds():
    d = drive()
    d.set_cmd_vel(0.2, 0.0, now=0.0)
    left, right = d.wheel_targets(now=0.1)
    assert left == pytest.approx(0.2) and right == pytest.approx(0.2)


def test_curve_uses_differential_kinematics():
    d = drive()
    d.set_cmd_vel(0.1, 1.0, now=0.0)
    left, right = d.wheel_targets(now=0.0)
    assert left == pytest.approx(0.1 - 0.065) and right == pytest.approx(0.1 + 0.065)


def test_pure_spin_turns_at_the_commanded_rate():
    d = drive()
    d.set_cmd_vel(0.0, 0.5, now=0.0)
    left, right = d.wheel_targets(now=0.0)
    assert (right - left) / 0.13 == pytest.approx(0.5)


def test_min_duty_lifts_small_commands_but_keeps_zero():
    d = DriveFirmware(Config(overrides={"dc": {"min_duty": 0.25}}))
    d.set_cmd_vel(0.0, 0.5, now=0.0)
    assert d.duty(now=0.0) == pytest.approx((-0.25, 0.25))
    d.set_cmd_vel(0.0, 1e-6, now=0.0)
    assert d.wheel_targets(now=0.0) == (0.0, 0.0)    # not lifted; the motor coasts
    d.set_cmd_vel(0.3, 0.0, now=0.0)
    assert d.duty(now=0.0)[0] > 0.25


def test_cmd_timeout_stops_the_wheels():
    d = drive()
    d.set_cmd_vel(0.2, 0.0, now=0.0)
    assert d.duty(now=0.4) != (0.0, 0.0)
    assert d.duty(now=0.6) == (0.0, 0.0)


def test_duty_is_clamped():
    d = drive()
    d.set_cmd_vel(5.0, 0.0, now=0.0)
    assert d.duty(now=0.0) == (1.0, 1.0)


JOINTS = {
    "a": {"min_angle": -90, "max_angle": 90, "init_angle": 0, "max_speed": 90, "enabled": True},
    "b": {"min_angle": -90, "max_angle": 90, "init_angle": 0, "max_speed": 0, "enabled": True},
}


def test_trajectory_points_apply_at_their_deadline_with_speed_limit():
    s = ServoFirmware(JOINTS)
    s.load_trajectory(["a"], [{"positions": [math.radians(45)], "time_from_start": {"sec": 1, "nanosec": 0}}], now=0.0)
    s.tick(now=0.5, dt=0.5)
    assert s.state()["a"] == pytest.approx(0.0)             # not due yet
    s.tick(now=1.0, dt=0.25)
    assert math.degrees(s.state()["a"]) == pytest.approx(22.5)  # 90 deg/s * 0.25 s
    s.tick(now=1.5, dt=0.5)
    assert math.degrees(s.state()["a"]) == pytest.approx(45.0)


def test_zero_speed_joint_jumps_and_limits_clamp():
    s = ServoFirmware(JOINTS)
    s.load_trajectory(["b"], [{"positions": [math.radians(120)]}], now=0.0)
    s.tick(now=0.0, dt=0.02)
    assert math.degrees(s.state()["b"]) == pytest.approx(90.0)


def test_unknown_joints_are_ignored():
    s = ServoFirmware(JOINTS)
    s.load_trajectory(["zzz"], [{"positions": [1.0]}], now=0.0)
    s.tick(now=0.0, dt=0.02)
    assert s.state() == {"a": 0.0, "b": 0.0}
