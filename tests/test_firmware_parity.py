"""The simulator's firmware ports against the real smabo-esp32 code.

``smabo_sim.firmware`` and ``smabo_sim.config.DEFAULTS`` are copies of the
board's ``dc_motors.py`` and ``config.py`` (MicroPython modules that CPython
cannot import as they are). These tests load the board's files from a
smabo-esp32 checkout next to this repository, with a fake ``machine`` module,
and check that the copies still behave the same. They are skipped when that
checkout is not there.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from smabo_sim.config import DEFAULTS, Config
from smabo_sim.firmware import DriveFirmware

ESP32 = Path(__file__).resolve().parents[2] / "smabo-esp32"

pytestmark = pytest.mark.skipif(not (ESP32 / "dc_motors.py").exists(),
                                reason="smabo-esp32 is not checked out next to smabo-sim")


class _Pin:
    OUT = 1

    def __init__(self, num, mode=None, pull=None):
        self._value = 0

    def value(self, v=None):
        if v is None:
            return self._value
        self._value = v


class _PWM:
    def __init__(self, pin, freq=None, duty=0):
        self._duty = duty

    def duty(self, d=None):
        if d is None:
            return self._duty
        self._duty = d


def _load(name: str):
    """Import one of the board's modules under a private name."""
    fake = types.ModuleType("machine")
    fake.Pin, fake.PWM = _Pin, _PWM
    saved = sys.modules.get("machine")
    sys.modules["machine"] = fake
    try:
        spec = importlib.util.spec_from_file_location(f"esp32_{name}", ESP32 / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is None:
            del sys.modules["machine"]
        else:
            sys.modules["machine"] = saved


def _signed_duty(motor) -> float:
    """What a board motor is driving, in [-1, 1] (+ forward, 0 coast or brake)."""
    sign = motor.in1.value() - motor.in2.value()
    return sign * motor.pwm.duty() / 1023.0


DC_KEYS = ("max_linear", "max_angular", "wheel_radius", "wheel_separation", "min_duty",
           "start_duty", "kick_ms", "brake", "speed_control", "speed_kp", "speed_ki",
           "speed_i_max", "cmd_timeout")


def test_dc_defaults_match_the_board():
    board = _load("config").DEFAULTS
    for key in DC_KEYS:
        assert DEFAULTS["dc"][key] == board["dc"][key], key
    for key in ("cpr", "publish_rate"):
        assert DEFAULTS["encoder"][key] == board["encoder"][key], key


# (linear, angular) per step, and the measured wheel speeds (m/s) the encoders report
SCENARIOS = {
    "forward, wheels lagging": ([(0.1, 0.0)] * 20, (0.05, 0.05)),
    "spin in place": ([(0.0, 0.6)] * 20, (-0.02, 0.02)),
    "reverse every step": ([(0.1, 0.0), (-0.1, 0.0)] * 10, (0.0, 0.0)),
    "stop and start again": ([(0.1, 0.0)] * 5 + [(0.0, 0.0)] * 3 + [(0.1, 0.0)] * 5, (0.08, 0.12)),
    "wheels too fast": ([(0.05, 0.0)] * 20, (0.2, 0.2)),
}


@pytest.mark.parametrize("cmds,meas", SCENARIOS.values(), ids=SCENARIOS.keys())
@pytest.mark.parametrize("closed_loop", [True, False], ids=["encoder_drive", "dc_drive"])
def test_drive_matches_the_board(cmds, meas, closed_loop):
    cfg = Config(None)
    cfg.update({"dc": {"kick_ms": 0}})       # the simulator does not model the kick-start
    board = _load("dc_motors").DiffDrive(cfg)
    sim = DriveFirmware(cfg)
    board.set_closed_loop(closed_loop)
    sim.set_closed_loop(closed_loop)
    dt = 0.05
    for k, (lin, ang) in enumerate(cmds):
        now = k * dt
        board.set_cmd_vel(lin, ang)
        sim.set_cmd_vel(lin, ang, now)
        board.update_speed(*meas, dt)
        sim.update_speed(*meas, dt)
        expected = (_signed_duty(board.left), _signed_duty(board.right))
        assert sim.duty(now) == pytest.approx(expected, abs=1.5 / 1023), f"step {k}"
