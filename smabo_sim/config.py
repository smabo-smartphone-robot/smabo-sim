"""Configuration store with the same schema as smabo-esp32's config.json.

``DEFAULTS`` mirrors ``smabo-esp32/config.py`` key for key, so smabo-web's
config editor (which talks to ``GET/POST /config``) sees exactly what it would
see on the real board. Hardware-only keys (pins, I2C, Wi-Fi) are kept for
parity but have no effect in the simulator.

Differences from the ESP32 defaults, chosen so the simulator works out of the
box against a local smabo-brain-ros:

  * ``brain``  → ``localhost:9090``, path ``/``.
  * ``modes``  → servos, encoder_drive and lidar on.
  * ``http``   → port 8080 (port 80 needs root on Linux/macOS).
  * ``sim``    → simulator-only settings (engine, world, noise, rates).

An ESP32 ``config.json`` can be loaded on top (``--config``) to simulate a
specific robot with its real geometry and servo limits.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any


def _servo(channel, lo, hi, init=0, max_speed=120, enabled=True, invert=False):
    """Same helper as smabo-esp32/config.py (500-2500 µs pulse range)."""
    return {
        "channel": channel,
        "min_angle": lo,
        "max_angle": hi,
        "min_us": 500,
        "max_us": 2500,
        "init_angle": init,
        "max_speed": max_speed,  # deg/s, 0 = jump instantly
        "enabled": enabled,
        "invert": invert,
    }


DEFAULTS: dict[str, Any] = {
    "wifi": {"ssid": "your-ssid", "password": "your-password", "hostname": "smabo-sim"},
    "brain": {
        "host": "localhost",
        "port": 9090,
        "path": "",          # "" = "/" (smabo-brain and smabo-brain-ros alike)
    },
    "http": {"port": 8080},
    "i2c": {"sda": 21, "scl": 22, "freq": 400000},
    "pca9685": {"address": 0x40, "freq": 50},
    "modes": {
        "servos": True,
        "dc_drive": False,
        "encoder_drive": True,
        "lidar": True,
    },
    "servos": {
        "behavior": "manual",
        "joints": {
            "left_hand":   _servo(0, -90, 90, 0,   0),
            "right_hand":  _servo(1, -90, 90, 0,   0),
            "head_pan":    _servo(2, -90, 90, 0, 120),
            "arm_joint_1": _servo(3, -90, 90, 0,  90),
            "arm_joint_2": _servo(4, -90, 90, 0,  90),
            "arm_joint_3": _servo(5, -90, 90, 0,  90),
            "arm_joint_4": _servo(6, -90, 90, 0,  90),
        },
        "random_groups": [],
        "joint_states_rate": 20.0,
    },
    "dc": {
        "pins": {"pwma": 32, "ain2": 33, "ain1": 25, "stby": 26,
                 "bin1": 27, "bin2": 14, "pwmb": 12},
        "pwm_freq": 1000,
        "max_linear": 0.30,
        "max_angular": 1.50,
        "wheel_radius": 0.030,
        "wheel_separation": 0.130,
        "min_duty": 0.0,
        "invert_left": False,
        "invert_right": False,
        "cmd_timeout": 0.5,
    },
    "encoder": {
        "left": {"a": 34, "b": 35},
        "right": {"a": 36, "b": 39},
        "cpr": 1440,
        "publish_rate": 20.0,
        "odom_frame": "odom",
        "base_frame": "base_link",
        "covariance": {"pose_xx": 0.001, "pose_yy": 0.001, "pose_aa": 0.001,
                       "twist_vv": 0.001, "twist_ww": 0.001},
    },
    "lidar": {
        "uart": 1, "rx": 20, "tx": -1, "baud": 230400,
        "frame_id": "laser",
        "bins": 360,
        "range_min": 0.05,
        "range_max": 12.0,
    },
    # ---- the smartphone (smabo-app's part of the robot) -----------------
    "phone": {
        "enabled": True,                # face, IMU and camera of the phone in the holder
        "camera_side": "robot_left",    # end of the phone with the front camera (= device top)
        "screen": {"width": 512, "height": 256, "fps": 30.0},
        "imu_rate": 20.0,               # Hz, smabo-app SensorService.startImu default
        # front camera; streamed to the brain over WebRTC like smabo-app (needs aiortc)
        "camera": {"width": 640, "height": 480, "hfov_deg": 60.0, "fps": 10.0, "stream": True},
        # GPS (/gps/fix, sensor_msgs/NavSatFix) like smabo-app's geolocator stream.
        # World x = east, y = north (ROS ENU); origin is where world (0, 0) lies.
        "gps": {"enabled": True, "rate": 1.0,
                "origin": {"latitude": 35.681236, "longitude": 139.767125, "altitude": 40.0},
                "map_rotation_deg": 0.0,        # world frame rotated CCW from ENU
                "accuracy": 3.0},               # m: reported accuracy and noise size
        # /speech/say read aloud on the host (macOS say / espeak-ng), like flutter_tts
        "tts": {"enabled": True, "language": "ja-JP", "voice": "", "rate": 0},
        # utterances to /speech/audio (typed text → speech, or the host microphone)
        "voice": {"enabled": True, "max_seconds": 10.0, "silence_ms": 1500, "silence_rms": 600.0},
    },
    # ---- simulator only -------------------------------------------------
    "sim": {
        "engine": "kinematic",         # kinematic | mujoco
        "world": "",                    # path to a world YAML ("" = bundled room)
        "physics_rate": 100.0,          # Hz
        "motor_time_constant": 0.08,    # s, first-order lag of each wheel's speed
        "robot_radius": 0.09,           # m, footprint used for collisions
        "laser_offset": [0.0, 0.0],     # m, laser frame in base_link (URDF laser_joint)
        "lidar_rate": 10.0,             # Hz, LD06 spins at ~10 Hz
        "lidar_noise": 0.005,           # m, std of range noise
        "encoder_noise": 0.01,          # relative std of measured wheel speed
        "seed": None,                   # RNG seed (None = random)
        # mujoco engine
        "description": "",              # smabo_description folder for meshes ("" = auto-detect, "none" = simple shapes)
        "timestep": 0.002,              # s, physics step
        "servo_kp": 1.5,                # N·m/rad, servo position gain (torque capped at SG90 stall)
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    """In-memory config with dot-path access, optionally persisted to a file.

    Parameters
    ----------
    path : str, optional
        JSON file (ESP32 ``config.json`` format) merged over ``DEFAULTS``.
        When ``persist`` is true, changes made over REST are written back.
    persist : bool
        Write changes back to ``path`` (like the ESP32 flash mirror).
    """

    def __init__(self, path: str | None = None, persist: bool = False,
                 overrides: dict | None = None) -> None:
        self.path = path
        self.persist = bool(path) and persist
        data = DEFAULTS
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = deep_merge(data, json.load(f))
        self.data: dict[str, Any] = deep_merge(data, overrides or {})

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read ``"a.b.c"``; returns ``default`` when any part is missing."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def update(self, patch: dict) -> None:
        """Deep-merge a patch (same semantics as the ESP32 ``Config.update``)."""
        self.data = deep_merge(self.data, patch)
        self.save()

    def save(self) -> None:
        if not self.persist:
            return
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)
