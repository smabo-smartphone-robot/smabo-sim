"""Simulated robot: the ESP32 ``Robot`` orchestrator on top of an engine.

Mirrors ``smabo-esp32/robot.py``:

  * the same advertise / subscribe sets and ``/esp32/config`` push on every
    brain connect (smabo-brain and smabo-brain-ros alike),
  * the same inbound topics (``/cmd_vel``, ``/servo/command``, ``/ping``),
  * the same outbound topics with the ``/esp32`` source prefix,
  * the same mode gating (servos / dc_drive / encoder_drive / lidar),
  * the same ``servos.behavior = "random"`` motion (``random_motion``),
  * the same REST semantics (``config_json`` / ``set_config`` / ``set_mode``).

One deliberate difference: header stamps use the host's wall clock, which on
a simulator host is also ROS time. (The ESP32 stamps with ``ticks_ms``, i.e.
milliseconds since boot.)
"""

from __future__ import annotations

import json
import math
import random
import time
from typing import Callable

import numpy as np

from .config import Config
from .engines import Engine
from .firmware import DriveFirmware, ServoFirmware
from .protocol import BRAIN_PATH, ESP32_ADVERTISE, ESP32_PREFIX, ESP32_SUBSCRIBE
from .random_motion import RandomMotion



def ros_stamp(t: float) -> dict:
    sec = int(t)
    return {"sec": sec, "nanosec": int((t - sec) * 1e9)}


class SimRobot:
    """Owns firmware + engine; produces outbound frames through ``send``.

    Parameters
    ----------
    cfg : Config
    engine : Engine
    send : callable(str)
        Sends one text frame to the brain (dropped while disconnected).
    clock, wall : callable() -> float
        Monotonic clock for control, wall clock for header stamps. Injected
        for tests.
    """

    def __init__(self, cfg: Config, engine: Engine, send: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time) -> None:
        self.cfg = cfg
        self.engine = engine
        self.send = send
        self.clock = clock
        self.wall = wall
        self.drive = DriveFirmware(cfg)
        self.servos = ServoFirmware(cfg.get("servos.joints") or {})
        self.random_motion = RandomMotion(cfg, random.Random(cfg.get("sim.seed")))
        self._last_wheel_pub = clock()
        self.on_brain_settings_changed: Callable[[], None] | None = None

    # ------------------------------------------------------------------ #
    # outbound helpers
    # ------------------------------------------------------------------ #
    def _frame(self, obj: dict) -> None:
        self.send(json.dumps(obj))

    def publish(self, topic: str, msg: dict) -> None:
        self._frame({"op": "publish", "topic": ESP32_PREFIX + topic, "msg": msg})

    def notice(self, message: str) -> None:
        self._frame({"op": "notice", "message": message})

    def mode(self, name: str) -> bool:
        return bool(self.cfg.get(f"modes.{name}", False))

    def drive_enabled(self) -> bool:
        return self.mode("dc_drive") or self.mode("encoder_drive")

    # ------------------------------------------------------------------ #
    # connection lifecycle
    # ------------------------------------------------------------------ #
    def ws_path(self) -> str:
        return self.cfg.get("brain.path") or BRAIN_PATH

    def on_brain_connect(self) -> None:
        for topic, typ in ESP32_ADVERTISE:
            self._frame({"op": "advertise", "topic": topic, "type": typ})
        for topic, typ in ESP32_SUBSCRIBE:
            self._frame({"op": "subscribe", "topic": topic, "type": typ})
        self.sync_config_to_brain()

    def sync_config_to_brain(self) -> None:
        """Publish the config (minus wifi) on /esp32/config for smabo-brain's odometry."""
        snapshot = {k: v for k, v in self.cfg.data.items() if k != "wifi"}
        self.publish("/config", {"data": json.dumps(snapshot)})

    # ------------------------------------------------------------------ #
    # inbound
    # ------------------------------------------------------------------ #
    def on_message(self, text: str) -> None:
        try:
            data = json.loads(text)
        except ValueError:
            return
        if data.get("op") == "publish":
            self._on_publish(data.get("topic"), data.get("msg") or {})

    def _on_publish(self, topic: str | None, msg: dict) -> None:
        now = self.clock()
        if topic == "/cmd_vel":
            if self.drive_enabled():
                lin = (msg.get("linear") or {}).get("x", 0.0)
                ang = (msg.get("angular") or {}).get("z", 0.0)
                self.drive.set_cmd_vel(float(lin), float(ang), now)
        elif topic == "/servo/command":
            if self.mode("servos"):
                self.servos.load_trajectory(msg.get("joint_names") or [], msg.get("points") or [], now)
        elif topic == "/ping":
            self.publish("/pong", {"data": (msg or {}).get("data", "")})

    # ------------------------------------------------------------------ #
    # periodic work (called by the runner)
    # ------------------------------------------------------------------ #
    def physics_step(self, dt: float) -> None:
        now = self.clock()
        if self.mode("servos"):
            self.random_motion.tick(now, self.servos)    # servos.behavior = "random"
            self.servos.tick(now, dt)
        self.engine.set_joint_targets(self.servos.state())
        if self.drive_enabled():
            self.engine.set_wheel_targets(*self.drive.wheel_targets(now))
        else:
            self.drive.stop()
            self.engine.set_wheel_targets(0.0, 0.0)
        self.engine.step(dt)

    def publish_wheel_vel(self) -> None:
        now = self.clock()
        dt = now - self._last_wheel_pub
        self._last_wheel_pub = now
        d_l, d_r = self.engine.read_wheel_distance()
        if not self.mode("encoder_drive") or dt <= 0:
            return
        self.publish("/wheel_vel", {"left": d_l / dt, "right": d_r / dt, "dt": dt})

    def publish_joint_states(self) -> None:
        if not self.mode("servos"):
            return
        state = self.servos.state()
        self.publish("/joint_states", {
            "header": {"stamp": ros_stamp(self.wall()), "frame_id": ""},
            "name": list(state.keys()),
            "position": list(state.values()),
            "velocity": [0.0] * len(state),
            "effort": [0.0] * len(state),
        })

    def publish_scan(self) -> None:
        if not self.mode("lidar"):
            return
        n = int(self.cfg.get("lidar.bins", 360)) or 360
        inc = 2.0 * math.pi / n
        rmin = self.cfg.get("lidar.range_min", 0.05)
        rmax = self.cfg.get("lidar.range_max", 12.0)
        ranges = self.engine.scan(np.arange(n) * inc, rmin, rmax)
        scan_time = 1.0 / max(self.cfg.get("sim.lidar_rate", 10.0), 1e-3)
        self.publish("/scan", {
            "header": {"stamp": ros_stamp(self.wall()), "frame_id": self.cfg.get("lidar.frame_id", "laser")},
            "angle_min": 0.0,
            "angle_max": inc * (n - 1),
            "angle_increment": inc,
            "time_increment": scan_time / n,
            "scan_time": scan_time,
            "range_min": rmin,
            "range_max": rmax,
            "ranges": [round(float(r), 4) for r in ranges],
            "intensities": [],
        })

    # ------------------------------------------------------------------ #
    # REST API (same contract as smabo-esp32/http_server.py → Robot)
    # ------------------------------------------------------------------ #
    def config_json(self) -> dict:
        return self.cfg.data

    def set_config(self, patch: dict) -> None:
        patch = patch or {}
        if "servos" in patch:
            # keep servos where they are when their spec changes (firmware persists last_angle)
            current = {n: {"last_angle": a * 180.0 / math.pi} for n, a in self.servos.state().items()}
            self.cfg.update({"servos": {"joints": current}})
        self.cfg.update(patch)
        if "servos" in patch:
            self.servos = ServoFirmware(self.cfg.get("servos.joints") or {})
        hw = [k for k in ("i2c", "pca9685", "wifi") if k in patch]
        if hw:
            self.notice("smabo-sim: ignoring hardware settings (no effect in simulation): " + ", ".join(hw))
        if "brain" in patch and self.on_brain_settings_changed:
            self.on_brain_settings_changed()
        self.sync_config_to_brain()

    def set_mode(self, partial: dict) -> None:
        modes = dict(self.cfg.get("modes") or {})
        modes.update(partial or {})
        if modes.get("encoder_drive") and modes.get("dc_drive"):
            modes["dc_drive"] = False      # firmware: encoder_drive wins
        self.cfg.update({"modes": modes})
        if not self.drive_enabled():
            self.drive.stop()
        self.sync_config_to_brain()
