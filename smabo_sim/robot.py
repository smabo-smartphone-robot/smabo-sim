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
import socket
import math
import random
import time
import logging
from typing import Callable

import numpy as np

from . import __version__
from .config import Config
from .engines import Engine
from .firmware import DriveFirmware, ServoFirmware
from .models import RobotModel
from .protocol import (BRAIN_PATH, ESP32_ADVERTISE, ESP32_PREFIX, ESP32_SUBSCRIBE,
                       brain_info_warning, hello_msg)
from .random_motion import RandomMotion


log = logging.getLogger(__name__)


def ros_stamp(t: float) -> dict:
    sec = int(t)
    return {"sec": sec, "nanosec": int((t - sec) * 1e9)}



# simulator-only keys: not on the ESP32's REST /config nor in /esp32/config
SIM_ONLY_KEYS = ("phone", "sim", "view")


def local_url(cfg, port: int) -> str:
    """``http://<this machine>:<port>`` as seen from the brain's side of the network.

    With the brain on this machine that is ``localhost``; otherwise the
    address of the interface that routes to the brain.
    """
    brain = str(cfg.get("brain.host") or "localhost")
    host = "localhost"
    if brain not in ("localhost", "127.0.0.1", "::1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((brain, 9))          # no packet is sent
                host = s.getsockname()[0]
        except OSError:
            pass
    return f"http://{host}:{port}"


SIM_ESP32_CLIENT = "smabo-sim (esp32)"
WIFI_PASSWORD_MASK = "****"   # GET /config never returns the Wi-Fi password

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
        self.http_port: int | None = None    # the REST port actually opened (when not http.port)
        self.clock = clock
        self.wall = wall
        self.drive = DriveFirmware(cfg)
        self.servos = ServoFirmware(cfg.get("servos.joints") or {})
        self.random_motion = RandomMotion(cfg, random.Random(cfg.get("sim.seed")))
        self._last_wheel_pub = clock()
        self.on_brain_settings_changed: Callable[[], None] | None = None
        self._protocol_warning: str | None = None

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
        self.publish_config_url()
        self.publish("/hello", hello_msg(SIM_ESP32_CLIENT, __version__))

    def config_url(self) -> str:
        """Our REST base URL as seen from the brain's side of the network.

        With the brain on this machine that is ``localhost``; otherwise the
        address of the interface that routes to the brain.
        """
        return local_url(self.cfg, self.http_port or int(self.cfg.get("http.port", 80)))

    def publish_config_url(self) -> None:
        """Tell smabo-web where the REST config API is (on connect and every 5 s)."""
        self.publish("/config_url", {"data": self.config_url()})

    def sync_config_to_brain(self) -> None:
        """Publish the config (minus wifi) on /esp32/config for smabo-brain's odometry."""
        snapshot = {k: v for k, v in self.config_json().items() if k != "wifi"}
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
        elif topic == "/brain/info":
            warning = brain_info_warning(msg, SIM_ESP32_CLIENT)
            if warning and warning != self._protocol_warning:
                log.warning(warning)
                self.notice(warning)
            self._protocol_warning = warning

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
        """What the ESP32's REST /config holds: no simulator or phone keys
        (the phone's settings are on its own REST /config, phone/settings.py)."""
        data = {k: v for k, v in self.cfg.data.items() if k not in SIM_ONLY_KEYS}
        wifi = data.get("wifi")
        if isinstance(wifi, dict) and wifi.get("password"):
            data["wifi"] = {**wifi, "password": WIFI_PASSWORD_MASK}   # write-only, like the board
        return data

    def set_config(self, patch: dict) -> None:
        patch = dict(patch or {})
        wifi = patch.get("wifi")
        if isinstance(wifi, dict) and wifi.get("password") == WIFI_PASSWORD_MASK:
            wifi = {k: v for k, v in wifi.items() if k != "password"}   # sent back unchanged
            if wifi:
                patch["wifi"] = wifi
            else:
                del patch["wifi"]
        sim_only = [k for k in SIM_ONLY_KEYS if patch.pop(k, None) is not None]
        if sim_only:
            self.notice("smabo-sim: not ESP32 settings, ignored: " + ", ".join(sim_only)
                        + " (phone settings: the phone's REST /config)")
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

    def apply_model(self, model: RobotModel) -> None:
        """Switch to a roadmap ``model``: its modes and servos (the caller swaps
        the engine). The brain gets the new config, smabo-web a notice."""
        self.cfg.update(model.config_patch(self.cfg.get("servos.joints") or {}))
        self.servos = ServoFirmware(self.cfg.get("servos.joints") or {})
        self.drive.stop()
        self._last_wheel_pub = self.clock()
        self.notice(f"smabo-sim: model → {model.name} ({model.label})")
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
