"""Simulated smartphone: smabo-app's part of the robot.

smabo-sim's ``SimRobot`` replaces smabo-esp32; ``SimPhone`` replaces smabo-app,
the phone in the robot's front holder. It connects to the brain as its own
client, exactly like the app does:

  * subscribes ``/expression`` (std_msgs/Int32), ``/look_at``
    (geometry_msgs/PoseStamped) and ``/speech/say`` (std_msgs/String), and
    animates the face (``face.Face``) on the phone screen in the MuJoCo world;
  * publishes ``/app/imu/data`` (sensor_msgs/Imu, frame ``imu_link``) at
    ``phone.imu_rate`` with the app's own maths: raw device-axis accelerometer
    and gyroscope plus its complementary-filter orientation
    (``SensorService._publishImu``);
  * publishes ``/app/gps/fix`` (sensor_msgs/NavSatFix, frame ``gps``) at
    ``phone.gps.rate``: the robot's world position turned into latitude /
    longitude around ``phone.gps.origin`` (world x = east, y = north), with
    noise of the reported ``accuracy`` (``RosMessages.navSatFix``);
  * reads ``/speech/say`` aloud with the host's synthesizer (``tts.Speaker``)
    and sends utterances on ``/app/speech/audio`` like the app's voice capture
    (``voice``): typed text synthesized to speech, or the host microphone;
  * streams what the phone's front camera sees (MuJoCo engine) to the brain
    over WebRTC, like the app (``webrtc.CameraStreamer``): the brain's
    webrtc_camera node turns it into ``/camera/image_raw`` for vision. The same
    camera is also viewable on smabo-sim's HTTP page.

The phone's axes follow the device convention: z out of the screen (forward),
y toward the end with the front camera (``phone.camera_side``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Callable

import numpy as np

from ..protocol import APP_ADVERTISE, APP_PREFIX, APP_SUBSCRIBE, BRAIN_PATH
from ..world import Pose2D
from . import voice
from .face import Face
from .tts import Speaker, synth_wav
from .webrtc import CameraStreamer

log = logging.getLogger(__name__)

GRAVITY = 9.81
EARTH_RADIUS = 6378137.0     # m, WGS-84 equatorial (local flat-earth approximation)


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w), same as the app's ``quaternionFromEuler``."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def device_axes(camera_side: str) -> np.ndarray:
    """Rows = device x, y, z expressed in base_link (see ``mujoco_model.phone_axes``)."""
    y = np.array([0.0, 1.0, 0.0]) if camera_side != "robot_right" else np.array([0.0, -1.0, 0.0])
    z = np.array([1.0, 0.0, 0.0])
    return np.stack([np.cross(y, z), y, z])


class SimPhone:
    """Owns the face and the phone sensors; talks to the brain through ``send``."""

    def __init__(self, cfg, engine, send: Callable[[str], None],
                 wall: Callable[[], float] = time.time) -> None:
        self.cfg = cfg
        self.engine = engine
        self.send = send
        self.wall = wall
        self.face = Face(int(cfg.get("phone.screen.width", 512)),
                         int(cfg.get("phone.screen.height", 256)), seed=cfg.get("sim.seed"))
        self._axes = device_axes(cfg.get("phone.camera_side", "robot_left"))
        self._rpy = [0.0, 0.0, 0.0]
        self._prev_pose: Pose2D | None = None
        self._prev_v = 0.0
        self.last_screen: np.ndarray | None = None
        self._rng = np.random.default_rng(cfg.get("sim.seed"))
        self.speaker = Speaker(cfg)
        self.utterances_sent = 0
        self.camera: CameraStreamer | None = None
        if cfg.get("phone.camera.stream", True) and getattr(engine, "has_phone", False):
            cam = CameraStreamer(engine.render_phone_camera, self.publish,
                                 fps=float(cfg.get("phone.camera.fps", 10.0)))
            if cam.available:
                self.camera = cam
            else:
                log.warning("aiortc is not installed: the phone camera is not streamed to the brain "
                            "(pip install 'smabo-sim[webrtc]'); it is still on the /phone page")

    # ------------------------------------------------------------------ #
    # connection
    # ------------------------------------------------------------------ #
    def ws_path(self) -> str:
        return BRAIN_PATH

    def on_brain_connect(self) -> None:
        for topic, typ in APP_ADVERTISE:
            self._frame({"op": "advertise", "topic": topic, "type": typ})
        for topic, typ in APP_SUBSCRIBE:
            self._frame({"op": "subscribe", "topic": topic, "type": typ})
        if self.camera is not None:
            # like the app after (re)connecting: offer the camera to the brain
            asyncio.ensure_future(self.camera.start())

    def publish(self, topic: str, msg: dict) -> None:
        """Publish as the app (``/app`` source prefix)."""
        self._frame({"op": "publish", "topic": APP_PREFIX + topic, "msg": msg})

    def _frame(self, obj: dict) -> None:
        self.send(json.dumps(obj))

    def on_message(self, text: str) -> None:
        try:
            data = json.loads(text)
        except ValueError:
            return
        if data.get("op") == "publish":
            self.handle(data.get("topic"), data.get("msg") or {})

    def handle(self, topic: str | None, msg: dict) -> None:
        if topic == "/expression":
            try:
                self.face.set_expression(int(msg.get("data", 1)))
            except (TypeError, ValueError):
                pass
        elif topic == "/look_at":
            pose = msg.get("pose", msg) or {}
            p = pose.get("position") or {}
            self.face.look_at(float(p.get("x", 1.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0)))
        elif topic == "/speech/say":
            text = str(msg.get("data", ""))
            self.face.say(text)
            self.speaker.speak(text)            # app: if (settings.ttsEnabled) tts.speak(text)
        elif topic == "/webrtc/answer" and self.camera is not None:
            asyncio.ensure_future(self.camera.handle_answer(msg))

    # ------------------------------------------------------------------ #
    # periodic work
    # ------------------------------------------------------------------ #
    def update_screen(self, dt: float) -> None:
        """Advance the face animation and push the frame to the phone screen."""
        self.face.tick(dt)
        img = self.face.render()
        self.last_screen = img
        if getattr(self.engine, "has_phone", False):
            self.engine.set_screen_image(img)

    def read_imu(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Accelerometer / gyroscope in device axes (MuJoCo sensors, else 2D motion)."""
        if getattr(self.engine, "has_phone", False):
            return self.engine.phone_imu()
        p = self.engine.pose()
        if self._prev_pose is None or dt <= 0:
            self._prev_pose = p
            acc_r, gyro_r = np.array([0.0, 0.0, GRAVITY]), np.zeros(3)
        else:
            q = self._prev_pose
            dyaw = math.atan2(math.sin(p.yaw - q.yaw), math.cos(p.yaw - q.yaw))
            wz = dyaw / dt
            dx, dy = p.x - q.x, p.y - q.y
            v = (dx * math.cos(p.yaw) + dy * math.sin(p.yaw)) / dt
            ax, ay = (v - self._prev_v) / dt, v * wz            # tangential, centripetal
            self._prev_pose, self._prev_v = p, v
            acc_r, gyro_r = np.array([ax, ay, GRAVITY]), np.array([0.0, 0.0, wz])
        return self._axes @ acc_r, self._axes @ gyro_r

    def publish_imu(self, dt: float) -> None:
        if dt <= 0 or dt > 1:
            return
        acc, gyro = self.read_imu(dt)
        r = self._rpy
        r[0] += gyro[0] * dt
        r[1] += gyro[1] * dt
        r[2] += gyro[2] * dt
        ax, ay, az = acc
        alpha = 0.98
        r[0] = alpha * r[0] + (1 - alpha) * math.atan2(ay, az)
        r[1] = alpha * r[1] + (1 - alpha) * math.atan2(-ax, math.hypot(ay, az))
        qx, qy, qz, qw = quaternion_from_euler(*r)
        t = self.wall()
        sec = int(t)
        zeros = [0.0] * 9
        self.publish("/imu/data", {
            "header": {"stamp": {"sec": sec, "nanosec": int((t - sec) * 1e9)}, "frame_id": "imu_link"},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
            "orientation_covariance": zeros,
            "angular_velocity": {"x": float(gyro[0]), "y": float(gyro[1]), "z": float(gyro[2])},
            "angular_velocity_covariance": zeros,
            "linear_acceleration": {"x": float(ax), "y": float(ay), "z": float(az)},
            "linear_acceleration_covariance": zeros,
        })

    def gps_fix(self, noise: bool = True) -> tuple[float, float, float]:
        """(latitude, longitude, altitude) of the robot, like the phone's GPS."""
        p = self.engine.pose()
        rot = math.radians(float(self.cfg.get("phone.gps.map_rotation_deg", 0.0)))
        east = p.x * math.cos(rot) - p.y * math.sin(rot)
        north = p.x * math.sin(rot) + p.y * math.cos(rot)
        if noise:
            sigma = float(self.cfg.get("phone.gps.accuracy", 3.0)) / math.sqrt(2.0)
            east += self._rng.normal(0.0, sigma)
            north += self._rng.normal(0.0, sigma)
        lat0 = float(self.cfg.get("phone.gps.origin.latitude", 35.681236))
        lon0 = float(self.cfg.get("phone.gps.origin.longitude", 139.767125))
        alt0 = float(self.cfg.get("phone.gps.origin.altitude", 0.0))
        lat = lat0 + math.degrees(north / EARTH_RADIUS)
        lon = lon0 + math.degrees(east / (EARTH_RADIUS * math.cos(math.radians(lat0))))
        return lat, lon, alt0

    def publish_gps(self, _dt: float = 0.0) -> None:
        if not self.cfg.get("phone.gps.enabled", True):
            return
        lat, lon, alt = self.gps_fix()
        acc = float(self.cfg.get("phone.gps.accuracy", 3.0))
        cov = acc * acc
        t = self.wall()
        sec = int(t)
        self.publish("/gps/fix", {
            "header": {"stamp": {"sec": sec, "nanosec": int((t - sec) * 1e9)}, "frame_id": "gps"},
            "status": {"status": 0, "service": 1},          # STATUS_FIX, SERVICE_GPS
            "latitude": lat, "longitude": lon, "altitude": alt,
            "position_covariance": [cov, 0.0, 0.0, 0.0, cov, 0.0, 0.0, 0.0, cov],
            "position_covariance_type": 1,                  # APPROXIMATED
        })

    # ------------------------------------------------------------------ #
    # voice input (the app's wake → record → /speech/audio)
    # ------------------------------------------------------------------ #
    def send_utterance(self, wav: bytes) -> None:
        """Send one recorded utterance (WAV bytes) to the brain for STT."""
        if not self.cfg.get("phone.voice.enabled", True):
            raise RuntimeError("voice input is disabled (phone.voice.enabled)")
        wav = voice.resample_wav(wav, voice.SAMPLE_RATE)      # brain expects 16 kHz mono
        self.publish("/speech/audio", voice.audio_msg(wav))
        self.utterances_sent += 1
        log.info("voice: sent %.1f s of audio to the brain",
                 (len(wav) - 44) / 2 / voice.SAMPLE_RATE)

    async def talk(self, text: str) -> None:
        """Say ``text`` to the robot: synthesize it and send it as an utterance."""
        wav = await asyncio.get_running_loop().run_in_executor(None, synth_wav, text, self.cfg)
        if wav is None:
            raise RuntimeError("no speech synthesizer (macOS 'say' or espeak-ng) to turn text into audio")
        self.send_utterance(wav)

    async def listen(self) -> bool:
        """Record one utterance from the host microphone and send it.

        Returns False when nothing was said. Needs ``sounddevice``.
        """
        if not voice.have_microphone():
            raise RuntimeError("microphone input needs sounddevice (pip install 'smabo-sim[voice]')")
        wav = await voice.record_utterance(
            max_seconds=float(self.cfg.get("phone.voice.max_seconds", 10.0)),
            silence_ms=int(self.cfg.get("phone.voice.silence_ms", voice.SILENCE_MS)),
            silence_rms=float(self.cfg.get("phone.voice.silence_rms", voice.SILENCE_RMS)))
        if wav is None:
            return False
        self.send_utterance(wav)
        return True
