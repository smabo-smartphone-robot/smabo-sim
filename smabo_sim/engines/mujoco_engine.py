"""MuJoCo engine: contact physics for the chassis, the wheels and the arm.

What is simulated (and what the brain therefore sees):

* Wheels are hinge joints driven like brushed DC motors: torque falls
  linearly from the TT motor's stall torque at rest to zero at the full-duty
  speed (``mujoco_model.wheel_kv``). Traction comes from tyre/floor friction, so the
  wheels can slip (e.g. when pushing an obstacle) and the encoders keep
  counting — odometry drifts like it does on the real robot.
* Encoders read the wheel joints' actual rotation (× radius) with
  ``sim.encoder_noise``.
* The ball caster is a frictionless sphere.
* The arm joints are position servos (SG90 torque limit, ``sim.servo_kp``)
  tracking the angles the servo firmware outputs, so the arm can be stopped by
  the shelf or the floor. ``/joint_states`` still reports the commanded angle,
  exactly as the ESP32 does (hobby servos have no position feedback).
* The LD06 casts 360 rays from the ``laser`` site against world obstacles
  (geom group 0), in the laser's own plane — if the chassis tilts, so does the
  scan.

The phone in the front holder (smabo-app's half of the robot) is modelled too:
its screen shows the face (``set_screen_image``), its front camera can be
rendered (``render_phone_camera``) and its accelerometer/gyro are read in the
phone's own axes (``phone_imu``).

The MuJoCo viewer (``smabo-sim --engine mujoco --viewer``) shares ``lock``
with ``step`` so drawing never sees a half-updated state.
"""

from __future__ import annotations

import contextlib
import logging
import math

import mujoco
import numpy as np

from ..world import Pose2D, World
from .base import Engine
from .mujoco_model import ARM_JOINTS, build_mjcf, find_description

log = logging.getLogger(__name__)

_OBSTACLES_ONLY = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


class MujocoEngine(Engine):
    def __init__(self, world: World, cfg) -> None:
        super().__init__(world, cfg)
        self.description = find_description(cfg.get("sim.description") or None)
        if self.description is None:
            log.warning("smabo_description meshes not found: drawing the robot with simple shapes "
                        "(set sim.description or SMABO_DESCRIPTION to the smabo_description folder)")
        else:
            log.info("robot meshes: %s", self.description / "meshes")
        self.xml = build_mjcf(world, cfg, self.description)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.lock = contextlib.nullcontext     # replaced by the viewer's lock
        self._rng = np.random.default_rng(cfg.get("sim.seed"))
        self._radius = cfg.get("dc.wheel_radius", 0.030)

        m = self.model
        self._base = m.body("base_link").id
        self._laser = m.site("laser").id
        self._root_qpos = m.jnt_qposadr[m.joint("root").id]
        self._root_qvel = m.jnt_dofadr[m.joint("root").id]
        self._wheel_qpos = [m.jnt_qposadr[m.joint(f"{s}_wheel_joint").id] for s in ("left", "right")]
        self._wheel_act = [m.actuator(f"{s}_wheel").id for s in ("left", "right")]
        self._arm_act = {name: m.actuator(name).id for name, *_ in ARM_JOINTS}
        self._arm_range = {name: tuple(m.jnt_range[m.joint(name).id]) for name in self._arm_act}
        self._sim_time_debt = 0.0
        self.has_phone = m.nsensor > 0 and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "phone_camera") >= 0
        if self.has_phone:
            self.screen_tex = m.texture("phone_screen").id
            self._screen_shape = (int(m.tex_height[self.screen_tex]), int(m.tex_width[self.screen_tex]),
                                  int(m.tex_nchannel[self.screen_tex]))
            self._accel = m.sensor("phone_accel").adr[0]
            self._gyro = m.sensor("phone_gyro").adr[0]
        self.screen_version = 0          # bumped on every new screen image
        self._renderer = None
        self._renderer_screen_version = -1
        self.last_scan_points = np.zeros((0, 3))   # world-frame lidar hits, for the viewer
        self.blocked = False
        self.reset(world.start)

    # ------------------------------------------------------------------ #
    def reset(self, pose: Pose2D) -> None:
        with self.lock():
            mujoco.mj_resetData(self.model, self.data)
            q = self.data.qpos
            a = self._root_qpos
            q[a:a + 3] = (pose.x, pose.y, self.model.body_pos[self._base][2])
            q[a + 3:a + 7] = (math.cos(pose.yaw / 2), 0.0, 0.0, math.sin(pose.yaw / 2))
            mujoco.mj_forward(self.model, self.data)
            self._wheel_last = [float(q[i]) for i in self._wheel_qpos]
            self._travel = [0.0, 0.0]

    def set_wheel_targets(self, left: float, right: float) -> None:
        self.data.ctrl[self._wheel_act[0]] = left / self._radius
        self.data.ctrl[self._wheel_act[1]] = right / self._radius

    def set_joint_targets(self, angles: dict[str, float]) -> None:
        for name, rad in angles.items():
            act = self._arm_act.get(name)
            if act is not None:
                lo, hi = self._arm_range[name]
                self.data.ctrl[act] = min(max(rad, lo), hi)

    def step(self, dt: float) -> None:
        ts = self.model.opt.timestep
        self._sim_time_debt += dt
        n = int(self._sim_time_debt / ts)
        if n <= 0:
            return
        self._sim_time_debt -= n * ts
        with self.lock():
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)
        self._accumulate_encoders()

    def _accumulate_encoders(self) -> None:
        noise = self.cfg.get("sim.encoder_noise", 0.0)
        for i, adr in enumerate(self._wheel_qpos):
            q = float(self.data.qpos[adr])
            d = (q - self._wheel_last[i]) * self._radius
            self._wheel_last[i] = q
            if noise > 0:
                d *= 1.0 + self._rng.normal(0.0, noise)
            self._travel[i] += d
        # "blocked": wheels turning but the chassis is not moving
        wheel_speed = np.abs(self.data.ctrl[self._wheel_act]).max() * self._radius
        body_speed = float(np.linalg.norm(self.data.qvel[self._root_qvel:self._root_qvel + 2]))
        self.blocked = wheel_speed > 0.05 and body_speed < 0.2 * wheel_speed

    def pose(self) -> Pose2D:
        p = self.data.xpos[self._base]
        R = self.data.xmat[self._base].reshape(3, 3)
        return Pose2D(float(p[0]), float(p[1]), math.atan2(R[1, 0], R[0, 0]))

    def read_wheel_distance(self) -> tuple[float, float]:
        out = (self._travel[0], self._travel[1])
        self._travel = [0.0, 0.0]
        return out

    def scan(self, angles: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
        angles = np.asarray(angles, dtype=float)
        n = len(angles)
        origin = self.data.site_xpos[self._laser].copy()
        R = self.data.site_xmat[self._laser].reshape(3, 3)
        local = np.stack([np.cos(angles), np.sin(angles), np.zeros(n)], axis=1)
        dirs = local @ R.T
        geomid = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n)
        mujoco.mj_multiRay(self.model, self.data, origin, dirs.ravel(), _OBSTACLES_ONLY, True, -1,
                           geomid, dist, None, n, range_max)
        hit = (geomid >= 0) & (dist >= 0)
        self.last_scan_points = origin + dirs[hit] * dist[hit, None]
        r = np.where(hit, dist, 0.0)
        noise = self.cfg.get("sim.lidar_noise", 0.0)
        if noise > 0:
            r = np.where(hit, r + self._rng.normal(0.0, noise, size=n), 0.0)
        r[(r < range_min) | (r > range_max)] = 0.0
        return r

    # ------------------------------------------------------------------ #
    # phone
    # ------------------------------------------------------------------ #
    def set_screen_image(self, rgb: np.ndarray) -> None:
        """Show an (H, W, 3) uint8 image on the phone screen."""
        if not self.has_phone:
            return
        h, w, n = self._screen_shape
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"screen image must be {w}x{h}, got {rgb.shape[1]}x{rgb.shape[0]}")
        adr = self.model.tex_adr[self.screen_tex]
        if n == 4:
            rgb = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
        self.model.tex_data[adr:adr + h * w * n] = rgb.reshape(-1)
        self.screen_version += 1

    def render_phone_camera(self) -> np.ndarray:
        """What the phone's front camera sees, as an (H, W, 3) uint8 RGB image.

        Must be called from one thread (the one that first called it): the
        offscreen OpenGL context belongs to that thread.
        """
        if not self.has_phone:
            raise RuntimeError("this model has no phone (phone.enabled is false)")
        if self._renderer is None:
            w = int(self.cfg.get("phone.camera.width", 640))
            h = int(self.cfg.get("phone.camera.height", 480))
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, w)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, h)
            self._renderer = mujoco.Renderer(self.model, h, w)
        r = self._renderer
        if self._renderer_screen_version != self.screen_version:
            # this context has its own copy of the screen texture
            r._gl_context.make_current()
            mujoco.mjr_uploadTexture(self.model, r._mjr_context, self.screen_tex)
            self._renderer_screen_version = self.screen_version
        with self.lock():
            r.update_scene(self.data, camera="phone_camera")
        return r.render().copy()

    def phone_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """(accelerometer m/s², gyroscope rad/s) in the phone's device axes.

        The accelerometer reads specific force like a real phone: +9.81 along
        the axis pointing up when at rest.
        """
        sd = self.data.sensordata
        return sd[self._accel:self._accel + 3].copy(), sd[self._gyro:self._gyro + 3].copy()
