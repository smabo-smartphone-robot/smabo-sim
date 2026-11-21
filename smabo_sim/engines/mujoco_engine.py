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
from .base import Engine, apply_target_change
from ..models import get_model
from .mujoco_model import build_mjcf, find_description

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
        self.robot_model = get_model(cfg.get("sim.model"))
        self._base = m.body("base_link").id
        # parts a model may lack (see smabo_sim.models): no laser site without
        # the lidar, no free joint / wheels unless on the TT-motor chassis
        self._laser = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "laser")
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self._root_qpos = m.jnt_qposadr[root] if root >= 0 else None
        self._root_qvel = m.jnt_dofadr[root] if root >= 0 else None
        mobile = self.robot_model.mobile
        self._wheel_qpos = [m.jnt_qposadr[m.joint(f"{s}_wheel_joint").id] for s in ("left", "right")] if mobile else []
        self._wheel_act = [m.actuator(f"{s}_wheel").id for s in ("left", "right")] if mobile else []
        self._arm_act = {name: m.actuator(name).id for name in self.robot_model.servos()}
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
        self._renderers: dict[str, mujoco.Renderer] = {}
        self._renderer_screen_version: dict[str, int] = {}
        self._targets = {t["name"]: m.body_mocapid[m.body(f"target_{t['name']}").id] for t in world.targets}
        self.last_scan_points = np.zeros((0, 3))   # world-frame lidar hits, for the viewer
        self.blocked = False
        self.reset(world.start)

    # ------------------------------------------------------------------ #
    def reset(self, pose: Pose2D) -> None:
        with self.lock():
            mujoco.mj_resetData(self.model, self.data)
            q = self.data.qpos
            a = self._root_qpos
            if a is not None:      # fixed models stay where the world file put them
                q[a:a + 3] = (pose.x, pose.y, self.model.body_pos[self._base][2])
                q[a + 3:a + 7] = (math.cos(pose.yaw / 2), 0.0, 0.0, math.sin(pose.yaw / 2))
            self._place_targets()
            mujoco.mj_forward(self.model, self.data)
            self._wheel_last = [float(q[i]) for i in self._wheel_qpos]
            self._travel = [0.0, 0.0]

    def set_wheel_targets(self, left: float, right: float) -> None:
        if not self._wheel_act:
            return
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
            self._place_targets()
        self._accumulate_encoders()

    # ------------------------------------------------------------------ #
    # vision targets (World.targets; their state lives there so it survives
    # a model switch, which builds a new engine on the same World)
    # ------------------------------------------------------------------ #
    def _place_targets(self) -> None:
        t_sim = self.data.time
        for t in self.world.targets:
            mid = self._targets[t["name"]]
            off = 0.0
            sw = t["sway"]
            if sw["on"]:
                off = sw["amplitude"] * math.sin(2 * math.pi * t_sim / sw["period"])
            # sideways = across the picture (perpendicular to where it faces)
            self.data.mocap_pos[mid] = (t["x"] - off * math.sin(t["yaw"]),
                                        t["y"] + off * math.cos(t["yaw"]), t["z"])
            self.data.mocap_quat[mid] = (math.cos(t["yaw"] / 2), 0.0, 0.0, math.sin(t["yaw"] / 2))

    def targets(self) -> list[dict]:
        """Vision targets with their current position (for /view and tests)."""
        out = []
        for t in self.world.targets:
            p = self.data.mocap_pos[self._targets[t["name"]]]
            out.append({"name": t["name"], "kind": t["kind"], "x": float(p[0]), "y": float(p[1]),
                        "z": float(p[2]), "yaw_deg": math.degrees(t["yaw"]), "size": t["size"],
                        "sway": dict(t["sway"])})
        return out

    def set_target(self, name: str, **change) -> None:
        """Move a target or set its sway: x, y, z (m), yaw_deg, sway_on,
        sway_amplitude (m), sway_period (s). Unknown names raise KeyError."""
        apply_target_change(self.world.target(name), change)
        with self.lock():
            self._place_targets()
            mujoco.mj_forward(self.model, self.data)

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
        if not self._wheel_act:
            return
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
        if self._laser < 0:        # this model has no lidar
            self.last_scan_points = np.zeros((0, 3))
            return np.zeros(n)
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
        r = self._renderer_for("phone", int(self.cfg.get("phone.camera.width", 640)),
                               int(self.cfg.get("phone.camera.height", 480)))
        with self.lock():
            r.update_scene(self.data, camera="phone_camera")
        return r.render().copy()

    # ------------------------------------------------------------------ #
    # offscreen rendering (the phone camera and the browser view)
    # ------------------------------------------------------------------ #
    def _renderer_for(self, name: str, width: int, height: int) -> mujoco.Renderer:
        """An offscreen renderer per use, created on first call.

        Each has its own OpenGL context, which belongs to the thread that
        created it, and its own copy of the phone screen texture (re-uploaded
        when the face changes).
        """
        r = self._renderers.get(name)
        if r is None or (r.width, r.height) != (width, height):
            if r is not None:
                r.close()
            g = self.model.vis.global_
            g.offwidth, g.offheight = max(g.offwidth, width), max(g.offheight, height)
            r = self._renderers[name] = mujoco.Renderer(self.model, height, width)
            self._renderer_screen_version[name] = -1
        if self.has_phone and self._renderer_screen_version[name] != self.screen_version:
            r._gl_context.make_current()
            mujoco.mjr_uploadTexture(self.model, r._mjr_context, self.screen_tex)
            self._renderer_screen_version[name] = self.screen_version
        return r

    def render_view(self, cam: dict, width: int, height: int, scan: bool = True) -> np.ndarray:
        """The world from an orbit camera, as an (H, W, 3) uint8 RGB image.

        ``cam``: ``azimuth`` / ``elevation`` (deg), ``distance`` (m), and either
        ``follow: True`` (orbit around the robot) or ``lookat`` ``[x, y, z]``.
        ``scan`` draws the latest lidar hits as red dots, like ``--viewer``.
        Same thread rule as ``render_phone_camera``.
        """
        r = self._renderer_for("view", width, height)
        c = mujoco.MjvCamera()
        c.azimuth, c.elevation, c.distance = cam["azimuth"], cam["elevation"], cam["distance"]
        if cam.get("follow", True):
            c.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            c.trackbodyid = self._base
        else:
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = cam.get("lookat", (0.0, 0.0, 0.0))
        with self.lock():
            r.update_scene(self.data, camera=c)
            pts = self.last_scan_points
        if scan and len(pts):
            scn = r.scene
            eye, size = np.eye(3).ravel(), np.array([0.012, 0.0, 0.0])
            rgba = np.array([1.0, 0.25, 0.2, 0.9], dtype=np.float32)
            n = min(len(pts), scn.maxgeom - scn.ngeom)
            for i in range(n):
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom + i], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size, pts[i], eye, rgba)
            scn.ngeom += n
        return r.render().copy()

    def close(self) -> None:
        """Free the offscreen renderers (call from the thread that rendered)."""
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def phone_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """(accelerometer m/s², gyroscope rad/s) in the phone's device axes.

        The accelerometer reads specific force like a real phone: +9.81 along
        the axis pointing up when at rest.
        """
        sd = self.data.sensordata
        return sd[self._accel:self._accel + 3].copy(), sd[self._gyro:self._gyro + 3].copy()
