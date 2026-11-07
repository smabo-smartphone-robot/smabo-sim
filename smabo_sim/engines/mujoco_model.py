"""Build the MuJoCo model (MJCF) for smabo-sim: the world plus the smabo robot.

The robot mirrors smabo_description (smabo-brain-ros) for the configuration
picked by ``sim.model`` (``smabo_sim.models``: one per roadmap step): the same
link frames, joint names, axes and limits, so joint angles mean exactly what
they mean in the URDF / MoveIt. The full tree (``full``, i.e. ``mount:=tt_motor
arm:=true`` plus the other modules where a model has them):

  [pan_stand_link (fixed on the floor) ─ pan_link ← head_pan (z)]   mount:=neck
  base_link (bottom centre of the base box; free joint only with tt_motor,
             otherwise fixed on the floor or on pan_link)
    ├─ chassis (TT motor chassis, ball caster)                      mount:=tt_motor
    │    ├─ left_wheel_link   ← left_wheel_joint  (hinge y, velocity motor)
    │    └─ right_wheel_link  ← right_wheel_joint
    ├─ laser site (z 0.115, URDF laser_joint)                       lidar
    ├─ arm_base_link                                                arm:=true
    │    └─ arm_link_1 ← arm_joint_1 (z) ─ arm_link_2 ← arm_joint_2 (y)
    │         ─ arm_link_3 ← arm_joint_3 (y) ─ arm_link_4 ← arm_joint_4 (x, gripper)
    ├─ hand_mount_link ─ left/right_hand_link ← left/right_hand (−y)  hand:=true
    ├─ breadboard_link                                              breadboard:=true
    └─ phone (front holder)

The ``phone`` model has no base: base_link is an empty frame on the floor and
the phone stands on a small stand in front of it.

Visual meshes are the smabo_description STL files when they can be found
(``sim.description`` or a sibling smabo-brain-ros checkout); otherwise simple
primitives are drawn. Collision always uses primitives for the chassis and
wheels (stable rolling contact) and, when available, the convex hull of each
arm link's collision mesh.

Geom groups (the viewer shows 0-2 by default):
  0 world obstacles (what the lidar sees)   1 robot visuals
  2 floor                                   3 robot collision shapes (hidden)
Contacts: the robot touches the world and floor but not itself.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from ..models import get_model
from ..world import World

# ---- geometry from smabo_description (metres, base_link frame) ------------
GROUND_Z = -0.04985                       # floor below base_link (tt_motor xacro)
WHEEL_JOINT = (-0.0215, 0.065, -0.01985)  # left wheel; right mirrors y
WHEEL_HALF_WIDTH = 0.013
CASTER = (0.041, 0.0, -0.0408, 0.009)     # x, y, z, radius (from the chassis mesh)
BASE_BOX = ((-0.05, 0.067), (-0.078, 0.078), (0.0, 0.10))
CHASSIS_BOX = ((-0.05, 0.05), (-0.0584, 0.0584), (-0.035, 0.003))
LASER_Z = 0.115
ARM_MOUNT = (0.010, 0.0, 0.090)
ARM_JOINTS = (
    # name, parent link, child link, origin xyz, origin rpy, axis
    ("arm_joint_1", "arm_base_link", "arm_link_1", (0, 0, 0.015), (0, 0, 0), (0, 0, 1)),
    ("arm_joint_2", "arm_link_1", "arm_link_2", (0, 0, 0.0365), (0, 0, 0), (0, 1, 0)),
    ("arm_joint_3", "arm_link_2", "arm_link_3", (0, 0, 0.0515), (0, 0, 0), (0, 1, 0)),
    ("arm_joint_4", "arm_link_3", "arm_link_4", (0.0125, -0.0048, 0.04345), (math.pi / 2, 0, 0), (1, 0, 0)),
)
# fallback capsule (from, to) per arm link when there are no meshes
ARM_CAPSULES = {
    "arm_base_link": ((0, 0, -0.01), (0, 0, 0.015), 0.012),
    "arm_link_1": ((0, 0, 0.0), (0, 0, 0.0365), 0.012),
    "arm_link_2": ((0, 0, 0.0), (0, 0, 0.0515), 0.009),
    "arm_link_3": ((0, 0, 0.0), (0.0125, 0, 0.065), 0.009),
    "arm_link_4": ((0, 0, 0.0), (0, 0, 0.028), 0.003),
}
MASS = {"base_link": 0.30, "chassis": 0.15, "wheel": 0.03, "caster": 0.005,
        "arm_base_link": 0.03, "arm_link_1": 0.03, "arm_link_2": 0.03,
        "arm_link_3": 0.03, "arm_link_4": 0.005, "hand": 0.02, "pan_link": 0.05}
# hands (smabo_hand.xacro): joint origins in hand_mount_link (= base_link)
HAND_JOINTS = (("left_hand", "left_hand_link", (0.0003, 0.052, 0.050), 1),
               ("right_hand", "right_hand_link", (-0.0003, -0.052, 0.050), -1))
HAND_BOX = (0.010, 0.0185, 0.030)          # fallback / collision half sizes (y extends outward)
# neck (smabo_neck.xacro)
PAN_JOINT_Z = 0.0438                      # head_pan above pan_stand_link (floor)
PAN_TO_BASE_Z = 0.015                     # base_link above pan_link
PAN_STAND = (0.057, 0.0508)               # fallback cylinder radius, height
PAN_PLATE = (0.055, 0.018)
BREADBOARD_BOX = ((-0.0725, -0.047), (-0.05, 0.05), (0.0005, 0.0978))
COLORS = {"base": "0.60 0.64 0.70 1", "part": "0.95 0.95 0.95 1", "mount": "0.80 0.82 0.86 1",
          "accent": "0.98 0.57 0.24 1", "dark": "0.15 0.15 0.18 1"}
ARM_COLOR = {"arm_base_link": "mount", "arm_link_1": "part", "arm_link_2": "part",
             "arm_link_3": "part", "arm_link_4": "accent"}
MESH_MODULE = {"base_link": "smabo_base", "chassis_link": "smabo_tt_motor",
               "left_wheel_link": "smabo_tt_motor", "right_wheel_link": "smabo_tt_motor",
               **{n: "smabo_arm" for n in ARM_COLOR},
               "hand_mount_link": "smabo_hand", "left_hand_link": "smabo_hand",
               "right_hand_link": "smabo_hand", "pan_stand_link": "smabo_neck",
               "pan_link": "smabo_neck", "breadboard_link": "smabo_breadboard"}

# ---- the smartphone (smabo-app) in the front holder ------------------------
# smabo_base's front plate ends at x = 0.053; the holder hooks (smartphone_long/
# short_*_height_11) leave an 11 mm slot in front of it, 0.156 m wide, so the
# phone sits in landscape, screen forward, centred on the plate.
PHONE_BACK_X = 0.054
PHONE_SIZE = (0.008, 0.150, 0.072)        # thickness (x), length (y), height (z)
PHONE_Z = 0.050
PHONE_SCREEN = (0.140, 0.064)             # visible screen (y, z)
PHONE_MASS = 0.18
PHONE_CAMERA_INSET = 0.010                # front camera from the phone's top edge


def phone_axes(camera_side: str) -> tuple[tuple, tuple, tuple]:
    """Phone device axes (Android/sensors_plus convention) in base_link.

    z = out of the screen (+x, forward). y = toward the top of the phone, i.e.
    the end with the front camera: ``robot_left`` (+y) or ``robot_right`` (−y).
    x = y × z.
    """
    ydev = (0, 1, 0) if camera_side != "robot_right" else (0, -1, 0)
    zdev = (1, 0, 0)
    xdev = tuple(int(v) for v in np.cross(ydev, zdev))
    return xdev, ydev, zdev


SERVO_TORQUE = 0.18        # N·m, SG90 stall torque
PAN_TORQUE = 1.0           # N·m, MG996R (head_pan) stall torque
WHEEL_TORQUE = 0.10        # N·m, TT gear motor (~6 V) stall torque at the wheel


def find_description(configured: str | None = None) -> Path | None:
    """Locate smabo_description (for its meshes). ``None`` = draw primitives.

    ``configured`` = "none" forces primitives.
    """
    if configured == "none":
        return None
    here = Path(__file__).resolve()
    candidates = [configured, os.environ.get("SMABO_DESCRIPTION"),
                  Path.cwd() / "../smabo-brain-ros/smabo_description",
                  here.parents[3] / "smabo-brain-ros/smabo_description"]
    for c in candidates:
        if c and (Path(c) / "meshes").is_dir():
            return Path(c).resolve()
    return None


def rpy_to_quat(r: float, p: float, y: float) -> tuple[float, float, float, float]:
    """URDF fixed-axis roll/pitch/yaw → MuJoCo quaternion (w, x, y, z)."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def _v(*xs) -> str:
    return " ".join(f"{float(x):.6g}" for x in xs)


def _box(bounds) -> tuple[str, str]:
    (x0, x1), (y0, y1), (z0, z1) = bounds
    return _v((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), _v((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)


# --------------------------------------------------------------------------- #
# world
# --------------------------------------------------------------------------- #
WALL_THICKNESS = 0.05


def _wall(name: str, x1, y1, x2, y2, height, outward=(0.0, 0.0), thick=WALL_THICKNESS) -> str:
    """Box along a segment; ``outward`` shifts it so one face lies on the segment."""
    length = math.hypot(x2 - x1, y2 - y1)
    yaw = math.atan2(y2 - y1, x2 - x1)
    cx, cy = (x1 + x2) / 2 + outward[0] * thick / 2, (y1 + y2) / 2 + outward[1] * thick / 2
    return (f'<geom name={quoteattr(name)} class="obstacle" type="box" pos="{_v(cx, cy, height / 2)}" '
            f'euler="0 0 {yaw:.6g}" size="{_v(length / 2 + thick / 2, thick / 2, height / 2)}"/>')


def world_xml(world: World) -> str:
    out = []
    for i, s in enumerate(world.shapes):
        kind = s.get("type")
        name = str(s.get("name") or f"{kind}_{i}")
        yaw = math.radians(float(s.get("yaw_deg", 0.0)))
        if kind == "room":
            h = float(s.get("height", 0.5))
            cx, cy, w, d = (float(s[k]) for k in ("x", "y", "w", "h"))
            # inner faces on the outline, so the lidar sees what the Nav2 map shows
            c, sn = math.cos(yaw), math.sin(yaw)
            for j, (lx1, ly1, lx2, ly2, ox, oy) in enumerate((
                    (-w / 2, -d / 2, w / 2, -d / 2, 0, -1), (w / 2, -d / 2, w / 2, d / 2, 1, 0),
                    (w / 2, d / 2, -w / 2, d / 2, 0, 1), (-w / 2, d / 2, -w / 2, -d / 2, -1, 0))):
                p1 = (cx + c * lx1 - sn * ly1, cy + sn * lx1 + c * ly1)
                p2 = (cx + c * lx2 - sn * ly2, cy + sn * lx2 + c * ly2)
                out.append(_wall(f"{name}_wall{j}", *p1, *p2, h, (c * ox - sn * oy, sn * ox + c * oy)))
        elif kind == "box":
            h = float(s.get("height", 0.4))
            out.append(f'<geom name={quoteattr(name)} class="obstacle" type="box" '
                       f'pos="{_v(s["x"], s["y"], h / 2)}" euler="0 0 {yaw:.6g}" '
                       f'size="{_v(float(s["w"]) / 2, float(s["h"]) / 2, h / 2)}"/>')
        elif kind == "wall":
            h = float(s.get("height", 0.5))
            out.append(_wall(name, float(s["x1"]), float(s["y1"]), float(s["x2"]), float(s["y2"]), h,
                             thick=float(s.get("thickness", 0.02))))
        elif kind == "cylinder":
            h = float(s.get("height", 0.5))
            out.append(f'<geom name={quoteattr(name)} class="obstacle" type="cylinder" '
                       f'pos="{_v(s["x"], s["y"], h / 2)}" size="{_v(s["r"], h / 2)}"/>')
        else:
            raise ValueError(f"unknown shape type: {kind!r}")
    return "\n    ".join(out)


# --------------------------------------------------------------------------- #
# robot
# --------------------------------------------------------------------------- #
class _Robot:
    def __init__(self, meshes: bool, cfg) -> None:
        self.meshes = meshes
        self.cfg = cfg
        self.model = get_model(cfg.get("sim.model"))
        # collision shapes are what you see when there are no meshes
        self.col = "collision" if meshes else "collision_visible"

    def visual(self, link: str, color: str, fallback: str = "") -> str:
        if self.meshes:
            return f'<geom class="visual" mesh="{link}" rgba="{COLORS[color]}"/>'
        return fallback

    def wheel(self, side: str, sign: int) -> str:
        x, y, z = WHEEL_JOINT
        r = self.cfg.get("dc.wheel_radius", 0.030)
        return f"""
        <body name="{side}_wheel_link" pos="{_v(x, sign * y, z)}">
          <joint name="{side}_wheel_joint" type="hinge" axis="0 1 0" damping="0.0002" armature="0.0001"/>
          <geom class="{self.col}" type="ellipsoid" size="{_v(r, WHEEL_HALF_WIDTH, r)}"
                mass="{MASS['wheel']}" rgba="{COLORS['dark']}" friction="1.0 0.005 0.0001"/>
          {self.visual(f'{side}_wheel_link', 'dark')}
        </body>"""

    def arm(self) -> str:
        def link_geoms(link: str) -> str:
            mass = MASS[link]
            color = COLORS[ARM_COLOR[link]]
            if self.meshes:
                col = (f'<geom class="collision" type="mesh" mesh="{link}_col" mass="{mass}"/>')
                return col + self.visual(link, ARM_COLOR[link])
            a, b, rad = ARM_CAPSULES[link]
            return (f'<geom class="{self.col}" type="capsule" fromto="{_v(*a, *b)}" size="{rad}" '
                    f'mass="{mass}" rgba="{color}"/>')

        lim = self._limits()
        xml = f'<body name="arm_base_link" pos="{_v(*ARM_MOUNT)}">{link_geoms("arm_base_link")}'
        closes = ["</body>"]
        for name, _parent, child, xyz, rpy, axis in ARM_JOINTS:
            lo, hi = lim.get(name, (-math.pi / 2, math.pi / 2))
            xml += (f'\n<body name="{child}" pos="{_v(*xyz)}" quat="{_v(*rpy_to_quat(*rpy))}">'
                    f'<joint name="{name}" type="hinge" axis="{_v(*axis)}" range="{_v(lo, hi)}" '
                    f'damping="0.01" armature="0.0005"/>{link_geoms(child)}')
            closes.append("</body>")
        return xml + "".join(reversed(closes))

    def _limits(self) -> dict[str, tuple[float, float]]:
        lim = {}
        for name, spec in (self.cfg.get("servos.joints") or {}).items():
            lim[name] = (math.radians(spec.get("min_angle", -90)), math.radians(spec.get("max_angle", 90)))
        return lim

    def hands(self) -> str:
        lim = self._limits()
        xml = f'<body name="hand_mount_link">{self.visual("hand_mount_link", "mount")}'
        hx, hy, hz = HAND_BOX
        for joint, link, pos, sign in HAND_JOINTS:
            lo, hi = lim.get(joint, (-math.pi / 2, math.pi / 2))
            xml += f"""
        <body name="{link}" pos="{_v(*pos)}">
          <joint name="{joint}" type="hinge" axis="0 -1 0" range="{_v(lo, hi)}" damping="0.005" armature="0.0003"/>
          <geom class="{self.col}" type="box" pos="{_v(0, sign * hy, 0)}" size="{_v(hx, hy, hz)}"
                mass="{MASS['hand']}" rgba="{COLORS['accent']}"/>
          {self.visual(link, 'accent')}
        </body>"""
        return xml + "</body>"

    def breadboard(self) -> str:
        pos, size = _box(BREADBOARD_BOX)
        return (f'<body name="breadboard_link">{self.visual("breadboard_link", "mount")}'
                + ("" if self.meshes else
                   f'<geom class="phone_frame" type="box" pos="{pos}" size="{size}" rgba="{COLORS["mount"]}"/>')
                + "</body>")

    def chassis(self) -> str:
        cpos, csize = _box(CHASSIS_BOX)
        cx, cy, cz, cr = CASTER
        return f"""
      <body name="chassis_link">
        <geom class="{self.col}" type="box" pos="{cpos}" size="{csize}" mass="{MASS['chassis']}" rgba="{COLORS['mount']}"/>
        <geom class="{self.col}" type="sphere" pos="{_v(cx, cy, cz)}" size="{cr}" mass="{MASS['caster']}"
              condim="1" priority="1" rgba="0.7 0.7 0.7 1"/>
        {self.visual('chassis_link', 'mount')}
        {self.wheel('left', 1)}
        {self.wheel('right', -1)}
      </body>"""

    def base_contents(self) -> str:
        m = self.model
        bpos, bsize = _box(BASE_BOX)
        lx, ly = (self.cfg.get("sim.laser_offset") or [0.0, 0.0])[:2]
        parts = [
            f'<geom class="{self.col}" type="box" pos="{bpos}" size="{bsize}" mass="{MASS["base_link"]}" rgba="{COLORS["base"]}"/>',
            self.visual("base_link", "base"),
        ]
        if m.lidar:
            parts.append(f'<site name="laser" pos="{_v(lx, ly, LASER_Z)}" size="0.012" rgba="0.1 0.6 1 1"/>')
        if m.mobile:
            parts.append(self.chassis())
        if m.arm:
            parts.append(self.arm())
        if m.hand:
            parts.append(self.hands())
        if m.breadboard:
            parts.append(self.breadboard())
        parts.append(self.phone())
        return "\n      ".join(parts)

    def body(self, x: float, y: float, yaw: float) -> str:
        m = self.model
        if not m.base:
            return self.phone_alone(x, y, yaw)
        if m.mobile:
            # on its wheels: free to drive, fall and be pushed
            z0 = -GROUND_Z + 0.002
            return f"""
    <body name="base_link" pos="{_v(x, y, z0)}" euler="0 0 {yaw:.6g}">
      <freejoint name="root"/>
      {self.base_contents()}
    </body>"""
        if m.mount == "neck":
            lo, hi = self._limits().get("head_pan", (-math.pi / 2, math.pi / 2))
            sr, sh = PAN_STAND
            pr, ph = PAN_PLATE
            stand = "" if self.meshes else (
                f'<geom class="phone_frame" type="cylinder" pos="0 0 {sh / 2:.6g}" size="{_v(sr, sh / 2)}" rgba="{COLORS["mount"]}"/>')
            plate = f'<geom class="{self.col}" type="cylinder" pos="0 0 {ph / 2:.6g}" size="{_v(pr, ph / 2)}" mass="{MASS["pan_link"]}" rgba="{COLORS["part"]}"/>'
            return f"""
    <body name="pan_stand_link" pos="{_v(x, y, 0)}" euler="0 0 {yaw:.6g}">
      {stand}{self.visual('pan_stand_link', 'mount')}
      <body name="pan_link" pos="0 0 {PAN_JOINT_Z}">
        <joint name="head_pan" type="hinge" axis="0 0 1" range="{_v(lo, hi)}" damping="0.02" armature="0.002"/>
        {plate}{self.visual('pan_link', 'part')}
        <body name="base_link" pos="0 0 {PAN_TO_BASE_Z}">
          {self.base_contents()}
        </body>
      </body>
    </body>"""
        # standing on the floor (fixed): the brain-only steps, hand, arm
        return f"""
    <body name="base_link" pos="{_v(x, y, 0)}" euler="0 0 {yaw:.6g}">
      {self.base_contents()}
    </body>"""

    def phone_alone(self, x: float, y: float, yaw: float) -> str:
        """The phone by itself on a small stand (no base); base_link is the floor frame."""
        t, _length, h = PHONE_SIZE
        stand = f'<geom class="phone_frame" type="box" pos="{_v(-0.012, 0, 0.012)}" size="0.012 0.04 0.012" rgba="{COLORS["dark"]}"/>'
        return f"""
    <body name="base_link" pos="{_v(x, y, 0)}" euler="0 0 {yaw:.6g}">
      {stand}
      {self.phone((t / 2, 0.0, h / 2 + 0.004))}
    </body>"""

    def phone(self, pos: tuple[float, float, float] | None = None) -> str:
        if not self.cfg.get("phone.enabled", True):
            return ""
        t, length, h = PHONE_SIZE
        side = self.cfg.get("phone.camera_side", "robot_left")
        xdev, ydev, _ = phone_axes(side)
        cam_y = (length / 2 - PHONE_CAMERA_INSET) * ydev[1]
        w, hh = int(self.cfg.get("phone.camera.width", 640)), int(self.cfg.get("phone.camera.height", 480))
        hfov = math.radians(float(self.cfg.get("phone.camera.hfov_deg", 60.0)))
        fovy = math.degrees(2 * math.atan(math.tan(hfov / 2) * hh / w))
        return f"""
      <body name="phone" pos="{_v(*(pos or (PHONE_BACK_X + t / 2, 0, PHONE_Z)))}">
        <geom class="{self.col}" type="box" size="{_v(t / 2, length / 2, h / 2)}" mass="{PHONE_MASS}" rgba="0.08 0.08 0.10 1"/>
        <geom class="phone_frame" type="box" size="{_v(t / 2, length / 2, h / 2)}" rgba="0.08 0.08 0.10 1"/>
        <geom name="phone_screen" class="visual" mesh="phone_screen" material="phone_screen" pos="{_v(t / 2 + 0.0003, 0, 0)}"/>
        <site name="phone_imu" xyaxes="{_v(*xdev, *ydev)}" size="0.004" rgba="0 0 0 0"/>
        <camera name="phone_camera" pos="{_v(t / 2 + 0.0005, cam_y, 0)}" xyaxes="0 -1 0 0 0 1" fovy="{fovy:.4g}"/>
        <geom class="visual" type="sphere" size="0.0025" pos="{_v(t / 2, cam_y, 0)}" rgba="0.02 0.02 0.03 1"/>
      </body>"""

    def actuators(self, wheel_kv: float) -> str:
        out = []
        if self.model.mobile:
            out += [f'<velocity name="{s}_wheel" joint="{s}_wheel_joint" kv="{wheel_kv:.6g}" '
                    f'forcerange="{-WHEEL_TORQUE} {WHEEL_TORQUE}"/>' for s in ("left", "right")]
        kp = float(self.cfg.get("sim.servo_kp", 1.5))
        for name in self.model.servos():
            torque = PAN_TORQUE if name == "head_pan" else SERVO_TORQUE
            gain = kp * (torque / SERVO_TORQUE)
            out.append(f'<position name="{name}" joint="{name}" kp="{gain:.6g}" '
                       f'forcerange="{-torque} {torque}"/>')
        return "\n    ".join(out)

    def mesh_links(self) -> list[str]:
        """Links of this model that have meshes."""
        m = self.model
        links = []
        if m.base:
            links.append("base_link")
        if m.mobile:
            links += ["chassis_link", "left_wheel_link", "right_wheel_link"]
        if m.arm:
            links += list(ARM_COLOR)
        if m.hand:
            links += ["hand_mount_link", "left_hand_link", "right_hand_link"]
        if m.mount == "neck":
            links += ["pan_stand_link", "pan_link"]
        if m.breadboard:
            links.append("breadboard_link")
        return links

    def assets(self) -> str:
        out = []
        if self.cfg.get("phone.enabled", True):
            sw, sh = PHONE_SCREEN
            tw = int(self.cfg.get("phone.screen.width", 512))
            th = int(self.cfg.get("phone.screen.height", 256))
            # a quad facing +x, u left→right and v top→bottom as seen from the front
            out.append(f'<texture name="phone_screen" type="2d" builtin="flat" width="{tw}" height="{th}" rgb1="0 0 0"/>')
            out.append('<material name="phone_screen" texture="phone_screen" emission="1" specular="0" shininess="0"/>')
            out.append(f'<mesh name="phone_screen" vertex="{_v(0, -sw / 2, -sh / 2, 0, sw / 2, -sh / 2, 0, sw / 2, sh / 2, 0, -sw / 2, sh / 2, -0.0002, 0, 0)}" '
                       'texcoord="0 1  1 1  1 0  0 0  0.5 0.5" face="0 1 2  0 2 3  4 1 0  4 2 1  4 3 2  4 0 3"/>')
        if not self.meshes:
            return "\n    ".join(out)
        for link in self.mesh_links():
            module = MESH_MODULE[link]
            out.append(f'<mesh name="{link}" file="{module}/visual/{link}.stl"/>')
            if link in ARM_COLOR:
                out.append(f'<mesh name="{link}_col" file="{module}/collision/{link}.stl"/>')
        return "\n    ".join(out)


def wheel_kv(cfg) -> float:
    """Velocity-motor gain reproducing a DC motor's torque-speed line.

    A brushed DC motor at duty ``d`` gives ``T = T_stall · (d − ω / ω_max)``.
    The firmware's duty is ``target / full_duty_speed``, so with the actuator
    ``T = kv · (ω_target − ω)`` this is ``kv = T_stall / ω_max``, where
    ``ω_max`` is the wheel speed at full duty. The response time then follows
    from the robot's mass (not from ``sim.motor_time_constant``).
    """
    r = cfg.get("dc.wheel_radius", 0.030)
    max_lin = cfg.get("dc.max_linear", 0.3)
    max_ang = cfg.get("dc.max_angular", 1.5)
    sep = cfg.get("dc.wheel_separation", 0.13)
    omega_max = max(max_lin + max_ang * sep / 2.0, 1e-6) / r
    return WHEEL_TORQUE / omega_max


def _phone_sensors(cfg) -> str:
    if not cfg.get("phone.enabled", True):
        return ""
    return ("<sensor>\n    <accelerometer name=\"phone_accel\" site=\"phone_imu\"/>\n"
            "    <gyro name=\"phone_gyro\" site=\"phone_imu\"/>\n  </sensor>")


def build_mjcf(world: World, cfg, description: Path | None) -> str:
    robot = _Robot(description is not None, cfg)
    meshdir = f'meshdir={quoteattr(str(description / "meshes"))}' if description else ""
    ts = float(cfg.get("sim.timestep", 0.002))
    st = world.start
    return f"""<mujoco model="smabo_sim_{world.name}">
  <compiler angle="radian" autolimits="true" {meshdir}/>
  <option timestep="{ts:.6g}" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/>
    <global azimuth="-130" elevation="-35" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.93 0.94 0.96" rgb2="0.70 0.74 0.80" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.87 0.89" rgb2="0.80 0.81 0.84"
             width="512" height="512" mark="edge" markrgb="0.7 0.7 0.72"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
    <material name="obstacle" rgba="0.55 0.62 0.72 1"/>
    {robot.assets()}
  </asset>
  <default>
    <default class="obstacle"><geom group="0" contype="1" conaffinity="2" material="obstacle" friction="0.3 0.005 0.0001" priority="1"/></default>
    <default class="floor"><geom group="2" contype="1" conaffinity="2"/></default>
    <default class="collision"><geom group="3" contype="2" conaffinity="1"/></default>
    <default class="collision_visible"><geom group="1" contype="2" conaffinity="1"/></default>
    <default class="visual"><geom group="1" contype="0" conaffinity="0" density="0" type="mesh"/></default>
    <default class="phone_frame"><geom group="1" contype="0" conaffinity="0" density="0"/></default>
  </default>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <geom name="floor" class="floor" type="plane" size="0 0 0.05" material="grid"/>
    {world_xml(world)}
    {robot.body(st.x, st.y, st.yaw)}
  </worldbody>
  <actuator>
    {robot.actuators(wheel_kv(cfg))}
  </actuator>
  {_phone_sensors(cfg)}
</mujoco>
"""
