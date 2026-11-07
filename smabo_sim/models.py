"""Robot configurations, one per step of the smabo roadmap.

The roadmap (smabo-smartphone-robot.github.io) builds smabo up step by step:
the phone alone, the base, then a module per guide. Each preset here is the
robot as it stands at that step, in smabo_description's terms (xacro args
``arm`` / ``hand`` / ``breadboard`` / ``mount``) plus what the ESP32 has wired
(encoders, LD06):

  name       roadmap step                           robot
  phone      (before the base)                      the phone alone, on a stand
  base       face / phone sensors / audio / vision  base + phone, on the floor
  hand       ハンド                                  base + both hands
  neck       首振り                                  base on the pan stand (head_pan)
  mobile     移動ロボット                            TT-motor chassis, no encoders (dc_drive)
  encmobile  エンコーダ付                            chassis + encoders (encoder_drive)
  arm        ロボットアーム / 動作計画               base + 3-DOF arm and gripper
  nav        ナビゲーション                          chassis + encoders + LD06 lidar
  full       (everything)                           chassis + encoders + lidar + arm

Picking a model sets ``sim.model`` and the ESP32 ``modes`` that model has
hardware for, and enables only its servos — what you would configure on the
real robot at that step.
"""

from __future__ import annotations

from dataclasses import dataclass

ARM_SERVOS = ("arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4")
HAND_SERVOS = ("left_hand", "right_hand")
NECK_SERVOS = ("head_pan",)


@dataclass(frozen=True)
class RobotModel:
    name: str
    label: str                  # shown in /view (Japanese, like the roadmap)
    roadmap: tuple[str, ...]    # roadmap node ids (docs/js/roadmap.js)
    base: bool = True           # False: the phone alone
    mount: str = "none"         # none | tt_motor | neck  (smabo_description mount:=)
    arm: bool = False
    hand: bool = False
    breadboard: bool = False
    encoder: bool = False       # wheel encoders wired (encoder_drive, /wheel_vel → odom)
    lidar: bool = False         # LD06 on top

    @property
    def mobile(self) -> bool:
        return self.mount == "tt_motor"

    def servos(self) -> tuple[str, ...]:
        return ((ARM_SERVOS if self.arm else ()) + (HAND_SERVOS if self.hand else ())
                + (NECK_SERVOS if self.mount == "neck" else ()))

    def modes(self) -> dict[str, bool]:
        """ESP32 ``modes`` for this hardware (same keys as config ``modes``)."""
        return {
            "servos": bool(self.servos()),
            "dc_drive": self.mobile and not self.encoder,
            "encoder_drive": self.mobile and self.encoder,
            "lidar": self.lidar,
        }

    def config_patch(self, joints: dict) -> dict:
        """Config patch selecting this model: ``sim.model``, ``modes`` and which
        servos are enabled (other joint settings are kept)."""
        mine = set(self.servos())
        return {
            "sim": {"model": self.name},
            "modes": self.modes(),
            "servos": {"joints": {n: {"enabled": n in mine} for n in joints}},
        }


MODELS: dict[str, RobotModel] = {m.name: m for m in (
    RobotModel("phone", "スマホだけ", (), base=False),
    RobotModel("base", "ベース+スマホ", ("brain", "web", "app", "face", "sensors", "voice", "imgproc",
                                       "baseparts", "esp32")),
    RobotModel("hand", "ハンド", ("hand",), hand=True, breadboard=True),
    RobotModel("neck", "首振り", ("neck",), mount="neck", breadboard=True),
    RobotModel("mobile", "移動ロボット", ("mobile",), mount="tt_motor", breadboard=True),
    RobotModel("encmobile", "エンコーダ付", ("encmobile",), mount="tt_motor", breadboard=True,
               encoder=True),
    RobotModel("arm", "ロボットアーム", ("arm", "plan"), arm=True),
    RobotModel("nav", "ナビゲーション", ("nav", "brainros"), mount="tt_motor", breadboard=True,
               encoder=True, lidar=True),
    RobotModel("full", "全部入り", (), mount="tt_motor", arm=True, encoder=True, lidar=True),
)}

DEFAULT_MODEL = "full"


def get_model(name: str | None) -> RobotModel:
    """The model called ``name`` (a preset name or a roadmap node id)."""
    name = (name or DEFAULT_MODEL).strip()
    if name in MODELS:
        return MODELS[name]
    for m in MODELS.values():
        if name in m.roadmap:
            return m
    raise ValueError(f"unknown model {name!r} (available: {', '.join(MODELS)})")
