import math
import random

from smabo_sim.config import Config
from smabo_sim.firmware import ServoFirmware
from smabo_sim.random_motion import RandomMotion

GROUP = {"name": "head", "joints": ["head_pan", "left_hand"], "interval": [0.5, 1.0]}


def run(cfg, seconds, dt=0.02, servos=None, rm=None):
    servos = servos or ServoFirmware(cfg.get("servos.joints"))
    rm = rm or RandomMotion(cfg, random.Random(1))
    seen = {n: set() for n in servos.joints}
    t = 0.0
    while t < seconds:
        rm.tick(t, servos)
        servos.tick(t, dt)
        for n, a in servos.state().items():
            seen[n].add(round(a, 4))
        t += dt
    return servos, rm, seen


def test_manual_behavior_keeps_joints_still():
    cfg = Config(overrides={"servos": {"behavior": "manual", "random_groups": [GROUP]}})
    _, _, seen = run(cfg, 10)
    assert all(len(v) == 1 for v in seen.values())


def test_random_moves_only_grouped_joints_within_limits():
    cfg = Config(overrides={"servos": {"behavior": "random", "random_groups": [GROUP]}})
    servos, _, seen = run(cfg, 20)
    assert len(seen["head_pan"]) > 20 and len(seen["left_hand"]) > 20
    assert len(seen["arm_joint_1"]) == 1                     # not in any group
    for n in ("head_pan", "left_hand"):
        spec = servos.joints[n]
        lo, hi = spec["min_angle"] * math.pi / 180, spec["max_angle"] * math.pi / 180
        assert all(lo - 1e-3 <= a <= hi + 1e-3 for a in seen[n])


def test_left_hand_with_instant_speed_still_moves_smoothly():
    # max_speed 0 ("instant") → the firmware's 120 deg/s default, not a jump
    cfg = Config(overrides={"servos": {"behavior": "random", "random_groups": [GROUP]}})
    servos = ServoFirmware(cfg.get("servos.joints"))
    assert servos.joints["left_hand"]["max_speed"] == 0
    prev = None
    _, _, _ = run(cfg, 0.05, servos=servos)
    for _ in range(200):
        servos.tick(0, 0.02)
        a = servos.state()["left_hand"] * 180 / math.pi
        if prev is not None:
            assert abs(a - prev) <= 120 * 0.02 + 1e-6
        prev = a


def test_switching_behavior_and_groups_live():
    cfg = Config(overrides={"servos": {"behavior": "manual", "random_groups": []}})
    servos, rm, _ = run(cfg, 2)
    cfg.update({"servos": {"behavior": "random", "random_groups": [GROUP]}})
    _, _, seen = run(cfg, 5, servos=servos, rm=rm)
    assert len(seen["head_pan"]) > 5
    cfg.update({"servos": {"behavior": "manual"}})
    servos.tick(0, 10.0)                                     # let it settle
    _, _, seen = run(cfg, 5, servos=servos, rm=rm)
    assert len(seen["head_pan"]) == 1


def test_trajectory_clears_the_random_speed_override():
    cfg = Config()
    servos = ServoFirmware(cfg.get("servos.joints"))
    servos.set_angle_deg("head_pan", 40, speed=10)           # slow random move
    servos.load_trajectory(["head_pan"], [{"positions": [0.0], "time_from_start": {"sec": 0}}], 0.0)
    servos.tick(0.0, 0.1)
    assert abs(servos.state()["head_pan"]) < 1e-9            # 120 deg/s config speed: stays at 0
    servos.set_angle_deg("head_pan", 40, speed=10)
    servos.tick(0.1, 1.0)
    assert abs(servos.state()["head_pan"] * 180 / math.pi - 10) < 0.01
