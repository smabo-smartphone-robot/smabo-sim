import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from smabo_sim.config import Config  # noqa: E402
from smabo_sim.engines import create_engine  # noqa: E402
from smabo_sim.robot import SimRobot  # noqa: E402
from smabo_sim.world import World  # noqa: E402

QUIET = {"sim": {"encoder_noise": 0.0, "lidar_noise": 0.0, "description": "none"}}


def engine(world=None, **sim):
    cfg = Config(overrides={"sim": {**QUIET["sim"], **sim}})
    return create_engine("mujoco", world or World.load(), cfg)


def run(e, left, right, seconds):
    travel = np.zeros(2)
    e.set_wheel_targets(left, right)
    for _ in range(int(seconds / 0.01)):
        e.step(0.01)
        travel += e.read_wheel_distance()
    return travel


def test_robot_stands_still_on_its_wheels():
    e = engine()
    run(e, 0.0, 0.0, 1.0)
    p = e.pose()
    assert abs(p.x) < 1e-3 and abs(p.y) < 1e-3 and abs(p.yaw) < 1e-3
    assert e.data.xpos[e._base][2] == pytest.approx(0.049, abs=1e-3)    # wheels on the floor (dg01d_e)


def test_straight_drive_matches_encoders():
    e = engine()
    travel = run(e, 0.2, 0.2, 2.0)
    p = e.pose()
    assert 0.3 < p.x < 0.4                      # 0.2 m/s minus motor spin-up
    assert abs(p.y) < 0.01 and abs(p.yaw) < 0.02
    assert travel == pytest.approx([p.x, p.x], abs=0.01)   # no slip on a free floor


def test_spin_direction_and_odometry():
    e = engine()
    travel = run(e, -0.065, 0.065, 1.5)       # +1 rad/s
    yaw = e.pose().yaw
    assert 1.0 < yaw < 1.5
    assert (travel[1] - travel[0]) / 0.13 == pytest.approx(yaw, abs=0.05)


def test_pushing_a_wall_slips_the_wheels_and_odom_drifts():
    world = World.from_dict({"shapes": [{"type": "wall", "x1": 0.2, "y1": -1, "x2": 0.2, "y2": 1}]})
    e = engine(world)
    travel = run(e, 0.3, 0.3, 3.0)
    x = e.pose().x
    assert x < 0.2                              # never through the wall
    assert travel.mean() > x + 0.05             # wheels slip, encoders count on → odom drifts
    assert e.blocked


def test_lidar_matches_the_2d_world():
    world = World.load()
    e = engine(world)
    angles = np.arange(360) * math.pi / 180
    r = e.scan(angles, 0.05, 12.0)
    origin = e.data.site_xpos[e._laser]
    expected = world.raycast(origin[0], origin[1], angles, 12.0)
    assert (r > 0).all()
    err = np.abs(r - expected)
    assert np.median(err) < 0.01 and np.percentile(err, 95) < 0.03


def test_arm_servo_follows_target_and_respects_limits():
    e = engine()
    e.set_joint_targets({"arm_joint_2": 0.6, "arm_joint_1": 5.0, "head_pan": 1.0})
    for _ in range(100):
        e.step(0.01)
    q = lambda n: e.data.qpos[e.model.jnt_qposadr[e.model.joint(n).id]]  # noqa: E731
    assert q("arm_joint_2") == pytest.approx(0.6, abs=0.05)
    assert q("arm_joint_1") == pytest.approx(math.pi / 2, abs=0.05)


def test_meshes_load_when_description_is_found():
    from smabo_sim.engines.mujoco_model import find_description
    if find_description() is None:
        pytest.skip("smabo_description not checked out next to smabo-sim")
    cfg = Config()
    e = create_engine("mujoco", World.load(), cfg)
    assert e.model.nmesh > 0


def test_sim_robot_drives_the_mujoco_engine():
    sent = []
    t = [0.0]
    e = engine()
    robot = SimRobot(e.cfg, e, sent.append, clock=lambda: t[0], wall=lambda: 1000.0)
    for _ in range(100):
        robot._on_publish("/cmd_vel", {"linear": {"x": 0.2}, "angular": {"z": 0.0}})
        robot.physics_step(0.01)
        t[0] += 0.01
    robot.publish_wheel_vel()
    robot.publish_scan()
    assert e.pose().x > 0.1
    topics = [s for s in sent if '"/esp32/wheel_vel"' in s or '"/esp32/scan"' in s]
    assert len(topics) == 2
