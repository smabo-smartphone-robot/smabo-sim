"""SimRobot speaks the ESP32 protocol: frames in, frames out."""

import json
import math

import pytest

from smabo_sim.config import Config
from smabo_sim.engines import create_engine
from smabo_sim.robot import SimRobot
from smabo_sim.world import World


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(overrides=None, world=None):
    cfg = Config(overrides={"sim": {"encoder_noise": 0.0, "lidar_noise": 0.0, "seed": 1,
                                    "motor_time_constant": 1e-6}, **(overrides or {})})
    world = world or World.from_dict({"shapes": [{"type": "room", "x": 0, "y": 0, "w": 10, "h": 10}]})
    out: list[dict] = []
    clock = Clock()
    robot = SimRobot(cfg, create_engine("kinematic", world, cfg),
                     lambda s: out.append(json.loads(s)), clock=clock, wall=lambda: 1000.25)
    return robot, out, clock


def run(robot, clock, seconds, dt=0.01):
    for _ in range(int(round(seconds / dt))):
        clock.t += dt
        robot.physics_step(dt)


def publish(robot, topic, msg):
    robot.on_message(json.dumps({"op": "publish", "topic": topic, "msg": msg}))


def test_rosbridge_connect_advertises_and_subscribes_like_the_esp32():
    robot, out, _ = make()
    robot.on_brain_connect()
    adv = {(f["topic"], f["type"]) for f in out if f["op"] == "advertise"}
    sub = {(f["topic"], f["type"]) for f in out if f["op"] == "subscribe"}
    assert ("/esp32/wheel_vel", "smabo_interfaces/msg/WheelVel") in adv
    assert ("/esp32/scan", "sensor_msgs/msg/LaserScan") in adv
    assert ("/cmd_vel", "geometry_msgs/msg/Twist") in sub
    assert ("/servo/command", "trajectory_msgs/msg/JointTrajectory") in sub
    assert robot.ws_path() == "/"


def test_connect_publishes_config_without_wifi():
    robot, out, _ = make()
    robot.on_brain_connect()
    pub = [f for f in out if f["op"] == "publish" and f["topic"] == "/esp32/config"]
    assert len(pub) == 1
    cfg = json.loads(pub[0]["msg"]["data"])
    assert cfg["dc"]["wheel_separation"] == 0.13
    assert "wifi" not in cfg
    # advertise comes before the publish (rosbridge requires it)
    assert out.index(pub[0]) > max(i for i, f in enumerate(out) if f["op"] == "advertise")


def test_cmd_vel_drives_and_wheel_vel_reports_speed():
    robot, out, clock = make()
    publish(robot, "/cmd_vel", {"linear": {"x": 0.2}, "angular": {"z": 0.0}})
    robot._last_wheel_pub = clock.t
    run(robot, clock, 0.2)
    robot.publish_wheel_vel()
    wv = [f for f in out if f.get("topic") == "/esp32/wheel_vel"][-1]["msg"]
    assert wv["left"] == pytest.approx(0.2, rel=1e-3) and wv["right"] == pytest.approx(0.2, rel=1e-3)
    assert wv["dt"] == pytest.approx(0.2)
    assert robot.engine.pose().x == pytest.approx(0.04, rel=1e-2)


def test_robot_stops_after_cmd_timeout():
    robot, _, clock = make()
    publish(robot, "/cmd_vel", {"linear": {"x": 0.2}})
    run(robot, clock, 2.0)
    assert robot.engine.pose().x == pytest.approx(0.1, abs=0.01)   # 0.5 s of motion


def test_wall_blocks_translation_but_encoders_keep_counting():
    world = World.from_dict({"shapes": [{"type": "wall", "x1": 0.15, "y1": -1, "x2": 0.15, "y2": 1}]})
    robot, _, clock = make(world=world)
    for _ in range(10):
        publish(robot, "/cmd_vel", {"linear": {"x": 0.2}})
        run(robot, clock, 0.1)
    assert robot.engine.pose().x < 0.07
    d_l, _ = robot.engine.read_wheel_distance()
    assert d_l == pytest.approx(0.2, rel=0.05)


def test_drive_ignored_when_drive_modes_off():
    robot, _, clock = make({"modes": {"encoder_drive": False, "dc_drive": False}})
    publish(robot, "/cmd_vel", {"linear": {"x": 0.2}})
    run(robot, clock, 0.3)
    assert robot.engine.pose().x == pytest.approx(0.0)


def test_servo_command_moves_joint_states():
    robot, out, clock = make()
    publish(robot, "/servo/command", {"joint_names": ["arm_joint_1"],
                                      "points": [{"positions": [math.radians(30)],
                                                  "time_from_start": {"sec": 0, "nanosec": 0}}]})
    run(robot, clock, 1.0)
    robot.publish_joint_states()
    js = [f for f in out if f.get("topic") == "/esp32/joint_states"][-1]["msg"]
    pos = dict(zip(js["name"], js["position"]))
    assert math.degrees(pos["arm_joint_1"]) == pytest.approx(30.0)
    assert js["header"]["stamp"] == {"sec": 1000, "nanosec": 250000000}


def test_scan_matches_ld06_message_shape_and_sees_walls():
    robot, out, _ = make()
    robot.publish_scan()
    msg = [f for f in out if f.get("topic") == "/esp32/scan"][-1]["msg"]
    assert msg["header"]["frame_id"] == "laser"
    assert len(msg["ranges"]) == 360 and msg["angle_min"] == 0.0
    assert msg["ranges"][0] == pytest.approx(5.0, abs=1e-3)      # +x wall
    assert msg["ranges"][90] == pytest.approx(5.0, abs=1e-3)     # +y wall (CCW)


def test_scan_off_when_lidar_mode_off():
    robot, out, _ = make({"modes": {"lidar": False}})
    robot.publish_scan()
    assert not [f for f in out if f.get("topic") == "/esp32/scan"]


def test_ping_is_echoed_on_pong():
    robot, out, _ = make()
    publish(robot, "/ping", {"data": "tok-1"})
    assert out[-1] == {"op": "publish", "topic": "/esp32/pong", "msg": {"data": "tok-1"}}


def test_set_mode_enforces_encoder_over_dc():
    robot, _, _ = make()
    robot.set_mode({"dc_drive": True})
    assert robot.cfg.get("modes.dc_drive") is False and robot.cfg.get("modes.encoder_drive") is True


def test_set_config_rebuilds_servos_keeping_position():
    robot, _, clock = make()
    publish(robot, "/servo/command", {"joint_names": ["head_pan"], "points": [{"positions": [math.radians(40)]}]})
    run(robot, clock, 1.0)
    robot.set_config({"servos": {"joints": {"head_pan": {"max_speed": 30}}}})
    assert math.degrees(robot.servos.state()["head_pan"]) == pytest.approx(40.0)
    assert robot.cfg.get("servos.joints.head_pan.max_speed") == 30
