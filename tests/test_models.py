"""Robot models per roadmap step (smabo_sim/models.py) and switching them."""

import asyncio
import json

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from smabo_sim.config import Config
from smabo_sim.engines import create_engine
from smabo_sim.http_api import build_app
from smabo_sim.models import MODELS, get_model
from smabo_sim.robot import SimRobot
from smabo_sim.world import World

QUIET = {"encoder_noise": 0.0, "lidar_noise": 0.0, "description": "none"}


def cfg_for(name: str) -> Config:
    cfg = Config(overrides={"sim": {**QUIET, "model": name}})
    cfg.update(get_model(name).config_patch(cfg.get("servos.joints")))
    return cfg


def drive(engine, seconds=1.0):
    engine.set_wheel_targets(0.1, 0.1)
    for _ in range(int(seconds / 0.01)):
        engine.step(0.01)
    return engine.pose()


def test_lookup_by_name_or_roadmap_id():
    assert get_model("encmobile").encoder
    assert get_model("plan").name == "arm"          # 動作計画 uses the arm robot
    assert get_model("imgproc").name == "base"
    assert get_model(None).name == "full"
    with pytest.raises(ValueError):
        get_model("tank")


def test_config_patch_sets_modes_and_enables_only_its_servos():
    cfg = cfg_for("neck")
    assert cfg.get("sim.model") == "neck"
    assert cfg.get("modes") == {"servos": True, "dc_drive": False, "encoder_drive": False, "lidar": False}
    enabled = {n for n, j in cfg.get("servos.joints").items() if j["enabled"]}
    assert enabled == {"head_pan"}
    assert cfg_for("mobile").get("modes.dc_drive") is True
    assert cfg_for("nav").get("modes") == {"servos": False, "dc_drive": False,
                                           "encoder_drive": True, "lidar": True}


@pytest.mark.parametrize("name", list(MODELS))
def test_kinematic_moves_and_scans_only_with_the_hardware(name):
    m = MODELS[name]
    e = create_engine("kinematic", World.load(), cfg_for(name))
    p = drive(e)
    assert (p.x > 0.05) == m.mobile
    assert (e.scan(np.linspace(0, 6.28, 16), 0.05, 12.0) > 0).any() == m.lidar


@pytest.mark.parametrize("name", list(MODELS))
def test_mujoco_builds_every_model_with_its_joints(name):
    pytest.importorskip("mujoco")
    import mujoco
    m = MODELS[name]
    e = create_engine("mujoco", World.load(), cfg_for(name))
    joints = {e.model.joint(i).name for i in range(e.model.njnt)}
    for servo in ("arm_joint_1", "left_hand", "head_pan"):
        assert (servo in joints) == (servo in m.servos())
    assert ("left_wheel_joint" in joints) == m.mobile
    assert (mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_SITE, "laser") >= 0) == m.lidar
    p = drive(e)
    assert (p.x > 0.05) == m.mobile
    if not m.mobile:
        assert e.read_wheel_distance() == (0.0, 0.0)


def test_mujoco_neck_turns_the_base_and_hands_move():
    pytest.importorskip("mujoco")
    e = create_engine("mujoco", World.load(), cfg_for("neck"))
    e.set_joint_targets({"head_pan": 0.6})
    for _ in range(150):
        e.step(0.01)
    assert e.pose().yaw == pytest.approx(0.6, abs=0.05)

    e = create_engine("mujoco", World.load(), cfg_for("hand"))
    e.set_joint_targets({"left_hand": 1.0, "right_hand": -1.0})
    for _ in range(150):
        e.step(0.01)
    q = e.data.qpos
    assert q[e.model.joint("left_hand").qposadr[0]] == pytest.approx(1.0, abs=0.05)
    assert q[e.model.joint("right_hand").qposadr[0]] == pytest.approx(-1.0, abs=0.05)


def test_apply_model_updates_config_and_tells_the_brain():
    sent = []
    cfg = Config(overrides={"sim": QUIET})
    robot = SimRobot(cfg, create_engine("kinematic", World.load(), cfg), sent.append)
    robot.apply_model(get_model("hand"))
    frames = [json.loads(s) for s in sent]
    assert any(f.get("op") == "notice" and "hand" in f["message"] for f in frames)
    snap = next(json.loads(f["msg"]["data"]) for f in frames if f.get("topic") == "/esp32/config")
    assert snap["sim"]["model"] == "hand" and snap["modes"]["servos"] is True
    assert snap["modes"]["encoder_drive"] is False


def test_view_model_routes():
    switched = []

    def switch(name):
        model = get_model(name)          # raises ValueError for unknown names
        cfg.update({"sim": {"model": model.name}})
        switched.append(model.name)
        return model

    cfg = Config(overrides={"sim": QUIET})
    robot = SimRobot(cfg, create_engine("kinematic", World.load(), cfg), lambda s: None)

    async def go():
        async with TestClient(TestServer(build_app(robot, None, switch))) as client:
            j = await (await client.get("/view/models")).json()
            assert j["current"] == "full" and j["switchable"]
            assert [m["name"] for m in j["models"]] == list(MODELS)
            r = await client.post("/view/model", json={"name": "neck"})
            assert r.status == 200 and (await r.json())["current"] == "neck"
            r = await client.post("/view/model", json={"name": "tank"})
            assert r.status == 400
        async with TestClient(TestServer(build_app(robot))) as client:
            r = await client.post("/view/model", json={"name": "arm"})
            assert r.status == 409
    asyncio.run(go())
    assert switched == ["neck"]
