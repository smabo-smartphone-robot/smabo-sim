"""Vision targets: pictures and balls for smabo-brain's vision modes."""

import asyncio
import math

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from smabo_sim.config import Config
from smabo_sim.engines import create_engine
from smabo_sim.http_api import build_app
from smabo_sim.robot import SimRobot
from smabo_sim.world import World


def test_bundled_room_has_one_target_per_vision_mode():
    w = World.load()
    kinds = {t["name"]: t["kind"] for t in w.targets}
    assert kinds == {"face": "image", "aruco_0": "image", "qr": "image", "red_ball": "color"}
    for t in w.targets:
        # default yaw: the picture faces the start pose
        assert math.cos(t["yaw"] - math.atan2(-t["y"], -t["x"])) == pytest.approx(1.0)


def test_target_parsing_and_errors(tmp_path):
    w = World.from_dict({"targets": [{"name": "b", "color": [10, 20, 30], "shape": "box", "x": 2,
                                      "yaw_deg": 90, "sway": {"on": True, "period": 0.1}}]})
    t = w.target("b")
    assert t["color"] == (10, 20, 30) and t["yaw"] == pytest.approx(math.pi / 2)
    assert t["sway"]["on"] and t["sway"]["period"] == 0.5         # clamped
    with pytest.raises(ValueError):
        World.from_dict({"targets": [{"color": "mauve"}]})
    with pytest.raises(ValueError):
        World.from_dict({"targets": [{"image": "missing.png"}]}, base_dir=str(tmp_path))
    with pytest.raises(ValueError):
        World.from_dict({"targets": [{"name": "a"}, {"name": "a"}]})


def mujoco_engine(**sim):
    pytest.importorskip("mujoco")
    cfg = Config(overrides={"sim": {"description": "none", "model": "base", **sim}})
    return create_engine("mujoco", World.load(), cfg)


def test_targets_sway_and_move():
    e = mujoco_engine()
    start = {t["name"]: (t["x"], t["y"]) for t in e.targets()}
    e.set_target("red_ball", sway_on=True, sway_amplitude=0.2, sway_period=2.0)
    for _ in range(50):                      # a quarter period: full amplitude
        e.step(0.01)
    ball = next(t for t in e.targets() if t["name"] == "red_ball")
    assert math.hypot(ball["x"] - start["red_ball"][0], ball["y"] - start["red_ball"][1]) == pytest.approx(0.2, abs=0.02)
    e.set_target("face", x=1.5, y=0.0)
    face = next(t for t in e.targets() if t["name"] == "face")
    assert (face["x"], face["y"]) == pytest.approx((1.5, 0.0))
    with pytest.raises(KeyError):
        e.set_target("nope", x=1)


def test_targets_keep_their_state_across_engines():
    """A model switch builds a new engine on the same World."""
    pytest.importorskip("mujoco")
    world = World.load()
    cfg = Config(overrides={"sim": {"description": "none", "model": "base"}})
    create_engine("mujoco", world, cfg).set_target("qr", sway_on=True)
    assert next(t for t in create_engine("mujoco", world, cfg).targets() if t["name"] == "qr")["sway"]["on"]


def test_phone_camera_sees_the_targets():
    cv2 = pytest.importorskip("cv2")         # optional: the simulator itself does not need OpenCV
    e = mujoco_engine()
    e.step(0.05)
    bgr = cv2.cvtColor(e.render_phone_camera(), cv2.COLOR_RGB2BGR)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    _corners, ids, _ = det.detectMarkers(bgr)
    assert ids is not None and 0 in ids.ravel()
    ok, texts, _pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(bgr)
    assert ok and "smabo" in texts
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    assert len(faces.detectMultiScale(gray, 1.2, 5, minSize=(40, 40))) >= 1


def test_phone_camera_shows_the_red_ball():
    e = mujoco_engine()
    e.step(0.05)
    rgb = e.render_phone_camera().astype(int)
    red = (rgb[..., 0] > 150) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)
    assert red.sum() > 200


def test_view_targets_routes():
    cfg = Config()
    robot = SimRobot(cfg, create_engine("kinematic", World.load(), cfg), lambda s: None)

    async def go():
        async with TestClient(TestServer(build_app(robot))) as client:
            j = await (await client.get("/view/targets")).json()
            assert [t["name"] for t in j["targets"]] == ["face", "aruco_0", "qr", "red_ball"]
            r = await client.post("/view/targets/qr", json={"x": 1.2, "sway": {"on": True}})
            qr = next(t for t in (await r.json())["targets"] if t["name"] == "qr")
            assert qr["x"] == 1.2 and qr["sway"]["on"] is True
            assert (await client.post("/view/targets/nope", json={"x": 1})).status == 404
            assert (await client.post("/view/targets/qr", json={"x": "far"})).status == 400
    asyncio.run(go())
