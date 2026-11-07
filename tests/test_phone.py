import asyncio
import json

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from smabo_sim.config import Config
from smabo_sim.engines import create_engine
from smabo_sim.http_api import build_app
from smabo_sim.phone import SimPhone, device_axes
from smabo_sim.phone.face import EXPRESSIONS, Face, gaze_from_look_at
from smabo_sim.robot import SimRobot
from smabo_sim.world import World


# ---- face --------------------------------------------------------------- #
def test_look_at_projection_matches_the_app():
    # parseLookAt: right follows -y, down follows -z, relative to forward distance
    assert gaze_from_look_at(1.0, -0.5, 0.25) == pytest.approx((0.5, -0.25))
    assert gaze_from_look_at(0.0, 1.0, 0.0) == (-1.0, 0.0)       # clamped, forward floor 0.05


def test_gaze_eases_toward_look_at_target():
    f = Face(seed=0)
    f.look_at(1.0, -0.6, 0.0)
    f.tick(0.05)
    assert 0.0 < f.state.gaze[0] < 0.6                           # k = 8/s, not a jump
    for _ in range(100):
        f.tick(0.02)
    assert f.state.gaze[0] == pytest.approx(0.6, abs=1e-3)
    assert f.following


def test_random_saccades_resume_without_look_at():
    f = Face(seed=0)
    f.look_at(1.0, 0.0, 0.0)
    for _ in range(int(4.0 / 0.05)):
        f.tick(0.05)
    assert not f.following


def test_blinks_only_for_blinking_expressions():
    f = Face(seed=0)
    closed = []
    for _ in range(int(8.0 / 0.01)):
        f.tick(0.01)
        closed.append(f.state.blink < 0.5)
    assert any(closed)
    f.set_expression(4)                                          # Smile: no blink
    f2 = []
    for _ in range(int(8.0 / 0.01)):
        f.tick(0.01)
        f2.append(f.state.blink)
    assert min(f2) == 1.0


def test_unknown_expression_falls_back_to_normal_and_all_render():
    f = Face(320, 160, seed=0)
    f.set_expression(99)
    assert f.state.expression == 1
    for i in EXPRESSIONS:
        f.set_expression(i)
        img = f.render()
        assert img.shape == (160, 320, 3) and img.dtype == np.uint8
        assert img.max() > 0


def test_speech_bubble_shows_then_expires():
    f = Face(seed=0)
    f.say("hello")
    plain = Face(seed=0).render()
    assert not np.array_equal(f.render(), plain)
    for _ in range(int(4.0 / 0.1)):
        f.tick(0.1)
    assert f.state.bubble == ""


# ---- SimPhone ----------------------------------------------------------- #
def kinematic_phone(sent):
    cfg = Config(overrides={"sim": {"encoder_noise": 0.0}})
    engine = create_engine("kinematic", World.load(), cfg)
    return SimPhone(cfg, engine, sent.append, wall=lambda: 1000.5), engine, cfg


def test_phone_subscribes_like_the_app_and_handles_topics():
    sent = []
    phone, _, _ = kinematic_phone(sent)
    phone.on_brain_connect()
    ops = [json.loads(s) for s in sent]
    assert {"op": "advertise", "topic": "/app/imu/data", "type": "sensor_msgs/msg/Imu"} in ops
    assert {"op": "advertise", "topic": "/app/gps/fix", "type": "sensor_msgs/msg/NavSatFix"} in ops
    assert {o["topic"] for o in ops if o["op"] == "subscribe"} == {
        "/expression", "/look_at", "/speech/say", "/webrtc/answer"}
    phone.on_message(json.dumps({"op": "publish", "topic": "/expression", "msg": {"data": 3}}))
    phone.on_message(json.dumps({"op": "publish", "topic": "/look_at", "msg": {
        "header": {}, "pose": {"position": {"x": 1.0, "y": 0.5, "z": 0.0}}}}))
    phone.on_message(json.dumps({"op": "publish", "topic": "/speech/say", "msg": {"data": "やあ"}}))
    assert phone.face.state.expression == 3
    assert phone.face.state.target == pytest.approx([-0.5, 0.0])
    assert phone.face.state.bubble == "やあ"


def test_device_axes_put_gravity_where_a_landscape_phone_feels_it():
    # camera (device top) on the robot's left: device x points down
    assert device_axes("robot_left") @ np.array([0, 0, 9.81]) == pytest.approx([-9.81, 0, 0])
    assert device_axes("robot_right") @ np.array([0, 0, 9.81]) == pytest.approx([9.81, 0, 0])


def test_imu_message_from_kinematic_motion():
    sent = []
    phone, engine, _ = kinematic_phone(sent)
    engine.set_wheel_targets(-0.065, 0.065)                      # spin +1 rad/s
    for _ in range(50):
        engine.step(0.01)
    phone.publish_imu(0.05)
    for _ in range(5):
        engine.step(0.01)
    phone.publish_imu(0.05)
    msg = json.loads(sent[-1])
    assert msg["topic"] == "/app/imu/data"
    m = msg["msg"]
    assert m["header"]["frame_id"] == "imu_link"
    assert m["linear_acceleration"]["x"] == pytest.approx(-9.81, abs=0.1)
    # device x points down (top on the robot's left), so a CCW yaw reads negative on x
    assert m["angular_velocity"]["x"] == pytest.approx(-1.0, abs=0.15)


mujoco = pytest.importorskip("mujoco")


def mujoco_phone():
    cfg = Config(overrides={"sim": {"description": "none", "encoder_noise": 0.0}})
    engine = create_engine("mujoco", World.load(), cfg)
    return SimPhone(cfg, engine, [].append), engine


def test_mujoco_phone_screen_camera_and_imu():
    phone, engine = mujoco_phone()
    for _ in range(30):
        engine.step(0.01)
    acc, gyro = engine.phone_imu()
    assert acc == pytest.approx([-9.81, 0, 0], abs=0.15)
    assert np.abs(gyro).max() < 0.05
    phone.update_screen(0.03)
    adr = engine.model.tex_adr[engine.screen_tex]
    h, w, n = engine._screen_shape
    tex = engine.model.tex_data[adr:adr + h * w * n].reshape(h, w, n)[..., :3]
    assert np.array_equal(tex, phone.last_screen)
    img = engine.render_phone_camera()
    assert img.shape == (480, 640, 3)
    assert img.std() > 5                                         # sees the room, not a blank frame


def test_phone_http_routes():
    async def go():
        phone, engine = mujoco_phone()
        phone.update_screen(0.03)
        robot = SimRobot(engine.cfg, engine, [].append)
        async with TestClient(TestServer(build_app(robot, phone))) as c:
            r = await c.get("/phone/screen.png")
            assert r.status == 200 and (await r.read())[:4] == b"\x89PNG"
            r = await c.get("/phone/camera.jpg")
            assert r.status == 200 and (await r.read())[:2] == b"\xff\xd8"
            r = await c.get("/phone")
            assert "camera.mjpg" in await r.text()
    asyncio.run(go())


def test_camera_route_explains_kinematic_engine():
    async def go():
        sent = []
        phone, engine, cfg = kinematic_phone(sent)
        robot = SimRobot(cfg, engine, sent.append)
        async with TestClient(TestServer(build_app(robot, phone))) as c:
            r = await c.get("/phone/camera.jpg")
            assert r.status == 404 and "mujoco" in (await r.json())["error"]
    asyncio.run(go())




def test_camera_streams_to_a_brain_style_webrtc_peer():
    """CameraStreamer ↔ an aiortc answerer built like brain.webrtc_hub."""
    pytest.importorskip("aiortc")
    from aiortc import RTCPeerConnection, RTCSessionDescription

    from smabo_sim.phone.webrtc import CameraStreamer

    async def go():
        img = np.zeros((120, 160, 3), np.uint8)
        img[:, :80] = (255, 0, 0)
        offers = []
        cam = CameraStreamer(lambda: img, lambda topic, msg: offers.append((topic, msg)), fps=15)
        brain = RTCPeerConnection()
        got = asyncio.get_running_loop().create_future()

        @brain.on("track")
        def on_track(track):
            async def pull():
                frame = await track.recv()
                if not got.done():
                    got.set_result(frame.to_ndarray(format="rgb24"))
            asyncio.ensure_future(pull())

        await cam.start()
        topic, msg = offers[-1]
        assert topic == "/webrtc/offer"
        offer = json.loads(msg["data"])
        await brain.setRemoteDescription(RTCSessionDescription(offer["sdp"], offer["type"]))
        answer = await brain.createAnswer()
        await brain.setLocalDescription(answer)
        await cam.handle_answer({"data": json.dumps({"sdp": brain.localDescription.sdp,
                                                     "type": brain.localDescription.type})})
        frame = await asyncio.wait_for(got, timeout=20)
        assert frame.shape == (120, 160, 3)
        assert frame[60, 20, 0] > 200 and frame[60, 140, 0] < 60   # red left half survives encoding
        await cam.close()
        await brain.close()

    asyncio.run(go())


def test_camera_offers_again_when_the_brain_never_answers(monkeypatch):
    """An offer lost while the brain restarts must not leave the camera stuck."""
    pytest.importorskip("aiortc")
    from smabo_sim.phone import webrtc

    monkeypatch.setattr(webrtc, "ANSWER_TIMEOUT", 0.3)

    async def go():
        offers = []
        cam = webrtc.CameraStreamer(lambda: np.zeros((32, 32, 3), np.uint8),
                                    lambda topic, msg: offers.append(topic), fps=5)
        await cam.start()
        for _ in range(100):
            if len(offers) >= 2:
                break
            await asyncio.sleep(0.05)
        await cam.close()
        assert offers[:2] == ["/webrtc/offer", "/webrtc/offer"]

    asyncio.run(go())


# ---- GPS ----------------------------------------------------------------- #
def test_gps_maps_world_east_north_to_lat_lon():
    from smabo_sim.world import Pose2D
    sent = []
    phone, engine, cfg = kinematic_phone(sent)
    lat0, lon0 = cfg.get("phone.gps.origin.latitude"), cfg.get("phone.gps.origin.longitude")
    engine.reset(Pose2D(0.0, 0.0, 0.0))
    assert phone.gps_fix(noise=False)[:2] == pytest.approx((lat0, lon0))
    engine.reset(Pose2D(0.0, 111.32, 0.0))                       # ~0.001° north
    lat, lon, _ = phone.gps_fix(noise=False)
    assert lat - lat0 == pytest.approx(0.0010, rel=0.01) and lon == pytest.approx(lon0)
    engine.reset(Pose2D(100.0, 0.0, 0.0))                        # east: longitude grows
    lat, lon, _ = phone.gps_fix(noise=False)
    assert lat == pytest.approx(lat0) and lon > lon0
    cfg.update({"phone": {"gps": {"map_rotation_deg": 90.0}}})   # world x now points north
    lat, lon, _ = phone.gps_fix(noise=False)
    assert lat > lat0 and lon == pytest.approx(lon0)


def test_gps_publishes_a_navsatfix_like_the_app():
    sent = []
    phone, _, cfg = kinematic_phone(sent)
    phone.publish_gps()
    frame = json.loads(sent[-1])
    assert frame["op"] == "publish" and frame["topic"] == "/app/gps/fix"
    m = frame["msg"]
    assert m["header"]["frame_id"] == "gps" and m["header"]["stamp"] == {"sec": 1000, "nanosec": 500000000}
    assert m["status"] == {"status": 0, "service": 1} and m["position_covariance_type"] == 1
    assert m["position_covariance"][0] == pytest.approx(9.0)    # accuracy 3 m
    lat0 = cfg.get("phone.gps.origin.latitude")
    assert abs(m["latitude"] - lat0) < 30 / 111320               # noise of a few metres
    cfg.update({"phone": {"gps": {"enabled": False}}})
    n = len(sent)
    phone.publish_gps()
    assert len(sent) == n
