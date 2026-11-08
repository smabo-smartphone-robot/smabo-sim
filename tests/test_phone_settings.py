"""The phone's REST /config: smabo-app's settings contract, on its own port."""

import asyncio
import json

from aiohttp.test_utils import TestClient, TestServer

from smabo_sim.config import Config
from smabo_sim.engines import create_engine
from smabo_sim.http_api import build_app, build_phone_app
from smabo_sim.phone import SimPhone
from smabo_sim.phone.settings import PhoneSettings
from smabo_sim.robot import SimRobot
from smabo_sim.world import World


def make(sent=None, changes=None):
    cfg = Config(overrides={"sim": {"seed": 1}})
    sent = [] if sent is None else sent
    phone = SimPhone(cfg, create_engine("kinematic", World.load(), cfg), sent.append)
    settings = PhoneSettings(cfg, phone, 8081, on_change=(changes.append if changes is not None else None),
                             version="0.1.0")
    phone.settings_json = settings.to_json
    return cfg, phone, settings


def test_settings_tree_follows_the_app_schema():
    _cfg, _phone, s = make()
    j = s.to_json()
    assert set(j) == {"brain", "http", "sensors", "voice", "speech", "face", "app"}
    assert j["http"] == {"port": 8081}
    assert j["sensors"]["imu"] == {"enabled": True, "rate_hz": 20}
    assert j["face"]["expressions"][:3] == ["Normal", "Sleepy", "Angry"]
    assert j["speech"]["tts"] is True and j["face"]["expression"] == "Normal"
    assert j["app"]["schema"] == 1


def test_patch_updates_config_ignores_unknown_and_read_only():
    changes = []
    cfg, phone, s = make(changes=changes)
    ignored = s.apply({"sensors": {"imu": {"rate_hz": 500, "enabled": False}},
                       "speech": {"bubble": "yes"}, "brain": {"host": "x"}, "nope": 1,
                       "face": {"expression": "ANGRY"}})
    assert cfg.get("phone.imu_rate") == 200 and cfg.get("phone.imu_enabled") is False   # clamped
    assert phone.face.expression.name == "Angry"
    assert s.apply({"face": {"expression": 3}}) == ["face.expression"]          # names only
    assert s.apply({"face": {"expression": "nope"}}) == ["face.expression"]
    assert sorted(ignored) == ["brain.host", "nope", "speech.bubble"]
    assert changes == [{"sensors.imu.rate_hz", "sensors.imu.enabled", "face.expression"}]
    assert s.apply({"sensors": {"imu": {"enabled": False}}}) == [] and len(changes) == 1   # no change, no notice


def test_settings_change_behaviour():
    sent = []
    cfg, phone, s = make(sent)
    s.apply({"sensors": {"imu": {"enabled": False}}})
    phone.publish_imu(0.05)
    assert not any("/app/imu/data" in f for f in sent)
    say = lambda t: phone.on_message(json.dumps({"op": "publish", "topic": "/speech/say", "msg": {"data": t}}))  # noqa: E731
    s.apply({"speech": {"bubble": False, "tts": False}})
    say("やあ")
    assert phone.face.state.bubble in ("", None)
    s.apply({"speech": {"bubble": True}})
    say("やあ")                                   # a repeat: ignored
    assert phone.face.state.bubble in ("", None)
    say("こんにちは")
    assert phone.face.state.bubble == "こんにちは"


def test_phone_announces_its_rest_config_on_connect():
    sent = []
    _cfg, phone, _s = make(sent)
    phone.config_url = "http://localhost:8081"
    phone.on_brain_connect()
    frames = [json.loads(f) for f in sent]
    assert {"op": "advertise", "topic": "/app/phone/config", "type": "std_msgs/msg/String"} in frames
    pubs = {f["topic"]: f["msg"]["data"] for f in frames if f.get("op") == "publish"}
    assert pubs["/app/phone/config_url"] == "http://localhost:8081"
    assert json.loads(pubs["/app/phone/config"])["http"]["port"] == 8081


def test_phone_rest_api():
    async def go():
        _cfg, phone, s = make()
        async with TestClient(TestServer(build_phone_app(s))) as client:
            r = await client.get("/config")
            assert r.status == 200 and r.headers["Access-Control-Allow-Origin"] == "*"
            assert (await r.json())["app"]["name"] == "smabo-sim"
            r = await client.post("/config", json={"voice": {"wake_word": "ねえ"}, "app": {"name": "x"}})
            assert await r.json() == {"ok": True, "ignored": ["app.name"]}
            assert (await (await client.get("/config")).json())["voice"]["wake_word"] == "ねえ"
            assert (await client.post("/config", data="{")).status == 400
            assert (await client.options("/config")).status == 204
            assert (await client.get("/mode")).status == 404
    asyncio.run(go())


def test_esp32_rest_has_no_phone_or_sim_keys():
    async def go():
        cfg = Config()
        notices = []
        robot = SimRobot(cfg, create_engine("kinematic", World.load(), cfg), notices.append)
        async with TestClient(TestServer(build_app(robot))) as client:
            body = await (await client.get("/config")).json()
            assert not {"phone", "sim", "view"} & set(body) and "dc" in body
            await client.post("/config", json={"phone": {"imu_rate": 5}, "dc": {"max_linear": 0.2}})
            assert cfg.get("phone.imu_rate") == 20.0 and cfg.get("dc.max_linear") == 0.2
            assert any("not ESP32 settings" in n for n in notices)
    asyncio.run(go())
