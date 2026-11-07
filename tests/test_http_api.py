"""REST API keeps the ESP32 contract smabo-web relies on."""

import asyncio

from aiohttp.test_utils import TestClient, TestServer

from smabo_sim.config import Config
from smabo_sim.engines import create_engine
from smabo_sim.http_api import build_app
from smabo_sim.robot import SimRobot
from smabo_sim.world import World


def with_client(fn):
    async def go():
        cfg = Config()
        robot = SimRobot(cfg, create_engine("kinematic", World.load(), cfg), lambda s: None)
        async with TestClient(TestServer(build_app(robot))) as client:
            await fn(client, robot)
    asyncio.run(go())


def test_get_config_returns_esp32_schema_with_cors():
    async def t(client, _robot):
        r = await client.get("/config")
        assert r.status == 200 and r.headers["Access-Control-Allow-Origin"] == "*"
        body = await r.json()
        assert body["dc"]["wheel_separation"] == 0.13
        assert set(body["modes"]) == {"servos", "dc_drive", "encoder_drive", "lidar"}
    with_client(t)


def test_post_config_and_mode_answer_ok():
    async def t(client, robot):
        r = await client.post("/config", json={"dc": {"max_linear": 0.2}})
        assert r.status == 200 and await r.json() == {"ok": True}
        assert robot.cfg.get("dc.max_linear") == 0.2
        r = await client.post("/mode", json={"lidar": False})
        assert await r.json() == {"ok": True} and robot.cfg.get("modes.lidar") is False
    with_client(t)


def test_bad_json_and_unknown_path():
    async def t(client, _robot):
        r = await client.post("/config", data="not json", headers={"Content-Type": "application/json"})
        assert r.status == 400 and await r.json() == {"error": "bad json"}
        r = await client.get("/nope")
        assert r.status == 404 and await r.json() == {"error": "not found"}
        assert r.headers["Access-Control-Allow-Origin"] == "*"
    with_client(t)


def test_cors_preflight():
    async def t(client, _robot):
        r = await client.options("/config")
        assert r.status == 204 and "POST" in r.headers["Access-Control-Allow-Methods"]
    with_client(t)
