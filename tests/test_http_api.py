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


def test_rest_port_has_only_the_boards_endpoints():
    """The board has no /view or /phone: those live on the simulator's own port."""
    async def t(client, _robot):
        for path in ("/view", "/view/state", "/phone"):
            r = await client.get(path)
            assert r.status == 404 and await r.json() == {"error": "not found"}
    with_client(t)


def test_sim_pages_on_their_own_app():
    from smabo_sim.http_api import build_sim_app

    async def go():
        cfg = Config()
        robot = SimRobot(cfg, create_engine("kinematic", World.load(), cfg), lambda s: None)
        async with TestClient(TestServer(build_sim_app(robot))) as client:
            r = await client.get("/", allow_redirects=False)
            assert r.status == 302 and r.headers["Location"] == "/view"
            assert (await client.get("/view/state")).status == 200
            assert (await client.get("/config")).status == 404     # REST stays on the board's port
    asyncio.run(go())


def test_config_url_uses_the_port_actually_opened():
    cfg = Config()
    robot = SimRobot(cfg, create_engine("kinematic", World.load(), cfg), lambda s: None)
    assert cfg.get("http.port") == 80 and cfg.get("sim.view_port") == 8090
    assert robot.config_url() == "http://localhost:80"
    robot.http_port = 8080                   # 80 couldn't be opened
    assert robot.config_url() == "http://localhost:8080"
