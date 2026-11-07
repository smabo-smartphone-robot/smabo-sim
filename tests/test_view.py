"""The browser view (/view): camera control and offscreen rendering."""

import asyncio

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from smabo_sim.config import Config
from smabo_sim.engines import create_engine
from smabo_sim.http_api import build_app
from smabo_sim.robot import SimRobot
from smabo_sim.view import ViewCamera
from smabo_sim.world import World


def test_camera_orbit_zoom_and_limits():
    c = ViewCamera()
    s = c.apply({"orbit": [10, 5], "zoom": 2.0}, (0.0, 0.0))
    assert s["azimuth"] == -150.0 and s["elevation"] == -35.0 and s["distance"] == 2.4
    s = c.apply({"orbit": [0, 500], "zoom": 1e-6}, (0.0, 0.0))
    assert s["elevation"] == -89.0 and s["distance"] == 0.15
    s = c.apply({"preset": "default"}, (0.0, 0.0))
    assert s["distance"] == 1.2 and s["follow"] is True


def test_pan_stops_following_and_starts_at_the_robot():
    c = ViewCamera()
    c.apply({"azimuth": 0.0, "distance": 1.0}, (0.0, 0.0))
    s = c.apply({"pan": [0.0, 0.5]}, (2.0, 1.0))
    assert s["follow"] is False
    assert s["lookat"][0] == pytest.approx(2.5) and s["lookat"][1] == pytest.approx(1.0)


def with_client(engine_name, fn):
    async def go():
        cfg = Config(overrides={"sim": {"description": "none"}, "view": {"width": 160, "height": 100}})
        robot = SimRobot(cfg, create_engine(engine_name, World.load(), cfg), lambda s: None)
        async with TestClient(TestServer(build_app(robot))) as client:
            await fn(client)
    asyncio.run(go())


def test_page_and_state_work_on_any_engine():
    async def t(client):
        r = await client.get("/view")
        assert r.status == 200 and "/view/stream.mjpg" in await r.text()
        s = await (await client.get("/view/state")).json()
        assert set(s) >= {"x", "y", "yaw"}
        r = await client.post("/view/camera", json={"orbit": [10, 0], "scan": False})
        assert (await r.json())["scan"] is False
        r = await client.get("/view/frame.jpg")
        assert r.status == 404   # kinematic has nothing to draw
    with_client("kinematic", t)


def test_mujoco_frame_is_a_jpeg():
    pytest.importorskip("mujoco")

    async def t(client):
        r = await client.get("/view/frame.jpg")
        assert r.status == 200 and r.content_type == "image/jpeg"
        body = await r.read()
        assert body[:2] == b"\xff\xd8"
    with_client("mujoco", t)


def test_render_view_size_and_camera_changes_the_image():
    pytest.importorskip("mujoco")
    cfg = Config(overrides={"sim": {"description": "none"}})
    e = create_engine("mujoco", World.load(), cfg)
    cam = ViewCamera()
    a = e.render_view(cam.state, 160, 100)
    b = e.render_view(cam.apply({"preset": "top"}, (0.0, 0.0)), 160, 100)
    assert a.shape == (100, 160, 3) and a.dtype == np.uint8
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 1.0
