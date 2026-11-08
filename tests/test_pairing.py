"""Pairing like the board and the app: invite → token, REST locked to the brain."""

import asyncio
import json
import os
import stat

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from smabo_sim.config import Config
from smabo_sim.engines import create_engine
from smabo_sim.http_api import build_app
from smabo_sim.link import BrainLink
from smabo_sim.pairing import RELAY_KEY_HEADER, Credentials
from smabo_sim.robot import SimRobot
from smabo_sim.world import World


def test_credentials_pair_save_and_reload(tmp_path):
    path = tmp_path / "esp32.json"
    c = Credentials("esp32", "smabo", path)
    assert c.first_frame() is None and not c.locked
    c.set_invite("ABCD-EFGH")
    assert c.first_frame() == {"op": "pair", "code": "ABCD-EFGH", "role": "esp32", "name": "smabo"}
    assert c.handle_reply({"op": "pair", "ok": True, "token": "tok", "relay_key": "the-relay-key", "name": "smabo"})
    assert c.first_frame() == {"op": "auth", "token": "tok"}
    assert "the-relay-key" not in path.read_text()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    again = Credentials("esp32", "smabo", path)
    assert again.paired and again.relay_allowed("the-relay-key") and not again.relay_allowed("other")
    assert not again.relay_allowed(None)


def test_refused_invite_is_dropped(tmp_path):
    c = Credentials("app", "phone", tmp_path / "app.json")
    c.set_invite("X")
    assert c.handle_reply({"op": "pair", "ok": False, "reason": "expired"}) is False
    assert c.first_frame() is None and c.status() == {"paired": False, "refused": "expired"}
    assert c.handle_reply({"op": "publish", "topic": "/x"}) is None


def _rest(creds, invites):
    cfg = Config()
    robot = SimRobot(cfg, create_engine("kinematic", World.load(), cfg), lambda s: None)
    return build_app(robot, creds, invites.append), robot


def test_rest_takes_an_invite_while_unpaired_then_answers_only_the_brain(tmp_path):
    async def go():
        creds = Credentials("esp32", "smabo", tmp_path / "esp32.json")
        invites = []
        app, robot = _rest(creds, invites)
        async with TestClient(TestServer(app)) as client:
            r = await client.post("/config", json={"brain": {"invite": "ABCD-EFGH"}, "dc": {"max_linear": 0.25}})
            assert r.status == 200 and invites == ["ABCD-EFGH"]
            assert robot.cfg.get("dc.max_linear") == 0.25 and robot.cfg.get("brain.invite") is None
            body = await (await client.get("/config")).json()
            assert body["brain"]["paired"] is False

            creds.handle_reply({"op": "pair", "ok": True, "token": "tok", "relay_key": "key"})
            r = await client.get("/config")
            assert r.status == 401 and "Access-Control-Allow-Origin" not in r.headers
            assert (await client.options("/config")).status == 401
            r = await client.get("/config", headers={RELAY_KEY_HEADER: "key"})
            assert r.status == 200 and "Access-Control-Allow-Origin" not in r.headers
            assert (await r.json())["brain"]["paired"] is True
    asyncio.run(go())


def test_link_pairs_then_authenticates(tmp_path):
    """A brain stand-in that accepts one invite and then the token it issued."""
    async def go():
        seen = []
        connected = asyncio.Event()

        async def ws_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            first = json.loads((await ws.receive()).data)
            seen.append(first)
            if first["op"] == "pair":
                await ws.send_json({"op": "pair", "ok": True, "token": "tok", "relay_key": "key", "name": "x"})
                await ws.close()          # make the link reconnect with the token
                return ws
            ok = first.get("token") == "tok"
            await ws.send_json({"op": "auth", "ok": ok})
            if ok:
                connected.set()
                async for _ in ws:
                    pass
            return ws

        app = web.Application()
        app.router.add_get("/", ws_handler)
        server = TestServer(app)
        await server.start_server()
        creds = Credentials("esp32", "smabo", tmp_path / "esp32.json")
        creds.set_invite("ABCD-EFGH")
        on_connect = []
        link = BrainLink(lambda: str(server.make_url("/")).replace("http", "ws"), lambda m: None,
                         lambda: on_connect.append(1), retry=0.05, credentials=creds)
        task = asyncio.create_task(link.run())
        try:
            await asyncio.wait_for(connected.wait(), 3)
            await asyncio.sleep(0.05)
        finally:
            task.cancel()
            await server.close()
        assert [f["op"] for f in seen] == ["pair", "auth"]
        # once after pairing (the brain keeps that connection), once after the token
        assert on_connect == [1, 1]
    asyncio.run(go())


def test_link_goes_on_with_a_no_auth_brain(tmp_path, monkeypatch):
    async def go():
        import smabo_sim.link as link_mod
        monkeypatch.setattr(link_mod, "AUTH_REPLY_TIMEOUT", 0.1)
        got = []

        async def ws_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for m in ws:
                got.append(json.loads(m.data)["op"])
            return ws

        app = web.Application()
        app.router.add_get("/", ws_handler)
        server = TestServer(app)
        await server.start_server()
        creds = Credentials("esp32", "smabo", tmp_path / "esp32.json")
        creds.handle_reply({"op": "pair", "ok": True, "token": "tok", "relay_key": "key"})
        link = BrainLink(lambda: str(server.make_url("/")).replace("http", "ws"), lambda m: None,
                         lambda: link.send(json.dumps({"op": "advertise"})), credentials=creds)
        task = asyncio.create_task(link.run())
        await asyncio.sleep(0.4)
        task.cancel()
        await server.close()
        assert got[:2] == ["auth", "advertise"]
    asyncio.run(go())
