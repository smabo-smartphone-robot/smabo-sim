"""The simulator's two HTTP servers.

``build_app`` — the REST config API, the same endpoints and CORS as
``smabo-esp32/http_server.py`` and nothing else, on ``http.port`` (80 like the
ESP32; 8080 where 80 can't be opened):

    GET  /config           → full live config (JSON, wifi.password shown as "****")
    POST /config  {patch}  → deep-merge a config patch
    POST /mode    {modes}  → enable/disable subsystems

``build_sim_app`` — what only the simulator has, on its own port
(``sim.view_port``, default 8090), so the REST port looks exactly like a board.
The phone (smabo-app's part):

    GET  /phone              → page showing the phone screen and front camera
    GET  /phone/screen.png   → the face currently on the phone screen
    GET  /phone/camera.jpg   → one frame from the phone's front camera (MuJoCo)
    GET  /phone/camera.mjpg  → the front camera as an MJPEG stream (MuJoCo)
    POST /phone/speech       → talk to the robot (sent on /app/speech/audio for STT):
                               {"text": "..."} synthesized to speech, {"listen": true}
                               records the host microphone, or a raw audio/wav body

The simulation itself, in a browser (no ``--viewer`` window needed; see view.py):

    GET  /                   → redirects to /view
    GET  /view               → page with the 3D view (drag to orbit, wheel to zoom)

smabo-web finds the REST API by itself (the simulator announces it on
/esp32/config_url, like the board does).
"""

from __future__ import annotations

import asyncio
import io
import json
import time

from aiohttp import web

from .robot import SimRobot
from .view import add_view_routes

_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class _BadJson(Exception):
    pass


def _bad_json() -> _BadJson:
    return _BadJson()


_PHONE_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>smabo-sim phone</title>
<style>body{margin:0;padding:16px;background:#111318;color:#cfd3dc;font:14px system-ui,sans-serif}
h2{font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#8a93a6;margin:16px 0 8px}
img{display:block;max-width:100%;border-radius:8px;background:#000}
form{display:flex;gap:8px;flex-wrap:wrap}input{flex:1;min-width:0;padding:8px;border-radius:6px;border:1px solid #333a48;background:#1a1d24;color:inherit}
button{padding:8px 12px;border-radius:6px;border:1px solid #333a48;background:#232836;color:inherit}.note{color:#8a93a6}</style></head><body>
<h2>Screen (face)</h2><img id="screen" alt="phone screen">
<h2>Front camera</h2><img src="/phone/camera.mjpg" alt="front camera (MuJoCo engine only)">
<h2>Talk to the robot</h2>
<form id="talk"><input id="text" placeholder="話しかける内容" autocomplete="off">
<button>Send</button> <button type="button" id="mic">🎤 Record</button></form>
<p id="voice" class="note"></p>
<script>const s=document.getElementById('screen');
setInterval(()=>{s.src='/phone/screen.png?'+Date.now()},100);
const note=document.getElementById('voice');
async function speech(body,label){note.textContent=label;
 try{const r=await fetch('/phone/speech',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json();note.textContent=j.ok?(j.sent?'Sent to the brain (/speech/recognized shows in smabo-web)':'Nothing was said'):'Error: '+j.error;}
 catch(e){note.textContent='Error: '+e;}}
document.getElementById('talk').onsubmit=e=>{e.preventDefault();const t=document.getElementById('text');
 if(t.value.trim())speech({text:t.value},'Synthesizing…');};
document.getElementById('mic').onclick=()=>speech({listen:true},'Listening… (stops after 1.5 s of silence)');
</script></body></html>"""


def _with_cors() -> web.Application:
    """An app answering like the ESP32 does: CORS on everything, JSON errors."""
    @web.middleware
    async def cors(request: web.Request, handler):
        if request.method == "OPTIONS":
            return web.Response(status=204, headers=_CORS)
        try:
            resp = await handler(request)
        except _BadJson:
            resp = web.json_response({"error": "bad json"}, status=400)
        except web.HTTPNotFound:
            resp = web.json_response({"error": "not found"}, status=404)
        except web.HTTPMethodNotAllowed:
            resp = web.json_response({"error": "not found"}, status=404)   # ESP32 answers 404 too
        if not resp.prepared:                 # streams set their headers themselves
            resp.headers.update(_CORS)
        return resp

    return web.Application(middlewares=[cors])


def build_app(robot: SimRobot) -> web.Application:
    """The REST config API: exactly the ESP32's endpoints."""
    async def _body(request: web.Request) -> dict:
        try:
            data = await request.json()
        except (json.JSONDecodeError, ValueError):
            raise _bad_json()
        if not isinstance(data, dict):
            raise _bad_json()
        return data

    async def get_config(_request: web.Request) -> web.Response:
        return web.json_response(robot.config_json())

    async def post_config(request: web.Request) -> web.Response:
        robot.set_config(await _body(request))
        return web.json_response({"ok": True})

    async def post_mode(request: web.Request) -> web.Response:
        robot.set_mode(await _body(request))
        return web.json_response({"ok": True})

    async def face_preview(request: web.Request) -> web.Response:
        q = request.query
        try:
            png = face.preview(int(q["id"]), int(q.get("w", 400)), int(q.get("h", 200)))
        except (KeyError, ValueError):
            png = None
        if png is None:
            return _missing()
        return web.Response(body=png, content_type="image/png", headers={"Cache-Control": "no-store"})

    app = _with_cors()
    app.router.add_get("/config", get_config)
    app.router.add_post("/config", post_config)
    app.router.add_post("/mode", post_mode)
    return app


def build_phone_app(settings, face=None) -> web.Application:
    """The phone's REST config API (smabo-app serves the same on port 8081):
    ``GET /config`` the settings tree, ``POST /config`` a partial patch, and
    with ``face`` (phone/expressions.py FaceRest) the ``/face/...`` expressions."""
    async def _body(request: web.Request) -> dict:
        try:
            data = await request.json()
        except (json.JSONDecodeError, ValueError):
            raise _bad_json()
        if not isinstance(data, dict):
            raise _bad_json()
        return data

    async def get_config(_request: web.Request) -> web.Response:
        return web.json_response(settings.to_json())

    async def post_config(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "ignored": settings.apply(await _body(request))})

    def _id(request: web.Request) -> int:
        try:
            return int(request.match_info["id"])
        except ValueError:
            raise web.HTTPNotFound()

    def _missing() -> web.Response:
        return web.json_response({"error": "no such expression"}, status=404)

    async def face_schema(_request: web.Request) -> web.Response:
        return web.json_response(face.schema())

    async def face_list(_request: web.Request) -> web.Response:
        return web.json_response(face.list())

    async def face_patch(request: web.Request) -> web.Response:
        ignored = face.patch(_id(request), await _body(request))
        return _missing() if ignored is None else web.json_response({"ok": True, "ignored": ignored})

    async def face_add(request: web.Request) -> web.Response:
        body = await _body(request)
        copy_of = body.pop("copy_of", None)
        new_id = face.add(int(copy_of) if isinstance(copy_of, (int, float)) and not isinstance(copy_of, bool)
                          else None, body)
        return _missing() if new_id is None else web.json_response({"ok": True, "id": new_id})

    async def face_delete(request: web.Request) -> web.Response:
        err = face.delete(_id(request))
        return web.json_response({"ok": True} if err is None else {"error": err}, status=200 if err is None else 400)

    async def face_preview(request: web.Request) -> web.Response:
        q = request.query
        try:
            png = face.preview(int(q["id"]), int(q.get("w", 400)), int(q.get("h", 200)))
        except (KeyError, ValueError):
            png = None
        if png is None:
            return _missing()
        return web.Response(body=png, content_type="image/png", headers={"Cache-Control": "no-store"})

    app = _with_cors()
    app.router.add_get("/config", get_config)
    app.router.add_post("/config", post_config)
    if face is not None:
        app.router.add_get("/face/schema", face_schema)
        app.router.add_get("/face/expressions", face_list)
        app.router.add_post("/face/expressions", face_add)
        app.router.add_post("/face/expressions/{id}", face_patch)
        app.router.add_delete("/face/expressions/{id}", face_delete)
        app.router.add_get("/face/preview.png", face_preview)
    return app


def build_sim_app(robot: SimRobot, phone=None, switch_model=None) -> web.Application:
    """The simulator-only pages: /view (3D view, models, targets) and /phone.

    ``switch_model(name)`` rebuilds the robot as another roadmap model
    (``POST /view/model``); without it the model is fixed."""
    async def index(_request):
        raise web.HTTPFound("/view")

    app = _with_cors()
    app.router.add_get("/", index)
    add_view_routes(app, lambda: robot.engine, robot.cfg, _CORS, switch_model)
    if phone is not None:
        _add_phone_routes(app, phone)
    return app


def _encode(img, fmt: str) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=fmt, **({"quality": 80} if fmt == "JPEG" else {}))
    return buf.getvalue()


def _add_phone_routes(app: web.Application, phone) -> None:
    engine = phone.engine
    fps = max(float(phone.cfg.get("phone.camera.fps", 10.0)), 0.5)
    cache = {"t": -1.0, "jpg": b""}

    def camera_jpeg() -> bytes:
        # one render per frame period, shared by every viewer
        now = time.monotonic()
        if now - cache["t"] >= 1.0 / fps:
            cache["jpg"] = _encode(engine.render_phone_camera(), "JPEG")
            cache["t"] = now
        return cache["jpg"]

    def has_camera() -> bool:
        return bool(getattr(engine, "has_phone", False))

    async def page(_request):
        return web.Response(text=_PHONE_PAGE, content_type="text/html")

    async def screen(_request):
        if phone.last_screen is None:
            raise web.HTTPNotFound()
        return web.Response(body=_encode(phone.last_screen, "PNG"), content_type="image/png",
                            headers={"Cache-Control": "no-store"})

    async def camera(_request):
        if not has_camera():
            return web.json_response({"error": "the phone camera needs --engine mujoco"}, status=404)
        return web.Response(body=camera_jpeg(), content_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    async def mjpeg(request):
        if not has_camera():
            return web.json_response({"error": "the phone camera needs --engine mujoco"}, status=404)
        resp = web.StreamResponse(headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-store", **_CORS})
        await resp.prepare(request)
        try:
            while True:
                jpg = camera_jpeg()
                await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(1.0 / fps)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def speech(request):
        try:
            if request.content_type in ("audio/wav", "audio/x-wav", "audio/wave"):
                phone.send_utterance(await request.read())
                return web.json_response({"ok": True, "sent": True})
            body = await request.json()
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except RuntimeError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=409)
        if not isinstance(body, dict):
            raise _bad_json()
        try:
            if body.get("listen"):
                sent = await phone.listen()
            elif str(body.get("text", "")).strip():
                await phone.talk(str(body["text"]))
                sent = True
            else:
                return web.json_response({"ok": False, "error": "give text, listen or a WAV body"}, status=400)
        except RuntimeError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=409)
        return web.json_response({"ok": True, "sent": sent})

    app.router.add_get("/phone", page)
    app.router.add_post("/phone/speech", speech)
    app.router.add_get("/phone/screen.png", screen)
    app.router.add_get("/phone/camera.jpg", camera)
    app.router.add_get("/phone/camera.mjpg", mjpeg)
