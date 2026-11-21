"""The simulation in a web browser: ``http://localhost:<http.port>/view``.

The MuJoCo world is rendered offscreen (no window) from an orbit camera and
streamed as MJPEG, so ``--viewer`` is not needed. Nothing is rendered while no
browser has the page open.

    GET  /view               → page: the stream, drag to orbit, wheel to zoom
    GET  /view/stream.mjpg   → the view as an MJPEG stream
    GET  /view/frame.jpg     → one frame
    GET  /view/camera        → the camera ({azimuth, elevation, distance, follow, lookat})
    POST /view/camera        → change it: {"orbit": [dx, dy]} (deg), {"zoom": factor},
                               {"pan": [right, forward]} (m, stops following),
                               {"follow": bool}, {"preset": "default" | "top"},
                               or any camera field directly
    GET  /view/state         → robot pose (ground truth) for the overlay
    GET  /view/models        → the roadmap models and the current one
    POST /view/model {name}  → rebuild the robot as that model (smabo_sim/models.py)
    GET  /view/targets       → vision targets (World.targets) and where they are
    POST /view/targets/{name} {x, y, z, yaw_deg, sway: {on, amplitude, period}}
                             → move a target / make it sway sideways

Config (``view.*``): ``width`` / ``height`` (default 960 × 600), ``fps``
(default 15), ``jpeg_quality`` (default 80). Needs ``--engine mujoco``.
"""

from __future__ import annotations

import asyncio
import io
import math
import time

from aiohttp import web

from .models import MODELS, get_model

_DEFAULT = {"azimuth": -140.0, "elevation": -30.0, "distance": 1.2, "follow": True,
            "lookat": [0.0, 0.0, 0.0]}
_TOP = {"azimuth": -90.0, "elevation": -89.0, "distance": 4.0}


class ViewCamera:
    """The orbit camera shared by every browser that has /view open."""

    def __init__(self) -> None:
        self.state = {**_DEFAULT, "lookat": list(_DEFAULT["lookat"])}
        self.version = 0

    def apply(self, req: dict, robot_xy: tuple[float, float]) -> dict:
        s = self.state
        if req.get("preset") == "top":
            s.update(_TOP)
        elif req.get("preset") == "default":
            s.update({**_DEFAULT, "lookat": list(_DEFAULT["lookat"])})
        for key in ("azimuth", "elevation", "distance"):
            if isinstance(req.get(key), (int, float)):
                s[key] = float(req[key])
        if isinstance(req.get("lookat"), list) and len(req["lookat"]) == 3:
            s["lookat"] = [float(v) for v in req["lookat"]]
        if "follow" in req:
            s["follow"] = bool(req["follow"])
            if not s["follow"]:
                s["lookat"] = [robot_xy[0], robot_xy[1], 0.05]
        if isinstance(req.get("orbit"), list) and len(req["orbit"]) == 2:
            s["azimuth"] -= float(req["orbit"][0])
            s["elevation"] -= float(req["orbit"][1])
        if isinstance(req.get("zoom"), (int, float)) and req["zoom"] > 0:
            s["distance"] *= float(req["zoom"])
        if isinstance(req.get("pan"), list) and len(req["pan"]) == 2:
            if s["follow"]:
                s["follow"] = False
                s["lookat"] = [robot_xy[0], robot_xy[1], 0.05]
            # move the look-at point in the ground plane, relative to the view direction
            right, fwd = (float(v) * s["distance"] for v in req["pan"])
            az = math.radians(s["azimuth"])
            s["lookat"][0] += fwd * math.cos(az) + right * math.sin(az)
            s["lookat"][1] += fwd * math.sin(az) - right * math.cos(az)
        s["azimuth"] = (s["azimuth"] + 180.0) % 360.0 - 180.0
        s["elevation"] = min(max(s["elevation"], -89.0), 89.0)
        s["distance"] = min(max(s["distance"], 0.15), 30.0)
        self.version += 1
        return dict(s)


def _jpeg(img, quality: int) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def add_view_routes(app: web.Application, get_engine, cfg, cors: dict, switch_model=None) -> None:
    """``get_engine()`` returns the current engine (it is replaced when the
    model changes); ``switch_model(name)`` does that change."""
    width = int(cfg.get("view.width", 960))
    height = int(cfg.get("view.height", 600))
    fps = max(float(cfg.get("view.fps", 15.0)), 0.5)
    quality = int(cfg.get("view.jpeg_quality", 80))
    camera = ViewCamera()
    show = {"scan": True}
    cache = {"t": -1.0, "version": -1, "jpg": b""}

    def has_view() -> bool:
        return hasattr(get_engine(), "render_view")

    def frame() -> bytes:
        # one render per frame period (or camera change), shared by every viewer
        now = time.monotonic()
        if now - cache["t"] >= 1.0 / fps or cache["version"] != camera.version:
            img = get_engine().render_view(camera.state, width, height, show["scan"])
            cache["jpg"] = _jpeg(img, quality)
            cache["t"], cache["version"] = now, camera.version
        return cache["jpg"]

    def need_mujoco():
        return web.json_response({"error": "the browser view needs --engine mujoco"}, status=404)

    def robot_xy() -> tuple[float, float]:
        p = get_engine().pose()
        return p.x, p.y

    async def page(_request):
        return web.Response(text=_PAGE, content_type="text/html")

    async def one(_request):
        if not has_view():
            return need_mujoco()
        return web.Response(body=frame(), content_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    async def stream(request):
        if not has_view():
            return need_mujoco()
        resp = web.StreamResponse(headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-store", **cors})
        await resp.prepare(request)
        try:
            while has_view():
                jpg = frame()
                await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(1.0 / fps)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def get_camera(_request):
        return web.json_response({**camera.state, "scan": show["scan"]})

    async def post_camera(request):
        try:
            req = await request.json()
        except ValueError:
            req = None
        if not isinstance(req, dict):
            return web.json_response({"error": "bad json"}, status=400)
        if "scan" in req:
            show["scan"] = bool(req["scan"])
        return web.json_response({**camera.apply(req, robot_xy()), "scan": show["scan"]})

    async def state(_request):
        engine = get_engine()
        p = engine.pose()
        return web.json_response({"x": p.x, "y": p.y, "yaw": math.degrees(p.yaw),
                                  "blocked": bool(getattr(engine, "blocked", False)),
                                  "engine": "mujoco" if has_view() else "kinematic",
                                  "model": get_model(cfg.get("sim.model")).name})

    def models_json() -> dict:
        return {"current": get_model(cfg.get("sim.model")).name,
                "switchable": switch_model is not None,
                "models": [{"name": m.name, "label": m.label, "roadmap": list(m.roadmap),
                            "parts": _parts(m), "modes": m.modes()} for m in MODELS.values()]}

    async def models(_request):
        return web.json_response(models_json())

    async def post_model(request):
        if switch_model is None:
            return web.json_response({"error": "this simulator cannot switch models"}, status=409)
        try:
            req = await request.json()
        except ValueError:
            req = None
        if not isinstance(req, dict) or not isinstance(req.get("name"), str):
            return web.json_response({"error": 'give {"name": "<model>"}'}, status=400)
        try:
            switch_model(req["name"])
        except ValueError as e:                  # unknown model
            return web.json_response({"error": str(e)}, status=400)
        except RuntimeError as e:                # e.g. --viewer is open
            return web.json_response({"error": str(e)}, status=409)
        cache["version"] = -1                    # next frame shows the new robot
        return web.json_response(models_json())

    app.router.add_get("/view", page)
    app.router.add_get("/view/stream.mjpg", stream)
    app.router.add_get("/view/frame.jpg", one)
    app.router.add_get("/view/camera", get_camera)
    app.router.add_post("/view/camera", post_camera)
    app.router.add_get("/view/state", state)
    async def targets(_request):
        return web.json_response({"targets": get_engine().targets()})

    async def post_target(request):
        try:
            req = await request.json()
        except ValueError:
            req = None
        if not isinstance(req, dict):
            return web.json_response({"error": "bad json"}, status=400)
        sway = req.get("sway") if isinstance(req.get("sway"), dict) else {}
        try:
            get_engine().set_target(request.match_info["name"],
                                    **{k: req.get(k) for k in ("x", "y", "z", "yaw_deg")},
                                    **{f"sway_{k}": sway.get(k) for k in ("on", "amplitude", "period")})
        except KeyError:
            return web.json_response({"error": "no such target"}, status=404)
        except (TypeError, ValueError):
            return web.json_response({"error": "numbers expected"}, status=400)
        return web.json_response({"targets": get_engine().targets()})

    app.router.add_get("/view/targets", targets)
    app.router.add_post("/view/targets/{name}", post_target)
    app.router.add_get("/view/models", models)
    app.router.add_post("/view/model", post_model)


def _parts(m) -> list[str]:
    """What a model is built from, for the /view tooltip."""
    out = ["スマホ"] if not m.base else ["ベース", "スマホ"]
    if m.mount == "tt_motor":
        out.append("TTモーター台車")
    elif m.mount == "dg01d_e":
        out.append("DG01D-E台車")
    if m.encoder:
        out.append("エンコーダ")
    if m.lidar:
        out.append("LiDAR")
    if m.mount == "neck":
        out.append("首振り台")
    if m.hand:
        out.append("ハンド")
    if m.arm:
        out.append("アーム")
    if m.breadboard:
        out.append("ブレッドボード")
    return out


_PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>smabo-sim view</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#101216;color:#cfd3dc;font:14px system-ui,sans-serif;height:100dvh;display:flex;flex-direction:column}
#stage{position:relative;flex:1;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden;
 touch-action:none;cursor:grab;user-select:none}
#stage.drag{cursor:grabbing}
#img{max-width:100%;max-height:100%;display:block;pointer-events:none}
#hud{position:absolute;left:12px;top:12px;padding:6px 10px;border-radius:6px;background:#0009;
 font:12px ui-monospace,monospace;font-variant-numeric:tabular-nums;white-space:pre}
#err{position:absolute;inset:0;display:none;align-items:center;justify-content:center;padding:16px;text-align:center;color:#e8a0a0}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:10px 16px;border-top:1px solid #262a33;background:#15181e}
button{padding:7px 12px;border-radius:6px;border:1px solid #333a48;background:#20242e;color:inherit;font:inherit;cursor:pointer}
button:hover{background:#2a2f3b}button:active{transform:translateY(1px)}
button[aria-pressed=true]{border-color:#e07b39;color:#f3b48a}
.hint{color:#7d8597;font-size:12px;margin-left:auto}
.models{border-top:none;padding-bottom:0}.models::before{content:"モデル";color:#7d8597;font-size:12px;margin-right:4px}
.targets::before{content:"ターゲットを揺らす"}.targets:empty{display:none}
.models button{padding:5px 10px;font-size:13px}button:disabled{opacity:.5;cursor:default}
#msg{color:#e8a0a0;font-size:12px}
a{color:#8fb3ff}
</style></head><body>
<div id="stage"><img id="img" alt="simulation view"><div id="hud">…</div><div id="err"></div></div>
<div class="bar models" id="models" role="group" aria-label="ロボットのモデル"></div>
<div class="bar models targets" id="targets" role="group" aria-label="画像処理のターゲット"></div>
<div class="bar">
 <button id="follow" aria-pressed="true">ロボットを追う</button>
 <button data-preset="default">視点リセット</button>
 <button data-preset="top">真上</button>
 <button data-zoom="0.8">＋</button><button data-zoom="1.25">－</button>
 <button id="scan" aria-pressed="true">LiDAR点</button>
 <span class="hint">ドラッグ: 回転 / Shift・右ドラッグ: 移動 / ホイール: ズーム ・ <a href="/phone">スマホ画面</a></span>
</div>
<script>
const img=document.getElementById('img'),stage=document.getElementById('stage'),hud=document.getElementById('hud'),
 err=document.getElementById('err'),fBtn=document.getElementById('follow'),sBtn=document.getElementById('scan');
function show(c){fBtn.setAttribute('aria-pressed',c.follow);sBtn.setAttribute('aria-pressed',c.scan);}
let busy=false,pending=null;
async function cam(body){
 if(busy){pending=Object.assign(pending||{},merge(pending,body));return;}
 busy=true;
 try{const r=await fetch('/view/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(r.ok)show(await r.json());}catch(e){}
 busy=false;
 if(pending){const p=pending;pending=null;cam(p);}
}
// accumulate drags that arrive while a request is in flight
function merge(a,b){a=a||{};const o={...b};
 for(const k of['orbit','pan'])if(a[k]&&b[k])o[k]=[a[k][0]+b[k][0],a[k][1]+b[k][1]];
 if(a.zoom&&b.zoom)o.zoom=a.zoom*b.zoom;return o;}
function start(){img.src='/view/stream.mjpg?'+Date.now();}
img.onerror=async()=>{try{const r=await fetch('/view/frame.jpg');if(!r.ok){const j=await r.json();
 err.textContent=j.error;err.style.display='flex';return;}}catch(e){}setTimeout(start,1500);};
start();
fetch('/view/camera').then(r=>r.json()).then(show);
let last=null,mode='orbit';
stage.addEventListener('contextmenu',e=>e.preventDefault());
stage.addEventListener('pointerdown',e=>{last=[e.clientX,e.clientY];
 mode=(e.shiftKey||e.button===2)?'pan':'orbit';stage.classList.add('drag');
 try{stage.setPointerCapture(e.pointerId);}catch(_){}});
stage.addEventListener('pointermove',e=>{if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];last=[e.clientX,e.clientY];
 if(mode==='orbit')cam({orbit:[dx*0.4,dy*0.4]});
 else{const h=stage.clientHeight||600;cam({pan:[-dx/h,dy/h]});}});
const end=()=>{last=null;stage.classList.remove('drag');};
stage.addEventListener('pointerup',end);stage.addEventListener('pointercancel',end);
stage.addEventListener('wheel',e=>{e.preventDefault();cam({zoom:Math.exp(e.deltaY*0.001)});},{passive:false});
fBtn.onclick=()=>cam({follow:fBtn.getAttribute('aria-pressed')!=='true'});
sBtn.onclick=()=>cam({scan:sBtn.getAttribute('aria-pressed')!=='true'});
document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>cam({preset:b.dataset.preset}));
document.querySelectorAll('[data-zoom]').forEach(b=>b.onclick=()=>cam({zoom:+b.dataset.zoom}));
const mBar=document.getElementById('models');
function showModels(j){mBar.innerHTML='';
 for(const m of j.models){const b=document.createElement('button');b.textContent=m.label;
  b.title=m.parts.join(' + ')+(m.roadmap.length?'（ロードマップ: '+m.roadmap.join(', ')+'）':'');
  b.setAttribute('aria-pressed',m.name===j.current);b.disabled=!j.switchable;
  b.onclick=()=>pick(m.name);mBar.appendChild(b);}
 const s=document.createElement('span');s.id='msg';mBar.appendChild(s);}
async function pick(name){const msg=document.getElementById('msg');msg.textContent='組み立て中…';
 mBar.querySelectorAll('button').forEach(b=>b.disabled=true);
 try{const r=await fetch('/view/model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  const j=await r.json();if(!r.ok){await loadModels();document.getElementById('msg').textContent=j.error;return;}
  showModels(j);start();}catch(e){await loadModels();document.getElementById('msg').textContent='切り替えに失敗しました';}}
async function loadModels(){try{showModels(await(await fetch('/view/models')).json());}catch(e){}}
loadModels();
const tBar=document.getElementById('targets');
const KIND={face:'顔',qr:'QR'};
function tLabel(t){if(t.kind==='color')return t.name.replace(/_/g,' ');
 if(t.name.startsWith('aruco'))return 'ArUco '+t.name.split('_').pop();return KIND[t.name]||t.name;}
function showTargets(list){tBar.innerHTML='';
 for(const t of list){const b=document.createElement('button');b.textContent=tLabel(t)+' ↔';
  b.title=`(${t.x.toFixed(2)}, ${t.y.toFixed(2)}) m・振れ幅 ${t.sway.amplitude} m・周期 ${t.sway.period} s`;
  b.setAttribute('aria-pressed',t.sway.on);
  b.onclick=async()=>{const r=await fetch('/view/targets/'+encodeURIComponent(t.name),{method:'POST',
   headers:{'Content-Type':'application/json'},body:JSON.stringify({sway:{on:!t.sway.on}})});
   if(r.ok)showTargets((await r.json()).targets);};
  tBar.appendChild(b);}}
fetch('/view/targets').then(r=>r.json()).then(j=>showTargets(j.targets)).catch(()=>{});
setInterval(async()=>{try{const s=await(await fetch('/view/state')).json();
 hud.textContent=`x ${s.x.toFixed(2)} m  y ${s.y.toFixed(2)} m  yaw ${s.yaw.toFixed(0)}°`+(s.blocked?'  壁に接触':'');}catch(e){hud.textContent='sim に接続できません';}},300);
</script></body></html>"""
