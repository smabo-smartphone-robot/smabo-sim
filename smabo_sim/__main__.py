"""smabo-sim command line.

    smabo-sim                         # run against ws://localhost:9090 (smabo-brain or smabo-brain-ros)
    smabo-sim run --host 192.168.1.20 --config my-esp32-config.json
    smabo-sim --engine mujoco --viewer   # MuJoCo physics + 3D viewer (macOS: run with mjpython)
    smabo-sim --engine mujoco --model hand   # the robot at a roadmap step (see smabo_sim/models.py)
    smabo-sim export-map smabo_sim/worlds/room.yaml -o maps/sim_room
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from pathlib import Path

from aiohttp import web

from .config import Config
from .engines import create_engine
from . import __version__
from .http_api import build_app, build_phone_app, build_sim_app
from .link import BrainLink
from .pairing import DEFAULT_DIR as DEFAULT_CRED_DIR, Credentials
from .models import MODELS, RobotModel, get_model
from .phone import SimPhone
from .phone.expressions import FaceRest
from .phone.settings import DEFAULT_PORT as PHONE_PORT, PhoneSettings
from .robot import SimRobot, local_url
from .world import World

log = logging.getLogger("smabo_sim")


async def _every(period, fn) -> None:
    """Call ``fn(dt)`` at a fixed rate, passing the measured interval.

    ``period`` may be a callable, read again after every call (a rate that
    can be changed while running, e.g. the phone's IMU rate)."""
    get = period if callable(period) else (lambda: period)
    loop = asyncio.get_running_loop()
    last = loop.time()
    nxt = last + get()
    while True:
        period = get()
        await asyncio.sleep(max(0.0, nxt - loop.time()))
        now = loop.time()
        fn(min(now - last, 0.1))
        last = now
        nxt += period
        if nxt < now:              # fell behind (e.g. laptop slept): resync
            nxt = now + period


async def run(cfg: Config, world: World, viewer: bool = False, http_port_given: bool = False,
              invite: str = "", phone_invite: str = "", credentials_dir: str = "") -> None:
    engine = create_engine(cfg.get("sim.engine", "kinematic"), world, cfg)
    view = _open_viewer(engine) if viewer else None
    link: BrainLink | None = None

    def send(text: str) -> None:
        if link is not None:
            link.send(text)

    robot = SimRobot(cfg, engine, send)
    # Pairing with the brain (pairing.py): tokens survive restarts in cred_dir
    cred_dir = Path(credentials_dir or DEFAULT_CRED_DIR).expanduser()
    robot_creds = Credentials("esp32", cfg.get("wifi.hostname") or "smabo-sim", cred_dir / "esp32.json")
    if invite:
        robot_creds.set_invite(invite)

    def url() -> str:
        return f"ws://{cfg.get('brain.host')}:{cfg.get('brain.port')}{robot.ws_path()}"

    link = BrainLink(url, robot.on_message, robot.on_brain_connect, credentials=robot_creds)

    # the phone (smabo-app's part) is a separate brain client, like the real app
    phone: SimPhone | None = None
    phone_link: BrainLink | None = None
    phone_creds = Credentials("app", "smabo-sim phone", cred_dir / "app.json")
    if phone_invite:
        phone_creds.set_invite(phone_invite)
    if cfg.get("phone.enabled", True):
        phone = SimPhone(cfg, engine, lambda text: phone_link.send(text) if phone_link else None)
        phone_link = BrainLink(
            lambda: f"ws://{cfg.get('brain.host')}:{cfg.get('brain.port')}{phone.ws_path()}",
            phone.on_message, phone.on_brain_connect, credentials=phone_creds)

    def invite_robot(code: str) -> None:
        robot_creds.set_invite(code)
        link.restart()

    def invite_phone(code: str) -> None:
        phone_creds.set_invite(code)
        if phone_link is not None:
            phone_link.restart()

    def switch_model(name: str) -> RobotModel:
        """Rebuild the robot as another roadmap model (from /view or POST /view/model)."""
        model = get_model(name)
        if view is not None:
            raise RuntimeError("the model cannot be switched while --viewer is open "
                               "(restart with --model, or use the browser view)")
        robot.apply_model(model)
        old = robot.engine
        new = create_engine(cfg.get("sim.engine", "kinematic"), world, cfg)
        robot.engine = new
        if phone is not None:
            phone.engine = new
        old.close()
        log.info("model: %s (%s)", model.name, model.label)
        return model

    def restart_links() -> None:
        link.restart()
        if phone_link is not None:
            phone_link.restart()

    robot.on_brain_settings_changed = restart_links

    # Like the board: the REST config API alone on http.port (80). The simulator's
    # own pages (/view, /phone) get a port of their own.
    rest = web.AppRunner(build_app(robot, robot_creds, invite_robot), access_log=None)
    await rest.setup()
    port = int(cfg.get("http.port", 80))
    try:
        await web.TCPSite(rest, "0.0.0.0", port).start()
    except OSError as e:
        if port != 80 or http_port_given:
            raise SystemExit(f"smabo-sim: can't open the REST port {port}: {e}")
        # Linux without root, or taken: the usual unprivileged stand-in
        log.warning("port 80 unavailable (%s); REST config API on 8080 instead", e.strerror or e)
        port = 8080
        try:
            await web.TCPSite(rest, "0.0.0.0", port).start()
        except OSError as e2:
            raise SystemExit(f"smabo-sim: can't open the REST port 80 or 8080: {e2} (use --http-port)")
    robot.http_port = port
    robot.publish_config_url()

    # The phone's own REST /config (smabo-app serves the same on 8081)
    if phone is not None:
        phone_port = int(cfg.get("phone.http_port", PHONE_PORT))

        def phone_changed(paths: set[str]) -> None:
            if "sensors.camera.enabled" in paths:
                phone.set_camera(bool(cfg.get("phone.camera.stream", True)))
            phone.publish_settings()           # (cfg.update already saved it with --save)

        settings = PhoneSettings(cfg, phone, phone_port, on_change=phone_changed, version=__version__)
        phone.settings_json = settings.to_json
        face_rest = FaceRest(cfg, phone.face, on_change=phone.publish_settings)
        phone_rest = web.AppRunner(build_phone_app(settings, face_rest, phone_creds, invite_phone),
                                   access_log=None)
        await phone_rest.setup()
        try:
            await web.TCPSite(phone_rest, "0.0.0.0", phone_port).start()
        except OSError as e:
            raise SystemExit(f"smabo-sim: can't open the phone's REST port {phone_port}: {e} (use --phone-port)")
        phone.config_url = local_url(cfg, phone_port)
        phone.publish_config_url()

    view_port = int(cfg.get("sim.view_port", 8090))
    sim_pages = web.AppRunner(build_sim_app(robot, phone, switch_model), access_log=None)
    await sim_pages.setup()
    try:
        await web.TCPSite(sim_pages, "0.0.0.0", view_port).start()
    except OSError as e:
        raise SystemExit(f"smabo-sim: can't open the view port {view_port}: {e} (use --view-port)")

    log.info("world '%s' (%d segments), engine '%s', model '%s'", world.name, len(world.segments),
             cfg.get("sim.engine"), get_model(cfg.get("sim.model")).name)
    log.info("REST config API (as on the ESP32) on http://localhost:%d", port)
    log.info("brain: %s", url())
    if hasattr(robot.engine, "render_view"):
        log.info("3D view in the browser: http://localhost:%d/view", view_port)
    if phone is not None:
        log.info("phone (smabo-app part): screen + camera on http://localhost:%d/phone", view_port)
        log.info("phone settings REST API (as smabo-app) on http://localhost:%d/config", phone_port)

    def status(_dt: float) -> None:
        p = robot.engine.pose()
        l, r = robot.drive.duty(robot.clock())
        face = ""
        if phone is not None:
            fs = phone.face.state
            face = (f" | face {phone.face.expression.name} gaze {fs.gaze[0]:+.2f},{fs.gaze[1]:+.2f}"
                    f"{' (look_at)' if phone.face.following else ''}"
                    f"{' says: ' + fs.bubble if fs.bubble else ''}")
            if phone.speaker.speaking:
                face += " (speaking)"
            if phone.camera is not None:
                face += f" | camera {phone.camera.state} ({phone.camera.frames_sent} frames)"
        log.info("%s | pose x=%+.2f y=%+.2f yaw=%+4.0f° | duty L=%+.2f R=%+.2f%s%s",
                 "connected" if link.connected else "waiting for brain",
                 p.x, p.y, math.degrees(p.yaw), l, r,
                 " | BLOCKED by wall" if getattr(robot.engine, "blocked", False) else "", face)

    jobs = [
        link.run(),
        _every(1.0 / cfg.get("sim.physics_rate", 100.0), robot.physics_step),
        _every(1.0 / max(cfg.get("encoder.publish_rate", 20.0), 1e-3), lambda _dt: robot.publish_wheel_vel()),
        _every(1.0 / max(cfg.get("servos.joint_states_rate", 20.0), 1e-3), lambda _dt: robot.publish_joint_states()),
        _every(1.0 / max(cfg.get("sim.lidar_rate", 10.0), 1e-3), lambda _dt: robot.publish_scan()),
        _every(2.0, status),
        _every(5.0, lambda _dt: robot.publish_config_url()),
    ]
    if phone is not None:
        jobs += [
            phone_link.run(),
            _every(1.0 / max(cfg.get("phone.screen.fps", 30.0), 1e-3), phone.update_screen),
            _every(lambda: 1.0 / max(float(cfg.get("phone.imu_rate", 20.0)), 1e-3), phone.publish_imu),
            _every(5.0, phone.publish_config_url),
            _every(1.0 / max(cfg.get("phone.gps.rate", 1.0), 1e-3), phone.publish_gps),
        ]
    if view is not None:
        jobs.append(_run_viewer(view, engine))
    tasks = [asyncio.ensure_future(j) for j in jobs]
    try:
        # the viewer task ends when its window is closed; anything else ending is an error
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            t.result()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runner.cleanup()
        if view is not None:
            view.close()


class ViewerError(RuntimeError):
    pass


def _open_viewer(engine):
    """Open MuJoCo's passive viewer on the engine's model (MuJoCo engine only)."""
    if not hasattr(engine, "model"):
        raise ViewerError("--viewer needs the MuJoCo engine: add --engine mujoco")
    import mujoco
    import mujoco.viewer
    try:
        view = mujoco.viewer.launch_passive(engine.model, engine.data,
                                            show_left_ui=False, show_right_ui=False)
    except RuntimeError as e:
        if sys.platform == "darwin":
            raise ViewerError("on macOS the MuJoCo viewer must be started with mjpython:\n"
                              "  .venv/bin/mjpython -m smabo_sim --engine mujoco --viewer") from e
        raise
    engine.lock = view.lock
    with view.lock():
        view.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        view.cam.trackbodyid = engine.model.body("base_link").id
        view.cam.distance = 1.2
        view.cam.elevation = -30
        view.cam.azimuth = -140
    log.info("MuJoCo viewer open (double-click to select, ctrl+drag to push, Tab/Shift+Tab for panels)")
    return view


async def _run_viewer(view, engine, rate: float = 30.0) -> None:
    """Redraw at ``rate`` Hz with the latest lidar hits as dots; return when closed."""
    import mujoco
    import numpy as np

    eye = np.eye(3).ravel()
    size = np.array([0.012, 0.0, 0.0])
    rgba = np.array([1.0, 0.25, 0.2, 0.9], dtype=np.float32)
    shown = -1
    while view.is_running():
        version = getattr(engine, "screen_version", 0)
        if version != shown and getattr(engine, "has_phone", False):
            view.update_texture(engine.screen_tex)
            shown = version
        with view.lock():
            scn = view.user_scn
            pts = engine.last_scan_points
            n = min(len(pts), scn.maxgeom)
            for i in range(n):
                mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE, size, pts[i], eye, rgba)
            scn.ngeom = n
        view.sync()
        await asyncio.sleep(1.0 / rate)
    log.info("viewer closed")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv.insert(0, "run")

    ap = argparse.ArgumentParser(prog="smabo-sim", description="Simulated smabo-esp32.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the simulated robot (default)")
    r.add_argument("--config", help="ESP32-format config.json merged over the defaults")
    r.add_argument("--save", action="store_true", help="write REST config changes back to --config")
    r.add_argument("--world", help="world YAML (default: bundled room)")
    r.add_argument("--host", help="brain host (default: localhost)")
    r.add_argument("--port", type=int, help="brain port (default: 9090)")
    r.add_argument("--http-port", type=int, help="REST config API port (default: 80 like the ESP32; 8080 if 80 can't be opened)")
    r.add_argument("--phone-port", type=int, help="the phone's settings REST API, as smabo-app (default: 8081)")
    r.add_argument("--view-port", type=int, help="simulator-only pages: /view, /phone (default: 8090)")
    r.add_argument("--engine", help="simulation engine: kinematic (default) or mujoco")
    r.add_argument("--model", help="robot for a roadmap step: " + " | ".join(MODELS)
                   + " (default: full; switchable at /view)")
    r.add_argument("--viewer", action="store_true", help="open the MuJoCo 3D viewer (needs --engine mujoco)")
    r.add_argument("--no-phone", action="store_true",
                   help="do not simulate the phone (e.g. a real smabo-app is connected)")
    r.add_argument("--mute", action="store_true", help="do not read /speech/say aloud (phone.tts.enabled = false)")
    r.add_argument("--seed", type=int, help="random seed for sensor noise")
    r.add_argument("--invite", help="pair the simulated ESP32 with the brain using this invite code "
                                    "(smabo-web → Devices → Add ESP32)")
    r.add_argument("--phone-invite", help="pair the simulated phone with the brain using this invite code")
    r.add_argument("--credentials-dir", help=f"where the tokens are kept (default: {DEFAULT_CRED_DIR})")
    r.add_argument("-v", "--verbose", action="store_true")

    e = sub.add_parser("export-map", help="write a Nav2 map (pgm + yaml) for a world")
    e.add_argument("world", help="world YAML")
    e.add_argument("-o", "--out", required=True, help="output path without extension, e.g. maps/sim_room")
    e.add_argument("--resolution", type=float, default=0.05)

    args = ap.parse_args(argv)

    if args.cmd == "export-map":
        pgm, yml = World.load(args.world).export_map(args.out, args.resolution)
        print(f"wrote {pgm}\nwrote {yml}")
        return 0

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    overrides: dict = {"brain": {}, "sim": {}, "http": {}, "phone": {}}
    if args.no_phone:
        overrides["phone"]["enabled"] = False
    if args.mute:
        overrides["phone"]["tts"] = {"enabled": False}
    if args.host:
        overrides["brain"]["host"] = args.host
    if args.port:
        overrides["brain"]["port"] = args.port
    if args.http_port:
        overrides["http"]["port"] = args.http_port
    if args.phone_port:
        overrides["phone"]["http_port"] = args.phone_port
    if args.view_port:
        overrides["sim"]["view_port"] = args.view_port
    if args.engine:
        overrides["sim"]["engine"] = args.engine
    if args.world:
        overrides["sim"]["world"] = args.world
    if args.seed is not None:
        overrides["sim"]["seed"] = args.seed

    cfg = Config(args.config, persist=args.save, overrides=overrides)
    if args.model:
        try:
            model = get_model(args.model)
        except ValueError as e:
            print(f"smabo-sim: {e}", file=sys.stderr)
            return 2
        cfg.update(model.config_patch(cfg.get("servos.joints") or {}))
    world = World.load(cfg.get("sim.world") or None)
    try:
        asyncio.run(run(cfg, world, viewer=args.viewer, http_port_given=bool(args.http_port),
                        invite=args.invite or "", phone_invite=args.phone_invite or "",
                        credentials_dir=args.credentials_dir or ""))
    except KeyboardInterrupt:
        pass
    except ViewerError as e:
        print(f"smabo-sim: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
