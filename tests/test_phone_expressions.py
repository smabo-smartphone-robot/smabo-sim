"""Face expressions over the phone's REST API: the same contract as smabo-app."""

import asyncio
import io
import json
import pathlib

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from smabo_sim.config import Config
from smabo_sim.engines import create_engine
from smabo_sim.http_api import build_phone_app
from smabo_sim.phone import SimPhone
from smabo_sim.phone.expressions import SCHEMA, FaceRest, apply_patch, expression_to_json
from smabo_sim.phone.face import Face
from smabo_sim.phone.settings import PhoneSettings
from smabo_sim.world import World

# generated from smabo-protocol/face_schema.json (smabo-app serves the same schema)
APP_SCHEMA = pathlib.Path(__file__).resolve().parent / "face_schema.json"


@pytest.mark.skipif(not APP_SCHEMA.exists(), reason="tests/face_schema.json is missing (run smabo-protocol/gen.py)")
def test_schema_is_smabo_apps():
    assert SCHEMA == json.loads(APP_SCHEMA.read_text(encoding="utf-8"))


def test_templates_on_the_wire_follow_the_schema():
    f = Face(seed=0)
    wink = expression_to_json(f.expressions[8])
    assert wink["per_eye"] is True and wink["right_eye"]["shape"] == "arc"
    assert wink["left_eye"]["iris_color"] == "#4FC3F7"
    assert set(wink["left_eye"]) == {fld["key"] for fld in SCHEMA["eye"]}
    assert set(wink) - {"id", "left_eye", "right_eye"} == {fld["key"] for fld in SCHEMA["expression"]}


def test_patch_is_typed_and_clamped():
    x = Face(seed=0).expressions[1]
    ignored = apply_patch(x, {"name": "Surprised", "background": "#102030", "id": 9,
                              "left_eye": {"shape": "heart", "lid_top": 2, "iris_frac": 0.3,
                                           "pupil_frac": 0.9, "iris_color": "red", "nope": 1},
                              "right_eye": "x"})
    assert sorted(ignored) == ["id", "left_eye.iris_color", "left_eye.nope", "right_eye"]
    assert (x.id, x.name, x.background) == (1, "Surprised", (16, 32, 48))
    assert x.left.shape == "heart" and x.left.lid_top == 1.0 and x.left.pupil_frac == 0.3


def test_shared_eyes_follow_the_left_and_per_eye_starts_from_it():
    x = Face(seed=0).expressions[1]
    apply_patch(x, {"left_eye": {"lid_top": 0.4}, "right_eye": {"lid_top": 0.9}})
    assert x.right.lid_top == 0.4
    apply_patch(x, {"per_eye": True})
    assert x.right.lid_top == 0.4
    apply_patch(x, {"right_eye": {"shape": "arc"}})
    assert (x.left.shape, x.right.shape) == ("round", "arc")


def test_edits_change_what_the_phone_draws():
    f = Face(seed=0)
    f.set_expression(1)
    base = f.render()
    for patch in ({"left_eye": {"rotation": 30, "lid_top": 0.4}}, {"left_eye": {"highlight_visible": False}},
                  {"left_eye": {"sclera_scale": 0.6}}, {"left_eye": {"shape": "arc", "shape_rotation": 90}}):
        g = Face(seed=0)
        apply_patch(g.expressions[1], patch)
        assert np.abs(g.render().astype(int) - base).sum() > 0, patch


def test_shared_eyes_mirror_the_left_eye():
    f = Face(seed=0)
    # (the app doesn't mirror the highlight either)
    apply_patch(f.expressions[1], {"left_eye": {"lid_top": 0.4, "rotation": 25, "highlight_visible": False}})
    f.state.gaze = [0.0, 0.0]
    img = f.render().astype(int)
    assert np.abs(img - img[:, ::-1]).mean() < 1.0          # symmetric face
    apply_patch(f.expressions[1], {"per_eye": True})          # right eye: its own, untilted lid
    assert np.abs(f.render().astype(int) - img).sum() > 0


def make_rest(sent=None):
    cfg = Config(overrides={"sim": {"seed": 1}})
    sent = [] if sent is None else sent
    phone = SimPhone(cfg, create_engine("kinematic", World.load(), cfg), sent.append)
    changes = []
    face = FaceRest(cfg, phone.face, on_change=lambda: changes.append(1))
    return cfg, phone, face, changes


def test_rest_add_edit_delete_and_persist():
    cfg, phone, face, changes = make_rest()
    new = face.add(copy_of=4, patch={"name": "Grin"})
    assert new == 9 and phone.face.expressions[9].left.shape == "arc"
    assert face.patch(9, {"left_eye": {"lid_bottom": 0.3}}) == []
    assert face.patch(42, {}) is None
    phone.face.set_expression(9)
    assert face.delete(9) is None and phone.face.expression.id == 1      # back to the first
    assert len(changes) == 3
    saved = cfg.get("phone.face.expressions")
    assert [x["id"] for x in saved] == list(range(1, 9))
    # a restart with the saved config keeps the edits
    apply_patch(phone.face.expressions[2], {})
    face.patch(2, {"name": "Tired"})
    cfg2 = Config(overrides={"phone": {"face": {"expressions": cfg.get("phone.face.expressions")}}})
    f2 = Face(seed=0)
    FaceRest(cfg2, f2)
    assert f2.expressions[2].name == "Tired"


def test_phone_rest_face_routes():
    async def go():
        cfg, phone, face, _ = make_rest()
        settings = PhoneSettings(cfg, phone, 8081)
        async with TestClient(TestServer(build_phone_app(settings, face))) as client:
            assert (await (await client.get("/face/schema")).json())["schema"] == 1
            j = await (await client.get("/face/expressions")).json()
            assert j["active"] == 1 and len(j["expressions"]) == 8
            r = await client.post("/face/expressions/3", json={"blink": False, "x": 1})
            assert await r.json() == {"ok": True, "ignored": ["x"]}
            assert (await client.post("/face/expressions/99", json={})).status == 404
            r = await client.post("/face/expressions", json={"copy_of": 3, "name": "Angrier"})
            assert await r.json() == {"ok": True, "id": 9}
            names = (await (await client.get("/config")).json())["face"]["expressions"]
            assert "Angrier" in names
            r = await client.get("/face/preview.png?id=3&w=200&h=100")
            assert r.status == 200 and r.content_type == "image/png"
            img = np.asarray(Image.open(io.BytesIO(await r.read())).convert("RGB"))
            assert img.shape == (100, 200, 3)
            assert tuple(img[50, 50]) == (0, 0, 0)                     # pupil
            assert np.abs(img[50, 68].astype(int) - (0xEF, 0x53, 0x50)).max() < 8   # red iris
            assert (await client.get("/face/preview.png?id=99")).status == 404
            assert (await client.get("/face/preview.png")).status == 404
            assert (await client.delete("/face/expressions/9")).status == 200
            r = await client.options("/face/expressions/1")
            assert "DELETE" in r.headers["Access-Control-Allow-Methods"]
    asyncio.run(go())


def test_an_old_image_shape_is_ignored():
    x = Face(seed=0).expressions[1]
    assert apply_patch(x, {"left_eye": {"shape": "image"}}) == ["left_eye.shape"]
    assert x.left.shape == "round"


def test_names_are_unique_ignoring_case():
    cfg, _phone, face, _ = make_rest()
    assert face.patch(3, {"name": "normal"}) == ["name"]                 # Normal is taken
    assert face.patch(3, {"name": "ANGRY"}) == []                        # its own name, new case
    new = face.add(copy_of=3)
    assert face.face.expressions[new].name == "ANGRY (copy)"
    assert face.face.expressions[face.add(copy_of=3)].name == "ANGRY (copy) 2"
    assert face.face.expressions[face.add(None, {"name": "sleepy"})].name == "Expression"
    # an older save that repeats a name loads with unique names
    saved = [dict(x, name="Same") for x in face.list()["expressions"][:2]]
    cfg.update({"phone": {"face": {"expressions": saved}}})
    again = FaceRest(cfg, Face(seed=0))
    assert [x.name for x in again.face.expressions.values()] == ["Same", "Same 2"]
