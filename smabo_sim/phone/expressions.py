"""Face expressions over the phone's REST API, as smabo-app serves them.

smabo-app (``lib/core/models/face_schema.dart``) serves on its settings port:

    GET    /face/schema            the editable fields; smabo-web builds its editor from it
    GET    /face/expressions       {"active": id, "expressions": [...]}
    POST   /face/expressions/<id>  a partial expression → {"ok": true, "ignored": [...]}
    POST   /face/expressions       {"copy_of"?: id, ...fields} → {"ok": true, "id": new id}
    DELETE /face/expressions/<id>  → {"ok": true}
    GET    /face/preview.png?id=<id>&w=<px>&h=<px>   the expression as drawn on the phone

An expression on the wire (colours "#RRGGBB", shape by name):

    {"id": 1, "name": "Normal", "background": "#000000", "blink": true, "per_eye": false,
     "left_eye": {"shape": "round", "lid_top": 0.0, ...}, "right_eye": {...}}

``SCHEMA`` must equal smabo-app's ``test/face_schema.json`` (a test compares
them when the two repositories sit side by side). Edited expressions are kept
in the simulator's config as ``phone.face.expressions`` (saved with --save).
"""

from __future__ import annotations

import copy
import io
import math
import re
from typing import Any, Callable

from PIL import Image

from .face import Expression, Eye, expression_key, render_expression, unique_name

SHAPES = ["round", "arc", "heart", "swirl", "squeeze"]
PREVIEW_MAX_SIDE = 1024
_ROUND = ["round"]
_ROUND_HEART = ["round", "heart"]
_LINED = ["round", "arc", "heart", "swirl", "squeeze"]


def _f(key, typ, label, group=None, lo=None, hi=None, step=None, options=None, shapes=None,
       when=None) -> dict:
    out: dict[str, Any] = {"key": key, "type": typ, "label": label}
    if group is not None:
        out["group"] = group
    if lo is not None:
        out["min"], out["max"], out["step"] = float(lo), float(hi), float(step)
    if options is not None:
        out["options"] = options
    if shapes is not None:
        out["shapes"] = shapes
    if when is not None:
        out["when"] = when                      # a bool field that must be true
    return out


# schema key → Eye attribute (colours are RGB tuples on the Eye)
EYE_ATTR = {
    "shape": "shape", "sclera_color": "sclera", "iris_color": "iris",
    "sclera_scale": "sclera_scale", "shape_rotation": "shape_rotation",
    "iris_frac": "iris_frac", "pupil_color": "pupil", "pupil_frac": "pupil_frac",
    "highlight_visible": "highlight_visible", "highlight_color": "highlight",
    "highlight_frac": "highlight_frac", "highlight_x": "highlight_x", "highlight_y": "highlight_y",
    "lid_top": "lid_top", "lid_bottom": "lid_bottom", "rotation": "rotation",
}

SCHEMA: dict = {
    "schema": 1,
    "expression": [
        _f("name", "string", "Name (unique: /expression picks it by name)"),
        _f("background", "color", "Background (also the eyelids)"),
        _f("blink", "bool", "Blink"),
        _f("per_eye", "bool", "Separate left / right eyes"),
    ],
    "eye": [
        _f("shape", "enum", "Shape", "Shape", options=SHAPES),
        _f("sclera_color", "color", "Sclera (white)", "Shape", shapes=_ROUND),
        _f("iris_color", "color", "Iris / line color", "Shape", shapes=_LINED),
        _f("sclera_scale", "number", "Size", "Shape", 0.3, 1.6, 0.01),
        _f("shape_rotation", "number", "Rotation (°)", "Shape", -180, 180, 1),
        _f("iris_frac", "number", "Iris size", "Iris & pupil", 0.1, 1.0, 0.01, shapes=_ROUND),
        _f("pupil_color", "color", "Pupil color", "Iris & pupil", shapes=_ROUND),
        _f("pupil_frac", "number", "Pupil size (≤ iris size)", "Iris & pupil", 0.05, 1.0, 0.01, shapes=_ROUND),
        _f("highlight_visible", "bool", "Highlight", "Highlight", shapes=_ROUND_HEART),
        _f("highlight_color", "color", "Highlight color", "Highlight", shapes=_ROUND_HEART, when="highlight_visible"),
        _f("highlight_frac", "number", "Highlight size", "Highlight", 0.05, 0.6, 0.01, shapes=_ROUND, when="highlight_visible"),
        _f("highlight_x", "number", "Highlight X", "Highlight", -1, 1, 0.01, shapes=_ROUND, when="highlight_visible"),
        _f("highlight_y", "number", "Highlight Y", "Highlight", -1, 1, 0.01, shapes=_ROUND, when="highlight_visible"),
        _f("lid_top", "number", "Upper lid", "Eyelid", 0, 1, 0.01),
        _f("lid_bottom", "number", "Lower lid", "Eyelid", 0, 1, 0.01),
        _f("rotation", "number", "Lid tilt (°)", "Eyelid", -180, 180, 1),
    ],
    "eyes": ["left_eye", "right_eye"],
    "notes": {
        "right_eye": "used only when per_eye is true; otherwise it mirrors left_eye",
    },
}
_EYE_FIELDS = {f["key"]: f for f in SCHEMA["eye"]}
_EXPR_FIELDS = {f["key"]: f for f in SCHEMA["expression"]}


def to_hex(rgb) -> str:
    return "#%02X%02X%02X" % tuple(int(c) for c in rgb[:3])


def from_hex(s: str):
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", s.strip()) if isinstance(s, str) else None
    if not m:
        return None
    v = int(m.group(1), 16)
    return ((v >> 16) & 255, (v >> 8) & 255, v & 255)


def _coerce(field: dict, value: Any) -> Any:
    """``value`` as the field's type, or None when it doesn't fit."""
    t = field["type"]
    if t == "bool":
        return value if isinstance(value, bool) else None
    if t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        return min(max(float(value), field.get("min", -math.inf)), field.get("max", math.inf))
    if t == "color":
        return from_hex(value)
    if t == "string":
        return value.strip() if isinstance(value, str) and value.strip() else None
    if t == "enum":
        return value if value in field["options"] else None
    return None


def eye_to_json(e: Eye) -> dict:
    out = {}
    for key, attr in EYE_ATTR.items():
        v = getattr(e, attr)
        f = _EYE_FIELDS[key]
        out[key] = to_hex(v) if f["type"] == "color" else (float(v) if f["type"] == "number" else v)
    return out


def expression_to_json(x: Expression) -> dict:
    return {"id": x.id, "name": x.name, "background": to_hex(x.background), "blink": x.blink,
            "per_eye": x.per_eye, "left_eye": eye_to_json(x.left), "right_eye": eye_to_json(x.right)}


def apply_patch(x: Expression, patch: dict, taken=()) -> list[str]:
    """Apply a partial expression to ``x`` in place. Returns the ignored paths.
    A name that one in ``taken`` already has (ignoring case) is ignored.

    Like the app's editor: with shared eyes (per_eye false) the right eye is a
    copy of the left, and turning per_eye on starts the right eye as that copy.
    """
    was_per_eye = x.per_eye
    used = {expression_key(n) for n in taken}
    ignored: list[str] = []
    for key, value in patch.items():
        if key in ("left_eye", "right_eye"):
            if not isinstance(value, dict):
                ignored.append(key)
                continue
            eye = x.left if key == "left_eye" else x.right
            for ek, ev in value.items():
                f = _EYE_FIELDS.get(ek)
                v = None if f is None else _coerce(f, ev)
                if v is None:
                    ignored.append(f"{key}.{ek}")
                else:
                    setattr(eye, EYE_ATTR[ek], v)
            eye.pupil_frac = min(eye.pupil_frac, eye.iris_frac)
            continue
        f = _EXPR_FIELDS.get(key)
        v = None if f is None else _coerce(f, value)
        if v is None or (key == "name" and expression_key(v) in used):
            ignored.append(key)                     # id, unknown keys, a name in use
        else:
            setattr(x, key, v)
    if not x.per_eye or (not was_per_eye and "right_eye" not in patch):
        x.right = copy.deepcopy(x.left)
    return ignored


def expression_from_json(j: dict) -> Expression:
    x = Expression(int(j["id"]), str(j.get("name") or f"Expression {j['id']}"), Eye())
    apply_patch(x, {k: v for k, v in j.items() if k != "id"})
    return x


class FaceRest:
    """``/face/...`` for the simulated phone (same contract as smabo-app)."""

    def __init__(self, cfg, face, on_change: Callable[[], None] | None = None) -> None:
        self.cfg = cfg
        self.face = face
        self.on_change = on_change
        saved = cfg.get("phone.face.expressions")
        if isinstance(saved, list) and saved:
            try:
                loaded = list(map(expression_from_json, saved))
                names: list[str] = []
                for x in loaded:                     # names are unique (older saves may repeat)
                    x.name = unique_name(x.name, names)
                    names.append(x.name)
                face.expressions = {x.id: x for x in loaded}
                face.set_expression(face.state.expression)
            except (KeyError, TypeError, ValueError):
                pass                                 # keep the templates

    def schema(self) -> dict:
        return copy.deepcopy(SCHEMA)

    def list(self) -> dict:
        return {"active": self.face.expression.id,
                "expressions": [expression_to_json(x) for x in self.face.expressions.values()]}

    def patch(self, expr_id: int, patch: dict) -> list[str] | None:
        x = self.face.expressions.get(expr_id)
        if x is None:
            return None
        ignored = apply_patch(x, patch, self._names_except(expr_id))
        self._changed()
        return ignored

    def add(self, copy_of: int | None = None, patch: dict | None = None) -> int | None:
        src = None if copy_of is None else self.face.expressions.get(copy_of)
        if copy_of is not None and src is None:
            return None
        new_id = max(self.face.expressions) + 1
        taken = self._names_except(None)
        if src is None:
            x = Expression(new_id, unique_name("Expression", taken), Eye())
        else:
            x = copy.deepcopy(src)
            x.id, x.name = new_id, unique_name(f"{src.name} (copy)", taken)
        apply_patch(x, patch or {}, taken)
        self.face.expressions[new_id] = x
        self._changed()
        return new_id

    def delete(self, expr_id: int) -> str | None:
        if expr_id not in self.face.expressions:
            return "no such expression"
        if len(self.face.expressions) <= 1:
            return "the last expression cannot be deleted"
        del self.face.expressions[expr_id]
        self.face.set_expression(self.face.state.expression)
        self._changed()
        return None

    def preview(self, expr_id: int, width: int = 400, height: int = 200) -> bytes | None:
        """Expression ``expr_id`` drawn as a PNG (sides clamped to 16..1024), or None."""
        x = self.face.expressions.get(expr_id)
        if x is None:
            return None
        w, h = (min(max(int(v), 16), PREVIEW_MAX_SIDE) for v in (width, height))
        buf = io.BytesIO()
        Image.fromarray(render_expression(x, w, h)).save(buf, "PNG")
        return buf.getvalue()

    def _names_except(self, expr_id: int | None) -> list[str]:
        return [x.name for x in self.face.expressions.values() if x.id != expr_id]

    def _changed(self) -> None:
        self.cfg.update({"phone": {"face": {"expressions": self.list()["expressions"]}}})
        if self.on_change:
            self.on_change()


__all__ = ["FaceRest", "SCHEMA", "apply_patch", "expression_to_json"]
