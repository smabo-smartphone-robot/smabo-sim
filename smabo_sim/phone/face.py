"""The smabo-app face, redrawn with Pillow for the simulated phone screen.

A port of smabo-app's ``ui/face`` (``face_screen.dart`` + ``eye_painter.dart``),
starting from the app's expression templates (``FaceExpression.templates``),
editable over the phone's REST API (``/face/expressions``, expressions.py):

  * two eyes at 1/4 and 3/4 of the width; shapes round / arc / heart / swirl /
    squeeze, size,
    shape rotation, eyelids (``lidTop`` / ``lidBottom``, tilted by ``rotation``)
    in the background colour; with shared eyes the right one mirrors the left;
  * the gaze eases toward its target with ``k = 8 /s`` (pupils move inside the
    sclera); ``/look_at`` sets the target exactly as ``parseLookAt`` does, and
    without one the eyes make random saccades like the app's "Random" mode;
  * blinks every 2-6 s lasting 0.18 s, for expressions that blink;
  * ``/speech/say`` text appears in a bubble for 2.5 + 0.18·len s (≤ 15 s).

Expressions are picked by name (``/expression``, std_msgs/String, ignoring
case); the integer id is internal. A deleted expression falls back to the
first one, the app's own fallback.
"""

from __future__ import annotations

import copy
import math
import os
import random
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _rgb(argb: int) -> tuple[int, int, int]:
    return ((argb >> 16) & 255, (argb >> 8) & 255, argb & 255)


@dataclass
class Eye:
    shape: str = "round"          # round | arc | heart | swirl | squeeze
    lid_top: float = 0.0
    lid_bottom: float = 0.0
    rotation: float = 0.0         # eyelid tilt, degrees
    shape_rotation: float = 0.0   # the eye shape itself, degrees
    sclera_scale: float = 1.0
    highlight_visible: bool = True
    iris: tuple = _rgb(0xFF4FC3F7)
    sclera: tuple = (255, 255, 255)
    pupil: tuple = (0, 0, 0)
    highlight: tuple = (255, 255, 255)
    iris_frac: float = 0.62
    pupil_frac: float = 0.41
    highlight_frac: float = 0.30
    highlight_x: float = -0.32
    highlight_y: float = -0.34


@dataclass
class Expression:
    id: int
    name: str
    left: Eye
    right: Eye = None             # None = a copy of left
    blink: bool = True
    background: tuple = (0, 0, 0)
    per_eye: bool = False         # False: the right eye mirrors the left

    def __post_init__(self) -> None:
        if self.right is None:
            self.right = copy.deepcopy(self.left)

    def eyes(self) -> tuple[Eye, Eye]:
        return self.left, (self.right if self.per_eye else self.left)


# FaceExpression.templates() in smabo-app
EXPRESSIONS: dict[int, Expression] = {e.id: e for e in (
    Expression(1, "Normal", Eye()),
    Expression(2, "Sleepy", Eye(lid_top=0.55, lid_bottom=0.12)),
    Expression(3, "Angry", Eye(iris=_rgb(0xFFEF5350), lid_top=0.22)),
    Expression(4, "Smile", Eye(shape="arc", iris=_rgb(0xFF66BB6A)), blink=False),
    Expression(5, "Sad", Eye(iris=_rgb(0xFF42A5F5), lid_top=0.3)),
    Expression(6, "Love", Eye(shape="heart", iris=_rgb(0xFFEC407A)), blink=False),
    Expression(7, "Dizzy", Eye(shape="swirl", iris=_rgb(0xFFBDBDBD)), blink=False),
    Expression(8, "Wink", Eye(), right=Eye(shape="arc"), per_eye=True),
)}


def expression_key(name: str) -> str:
    """``name`` as expressions are matched: case and surrounding spaces don't count."""
    return name.strip().lower()


def unique_name(base: str, taken) -> str:
    """``base``, or "base 2", "base 3", … whichever no name in ``taken`` matches."""
    used = {expression_key(n) for n in taken}
    base = base.strip() or "Expression"
    if expression_key(base) not in used:
        return base
    i = 2
    while expression_key(f"{base} {i}") in used:
        i += 1
    return f"{base} {i}"


def render_expression(expr: Expression, width: int, height: int) -> np.ndarray:
    """``expr`` alone (eyes open, looking straight ahead, no bubble) as an
    (height, width, 3) uint8 RGB image: the preview smabo-web shows."""
    f = Face(width, height)
    f.expressions = {expr.id: expr}
    f.state.expression = expr.id
    return f.render()


def gaze_from_look_at(x: float, y: float, z: float) -> tuple[float, float]:
    """``RosMessages.parseLookAt``: robot-frame direction → screen gaze (right, down)."""
    fwd = max(abs(x), 0.05)
    return (max(-1.0, min(1.0, -y / fwd)), max(-1.0, min(1.0, -z / fwd)))


def _find_font(size: int) -> ImageFont.ImageFont:
    """A font that can show Japanese if the OS has one; Pillow's default otherwise."""
    candidates = {
        "darwin": ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                   "/System/Library/Fonts/Hiragino Sans GB.ttc",
                   "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"],
        "win32": [os.path.expandvars(r"%WINDIR%\Fonts\YuGothB.ttc"),
                  os.path.expandvars(r"%WINDIR%\Fonts\meiryo.ttc"),
                  os.path.expandvars(r"%WINDIR%\Fonts\msgothic.ttc")],
    }.get(sys.platform, ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                         "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
                         "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"])
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


@dataclass
class FaceState:
    expression: int = 1
    gaze: list = field(default_factory=lambda: [0.0, 0.0])
    target: list = field(default_factory=lambda: [0.0, 0.0])
    blink: float = 1.0            # 1 open .. 0 closed
    bubble: str = ""


class Face:
    """Animated face. ``tick(dt)`` advances it, ``render()`` draws one frame."""

    FOLLOW_TIMEOUT = 3.0          # s without /look_at → back to random saccades

    def __init__(self, width: int = 512, height: int = 256, seed: int | None = None) -> None:
        self.width, self.height = width, height
        self.state = FaceState()
        # this phone's expressions: the templates until edited over REST
        self.expressions: dict[int, Expression] = copy.deepcopy(EXPRESSIONS)
        self._rng = random.Random(seed)
        self._t = 0.0
        self._next_blink = self._t + 2.0 + self._rng.random() * 4.0
        self._blink_phase = -1.0
        self._next_saccade = 0.0
        self._last_look_at = -math.inf
        self._bubble_until = 0.0
        self._ss = 2                                   # supersampling for anti-aliasing
        self._font = _find_font(int(height * 0.085) * self._ss)

    # ------------------------------------------------------------------ #
    # inputs (topics the app subscribes to)
    # ------------------------------------------------------------------ #
    def set_expression(self, expr_id: int) -> None:
        """Show expression ``expr_id`` (the internal id; the first one if unknown)."""
        self.state.expression = expr_id if expr_id in self.expressions else next(iter(self.expressions))

    def find(self, name: str) -> Expression | None:
        """The expression called ``name`` (ignoring case and surrounding spaces), or None."""
        k = expression_key(name)
        return next((x for x in self.expressions.values() if expression_key(x.name) == k), None)

    def set_expression_by_name(self, name: str) -> bool:
        """Show the expression called ``name`` (``/expression``). False if there is none."""
        x = self.find(name)
        if x is not None:
            self.state.expression = x.id
        return x is not None

    @property
    def expression(self) -> Expression:
        """The one shown (the first one if it was deleted)."""
        e = self.expressions.get(self.state.expression)
        if e is None:
            self.state.expression = next(iter(self.expressions))
            e = self.expressions[self.state.expression]
        return e

    def look_at(self, x: float, y: float, z: float) -> None:
        self.state.target = list(gaze_from_look_at(x, y, z))
        self._last_look_at = self._t

    def say(self, text: str) -> None:
        text = (text or "").strip()
        self.state.bubble = text
        self._bubble_until = self._t + min(15.0, max(2.5, 2.5 + len(text) * 0.18))

    @property
    def following(self) -> bool:
        return self._t - self._last_look_at < self.FOLLOW_TIMEOUT

    # ------------------------------------------------------------------ #
    def tick(self, dt: float) -> None:
        s = self.state
        self._t += dt
        if not self.following and self._t >= self._next_saccade:
            s.target = [(self._rng.random() * 2 - 1) * 0.8, (self._rng.random() * 2 - 1) * 0.6]
            self._next_saccade = self._t + 1.0 + self._rng.random() * 2.5
        a = min(1.0, max(0.0, 8.0 * dt))
        s.gaze = [g + (t - g) * a for g, t in zip(s.gaze, s.target)]

        if not self.expression.blink:
            s.blink, self._blink_phase = 1.0, -1.0
        else:
            if self._blink_phase < 0 and self._t >= self._next_blink:
                self._blink_phase = 0.0
            if self._blink_phase >= 0:
                self._blink_phase += dt / 0.18
                p = self._blink_phase
                s.blink = min(1.0, max(0.0, 1 - p / 0.5 if p < 0.5 else (p - 0.5) / 0.5))
                if self._blink_phase >= 1.0:
                    s.blink, self._blink_phase = 1.0, -1.0
                    self._next_blink = self._t + 2.0 + self._rng.random() * 4.0
        if s.bubble and self._t >= self._bubble_until:
            s.bubble = ""

    # ------------------------------------------------------------------ #
    def render(self) -> np.ndarray:
        """Current face as an (height, width, 3) uint8 RGB image."""
        ss = self._ss
        W, H = self.width * ss, self.height * ss
        expr = self.expression
        img = Image.new("RGB", (W, H), expr.background)
        left, right = expr.eyes()
        # shared eyes: the right one mirrors the left's tilt and rotation (eye_painter.dart)
        sign = 1.0 if expr.per_eye else -1.0
        for cx, eye, is_left, rot, shape_rot in (
                (W * 0.25, left, True, left.rotation, left.shape_rotation),
                (W * 0.75, right, False, sign * right.rotation, sign * right.shape_rotation)):
            self._eye(img, (cx, H * 0.5), eye, expr.background, is_left, rot, shape_rot)
        if self.state.bubble:
            self._bubble(img, self.state.bubble)
        return np.asarray(img.resize((self.width, self.height), Image.LANCZOS))

    def _eye(self, img: Image.Image, center, eye: Eye, bg, is_left: bool,
             lid_rot: float = 0.0, shape_rot: float = 0.0) -> None:
        W, H = img.size
        max_r = min(W * 0.22, H * 0.46)
        r = min(max(H * 0.36 * eye.sclera_scale, 8.0), max_r)
        cx, cy = center
        box = (int(cx - r), int(cy - r), int(cx + r) + 1, int(cy + r) + 1)
        layer = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), bg)
        d = ImageDraw.Draw(layer)
        ox, oy = cx - box[0], cy - box[1]
        t = self.state.blink
        line = eye.iris
        if t > 0.04:
            if eye.shape == "round":
                self._round(d, ox, oy, r, eye)
            elif eye.shape == "arc":
                pts = _quad_bezier((ox - r * 0.72, oy + r * 0.18), (ox, oy - r * 0.55),
                                   (ox + r * 0.72, oy + r * 0.18))
                _stroke(d, pts, line, r * 0.2)
            elif eye.shape == "heart":
                self._heart(d, ox, oy, r, eye)
            elif eye.shape == "swirl":
                pts = [(ox + f * r * 0.85 * math.cos(f * 6 * math.pi), oy + f * r * 0.85 * math.sin(f * 6 * math.pi))
                       for f in np.linspace(0, 1, 141)]
                _stroke(d, pts, line, r * 0.12)
            elif eye.shape == "squeeze":
                bx = ox + (-r * 0.5 if is_left else r * 0.5)
                vx = ox + (r * 0.48 if is_left else -r * 0.48)
                _stroke(d, [(bx, oy - r * 0.5), (vx, oy), (bx, oy + r * 0.5)], line, r * 0.18)
        if shape_rot:                        # Flutter's rotate is clockwise on screen
            layer = layer.rotate(-shape_rot, resample=Image.BICUBIC, center=(ox, oy), fillcolor=bg)
        top = min(1.0, max(eye.lid_top, 1.0 - t))
        curve = r * 0.16
        lids = Image.new("L", layer.size, 0)
        ld = ImageDraw.Draw(lids)
        if top > 0:
            edge = oy - r + top * 2 * r
            ld.polygon([(ox - r, oy - r), (ox + r, oy - r), (ox + r, edge),
                        *_quad_bezier((ox + r, edge), (ox, edge + curve), (ox - r, edge))[1:]], fill=255)
        if eye.lid_bottom > 0:
            edge = oy + r - eye.lid_bottom * 2 * r
            ld.polygon([(ox - r, oy + r), (ox + r, oy + r), (ox + r, edge),
                        *_quad_bezier((ox + r, edge), (ox, edge - curve), (ox - r, edge))[1:]], fill=255)
        if top > 0 or eye.lid_bottom > 0:
            if lid_rot:                      # the lids tilt, the eye stays upright
                lids = lids.rotate(-lid_rot, resample=Image.BICUBIC, center=(ox, oy))
            layer.paste(bg, (0, 0), lids)
        mask = Image.new("L", layer.size, 0)
        ImageDraw.Draw(mask).ellipse((ox - r, oy - r, ox + r, oy + r), fill=255)
        img.paste(layer, box[:2], mask)

    def _round(self, d: ImageDraw.ImageDraw, ox, oy, r, eye: Eye) -> None:
        _disc(d, ox, oy, r, eye.sclera)
        ir, pr = r * eye.iris_frac, r * eye.pupil_frac
        off = max(0.0, r - ir)
        cx, cy = ox + self.state.gaze[0] * off, oy + self.state.gaze[1] * off
        _disc(d, cx, cy, ir, eye.iris)
        _disc(d, cx, cy, pr, eye.pupil)
        if eye.highlight_visible:
            hr = ir * eye.highlight_frac
            hoff = max(0.0, ir - hr)
            _disc(d, cx + eye.highlight_x * hoff, cy + eye.highlight_y * hoff, hr, eye.highlight)

    @staticmethod
    def _heart(d: ImageDraw.ImageDraw, ox, oy, r, eye: Eye) -> None:
        w = h = r * 1.35
        cy = oy + r * 0.08
        a = (ox, cy + h * 0.35)
        pts = _cubic(a, (ox - w * 0.5, cy - h * 0.08), (ox - w * 0.5, cy - h * 0.55), (ox, cy - h * 0.22))
        pts += _cubic((ox, cy - h * 0.22), (ox + w * 0.5, cy - h * 0.55), (ox + w * 0.5, cy - h * 0.08), a)[1:]
        d.polygon(pts, fill=eye.iris)
        if eye.highlight_visible:
            _disc(d, ox - w * 0.18, cy - h * 0.18, r * 0.12, eye.highlight)

    def _bubble(self, img: Image.Image, text: str) -> None:
        W, H = img.size
        d = ImageDraw.Draw(img)
        if len(text) > 40:
            text = text[:39] + "…"
        bbox = d.textbbox((0, 0), text, font=self._font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = H * 0.035
        cx, cy = W / 2, H * 0.19                       # Alignment(0, -0.62)
        rect = (cx - tw / 2 - pad * 1.6, cy - th / 2 - pad, cx + tw / 2 + pad * 1.6, cy + th / 2 + pad)
        d.rounded_rectangle(rect, radius=pad * 1.2, fill=(255, 255, 255))
        d.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), text, font=self._font, fill=(20, 20, 24))


# ---- drawing helpers ------------------------------------------------------- #
def _disc(d, x, y, r, fill) -> None:
    d.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def _stroke(d, pts, fill, width) -> None:
    w = max(1, int(round(width)))
    d.line(pts, fill=fill, width=w, joint="curve")
    for x, y in (pts[0], pts[-1]):                     # round caps
        _disc(d, x, y, w / 2, fill)


def _quad_bezier(p0, p1, p2, n: int = 24) -> list:
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in np.linspace(0, 1, n)]


def _cubic(p0, p1, p2, p3, n: int = 24) -> list:
    out = []
    for t in np.linspace(0, 1, n):
        u = 1 - t
        out.append((u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
                    u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1]))
    return out


__all__ = ["EXPRESSIONS", "Expression", "Eye", "Face", "expression_key", "gaze_from_look_at",
           "unique_name"]
