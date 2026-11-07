"""2D world: line segments for lidar ray casting, collisions and Nav2 maps.

A world YAML lists shapes in the ``map`` frame (metres, degrees)::

    name: room
    start: {x: 0.0, y: 0.0, yaw_deg: 0}
    shapes:
      - {type: room,     x: 0, y: 0, w: 6, h: 5}          # 4 walls (outline), centred
      - {type: box,      x: 1.8, y: 0.8, w: 0.3, h: 0.8, yaw_deg: 0, name: shelf}
      - {type: wall,     x1: -1, y1: 1, x2: 0, y2: 1}
      - {type: cylinder, x: 0.5, y: -1.5, r: 0.15}

Any shape may also give ``height`` (m) for 3D engines such as MuJoCo
(defaults: walls 0.5, boxes 0.4, cylinders 0.5).

Only the outline of each shape matters: the LD06 sees a single horizontal
plane, so the world is exactly what it can hit at its mounting height.
The same file is exported as a Nav2 map (``export_map``) so AMCL localises the
simulated robot against the very walls its simulated lidar sees.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

BUNDLED_WORLD = os.path.join(os.path.dirname(__file__), "worlds", "room.yaml")


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # rad, CCW from +x


@dataclass
class World:
    name: str = "world"
    segments: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # [x1,y1,x2,y2]
    start: Pose2D = field(default_factory=Pose2D)
    shapes: list = field(default_factory=list)   # raw shape dicts (3D engines extrude them)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | None = None) -> "World":
        with open(path or BUNDLED_WORLD, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        segs: list[tuple[float, float, float, float]] = []
        for s in d.get("shapes") or []:
            segs.extend(_shape_segments(s))
        st = d.get("start") or {}
        start = Pose2D(float(st.get("x", 0.0)), float(st.get("y", 0.0)),
                       math.radians(float(st.get("yaw_deg", 0.0))))
        arr = np.array(segs, dtype=float).reshape(-1, 4)
        return cls(name=str(d.get("name", "world")), segments=arr, start=start,
                   shapes=[dict(s) for s in d.get("shapes") or []])

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def raycast(self, ox: float, oy: float, angles: np.ndarray, max_range: float) -> np.ndarray:
        """Distance along each ray to the nearest segment (``inf`` if none within range)."""
        angles = np.asarray(angles, dtype=float)
        out = np.full(angles.shape, np.inf)
        if len(self.segments) == 0:
            return out
        dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]           # (M,1)
        p = self.segments[:, :2][None, :, :]                                # (1,N,2)
        s = (self.segments[:, 2:] - self.segments[:, :2])[None, :, :]       # (1,N,2)
        qx, qy = p[..., 0] - ox, p[..., 1] - oy                             # p - o
        denom = dx * s[..., 1] - dy * s[..., 0]                             # cross(d, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (qx * s[..., 1] - qy * s[..., 0]) / denom                   # along ray
            t = (qx * dy - qy * dx) / denom                                 # along segment
        hit = (np.abs(denom) > 1e-12) & (u >= 0) & (t >= 0) & (t <= 1)
        u = np.where(hit, u, np.inf)
        out = u.min(axis=1)
        out[out > max_range] = np.inf
        return out

    def clearance(self, x: float, y: float) -> float:
        """Distance from (x, y) to the nearest segment (``inf`` for an empty world)."""
        if len(self.segments) == 0:
            return math.inf
        return float(_point_segment_dist(np.array([[x, y]]), self.segments).min())

    def bounds(self, margin: float = 0.5) -> tuple[float, float, float, float]:
        if len(self.segments) == 0:
            return (-5.0, -5.0, 5.0, 5.0)
        xs = self.segments[:, [0, 2]]
        ys = self.segments[:, [1, 3]]
        return (xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin)

    # ------------------------------------------------------------------ #
    # Nav2 map export
    # ------------------------------------------------------------------ #
    def occupancy(self, resolution: float = 0.05, margin: float = 0.5) -> tuple[np.ndarray, tuple[float, float]]:
        """Rasterise segments into a PGM-style image (0 = occupied, 254 = free).

        Returns the image (row 0 = top = max y, as map_server expects) and the
        world coordinates of its bottom-left corner (the map ``origin``).
        """
        x0, y0, x1, y1 = self.bounds(margin)
        w = int(math.ceil((x1 - x0) / resolution))
        h = int(math.ceil((y1 - y0) / resolution))
        cx = x0 + (np.arange(w) + 0.5) * resolution
        cy = y0 + (np.arange(h) + 0.5) * resolution
        gx, gy = np.meshgrid(cx, cy)                     # row i = y index from bottom
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        dist = np.full(len(pts), np.inf)
        for chunk in np.array_split(np.arange(len(pts)), max(1, len(pts) // 20000)):
            dist[chunk] = _point_segment_dist(pts[chunk], self.segments).min(axis=1) \
                if len(self.segments) else np.inf
        occ = (dist <= resolution * 0.75).reshape(h, w)
        img = np.where(occ, 0, 254).astype(np.uint8)
        return img[::-1, :], (x0, y0)

    def export_map(self, stem: str, resolution: float = 0.05) -> tuple[str, str]:
        """Write ``<stem>.pgm`` and ``<stem>.yaml`` for nav2_map_server."""
        img, (ox, oy) = self.occupancy(resolution)
        os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
        pgm, yml = f"{stem}.pgm", f"{stem}.yaml"
        h, w = img.shape
        with open(pgm, "wb") as f:
            f.write(f"P5\n# smabo-sim world: {self.name}\n{w} {h}\n255\n".encode())
            f.write(img.tobytes())
        with open(yml, "w", encoding="utf-8") as f:
            f.write(
                f"# Generated by smabo-sim from world '{self.name}'. Regenerate with\n"
                f"#   smabo-sim export-map <world.yaml> -o <stem>\n"
                f"image: {os.path.basename(pgm)}\n"
                f"mode: trinary\n"
                f"resolution: {resolution}\n"
                f"origin: [{ox:.3f}, {oy:.3f}, 0.0]\n"
                f"negate: 0\n"
                f"occupied_thresh: 0.65\n"
                f"free_thresh: 0.25\n"
            )
        return pgm, yml


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _rect_corners(cx, cy, w, h, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for lx, ly in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + c * lx - s * ly, cy + s * lx + c * ly))
    return pts


def _loop(pts):
    return [(*pts[i], *pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _shape_segments(s: dict):
    kind = s.get("type")
    yaw = math.radians(float(s.get("yaw_deg", 0.0)))
    if kind in ("room", "box"):
        return _loop(_rect_corners(float(s["x"]), float(s["y"]), float(s["w"]), float(s["h"]), yaw))
    if kind == "wall":
        return [(float(s["x1"]), float(s["y1"]), float(s["x2"]), float(s["y2"]))]
    if kind == "cylinder":
        n = int(s.get("sides", 16))
        r, cx, cy = float(s["r"]), float(s["x"]), float(s["y"])
        return _loop([(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
                      for k in range(n)])
    raise ValueError(f"unknown shape type: {kind!r}")


def _point_segment_dist(pts: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """(P,2) points × (N,4) segments → (P,N) distances."""
    a = segs[None, :, :2]
    ab = (segs[:, 2:] - segs[:, :2])[None, :, :]
    ap = pts[:, None, :] - a
    denom = (ab ** 2).sum(-1)
    t = np.clip(np.where(denom > 0, (ap * ab).sum(-1) / np.where(denom > 0, denom, 1), 0), 0, 1)
    closest = a + t[..., None] * ab
    return np.linalg.norm(pts[:, None, :] - closest, axis=-1)
