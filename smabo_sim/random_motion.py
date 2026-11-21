"""Lifelike random servo motion, ported from ``smabo-esp32/random_motion.py``.

With ``servos.behavior = "random"``, the joints listed in each of
``servos.random_groups`` fire together on that group's own timer:

    {"name": "head", "joints": ["head_pan"], "interval": [1.0, 3.0],
     # optional tuning, same defaults as the firmware:
     "saccade_prob": 0.18, "drift": 0.07, "center_pull": 0.12,
     "drift_speed": 0.4, "long_pause_prob": 0.22}

Most fires are a slow, small drift pulled back toward the joint's rest angle;
now and then a quicker, larger "saccade". Hold times are usually short with
the odd long settle. The maths is the firmware's, step for step.

The firmware runs one asyncio task per group; here ``tick(now, servos)`` is
called every physics step and fires each group when its time comes, which
keeps it deterministic (seeded) and follows config changes (behaviour,
groups, joints, tuning) on the next fire without restarting anything.
"""

from __future__ import annotations

import random


class RandomMotion:
    """Per-group random motion driving a ``firmware.ServoFirmware``."""

    def __init__(self, cfg, rng: random.Random | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or random.Random()
        self._pos: dict[str, float] = {}     # joint -> last random target (random walk)
        self._next: dict[str, float] = {}    # group -> time of its next fire

    @property
    def active(self) -> bool:
        return self.cfg.get("servos.behavior", "manual") == "random"

    def tick(self, now: float, servos) -> None:
        if not self.active:
            self._next.clear()               # resume fresh when switched back on
            return
        groups = [g for g in (self.cfg.get("servos.random_groups") or []) if isinstance(g, dict)]
        names = {g.get("name", "") for g in groups}
        for name in list(self._next):
            if name not in names:
                del self._next[name]
        for g in groups:
            name = g.get("name", "")
            if now >= self._next.setdefault(name, now):   # a new group fires at once
                self._next[name] = now + self._fire(g, servos)

    # ------------------------------------------------------------------ #
    # one fire of one group (= one iteration of the firmware's run_group)
    # ------------------------------------------------------------------ #
    def _fire(self, g: dict, servos) -> float:
        """Move every joint of ``g`` and return how long to hold (s)."""
        rnd = self.rng
        saccade_prob = _gp(g, "saccade_prob", 0.18)
        drift_frac = _gp(g, "drift", 0.07)
        center_pull = _gp(g, "center_pull", 0.12)
        drift_speed = _gp(g, "drift_speed", 0.4)
        long_prob = _gp(g, "long_pause_prob", 0.22)

        saccade = rnd.random() < saccade_prob
        for jname in g.get("joints") or []:
            spec = servos.joints.get(jname)
            if spec is None:
                continue
            lo, hi = spec["min_angle"], spec["max_angle"]
            span = hi - lo
            if span <= 0:
                continue
            center = _center(spec)
            cur = self._pos.get(jname, center)
            base = spec.get("max_speed", 0)
            if base <= 0:
                base = 120.0                  # smooth default when configured "instant"
            if saccade:
                margin = span * 0.12
                target = rnd.uniform(lo + margin, hi - margin)
                speed = base
            else:
                target = (cur + self._gauss(span * drift_frac)
                          + center_pull * (center - cur)
                          + rnd.uniform(-1.0, 1.0) * span * 0.01)
                speed = base * drift_speed * rnd.uniform(0.7, 1.0)
            target = min(max(target, lo), hi)
            self._pos[jname] = target
            servos.set_angle_deg(jname, target, speed=speed)

        lo_t, hi_t = g.get("interval") or [1.0, 3.0]
        t = rnd.uniform(float(lo_t), float(hi_t))
        if saccade:
            t *= 0.6
        elif rnd.random() < long_prob:
            t *= rnd.uniform(1.8, 3.5)
        return max(t, 0.05)

    def _gauss(self, sigma: float) -> float:
        """~N(0, sigma) as the firmware computes it (12 uniforms − 6)."""
        return (sum(self.rng.random() for _ in range(12)) - 6.0) * sigma


def _gp(g: dict, key: str, default: float) -> float:
    try:
        return float(g.get(key, default))
    except (TypeError, ValueError):
        return default


def _center(spec: dict) -> float:
    """Rest angle: ``init_angle`` if within limits, else the midpoint."""
    lo, hi = spec["min_angle"], spec["max_angle"]
    c = spec.get("init_angle", 0)
    if c < lo or c > hi:
        c = (lo + hi) / 2.0
    return c
