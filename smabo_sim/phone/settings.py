"""The phone's settings as smabo-app exposes them: its REST ``/config``.

smabo-app serves ``GET /config`` / ``POST /config`` on port 8081 (the ESP32's
REST API, same shape: a JSON tree, POST deep-merges a partial patch) and
announces itself on ``/app/phone/config_url``; after every change it publishes
the whole tree on ``/app/phone/config`` so open smabo-web pages refresh. This
module is the simulator's copy of that contract (``lib/core/models/
phone_config.dart`` in smabo-app), mapped onto the simulated phone's settings
(``phone.*`` in the simulator's config, which are not on the ESP32's REST port).

    {
      "brain":   {"host": str, "port": int},                 read-only here
      "http":    {"port": int},                              read-only
      "sensors": {"imu":    {"enabled": bool, "rate_hz": int},
                  "gps":    {"enabled": bool},
                  "camera": {"enabled": bool, "front": bool}},
      "voice":   {"enabled": bool, "wake_word": str, "stt_locale": str},
      "speech":  {"tts": bool, "tts_language": str, "bubble": bool, "ignore_repeated": bool},
      "face":    {"auto_hide": bool, "auto_hide_seconds": int,
                  "expression": str,          the name shown (matched ignoring case)
                  "expressions": [str]},      the names, read-only
      "app":     {"name": str, "version": str, "schema": int}  read-only
    }

Unknown keys and read-only keys in a patch are ignored and listed back, like
the ESP32 ignores what it doesn't know.
"""

from __future__ import annotations

from typing import Any, Callable

SCHEMA = 1
DEFAULT_PORT = 8081

# schema path → (simulator config key, type, default)
FIELDS: dict[str, tuple[str, type, Any]] = {
    "sensors.imu.enabled":     ("phone.imu_enabled", bool, True),
    "sensors.imu.rate_hz":     ("phone.imu_rate", int, 20),
    "sensors.gps.enabled":     ("phone.gps.enabled", bool, True),
    "sensors.camera.enabled":  ("phone.camera.stream", bool, True),
    "sensors.camera.front":    ("phone.camera.front", bool, True),
    "voice.enabled":           ("phone.voice.enabled", bool, True),
    "voice.wake_word":         ("phone.voice.wake_word", str, "smabo"),
    "voice.stt_locale":        ("phone.voice.stt_locale", str, "ja_JP"),
    "speech.tts":              ("phone.tts.enabled", bool, True),
    "speech.tts_language":     ("phone.tts.language", str, "ja-JP"),
    "speech.bubble":           ("phone.speech.bubble", bool, True),
    "speech.ignore_repeated":  ("phone.speech.ignore_repeated", bool, True),
    "face.auto_hide":          ("phone.face.auto_hide", bool, True),
    "face.auto_hide_seconds":  ("phone.face.auto_hide_seconds", int, 5),
}
LIMITS = {"sensors.imu.rate_hz": (1, 200), "face.auto_hide_seconds": (1, 3600)}


def _flatten(tree: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in tree.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, path + "."))
        else:
            out[path] = v
    return out


def _set(tree: dict, path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    for p in parents:
        tree = tree.setdefault(p, {})
    tree[leaf] = value


def _coerce(value: Any, typ: type) -> Any:
    """``value`` as ``typ``, or raise ValueError (no silent str → bool etc.)."""
    if typ is bool:
        if isinstance(value, bool):
            return value
    elif typ is int:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value).is_integer():
            return int(value)
    elif typ is str:
        if isinstance(value, str):
            return value.strip()
    raise ValueError


class PhoneSettings:
    """The simulated phone's REST ``/config``: read, patch, and react."""

    def __init__(self, cfg, phone, port: int = DEFAULT_PORT,
                 on_change: Callable[[set[str]], None] | None = None,
                 version: str = "") -> None:
        self.cfg = cfg
        self.phone = phone
        self.port = port
        self.on_change = on_change          # called with the changed schema paths
        self.version = version

    def to_json(self) -> dict:
        tree: dict = {
            "brain": {"host": str(self.cfg.get("brain.host", "localhost")),
                      "port": int(self.cfg.get("brain.port", 9090))},
            "http": {"port": self.port},
        }
        for path, (key, typ, default) in FIELDS.items():
            _set(tree, path, typ(self.cfg.get(key, default)))
        tree["face"]["expression"] = self.phone.face.expression.name
        tree["face"]["expressions"] = [e.name for e in self.phone.face.expressions.values()]
        tree["app"] = {"name": "smabo-sim", "version": self.version, "schema": SCHEMA}
        return tree

    def apply(self, patch: dict) -> list[str]:
        """Apply a partial tree. Returns the paths that were ignored."""
        ignored: list[str] = []
        updates: dict = {}
        changed: set[str] = set()
        for path, value in _flatten(patch).items():
            if path == "face.expression":
                x = self.phone.face.find(value) if isinstance(value, str) else None
                if x is None:
                    ignored.append(path)
                    continue
                if x.id != self.phone.face.state.expression:
                    self.phone.face.set_expression(x.id)
                    changed.add(path)
                continue
            field = FIELDS.get(path)
            if field is None:                      # unknown or read-only (brain, http, app, list)
                ignored.append(path)
                continue
            key, typ, default = field
            try:
                v = _coerce(value, typ)
            except ValueError:
                ignored.append(path)
                continue
            lo_hi = LIMITS.get(path)
            if lo_hi:
                v = min(max(v, lo_hi[0]), lo_hi[1])
            if typ is str and not v:
                ignored.append(path)
                continue
            if v != typ(self.cfg.get(key, default)):
                _set(updates, key, v)
                changed.add(path)
        if updates:
            self.cfg.update(updates)
        if changed and self.on_change:
            self.on_change(changed)
        return ignored


__all__ = ["DEFAULT_PORT", "FIELDS", "PhoneSettings", "SCHEMA"]
