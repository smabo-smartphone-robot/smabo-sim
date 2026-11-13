"""Pairing with smabo-brain, as smabo-esp32 and smabo-app do it.

A paired device shows its token as the first frame on every connection
(``{"op": "auth", "token": ...}``). To pair, it shows a one-time invite code
from smabo-web instead (``{"op": "pair", "code": ...}``) and the brain answers
with a token and a relay key. The device keeps the token and only the hash of
the relay key; from then on its REST API answers only requests that carry the
key (``X-Smabo-Relay-Key``, sent by the brain when it relays smabo-web's
settings) and sends no CORS headers.

The invite reaches the device as ``{"brain": {"invite": "ABCD-EFGH"}}`` posted
to its REST ``/config`` (open while the device is not paired), or with
``--invite`` / ``--phone-invite`` on the command line.

The simulator keeps the token in ``~/.smabo/sim/<esp32|app>.json`` (the board
keeps it in config.json, the app in its preferences). When the brain says it
no longer knows the token (the device was removed in smabo-web, or the brain
was reset), the device forgets it and its REST API takes invites again.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("smabo_sim.pairing")

RELAY_KEY_HEADER = "X-Smabo-Relay-Key"
DEFAULT_DIR = Path.home() / ".smabo" / "sim"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Credentials:
    """One device's token, pending invite and relay-key hash."""

    def __init__(self, role: str, name: str, path: Path | None = None) -> None:
        self.role = role
        self.name = name
        self.path = path
        self.token = ""
        self.relay_key_sha256 = ""
        self.invite = ""
        self.refused = ""            # last reason the brain gave, for the logs / REST
        # False while the brain says it does not check tokens (it runs without
        # --pairing): then smabo-web talks to this REST directly, so it opens.
        self.brain_checks_tokens = True
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.token = str(data.get("token", ""))
                self.relay_key_sha256 = str(data.get("relay_key_sha256", ""))
            except (OSError, ValueError, AttributeError):
                log.warning("%s: %s is unreadable; this device is not paired", role, path)

    @property
    def paired(self) -> bool:
        return bool(self.token)

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"token": self.token, "relay_key_sha256": self.relay_key_sha256}, f)

    def set_invite(self, code: str) -> None:
        self.invite = str(code).strip()

    def first_frame(self) -> dict | None:
        """What to send right after connecting, or None (an unpaired device without an invite)."""
        if self.invite:
            return {"op": "pair", "code": self.invite, "role": self.role, "name": self.name}
        if self.token:
            return {"op": "auth", "token": self.token}
        return None

    def handle_reply(self, frame: dict) -> bool | None:
        """The brain's answer to ``first_frame``: True accepted, False refused, None not an answer."""
        op = frame.get("op")
        if op not in ("auth", "pair"):
            return None
        ok = frame.get("ok") is True
        reason = str(frame.get("reason") or "")
        if reason == "no_auth" or frame.get("auth") is False:
            # The brain does not check tokens: keep ours for when it does.
            self.brain_checks_tokens = False
            self.invite = ""
            self.refused = ""
            return True
        self.brain_checks_tokens = True
        if op == "pair":
            self.invite = ""                       # one try per code
            if ok and frame.get("token"):
                self.token = str(frame["token"])
                key = frame.get("relay_key")
                self.relay_key_sha256 = _sha256(key) if isinstance(key, str) and key else ""
                self._save()
                log.info("%s: paired with the brain as %r", self.role, frame.get("name"))
            elif not ok:
                log.warning("%s: the brain refused the invite code (%s). Get a new one in "
                            "smabo-web → Devices", self.role, reason)
        elif not ok:
            if self.token and reason == "unauthorized":
                # Removed from the brain (or the brain was reset): forget the
                # pairing so the REST API takes a new invite again.
                log.warning("%s: the brain no longer knows this device; it is unpaired now. "
                            "Pair it again from smabo-web → Devices", self.role)
                self.token = ""
                self.relay_key_sha256 = ""
                self._save()
            elif self.token:
                log.warning("%s: the brain refused this device's token (%s)", self.role, reason)
            else:
                log.warning("%s: the brain needs pairing. Get an invite code in smabo-web → Devices "
                            "and post {\"brain\": {\"invite\": CODE}} to this device's /config "
                            "(or start with --%sinvite CODE)", self.role,
                            "phone-" if self.role == "app" else "")
        self.refused = "" if ok else reason or "refused"
        return ok

    def relay_allowed(self, key: str | None) -> bool:
        """Whether a REST request may proceed: anything while unpaired, else the brain's key only."""
        if not self.locked:
            return True
        return bool(key) and hmac.compare_digest(_sha256(key), self.relay_key_sha256)

    @property
    def locked(self) -> bool:
        """Paired with a relay key and the brain checks tokens: REST only for the brain, no CORS."""
        return bool(self.relay_key_sha256) and self.brain_checks_tokens

    def status(self) -> dict:
        """For GET /config: whether this device is paired (never the token)."""
        return {"paired": self.paired, **({"refused": self.refused} if self.refused else {})}
