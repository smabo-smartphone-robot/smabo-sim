"""What the simulated ESP32 and phone exchange with the brain.

Same handshake as the real devices (smabo-esp32's brain_link.py, smabo-app's
ros_compat.dart): connect to ``ws://<brain>:9090/``, advertise what we publish,
subscribe to what we read, publish with the source prefix. Works with
smabo-brain and smabo-brain-ros alike. The tables come from smabo-protocol
(protocol_gen.py, generated).
"""

import json

# Topic tables and protocol number, generated from smabo-protocol.
from .protocol_gen import (APP_ADVERTISE, APP_PREFIX, APP_SUBSCRIBE,  # noqa: F401
                           ESP32_ADVERTISE, ESP32_PREFIX, ESP32_SUBSCRIBE, PROTOCOL_VERSION)

BRAIN_PATH = "/"   # both brains; config "brain.path" overrides


def hello_msg(client: str, version: str) -> dict:
    """``/hello`` payload (std_msgs/String) for ``client`` ("smabo-sim (esp32)" …)."""
    return {"data": json.dumps({"protocol": PROTOCOL_VERSION, "client": client,
                                "version": version})}


def brain_info_warning(msg: dict, client: str) -> str | None:
    """``/brain/info`` → a warning when the brain speaks another protocol, else None."""
    try:
        info = json.loads((msg or {}).get("data") or "{}")
    except (TypeError, ValueError):
        return None
    protocol = info.get("protocol") if isinstance(info, dict) else None
    if protocol is None or protocol == PROTOCOL_VERSION:
        return None
    return (f"protocol mismatch: brain {info.get('brain')} {info.get('version')} speaks "
            f"{protocol}, {client} speaks {PROTOCOL_VERSION}")
