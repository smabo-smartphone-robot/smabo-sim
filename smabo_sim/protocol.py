"""What the simulated ESP32 and phone exchange with the brain.

Same handshake as the real devices (smabo-esp32's brain_link.py, smabo-app's
ros_compat.dart): connect to ``ws://<brain>:9090/``, advertise what we publish,
subscribe to what we read, publish with the source prefix. Works with
smabo-brain and smabo-brain-ros alike. The tables mirror smabo-brain's
brain/topics.py, whose tests check that they match.
"""

import json

# smabo-brain's PROTOCOL_VERSION for the tables below; sent on /hello and
# compared with the brain's /brain/info after every connect.
PROTOCOL_VERSION = 1

ESP32_PREFIX = "/esp32"
ESP32_ADVERTISE = (
    ("/esp32/config",       "std_msgs/msg/String"),
    ("/esp32/wheel_vel",    "smabo_interfaces/msg/WheelVel"),
    ("/esp32/joint_states", "sensor_msgs/msg/JointState"),
    ("/esp32/scan",         "sensor_msgs/msg/LaserScan"),
    ("/esp32/pong",         "std_msgs/msg/String"),
    ("/esp32/config_url",   "std_msgs/msg/String"),
    ("/esp32/hello",        "std_msgs/msg/String"),
)
ESP32_SUBSCRIBE = (
    ("/cmd_vel",       "geometry_msgs/msg/Twist"),
    ("/servo/command", "trajectory_msgs/msg/JointTrajectory"),
    ("/ping",          "std_msgs/msg/String"),
    ("/brain/info",    "std_msgs/msg/String"),
)

APP_PREFIX = "/app"
APP_ADVERTISE = (
    ("/app/imu/data",       "sensor_msgs/msg/Imu"),
    ("/app/gps/fix",        "sensor_msgs/msg/NavSatFix"),
    ("/app/speech/audio",   "audio_common_msgs/msg/AudioData"),
    ("/app/webrtc/offer",   "std_msgs/msg/String"),
    ("/app/webrtc/app_ice", "std_msgs/msg/String"),
    ("/app/phone/config",     "std_msgs/msg/String"),
    ("/app/phone/config_url", "std_msgs/msg/String"),
    ("/app/hello",          "std_msgs/msg/String"),
)
APP_SUBSCRIBE = (
    ("/expression",    "std_msgs/msg/String"),
    ("/look_at",       "geometry_msgs/msg/PoseStamped"),
    ("/speech/say",    "std_msgs/msg/String"),
    ("/webrtc/answer", "std_msgs/msg/String"),
    ("/brain/info",    "std_msgs/msg/String"),
)

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
