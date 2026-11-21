"""smabo-sim: a simulated smabo-esp32.

smabo-sim replaces the ESP32 board one-for-one. It speaks the same protocol
over the same channels, so smabo-brain(-ros) and smabo-web need no changes:

  * WebSocket client to the brain (``/``, smabo-brain or smabo-brain-ros),
    publishing ``/esp32/wheel_vel``, ``/esp32/joint_states``,
    ``/esp32/scan`` and receiving ``/cmd_vel``, ``/servo/command``.
  * REST config API (``GET/POST /config``, ``POST /mode``) for smabo-web.

Layers (see README):

  firmware  — ports of the ESP32 logic (drive mixing, servo follower, publish
              rates, config/modes). Same behaviour as the real board.
  engine    — the simulated hardware (wheels, encoders, lidar, servos).
              Swappable: ``kinematic`` now, a MuJoCo engine later.
  world     — walls / obstacles from a YAML file; also exported as a Nav2 map.
"""

__version__ = "0.2.0"
