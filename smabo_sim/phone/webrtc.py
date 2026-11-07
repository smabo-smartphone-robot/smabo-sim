"""Stream the simulated phone camera to the brain over WebRTC, like smabo-app.

smabo-app (``services/webrtc_service.dart``) sends its camera to the brain as
a WebRTC video track; the brain (``brain.webrtc_hub`` in smabo-brain, wrapped by
``webrtc_camera_node`` in smabo-brain-ros) answers, pulls frames for vision and
republishes them as ``/camera/image_raw``. This module is the same sending
peer, with the MuJoCo phone camera as the video source:

    sim → /app/webrtc/offer   {data: '{"sdp": ..., "type": "offer"}'}
    brain → /webrtc/answer    {data: '{"sdp": ..., "type": "answer"}'}

aiortc gathers ICE while setting the local description, so our candidates go
inside the offer SDP and nothing is trickled (the app trickles; the brain
accepts both). The brain's candidates arrive inside the answer.

If the connection drops (e.g. ICE consent expires), a new offer follows after
``RETRY_DELAY``; an offer nobody answers (brain restarting) is repeated after
``ANSWER_TIMEOUT``.

aiortc is optional (``pip install 'smabo-sim[webrtc]'``); without it the camera
is still viewable over smabo-sim's HTTP page.
"""

from __future__ import annotations

import asyncio
import fractions
import json
import logging
import time
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

VIDEO_CLOCK = 90000
RETRY_DELAY = 1.0   # s before re-offering after the connection failed or closed
ANSWER_TIMEOUT = 10.0   # s to wait for the brain's answer before offering again


def have_aiortc() -> bool:
    """aiortc (and PyAV) installed? Checked without importing them.

    They are imported only when the first stream starts: loading FFmpeg before
    the MuJoCo viewer has opened its window is avoided on purpose.
    """
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("aiortc", "av"))


def _make_track(render: Callable[[], np.ndarray], fps: float):
    from aiortc import VideoStreamTrack
    from av import VideoFrame

    class SimCameraTrack(VideoStreamTrack):
        """A video track whose frames are rendered on demand at ``fps``."""

        kind = "video"

        def __init__(self) -> None:
            super().__init__()
            self._period = 1.0 / max(fps, 1.0)
            self._start: float | None = None
            self._next = 0.0
            self.frames = 0

        async def recv(self):
            now = time.monotonic()
            if self._start is None:
                self._start = self._next = now
            wait = self._next - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next = max(self._next + self._period, time.monotonic())
            frame = VideoFrame.from_ndarray(render(), format="rgb24")
            frame.pts = int((time.monotonic() - self._start) * VIDEO_CLOCK)
            frame.time_base = fractions.Fraction(1, VIDEO_CLOCK)
            self.frames += 1
            return frame

    return SimCameraTrack()


class CameraStreamer:
    """One send-only peer connection to the brain, re-offered when it drops.

    Parameters
    ----------
    render : callable() -> (H, W, 3) uint8 RGB
        Renders one camera frame (called on the asyncio thread).
    publish : callable(topic, msg)
        Publishes a rosbridge message as the app (``/app`` prefix added here).
    fps : float
        Camera frame rate (the brain throttles vision with ``capture_fps``).
    """

    def __init__(self, render: Callable[[], np.ndarray], publish: Callable[[str, dict], None],
                 fps: float = 15.0) -> None:
        self._render = render
        self._publish = publish
        self._fps = fps
        self._pc = None
        self._track = None
        self._retry: asyncio.TimerHandle | None = None
        self.state = "idle"

    @property
    def available(self) -> bool:
        return have_aiortc()

    @property
    def frames_sent(self) -> int:
        return self._track.frames if self._track else 0

    async def start(self) -> None:
        """(Re)create the peer connection and send a fresh offer."""
        if not self.available:
            return
        from aiortc import RTCConfiguration, RTCPeerConnection

        await self.close()
        # like smabo-app: no STUN/TURN, the brain is on the local network. (With
        # aiortc's default Google STUN the pair can end up hair-pinning through
        # the router's public address, and ICE consent then expires.)
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self._pc = pc
        self._track = _make_track(self._render, self._fps)
        pc.addTrack(self._track)

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            if pc is not self._pc:
                return
            self.state = pc.connectionState
            log.info("WebRTC camera: %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                self._schedule_retry()

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)            # gathers ICE into the SDP
        self.state = "offered"
        self._publish("/webrtc/offer", {"data": json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})})
        log.info("WebRTC camera: offer sent to brain")

        def unanswered() -> None:
            if pc is self._pc and pc.signalingState == "have-local-offer":
                log.info("WebRTC camera: no answer from brain, offering again")
                self._schedule_retry(0.0)

        asyncio.get_running_loop().call_later(ANSWER_TIMEOUT, unanswered)

    async def handle_answer(self, msg: dict) -> None:
        if self._pc is None or self._pc.signalingState != "have-local-offer":
            return
        from aiortc import RTCSessionDescription

        try:
            data = json.loads(msg.get("data") or "{}")
            await self._pc.setRemoteDescription(RTCSessionDescription(data["sdp"], data["type"]))
            log.info("WebRTC camera: answer applied")
        except Exception as e:  # malformed answer: keep waiting / retry later
            log.warning("WebRTC camera: bad answer (%s)", e)

    def _schedule_retry(self, delay: float = RETRY_DELAY) -> None:
        if self._retry is not None:
            return
        loop = asyncio.get_running_loop()

        def fire() -> None:
            self._retry = None
            asyncio.ensure_future(self.start())

        self._retry = loop.call_later(delay, fire)

    async def close(self) -> None:
        if self._retry is not None:
            self._retry.cancel()
            self._retry = None
        pc, self._pc = self._pc, None
        if pc is not None:
            await pc.close()
        self.state = "idle"

