"""WebSocket client to the brain (smabo-brain or smabo-brain-ros), auto-reconnecting.

Equivalent of ``smabo-esp32/ws_client.py``: connect out to
``ws://<brain.host>:<brain.port><path>``, call ``on_connect`` after every
(re)connect, retry every 3 s.

With ``credentials`` (pairing.py) the first frame is the token or an invite
code, and ``on_connect`` runs only once the brain has accepted it. A brain
without pairing (the default) says so at once; smabo-brain-ros does not
answer that frame, and after ``AUTH_REPLY_TIMEOUT`` the link goes on as before.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Callable

import aiohttp

if TYPE_CHECKING:
    from .pairing import Credentials

log = logging.getLogger("smabo_sim.link")

AUTH_REPLY_TIMEOUT = 2.0


class BrainLink:
    def __init__(self, url: Callable[[], str], on_message: Callable[[str], None],
                 on_connect: Callable[[], None], retry: float = 3.0,
                 credentials: "Credentials | None" = None) -> None:
        self._url = url
        self.credentials = credentials
        self._on_message = on_message
        self._on_connect = on_connect
        self._retry = retry
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        self.connected = False

    def send(self, text: str) -> None:
        """Queue a frame; dropped while disconnected or if the queue is full."""
        if not self.connected:
            return
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            pass  # brain is not keeping up; drop like a lossy radio link

    def restart(self) -> None:
        """Drop the current connection so the next attempt uses new settings."""
        if self._ws is not None and not self._ws.closed:
            asyncio.ensure_future(self._ws.close())

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            while True:
                url = self._url()
                refused = False
                try:
                    async with session.ws_connect(url, heartbeat=20, max_msg_size=0) as ws:
                        self._ws = ws
                        if await self._authenticate(ws):
                            await self._serve(ws, url)
                        else:
                            refused = True
                except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as e:
                    log.warning("cannot reach %s (%s)", url, e.__class__.__name__)
                finally:
                    if self.connected:
                        log.warning("disconnected from %s", url)
                    self.connected = False
                    self._ws = None
                    while not self._queue.empty():
                        self._queue.get_nowait()
                # a refused token stays refused until someone pairs this device again
                await asyncio.sleep(max(self._retry, 15.0) if refused else self._retry)

    async def _serve(self, ws: aiohttp.ClientWebSocketResponse, url: str) -> None:
        self.connected = True
        log.info("connected to %s", url)
        self._on_connect()
        writer = asyncio.create_task(self._writer(ws))
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if not self._auth_reply(msg.data):
                        self._on_message(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        finally:
            writer.cancel()

    def _auth_reply(self, text: str) -> bool:
        """Handle an auth / pair answer (also the refusal an unpaired device gets later)."""
        if self.credentials is None or '"op"' not in text:
            return False
        try:
            frame = json.loads(text)
        except ValueError:
            return False
        return isinstance(frame, dict) and self.credentials.handle_reply(frame) is not None

    async def _authenticate(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        """Send the token or invite; False when the brain refused it."""
        first = self.credentials.first_frame() if self.credentials else None
        if first is None:
            return True
        await ws.send_str(json.dumps(first))
        try:
            msg = await ws.receive(timeout=AUTH_REPLY_TIMEOUT)
        except asyncio.TimeoutError:
            return True                      # smabo-brain-ros says nothing
        if msg.type != aiohttp.WSMsgType.TEXT:
            return False
        try:
            frame = json.loads(msg.data)
        except ValueError:
            frame = None
        ok = self.credentials.handle_reply(frame) if isinstance(frame, dict) else None
        if ok is None:
            self._on_message(msg.data)       # not an answer: an old brain talking already
            return True
        return ok

    async def _writer(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            text = await self._queue.get()
            await ws.send_str(text)
