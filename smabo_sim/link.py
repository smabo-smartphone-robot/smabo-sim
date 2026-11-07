"""WebSocket client to the brain (smabo-brain or smabo-brain-ros), auto-reconnecting.

Equivalent of ``smabo-esp32/ws_client.py``: connect out to
``ws://<brain.host>:<brain.port><path>``, call ``on_connect`` after every
(re)connect, retry every 3 s.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import aiohttp

log = logging.getLogger("smabo_sim.link")


class BrainLink:
    def __init__(self, url: Callable[[], str], on_message: Callable[[str], None],
                 on_connect: Callable[[], None], retry: float = 3.0) -> None:
        self._url = url
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
                try:
                    async with session.ws_connect(url, heartbeat=20, max_msg_size=0) as ws:
                        self._ws = ws
                        self.connected = True
                        log.info("connected to %s", url)
                        self._on_connect()
                        writer = asyncio.create_task(self._writer(ws))
                        try:
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    self._on_message(msg.data)
                                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                        finally:
                            writer.cancel()
                except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as e:
                    log.warning("cannot reach %s (%s)", url, e.__class__.__name__)
                finally:
                    if self.connected:
                        log.warning("disconnected from %s", url)
                    self.connected = False
                    self._ws = None
                    while not self._queue.empty():
                        self._queue.get_nowait()
                await asyncio.sleep(self._retry)

    async def _writer(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            text = await self._queue.get()
            await ws.send_str(text)
