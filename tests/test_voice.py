import asyncio
import base64
import json
import shutil
import sys
import types

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from smabo_sim.config import Config
from smabo_sim.engines import create_engine
from smabo_sim.http_api import build_app
from smabo_sim.phone import SimPhone, tts, voice
from smabo_sim.robot import SimRobot
from smabo_sim.world import World


def make_phone(sent, **phone_cfg):
    cfg = Config(overrides={"sim": {"encoder_noise": 0.0}, "phone": phone_cfg})
    engine = create_engine("kinematic", World.load(), cfg)
    return SimPhone(cfg, engine, sent.append, wall=lambda: 1000.0), cfg, engine


class FakePopen:
    started: list = []

    def __init__(self, cmd, **_kw):
        self.cmd, self.alive = cmd, True
        FakePopen.started.append(self)

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.alive = False


@pytest.fixture
def fake_say(monkeypatch):
    FakePopen.started = []
    monkeypatch.setattr(tts, "_backend", lambda: "say")
    monkeypatch.setattr(tts, "_say_voice", lambda lang: "Kyoko" if lang.startswith("ja") else "")
    monkeypatch.setattr(tts, "_popen", FakePopen)
    return FakePopen


# ---- reading /speech/say aloud ------------------------------------------- #
def test_speech_say_is_read_aloud_and_interrupts(fake_say):
    sent = []
    phone, _, _ = make_phone(sent)
    phone.handle("/speech/say", {"data": "こんにちは"})
    phone.handle("/speech/say", {"data": "さようなら"})
    first, second = fake_say.started
    assert first.cmd == ["say", "-v", "Kyoko", "--", "こんにちは"]
    assert not first.alive and second.alive                     # like tts.stop() + speak()
    assert phone.face.state.bubble == "さようなら"
    assert phone.speaker.speaking


def test_muted_phone_only_shows_the_bubble(fake_say):
    phone, _, _ = make_phone([], tts={"enabled": False})
    phone.handle("/speech/say", {"data": "やあ"})
    assert fake_say.started == [] and phone.face.state.bubble == "やあ"


def test_espeak_command_uses_language_and_rate():
    cmd = tts._command("espeak-ng", "hello", "en-US", "", 160)
    assert cmd == ["espeak-ng", "-v", "en", "-s", "160", "--", "hello"]


# ---- WAV helpers ----------------------------------------------------------- #
def test_wav_round_trip_and_resample():
    pcm = (np.sin(np.linspace(0, 200, 8000)) * 8000).astype(np.int16)
    wav = voice.pcm_to_wav(pcm, 8000)
    back, rate = voice.wav_to_pcm(wav)
    assert rate == 8000 and np.array_equal(back, pcm)
    up, rate = voice.wav_to_pcm(voice.resample_wav(wav, 16000))
    assert rate == 16000 and len(up) == 16000
    with pytest.raises(ValueError):
        voice.wav_to_pcm(b"not a wav")


# ---- utterances to /speech/audio ------------------------------------------- #
def test_utterance_is_sent_as_base64_audiodata():
    sent = []
    phone, _, _ = make_phone(sent)
    phone.on_brain_connect()
    ops = [json.loads(s) for s in sent]
    assert {"op": "advertise", "topic": "/app/speech/audio",
            "type": "audio_common_msgs/msg/AudioData"} in ops
    phone.send_utterance(voice.pcm_to_wav(np.zeros(4410, np.int16), 44100))   # resampled
    frame = json.loads(sent[-1])
    assert frame["topic"] == "/app/speech/audio"
    pcm, rate = voice.wav_to_pcm(base64.b64decode(frame["msg"]["data"]))
    assert rate == 16000 and len(pcm) == 1600
    phone.cfg.update({"phone": {"voice": {"enabled": False}}})
    with pytest.raises(RuntimeError):
        phone.send_utterance(voice.pcm_to_wav(np.zeros(10, np.int16)))


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say"), reason="needs macOS say")
def test_typed_text_is_synthesized_to_speech():
    sent = []
    phone, _, _ = make_phone(sent)
    asyncio.run(phone.talk("こんにちは"))
    pcm, rate = voice.wav_to_pcm(base64.b64decode(json.loads(sent[-1])["msg"]["data"]))
    assert rate == 16000 and len(pcm) > 8000 and np.abs(pcm).max() > 1000


def test_microphone_recording_stops_after_silence(monkeypatch):
    """The app's rule: record until 1.5 s of silence once voice was heard."""
    blocks = [0] * 5 + [3000] * 10 + [0] * 30                  # quiet, speech, quiet (0.1 s each)

    class FakeStream:
        def __init__(self, callback, blocksize, **_kw):
            self.cb, self.n = callback, blocksize

        def __enter__(self):
            async def feed():
                for amp in blocks:
                    self.cb(np.full((self.n, 1), amp, np.int16), self.n, None, None)
                    await asyncio.sleep(0.1)
            self.task = asyncio.ensure_future(feed())
            return self

        def __exit__(self, *a):
            self.task.cancel()

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=FakeStream))
    wav = asyncio.run(voice.record_utterance(max_seconds=10))
    pcm, rate = voice.wav_to_pcm(wav)
    assert rate == 16000
    assert 2.9 <= len(pcm) / rate <= 3.3                       # 0.5 + 1.0 speech + ~1.5 silence


def test_microphone_gives_up_when_nothing_is_said(monkeypatch):
    class Quiet:
        def __init__(self, callback, blocksize, **_kw):
            self.cb, self.n = callback, blocksize

        def __enter__(self):
            async def feed():
                while True:
                    self.cb(np.zeros((self.n, 1), np.int16), self.n, None, None)
                    await asyncio.sleep(0.05)
            self.task = asyncio.ensure_future(feed())
            return self

        def __exit__(self, *a):
            self.task.cancel()

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=Quiet))
    assert asyncio.run(voice.record_utterance(wait_seconds=0.3)) is None


# ---- HTTP ------------------------------------------------------------------ #
def test_speech_endpoint(monkeypatch):
    async def go():
        sent = []
        phone, cfg, engine = make_phone(sent)
        robot = SimRobot(cfg, engine, lambda _s: None)
        talked = []

        async def fake_talk(text):
            talked.append(text)
        monkeypatch.setattr(phone, "talk", fake_talk)
        async with TestClient(TestServer(build_app(robot, phone))) as c:
            r = await c.post("/phone/speech", json={"text": "前に進んで"})
            assert (await r.json()) == {"ok": True, "sent": True} and talked == ["前に進んで"]
            wav = voice.pcm_to_wav(np.zeros(1600, np.int16))
            r = await c.post("/phone/speech", data=wav, headers={"Content-Type": "audio/wav"})
            assert r.status == 200 and json.loads(sent[-1])["topic"] == "/app/speech/audio"
            r = await c.post("/phone/speech", data=b"junk", headers={"Content-Type": "audio/wav"})
            assert r.status == 400
            r = await c.post("/phone/speech", json={})
            assert r.status == 400
            assert "Talk to the robot" in await (await c.get("/phone")).text()

    asyncio.run(go())
