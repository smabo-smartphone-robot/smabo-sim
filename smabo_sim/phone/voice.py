"""Talking to the robot: the phone's half of the voice pipeline.

smabo-app spots the wake word on the phone, then records the utterance (16 kHz
mono PCM) until 1.5 s of silence and sends it as a WAV on ``/app/speech/audio``
(audio_common_msgs/AudioData; over rosbridge the uint8[] is a base64 string).
The brain transcribes it and publishes ``/speech/recognized``.

The simulated phone sends the same message. The utterance comes from:

  * text, synthesized to speech on the host (``tts.synth_wav``): no microphone
    needed, and it exercises the brain's real STT;
  * the host microphone (``record_utterance``, needs ``sounddevice``), recorded
    with the app's own silence rule — the equivalent of tapping the face
    (manual trigger; wake-word spotting is not simulated);
  * a WAV file.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import wave

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000        # SpeechService._sampleRate
SILENCE_MS = 1500          # SpeechService._silenceMs
SILENCE_RMS = 600.0        # SpeechService._silenceRms (int16 RMS)


def pcm_to_wav(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    """int16 mono samples → WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buf.getvalue()


def wav_to_pcm(data: bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV bytes → (int16 mono samples, rate). Raises ValueError."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError("only 16-bit PCM WAV is supported")
            ch, rate = w.getnchannels(), w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    except (wave.Error, EOFError) as e:
        raise ValueError(f"not a WAV file: {e}") from e
    if ch > 1:
        pcm = pcm[::ch]
    return pcm, rate


def resample_wav(data: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Any 16-bit WAV → mono WAV at ``rate`` (linear interpolation)."""
    pcm, src = wav_to_pcm(data)
    if src != rate and len(pcm):
        n = max(1, int(round(len(pcm) * rate / src)))
        x = np.linspace(0.0, len(pcm) - 1, n)
        pcm = np.interp(x, np.arange(len(pcm)), pcm.astype(np.float64)).round().astype(np.int16)
    return pcm_to_wav(pcm, rate)


def audio_msg(wav: bytes) -> dict:
    """audio_common_msgs/AudioData as rosbridge JSON (uint8[] → base64)."""
    return {"data": base64.b64encode(wav).decode("ascii")}


def have_microphone() -> bool:
    import importlib.util
    return importlib.util.find_spec("sounddevice") is not None


async def record_utterance(max_seconds: float = 10.0, silence_ms: int = SILENCE_MS,
                           silence_rms: float = SILENCE_RMS, wait_seconds: float = 5.0) -> bytes | None:
    """Record from the host microphone until ``silence_ms`` of silence after speech.

    Like ``SpeechService``: stops after a continuous quiet stretch once voice was
    heard, gives up if nothing is said within ``wait_seconds``, and never runs
    longer than ``max_seconds``. Returns WAV bytes, or None if nothing was said.
    """
    import sounddevice as sd       # optional: pip install 'smabo-sim[voice]'

    loop = asyncio.get_running_loop()
    chunks: list[np.ndarray] = []
    state = {"heard": False, "last_voice": 0.0}
    done = loop.create_future()
    start = time.monotonic()

    def finish() -> None:
        if not done.done():
            done.set_result(None)

    def on_audio(indata, _frames, _time, _status) -> None:
        block = indata[:, 0].copy()
        chunks.append(block)
        now = time.monotonic()
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) if len(block) else 0.0
        if rms > silence_rms:
            state["heard"], state["last_voice"] = True, now
        quiet_for = now - state["last_voice"]
        if ((state["heard"] and quiet_for * 1000 >= silence_ms)
                or (not state["heard"] and now - start >= wait_seconds)
                or now - start >= max_seconds):
            loop.call_soon_threadsafe(finish)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=SAMPLE_RATE // 10, callback=on_audio):
        await asyncio.wait_for(done, timeout=max_seconds + 2.0)
    if not state["heard"] or not chunks:
        return None
    return pcm_to_wav(np.concatenate(chunks))
