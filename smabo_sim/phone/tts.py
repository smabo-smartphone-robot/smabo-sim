"""Speech synthesis on the host: the phone's voice.

smabo-app reads ``/speech/say`` aloud with flutter_tts (``TtsService``); the
simulated phone does the same through the host's own synthesizer:

  * macOS: ``say`` (a voice for ``phone.tts.language``, e.g. Kyoko for ja-JP),
  * Linux: ``espeak-ng`` (or ``espeak``).

Like the app, a new utterance interrupts the current one. ``synth_wav`` renders
text to a 16 kHz mono WAV instead of the speakers; ``voice.py`` uses it to
"talk to" the robot without a microphone.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile

log = logging.getLogger(__name__)

_popen = subprocess.Popen     # starts the speaking process (swapped out in tests)


def _backend() -> str | None:
    if sys.platform == "darwin" and shutil.which("say"):
        return "say"
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            return exe
    return None


_VOICE_CACHE: dict[str, str] = {}


def _say_voice(language: str) -> str:
    """A macOS ``say`` voice for ``language`` ("ja-JP" → "Kyoko"), or "" (default)."""
    lang = language.replace("-", "_")
    if lang in _VOICE_CACHE:
        return _VOICE_CACHE[lang]
    voice = ""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
        names = []
        for line in out.splitlines():
            head = line.split("#", 1)[0].rstrip()
            parts = head.rsplit(None, 1)
            if len(parts) == 2 and parts[1].lower() == lang.lower():
                names.append(parts[0].split(" (")[0].strip())
        # the plain system voices read best; prefer them over the novelty ones
        for preferred in ("Kyoko", "Otoya", "Samantha", "Daniel"):
            if preferred in names:
                voice = preferred
                break
        else:
            voice = names[0] if names else ""
    except (OSError, subprocess.SubprocessError):
        pass
    _VOICE_CACHE[lang] = voice
    return voice


def _command(backend: str, text: str, language: str, voice: str, rate: float,
             out_wav: str | None = None) -> list[str]:
    if backend == "say":
        cmd = ["say"]
        v = voice or _say_voice(language)
        if v:
            cmd += ["-v", v]
        if rate > 0:
            cmd += ["-r", str(int(rate))]
        if out_wav:
            cmd += ["-o", out_wav, "--file-format=WAVE", "--data-format=LEI16@16000"]
        return cmd + ["--", text]
    cmd = [backend, "-v", voice or language.split("-")[0].lower()]
    if rate > 0:
        cmd += ["-s", str(int(rate))]
    if out_wav:
        cmd += ["-w", out_wav]
    return cmd + ["--", text]


class Speaker:
    """Reads text aloud on the host (``/speech/say``), interrupting itself."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.backend = _backend()
        self._proc = None
        self.spoken = 0
        if self.backend is None and self.enabled:
            log.warning("no speech synthesizer (macOS 'say' or espeak-ng) found: "
                        "/speech/say is shown in the bubble only")

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("phone.tts.enabled", True))

    @property
    def available(self) -> bool:
        return self.backend is not None

    def speak(self, text: str) -> bool:
        """Start reading ``text``; returns whether anything was started."""
        if not (self.enabled and self.available) or not text.strip():
            return False
        self.stop()
        cmd = _command(self.backend, text, self.cfg.get("phone.tts.language", "ja-JP"),
                       self.cfg.get("phone.tts.voice", ""), float(self.cfg.get("phone.tts.rate", 0)))
        try:
            self._proc = _popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            log.warning("speech synthesis failed: %s", e)
            return False
        self.spoken += 1
        return True

    @property
    def speaking(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if self.speaking:
            self._proc.terminate()
        self._proc = None


def synth_wav(text: str, cfg) -> bytes | None:
    """Render ``text`` to 16 kHz mono 16-bit WAV bytes with the host synthesizer."""
    backend = _backend()
    if backend is None or not text.strip():
        return None
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = _command(backend, text, cfg.get("phone.tts.language", "ja-JP"),
                       cfg.get("phone.tts.voice", ""), float(cfg.get("phone.tts.rate", 0)), out_wav=path)
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        with open(path, "rb") as f:
            data = f.read()
        if backend != "say":                    # espeak writes 22.05 kHz: resample
            from .voice import resample_wav
            data = resample_wav(data, 16000)
        return data
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("speech synthesis to WAV failed: %s", e)
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
