import pytest


class _SilentPopen:
    """Stands in for the speech synthesizer process so tests stay quiet."""

    def __init__(self, cmd, **_kw):
        self.cmd = cmd

    def poll(self):
        return 0

    def terminate(self):
        pass


@pytest.fixture(autouse=True)
def _no_speech_out_loud(monkeypatch):
    from smabo_sim.phone import tts
    monkeypatch.setattr(tts, "_popen", _SilentPopen)
