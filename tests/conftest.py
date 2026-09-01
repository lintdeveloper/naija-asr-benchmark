from __future__ import annotations

import numpy as np
import pytest

from naija_asr_benchmark.fleurs import AudioClip, Utterance


@pytest.fixture
def clip() -> AudioClip:
    """One second of 16kHz audio, ramped so a round-trip can be checked exactly."""
    return AudioClip(samples=np.linspace(-1, 1, 16_000, dtype=np.float32), sampling_rate=16_000)


@pytest.fixture
def utterance(clip: AudioClip) -> Utterance:
    return Utterance(audio=clip, transcription="an kwatanta faretin", fields=("audio", "id"))
