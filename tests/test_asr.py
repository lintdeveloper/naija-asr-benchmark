"""Transcription, focused on the aliasing bug that killed a 20-clip run.

`AutomaticSpeechRecognitionPipeline.preprocess` pops "array" and "sampling_rate"
out of the dict it is given — it mutates the caller's object. So the forced
attempt and the auto-detect fallback must not share one payload, or the second
call receives an empty dict and fails with a misleading complaint about a
missing "raw" key.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from naija_asr_benchmark import asr
from naija_asr_benchmark.fleurs import AudioClip, Utterance


@pytest.fixture
def utterance() -> Utterance:
    return Utterance(
        audio=AudioClip(samples=np.zeros(16_000, dtype=np.float32), sampling_rate=16_000),
        transcription="an kwatanta faretin",
        fields=("audio", "transcription"),
    )


class MutatingPipeline:
    """Mimics transformers: pops the keys, and rejects a dict already drained."""

    def __init__(self, *, fail_forced: bool) -> None:
        self.fail_forced = fail_forced
        self.calls = 0

    def __call__(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, str]:
        self.calls += 1
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError('the dict needs to contain a "raw" key')
        inputs.pop("array", None)
        inputs.pop("sampling_rate", None)
        if self.fail_forced and "generate_kwargs" in kwargs:
            raise ValueError("this checkpoint does not accept that language code")
        return {"text": " hypothesis "}


def test_forced_transcription_succeeds_and_is_reported_as_forced(utterance: Utterance) -> None:
    engine = MutatingPipeline(fail_forced=False)
    out = asr.transcribe(utterance, "ha", asr=engine)
    assert out.language_forced is True
    assert out.hypothesis == "hypothesis"
    assert engine.calls == 1


def test_the_fallback_gets_an_intact_payload(utterance: Utterance) -> None:
    # The regression. With one shared dict the second call raised
    # 'the dict needs to contain a "raw" key' and killed the run at clip 14 of 20.
    engine = MutatingPipeline(fail_forced=True)
    out = asr.transcribe(utterance, "ha", asr=engine)
    assert engine.calls == 2, "it must retry"
    assert out.language_forced is False, "auto-detect must be reported honestly"
    assert out.hypothesis == "hypothesis"


def test_the_utterance_waveform_is_not_consumed(utterance: Utterance) -> None:
    # The pipeline drains the dict, not the Utterance — so the same clip can be
    # scored again by another model in Milestone 3.
    asr.transcribe(utterance, "ha", asr=MutatingPipeline(fail_forced=True))
    assert len(utterance.audio.samples) == 16_000
    assert utterance.audio.sampling_rate == 16_000


@pytest.mark.parametrize(
    ("returned", "expected"),
    [
        ({"text": " padded "}, "padded"),
        ([{"text": "from a list"}], "from a list"),
        ({"text": None}, ""),
        ({}, ""),
    ],
)
def test_reads_the_transcript_out_of_either_shape(
    utterance: Utterance, returned: Any, expected: str
) -> None:
    # transformers returns a dict for one input but a list of dicts in some
    # versions. mypy --strict found this; no test had.
    class Fixed(MutatingPipeline):
        def __call__(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
            super().__call__(inputs, **kwargs)
            return returned

    assert asr.transcribe(utterance, "ha", asr=Fixed(fail_forced=False)).hypothesis == expected
