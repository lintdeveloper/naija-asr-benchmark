"""The Milestone 1 loop, with the model and dataset stubbed out.

No network and no checkpoint: this asserts the loop wires fetch → transcribe →
score → persist correctly, which is the part that can silently produce a
plausible-looking wrong number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from naija_asr_benchmark import asr, evaluate, fleurs
from naija_asr_benchmark.errors import SmokeError


def _utterance(text: str) -> fleurs.Utterance:
    return fleurs.Utterance(
        audio=fleurs.AudioClip(
            samples=np.zeros(16_000, dtype=np.float32), sampling_rate=16_000
        ),
        transcription=text,
        fields=("audio", "transcription"),
    )


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    refs = ["an kwatanta faretin", "aristotle masanin falsafa", "tsibirin na kwanciye"]
    monkeypatch.setattr(fleurs, "resolve_config", lambda lang, **_: f"{lang}_ng")
    monkeypatch.setattr(
        fleurs, "fetch_samples", lambda cfg, n, **_: [_utterance(r) for r in refs[:n]]
    )
    monkeypatch.setattr(asr, "load", lambda *_a, **_k: object())

    def fake_transcribe(u: fleurs.Utterance, _lang: str, **_kw: Any) -> asr.Transcription:
        # Mangle the first word only: a partial error, so WER is neither 0 nor 1.
        words = u.transcription.split()
        return asr.Transcription(
            reference=u.transcription,
            hypothesis=" ".join(["WRONG", *words[1:]]),
            language_forced=True,
        )

    monkeypatch.setattr(asr, "transcribe", fake_transcribe)


def test_produces_one_pooled_wer_over_all_clips(stubbed: None) -> None:
    result = evaluate.run("ha", clips=3)
    assert result.score.count == 3
    # one wrong word per clip, three clips
    assert result.score.utterances[0].substitutions == 1
    assert 0 < result.score.wer < 1
    assert result.config == "ha_ng"


def test_loads_the_model_once_not_per_clip(
    monkeypatch: pytest.MonkeyPatch, stubbed: None
) -> None:
    loads = 0

    def counting_load(*_a: Any, **_k: Any) -> object:
        nonlocal loads
        loads += 1
        return object()

    monkeypatch.setattr(asr, "load", counting_load)
    evaluate.run("ha", clips=3)
    assert loads == 1, "reloading the checkpoint per clip measures the loader, not the model"


def test_reports_progress_for_every_clip(stubbed: None) -> None:
    seen: list[tuple[int, int]] = []
    evaluate.run("ha", clips=3, on_clip=lambda d, t, _r: seen.append((d, t)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_blank_references_raise_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fleurs, "resolve_config", lambda lang, **_: "ha_ng")
    monkeypatch.setattr(fleurs, "fetch_samples", lambda *_a, **_k: [_utterance("")])
    monkeypatch.setattr(asr, "load", lambda *_a, **_k: object())
    monkeypatch.setattr(
        asr,
        "transcribe",
        lambda u, _l, **_k: asr.Transcription(
            reference="", hypothesis="x", language_forced=True
        ),
    )
    with pytest.raises(SmokeError) as caught:
        evaluate.run("ha", clips=1)
    assert "schema changed" in caught.value.hint


def test_persist_writes_every_utterance_and_says_normalisation_is_none(
    stubbed: None, tmp_path: Path
) -> None:
    result = evaluate.run("ha", clips=3)
    path = evaluate.persist(result, tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["clips"] == 3
    assert len(saved["utterances"]) == 3
    # Milestone 4 hand-categorises ~400 errors; re-running inference for them would be wasteful.
    assert saved["utterances"][0]["reference"] and saved["utterances"][0]["hypothesis"]
    # A number without its normalisation is not interpretable (§5.3).
    assert "none" in saved["normalisation"]
    # Long-audio handling changes the number too, so it is recorded alongside.
    assert "sequential" in saved["long_form"]
    assert saved["fleurs_config"] == "ha_ng"
