"""The parent/child plumbing in fetch_samples, without touching the network.

FLEURS streaming leaves no local parquet cache, so there is nothing to replay
offline, and a flaky CDN produces false failures. These are synthetic on purpose.

The deadlock check is the one that matters. The first version of fetch_samples
joined the child before reading the queue, which hangs as soon as the payload
exceeds the pipe buffer: the child blocks in put() waiting for a reader while
the parent blocks in join() waiting for the child. Three 64KB waveforms were
enough to reproduce it, and real FLEURS rows are about a megabyte each — so
every language would have reported a bogus timeout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from naija_asr_benchmark import fleurs
from naija_asr_benchmark.errors import SmokeError


def _row(index: int) -> dict[str, Any]:
    return {
        "samples": np.linspace(-1, 1, 16_000, dtype=np.float32),
        "sampling_rate": 16_000,
        "transcription": f"utterance {index}",
        "fields": ("audio", "id", "transcription"),
    }


# Module-level so `spawn` can pickle them by reference.
def _worker_ok(_config: str, count: int, _streaming: bool, sink: Any) -> None:
    sink.put(("ok", [_row(i) for i in range(count)]))


def _worker_error(_config: str, _count: int, _streaming: bool, sink: Any) -> None:
    sink.put(("error", "OSError: libsndfile not found"))


def _worker_empty(_config: str, _count: int, _streaming: bool, sink: Any) -> None:
    sink.put(("ok", []))


def _worker_silent(_config: str, _count: int, _streaming: bool, _sink: Any) -> None:
    # Never puts anything: stands in for a fetch that blocks unkillably.
    import time

    time.sleep(120)


def _patch_worker(monkeypatch: pytest.MonkeyPatch, worker: Any) -> None:
    monkeypatch.setattr(fleurs, "_fetch_worker", worker)


def test_returns_utterances_with_the_waveform_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_worker(monkeypatch, _worker_ok)
    utterances = fleurs.fetch_samples("ha_ng", 3, timeout_s=60)

    assert len(utterances) == 3
    clip = utterances[0].audio
    assert isinstance(clip.samples, np.ndarray), "the ASR pipeline needs an array, not a list"
    assert len(clip.samples) == 16_000
    assert clip.samples[0] == pytest.approx(-1.0)
    assert clip.samples[-1] == pytest.approx(1.0)
    assert clip.duration_s == pytest.approx(1.0)
    assert utterances[0].transcription == "utterance 0"


def test_a_payload_past_the_pipe_buffer_does_not_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 8 x 64KB is comfortably past a typical pipe buffer. This is the regression.
    _patch_worker(monkeypatch, _worker_ok)
    assert len(fleurs.fetch_samples("ha_ng", 8, timeout_s=90)) == 8


def test_a_child_that_never_answers_raises_rather_than_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_worker(monkeypatch, _worker_silent)
    with pytest.raises(SmokeError) as caught:
        fleurs.fetch_samples("ig_ng", 5, timeout_s=3)
    assert "produced no sample within 3s" in str(caught.value)
    assert "NAIJA_ASR_FETCH_TIMEOUT_S" in caught.value.hint, "must say how to raise the ceiling"


def test_a_child_error_surfaces_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_worker(monkeypatch, _worker_error)
    with pytest.raises(SmokeError) as caught:
        fleurs.fetch_samples("ha_ng", 5, timeout_s=60)
    assert "libsndfile" in str(caught.value), "the child's diagnosis must not be swallowed"


def test_an_empty_split_is_a_failure_not_an_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_worker(monkeypatch, _worker_empty)
    with pytest.raises(SmokeError, match="empty"):
        fleurs.fetch_samples("ha_ng", 5, timeout_s=60)


def test_the_default_ceiling_is_generous() -> None:
    # 180s was the first value and it failed a WORKING Hausa fetch.
    assert fleurs.DEFAULT_FETCH_TIMEOUT_S >= 600
