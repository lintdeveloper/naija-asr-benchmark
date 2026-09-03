"""Milestone 1 -- the working loop.

N clips, one model, one language, one WER number. Deliberately narrow: the plan
scopes this as plumbing, not quality, and expects 80-100% WER on Hausa. A good
score here would mean something is wrong, not right.

What it is not, on purpose: no normalisation (Milestone 2, §5.3), no degradation
conditions (Milestone 2, §5.4), no multi-model grid (Milestone 3), no GPU. The
sequencing rule is explicit -- never debug a harness and a cloud bill at once.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__, asr, fleurs, scoring
from .errors import SmokeError


@dataclass(frozen=True, slots=True)
class Run:
    language: str
    config: str
    model: str
    score: scoring.Score
    seconds: float
    started_at: str
    streaming: bool = False


def run(
    language: str,
    *,
    clips: int = 20,
    model: str = asr.DEFAULT_MODEL,
    device: str = "cpu",
    timeout_s: int = fleurs.DEFAULT_FETCH_TIMEOUT_S,
    streaming: bool = False,
    on_clip: Any | None = None,
) -> Run:
    """Fetch, transcribe and score. `on_clip(i, total, transcription)` reports progress.

    `streaming` defaults to False: the split is downloaded and cached once so
    repeated runs are fast and, more importantly, reproducible. Streaming a
    770 MB parquet to read twenty clips fails on a flaky link and makes the
    result depend on the network.
    """
    started = datetime.now(UTC).isoformat(timespec="seconds")
    config = fleurs.resolve_config(language)
    utterances = fleurs.fetch_samples(config, clips, timeout_s=timeout_s, streaming=streaming)

    engine = asr.load(model, device)
    pairs: list[tuple[str, str]] = []
    latencies: list[float] = []

    t0 = time.perf_counter()
    for index, utterance in enumerate(utterances):
        clip_started = time.perf_counter()
        result = asr.transcribe(utterance, language, model=model, device=device, asr=engine)
        latencies.append(time.perf_counter() - clip_started)
        pairs.append((result.reference, result.hypothesis))
        if on_clip is not None:
            on_clip(index + 1, len(utterances), result)
    elapsed = time.perf_counter() - t0

    try:
        result_score = scoring.score(pairs, latencies)
    except ValueError as exc:
        raise SmokeError(
            str(exc),
            "Every reference was blank, which means the dataset schema changed — "
            "check the `transcription` field still exists on FLEURS rows.",
        ) from exc

    return Run(
        language=language,
        config=config,
        model=model,
        score=result_score,
        seconds=elapsed,
        started_at=started,
        streaming=streaming,
    )


def persist(result: Run, directory: Path = Path("results")) -> Path:
    """Write the run as JSON, including every utterance.

    Per-utterance rows are kept because Milestone 4's error taxonomy needs ~400
    hand-categorised errors, and re-running inference to get them back would be
    wasteful. `results/` is gitignored — these are artifacts, not source.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = result.started_at.replace(":", "").replace("-", "")
    path = directory / f"{result.language}-{result.model.split('/')[-1]}-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "harness_version": __version__,
                "started_at": result.started_at,
                "language": result.language,
                "fleurs_config": result.config,
                "model": result.model,
                "normalisation": "none (raw) — Milestone 1 scores text as produced",
                # How long audio was handled is a methodological choice, not a
                # detail: chunking would introduce boundary errors into a number
                # meant to measure the model.
                "long_form": "sequential (return_timestamps=True), not chunked",
                "source": "cached local split" if not result.streaming else "streamed",
                "clips": result.score.count,
                "reference_words": result.score.reference_words,
                "empty_hypotheses": result.score.empty_hypotheses,
                "degenerate_hypotheses": result.score.degenerate_hypotheses,
                "cer_excluding_degenerate": result.score.cer_excluding_degenerate,
                "wer": result.score.wer,
                "cer": result.score.cer,
                "seconds": result.seconds,
                "utterances": [
                    {
                        "index": u.index,
                        "reference": u.reference,
                        "hypothesis": u.hypothesis,
                        "wer": u.wer,
                        "cer": u.cer,
                        "substitutions": u.substitutions,
                        "deletions": u.deletions,
                        "insertions": u.insertions,
                        "hits": u.hits,
                        "latency_s": round(u.latency_s, 3),
                        "degenerate": u.degenerate,
                    }
                    for u in result.score.utterances
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
