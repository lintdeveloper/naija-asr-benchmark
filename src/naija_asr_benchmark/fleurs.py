"""FLEURS access: config resolution, and a fetch that cannot hang the caller."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
from dataclasses import dataclass
from multiprocessing.queues import Queue as MPQueue
from typing import Any

from .errors import SmokeError

DATASET = "google/fleurs"

# FLEURS uses {iso639-1}_{region} config names. All four were confirmed against
# the live config list on 2026-08-27 (103 configs). `resolve_config` still checks
# at runtime — cheap insurance against the dataset being restructured.
LANGUAGE_CONFIGS: dict[str, str] = {
    "ha": "ha_ng",  # Hausa (Nigeria)
    "yo": "yo_ng",  # Yorùbá (Nigeria)
    "ig": "ig_ng",  # Igbo (Nigeria)
    "en": "en_us",  # English — the control arm
}

# Generous on purpose. The deadline exists so that one can be enforced AT ALL
# (see `fetch_samples`), not so that it is tight. An earlier 180s ceiling failed
# a working Hausa fetch; ig_ng has blocked past 25 minutes while ha/yo/en
# complete in a few. A first run also pays for a cold HuggingFace cache.
DEFAULT_FETCH_TIMEOUT_S = int(os.environ.get("NAIJA_ASR_FETCH_TIMEOUT_S", "900"))


@dataclass(frozen=True, slots=True)
class AudioClip:
    """A decoded waveform. `samples` is a numpy float array."""

    samples: Any
    sampling_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sampling_rate


@dataclass(frozen=True, slots=True)
class Utterance:
    audio: AudioClip
    transcription: str
    fields: tuple[str, ...]
    """Field names present on the source row — useful when the schema shifts."""


def resolve_config(language: str, *, configs: list[str] | None = None) -> str:
    """Return the FLEURS config name for `language`, checked against the hub.

    `configs` is injectable so this is testable without network access.
    """
    if language not in LANGUAGE_CONFIGS:
        raise SmokeError(
            f"unknown language {language!r}",
            f"Known: {', '.join(sorted(LANGUAGE_CONFIGS))}.",
        )
    expected = LANGUAGE_CONFIGS[language]

    if configs is None:
        from datasets import get_dataset_config_names

        try:
            configs = list(get_dataset_config_names(DATASET))
        except Exception as exc:
            raise SmokeError(
                f"could not list configs for {DATASET}: {exc}",
                "Usually a network problem. If it mentions trust_remote_code, the "
                "dataset still ships a loading script — check the datasets<4 pin held.",
            ) from exc

    if expected in configs:
        return expected

    candidates = sorted(c for c in configs if c.startswith(f"{language}_"))
    if candidates:
        return candidates[0]
    raise SmokeError(
        f"no config starts with {language + '_'!r}",
        f"Available sample: {', '.join(sorted(configs)[:12])} …",
    )


def _fetch_worker(  # pragma: no cover
    config: str, count: int, streaming: bool, sink: MPQueue[Any]
) -> None:
    """Runs in a child process. See `fetch_samples` for why it is a child."""
    try:
        from datasets import load_dataset

        if streaming:
            source: Any = load_dataset(DATASET, config, split="test", streaming=True)
        else:
            # Downloads and caches the split, so every later run reads from disk.
            # Slow once, then repeatable — which a benchmark needs regardless of
            # the network.
            full = load_dataset(DATASET, config, split="test")
            source = full.select(range(min(count, len(full))))

        rows: list[dict[str, Any]] = []
        for index, row in enumerate(source):
            if index >= count:
                break
            audio = row.get("audio") or {}
            # Send only what callers use. The raw row also carries loader
            # internals, not all of which pickle. The waveform stays a numpy
            # array: it pickles fine and the ASR pipeline wants an array.
            rows.append(
                {
                    "samples": audio.get("array"),
                    "sampling_rate": audio.get("sampling_rate"),
                    "transcription": (
                        row.get("transcription") or row.get("raw_transcription") or ""
                    ),
                    "fields": tuple(sorted(row.keys())),
                }
            )
        sink.put(("ok", rows))
    except Exception as exc:  # reported verbatim to the parent, whatever it is
        sink.put(("error", f"{type(exc).__name__}: {exc}"))


def fetch_samples(
    config: str,
    count: int = 5,
    *,
    timeout_s: int = DEFAULT_FETCH_TIMEOUT_S,
    streaming: bool = True,
) -> list[Utterance]:
    """Stream the first `count` test rows, bounded by a wall-clock deadline.

    `streaming=True` reads a handful of rows without downloading the whole split,
    which is what the Milestone 0 smoke test wants. **It is the wrong default for
    evaluation.** The Hausa test parquet is 770 MB and streaming re-fetches it on
    every run, so on a slow or flaky link a twenty-clip run can fail repeatedly
    without ever scoring anything — observed three times in a row. Worse, a
    benchmark whose numbers depend on the network is not reproducible.

    `streaming=False` downloads and caches the split once, then every later run
    reads from disk. Slow once, then fast and repeatable. The plan says as much:
    Milestone 1 switches to a local copy.

    The **child process** is about something else: a HuggingFace fetch can block
    in a way that cannot be interrupted from inside Python. A probe with `signal.alarm(90)`
    ran past 330
    seconds and needed SIGTERM from outside, because the block sits below the
    level at which Python delivers signals.

    So the deadline has to be enforced by a parent watching a child, not by the
    work watching itself. Milestone 1 loops over four languages and several
    models; without this, one bad shard hangs the whole run.
    """
    context = mp.get_context("spawn")
    sink: MPQueue[Any] = context.Queue()
    child = context.Process(
        target=_fetch_worker, args=(config, count, streaming, sink), daemon=True
    )
    child.start()

    # Read BEFORE joining. Joining first deadlocks: five FLEURS rows are several
    # megabytes and a pipe buffer is tens of kilobytes, so the child blocks in
    # put() waiting for a reader while the parent blocks in join() waiting for
    # the child. Neither moves, and it surfaces as a bogus timeout on every
    # language. `get` drains the pipe as the child writes.
    try:
        status, payload = sink.get(timeout=timeout_s)
    except queue_mod.Empty:
        child.kill()
        child.join(5)
        raise SmokeError(
            f"the {config!r} split produced no sample within {timeout_s}s",
            "The config name resolved, so this is data access rather than a typo. "
            f"The {config} test split is several hundred megabytes; a first "
            "non-streaming run has to download all of it. "
            "On a slow connection raise it: NAIJA_ASR_FETCH_TIMEOUT_S=1800. If it "
            "never returns the split may be unavailable — but a flaky CDN looks "
            "identical, so confirm another language fetches before concluding that.",
        ) from None

    child.join(30)
    if child.is_alive():
        child.kill()
        child.join(5)

    if status == "error":
        raise SmokeError(
            f"could not load {config!r}: {payload}",
            "A missing audio backend means soundfile did not install cleanly. A "
            "trust_remote_code error means datasets>=4 slipped past the pin.",
        )

    if not payload:
        raise SmokeError(f"{config!r} loaded but the test split was empty")

    return [
        Utterance(
            audio=AudioClip(samples=row["samples"], sampling_rate=row["sampling_rate"]),
            transcription=row["transcription"],
            fields=row["fields"],
        )
        for row in payload
    ]
