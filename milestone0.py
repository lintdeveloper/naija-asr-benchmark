#!/usr/bin/env python3
"""
Milestone 0 — environment smoke test.

Goal: prove the toolchain works end to end. Load FLEURS, read a Hausa sample,
run one clip through whisper-tiny, print the hypothesis beside the reference.

This is NOT a quality measurement. whisper-tiny on Hausa will produce something
close to nonsense; that is the expected and correct outcome. If you see any
Hausa-ish text at all next to a reference transcript, the milestone is done.

Deliberate choices:
  * streaming=True — FLEURS test splits are large. Streaming reads a handful of
    samples without downloading gigabytes. Milestone 1 switches to a full local
    copy so runs are repeatable.
  * whisper-tiny — ~39M params, runs on a laptop CPU in seconds. Do not use a
    large model here; you are testing plumbing, not accuracy.

Usage:
    ./.venv/bin/python milestone0.py
    ./.venv/bin/python milestone0.py --lang yo   # try Yorùbá instead
"""

from __future__ import annotations

import argparse
import sys
import textwrap

# FLEURS uses {iso639-1}_{region} config names. All four below were confirmed
# against the HuggingFace datasets-server on 2026-08-07 (103 configs total;
# ha_ng has train/validation/test). resolve_config() still checks at runtime —
# cheap insurance against the dataset being restructured.
EXPECTED_CONFIGS = {
    "ha": "ha_ng",  # Hausa (Nigeria)
    "yo": "yo_ng",  # Yorùbá (Nigeria)
    "ig": "ig_ng",  # Igbo (Nigeria)
    "en": "en_us",  # English — the control arm
}

DATASET = "google/fleurs"
MODEL = "openai/whisper-tiny"


def rule(title: str) -> None:
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def fail(msg: str, hint: str = "") -> None:
    print(f"\n  ✗ {msg}")
    if hint:
        print(textwrap.indent(textwrap.fill(hint, 68), "    "))
    sys.exit(1)


def check_environment() -> str:
    """Report versions and pick a compute device. Returns the device string."""
    rule("1. Environment")
    print(f"  python      {sys.version.split()[0]}")

    try:
        import torch
        import transformers
        import datasets
    except ImportError as e:
        fail(f"missing dependency: {e.name}", "Run ./setup.sh first.")

    print(f"  torch       {torch.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  datasets    {datasets.__version__}")

    # Apple Silicon exposes MPS. For a single tiny-model clip, CPU is fine and
    # avoids a class of MPS dtype quirks — so prefer CPU here and revisit at M3.
    if torch.backends.mps.is_available():
        print("  device      cpu  (MPS available; using CPU for this smoke test)")
    else:
        print("  device      cpu")
    return "cpu"


def resolve_config(lang: str) -> str:
    """
    Confirm the FLEURS config name for `lang`.

    The plan flags `ha_ng` as unverified. Rather than assume, list what the hub
    actually offers and match on prefix, so a wrong guess produces a useful
    answer instead of a stack trace.
    """
    rule("2. Resolving the FLEURS config")
    from datasets import get_dataset_config_names

    expected = EXPECTED_CONFIGS[lang]
    try:
        configs = get_dataset_config_names(DATASET)
    except Exception as e:
        fail(
            f"could not list configs for {DATASET}: {e}",
            "Usually a network problem. If it mentions trust_remote_code, the "
            "dataset still ships a loading script — re-run with datasets<3.",
        )

    print(f"  {len(configs)} configs available")

    if expected in configs:
        print(f"  ✓ '{expected}' exists — the plan's guess was right")
        return expected

    # Guess was wrong: show every config for this language so the fix is obvious.
    candidates = [c for c in configs if c.startswith(f"{lang}_")]
    print(f"  ✗ '{expected}' NOT found")
    if candidates:
        print(f"  → candidates for '{lang}': {', '.join(candidates)}")
        chosen = candidates[0]
        print(f"  → using '{chosen}'. Update EXPECTED_CONFIGS in this file.")
        return chosen
    fail(
        f"no config starts with '{lang}_'",
        f"Sample of what is available: {', '.join(sorted(configs)[:12])} …",
    )


def load_samples(config: str, n: int) -> list[dict]:
    """Stream the first n test samples. Streaming avoids a multi-GB download."""
    rule(f"3. Loading {n} samples from {DATASET}/{config} (streaming)")
    from datasets import load_dataset

    try:
        stream = load_dataset(DATASET, config, split="test", streaming=True)
        samples = []
        for i, s in enumerate(stream):
            if i >= n:
                break
            samples.append(s)
    except Exception as e:
        fail(
            f"could not load the dataset: {e}",
            "If this mentions a missing audio backend, check that soundfile "
            "installed cleanly. If it mentions trust_remote_code, your datasets "
            "version is trying to run a loading script — the pin in "
            "requirements.txt should prevent that.",
        )

    if not samples:
        fail("dataset loaded but the test split was empty.")

    print(f"  ✓ pulled {len(samples)} samples")
    print(f"  fields: {', '.join(sorted(samples[0].keys()))}")
    return samples


def show_references(samples: list[dict]) -> None:
    rule("4. Reference transcripts")
    for i, s in enumerate(samples, 1):
        text = s.get("transcription") or s.get("raw_transcription") or "<no text field>"
        audio = s["audio"]
        secs = len(audio["array"]) / audio["sampling_rate"]
        print(f"\n  [{i}] {secs:5.1f}s @ {audio['sampling_rate']}Hz")
        print(textwrap.indent(textwrap.fill(text, 66), "      "))


def transcribe_one(sample: dict, lang: str, device: str) -> None:
    """Run one clip through whisper-tiny and print hypothesis vs reference."""
    rule(f"5. Transcribing one clip with {MODEL}")
    from transformers import pipeline

    print("  loading model (first run downloads ~150MB) …")
    try:
        asr = pipeline("automatic-speech-recognition", model=MODEL, device=device)
    except Exception as e:
        fail(f"could not load {MODEL}: {e}", "Usually a network or disk-space issue.")

    audio = sample["audio"]
    payload = {"array": audio["array"], "sampling_rate": audio["sampling_rate"]}

    # Forcing the language stops Whisper guessing (it often guesses wrong on
    # low-resource audio). Not every checkpoint accepts every code, so fall back
    # to auto-detect rather than dying.
    try:
        out = asr(payload, generate_kwargs={"language": lang, "task": "transcribe"})
        forced = True
    except Exception:
        print(f"  note: could not force language='{lang}', using auto-detect")
        out = asr(payload)
        forced = False

    reference = sample.get("transcription") or sample.get("raw_transcription") or ""
    hypothesis = (out.get("text") or "").strip()

    print(f"\n  language forcing: {'on' if forced else 'off (auto-detect)'}")
    print("\n  REFERENCE (ground truth)")
    print(textwrap.indent(textwrap.fill(reference, 66), "    "))
    print("\n  HYPOTHESIS (whisper-tiny)")
    print(textwrap.indent(textwrap.fill(hypothesis or "<empty>", 66), "    "))


def main() -> None:
    ap = argparse.ArgumentParser(description="Milestone 0 smoke test")
    ap.add_argument("--lang", default="ha", choices=sorted(EXPECTED_CONFIGS),
                    help="language to test (default: ha)")
    ap.add_argument("--samples", type=int, default=5, help="samples to preview")
    args = ap.parse_args()

    print(f"\nMilestone 0 — smoke test — language '{args.lang}'")

    device = check_environment()
    config = resolve_config(args.lang)
    samples = load_samples(config, args.samples)
    show_references(samples)
    transcribe_one(samples[0], args.lang, device)

    rule("Done")
    print(textwrap.dedent(f"""
      Milestone 0 is complete if a hypothesis printed above — even a bad one.
      whisper-tiny scoring poorly on {args.lang} is the expected result.

      Record for the plan:
        · confirmed FLEURS config for '{args.lang}': {config}
        · re-run with --lang yo / ig / en to confirm the other three

      Still outstanding for Milestone 0 (manual, see README §PazaBench):
        · does PazaBench already cover Hausa / Yorùbá / Igbo?
          If yes, adopt its clean-condition numbers and Milestone 3 shrinks.
    """).rstrip())


if __name__ == "__main__":
    main()
