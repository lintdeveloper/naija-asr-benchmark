"""Milestone 0 — the environment smoke test.

Proves the toolchain works end to end: resolve a FLEURS config, read samples,
run one clip through whisper-tiny, print the hypothesis beside the reference.

This is NOT a quality measurement. whisper-tiny on Hausa produces something
close to nonsense, and that is the correct outcome — any Hausa-ish text beside a
reference means the milestone is done. Do not tune anything here.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__, asr, console, environment, fleurs
from .errors import SmokeError

SAMPLE_COUNT = 5


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="naija-asr-benchmark",
        description="Milestone 0 smoke test for Nigerian-language ASR evaluation.",
    )
    parser.add_argument(
        "--lang",
        default="ha",
        choices=sorted(fleurs.LANGUAGE_CONFIGS),
        help="language to probe (default: ha)",
    )
    parser.add_argument(
        "--model",
        default=asr.DEFAULT_MODEL,
        help=f"ASR checkpoint (default: {asr.DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=fleurs.DEFAULT_FETCH_TIMEOUT_S,
        metavar="SECONDS",
        help="deadline for the dataset fetch (default: %(default)s)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def _report_environment() -> str:
    console.rule("1. Environment")
    env = environment.describe()
    console.detail("python", env.python)
    console.detail("torch", env.torch)
    console.detail("transformers", env.transformers)
    console.detail("datasets", env.datasets)
    device = f"{env.device}  ({env.device_note})" if env.device_note else env.device
    console.detail("device", device)
    return env.device


def _report_config(language: str) -> str:
    console.rule("2. Resolving the FLEURS config")
    expected = fleurs.LANGUAGE_CONFIGS[language]
    config = fleurs.resolve_config(language)
    if config == expected:
        console.ok(f"{config!r} exists — the recorded name is right")
    else:
        console.warn(f"{expected!r} not found")
        console.note(f"using {config!r} — update LANGUAGE_CONFIGS in fleurs.py")
    return config


def _report_references(utterances: list[fleurs.Utterance]) -> None:
    console.rule("4. Reference transcripts")
    for index, utterance in enumerate(utterances, 1):
        clip = utterance.audio
        print(f"\n  [{index}]  {clip.duration_s:5.1f}s @ {clip.sampling_rate}Hz")
        console.block(utterance.transcription or "<no text field>", "      ")


def _report_transcription(result: asr.Transcription, model: str) -> None:
    forcing = "on" if result.language_forced else "off (auto-detect)"
    print(f"\n  language forcing: {forcing}")
    print("\n  REFERENCE (ground truth)")
    console.block(result.reference)
    print(f"\n  HYPOTHESIS ({model})")
    console.block(result.hypothesis or "<empty>")


def run(args: argparse.Namespace) -> None:
    device = _report_environment()
    config = _report_config(args.lang)

    console.rule(
        f"3. Loading {SAMPLE_COUNT} samples from {fleurs.DATASET}/{config} (streaming)"
    )
    utterances = fleurs.fetch_samples(config, SAMPLE_COUNT, timeout_s=args.timeout)
    console.ok(f"pulled {len(utterances)} samples")
    console.detail("fields", ", ".join(utterances[0].fields))

    _report_references(utterances)

    console.rule(f"5. Transcribing one clip with {args.model}")
    print("  loading model (first run downloads ~150MB) …")
    result = asr.transcribe(utterances[0], args.lang, model=args.model, device=device)
    _report_transcription(result, args.model)

    console.rule("Done")
    console.block(
        "Milestone 0 is complete if a hypothesis printed above — even a bad one. "
        f"A poor score on {args.lang} is the expected result, not a problem to fix.",
        "  ",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run(args)
    except SmokeError as exc:
        print(f"\n  ✗ {exc}", file=sys.stderr)
        if exc.hint:
            console.block(exc.hint)
        return 1
    except KeyboardInterrupt:
        print("\n  interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
