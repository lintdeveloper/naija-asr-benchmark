"""Command line entry point.

    naija-asr-benchmark smoke      Milestone 0 -- does the toolchain work at all
    naija-asr-benchmark evaluate   Milestone 1 -- N clips, one model, one WER

Neither is a quality measurement. The plan expects 80-100% WER on Hausa with a
tiny model; a good score would mean something is wrong, not right.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__, asr, console, environment, evaluate, fleurs
from .errors import SmokeError

SAMPLE_COUNT = 5


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="naija-asr-benchmark",
        description="Nigerian-language ASR evaluation under real deployment conditions.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--lang",
            default="ha",
            choices=sorted(fleurs.LANGUAGE_CONFIGS),
            help="language to evaluate (default: ha)",
        )
        p.add_argument(
            "--model",
            default=asr.DEFAULT_MODEL,
            help=f"ASR checkpoint (default: {asr.DEFAULT_MODEL})",
        )
        p.add_argument(
            "--timeout",
            type=int,
            default=fleurs.DEFAULT_FETCH_TIMEOUT_S,
            metavar="SECONDS",
            help="deadline for the dataset fetch (default: %(default)s)",
        )

    smoke = sub.add_parser("smoke", help="Milestone 0 — prove the toolchain works")
    shared(smoke)

    ev = sub.add_parser("evaluate", help="Milestone 1 — N clips, one model, one WER")
    shared(ev)
    ev.add_argument(
        "--clips", type=int, default=20, metavar="N", help="clips to score (default: 20)"
    )
    ev.add_argument(
        "--no-save", action="store_true", help="do not write a JSON result to results/"
    )
    ev.add_argument(
        "--streaming",
        action="store_true",
        help="stream instead of caching the split — faster to start, not reproducible",
    )

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


def run_smoke(args: argparse.Namespace) -> None:
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


def run_evaluate(args: argparse.Namespace) -> None:
    device = _report_environment()
    config = _report_config(args.lang)

    console.rule(f"3. Scoring {args.clips} clips — {args.model} on {config}")
    if args.streaming:
        print(f"  streaming {args.clips} clips (not reproducible) …")
    else:
        print(f"  reading {args.clips} clips from the cached split …")
        print("  a first run downloads it in full — several hundred MB per language")

    def progress(done: int, total: int, result: asr.Transcription) -> None:
        preview = (result.hypothesis or "<empty>").replace("\n", " ")[:44]
        print(f"    [{done:>3}/{total}]  {preview}")

    outcome = evaluate.run(
        args.lang,
        clips=args.clips,
        model=args.model,
        device=device,
        timeout_s=args.timeout,
        streaming=args.streaming,
        on_clip=progress,
    )
    s = outcome.score

    console.rule("4. Result")
    console.detail("clips", str(s.count))
    console.detail("ref words", str(s.reference_words))
    console.detail("WER", f"{s.wer * 100:.1f}%   (raw — no normalisation)")
    console.detail("CER", f"{s.cer * 100:.1f}%")
    console.detail("empty", f"{s.empty_hypotheses} blank hypotheses")
    console.detail("elapsed", f"{outcome.seconds:.1f}s for {s.count} clips")

    if not args.no_save:
        path = evaluate.persist(outcome)
        console.detail("saved", str(path))

    console.rule("Done")
    console.block(
        f"Milestone 1 is complete when a single WER number exists. The plan expects "
        f"80-100% on {args.lang} with a tiny model, so {s.wer * 100:.0f}% is the expected "
        "outcome and not a problem to fix. CER below WER means the model is hearing the "
        "phonetics and writing them in the wrong orthography.",
        "  ",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "evaluate":
            run_evaluate(args)
        else:
            run_smoke(args)
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
