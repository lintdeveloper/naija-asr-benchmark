"""Run a clip through an ASR model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import SmokeError
from .fleurs import Utterance

DEFAULT_MODEL = "openai/whisper-tiny"


@dataclass(frozen=True, slots=True)
class Transcription:
    reference: str
    hypothesis: str
    language_forced: bool
    """False means Whisper auto-detected, which it often gets wrong on
    low-resource audio — worth knowing when reading the hypothesis."""


def load(model: str = DEFAULT_MODEL, device: str = "cpu") -> Any:
    """Load an ASR pipeline once.

    Split out because Milestone 1 loops over clips: loading the checkpoint per
    clip dominated the runtime and measured the loader rather than the model.
    """
    from transformers import pipeline

    try:
        return pipeline("automatic-speech-recognition", model=model, device=device)
    except Exception as exc:
        raise SmokeError(
            f"could not load {model}: {exc}",
            "Usually network or disk space; whisper-tiny is about 150MB.",
        ) from exc


def transcribe(
    utterance: Utterance,
    language: str,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    asr: Any | None = None,
) -> Transcription:
    if asr is None:
        asr = load(model, device)

    def payload() -> dict[str, Any]:
        """A FRESH dict per attempt.

        `AutomaticSpeechRecognitionPipeline.preprocess` does
        `inputs.pop("array")` and `inputs.pop("sampling_rate")` on the dict it is
        handed — it mutates the caller's object. Reusing one payload across the
        forced attempt and the auto-detect fallback therefore hands the second
        call an empty dict, which fails with a misleading complaint about a
        missing "raw" key rather than about the real problem.

        Latent through Milestone 0, which only ever ran one clip whose first
        attempt succeeded. It surfaced on clip 14 of 20.
        """
        return {
            "array": utterance.audio.samples,
            "sampling_rate": utterance.audio.sampling_rate,
        }

    # Forcing the language stops Whisper guessing. Not every checkpoint accepts
    # every code, so fall back to auto-detect rather than failing the run.
    try:
        result = asr(payload(), generate_kwargs={"language": language, "task": "transcribe"})
        forced = True
    except (ValueError, KeyError):
        result = asr(payload())
        forced = False

    return Transcription(
        reference=utterance.transcription,
        hypothesis=_text_of(result),
        language_forced=forced,
    )


def _text_of(result: Any) -> str:
    """Pull the transcript out of whatever shape the pipeline returned.

    `transformers.pipeline` returns a dict for a single input but a list of dicts
    in some versions and configurations. Assuming the dict form type-checks fine
    against `Any` and then raises AttributeError at runtime on the other path —
    mypy found this, not a test run.
    """
    payload = result[0] if isinstance(result, list) and result else result
    if isinstance(payload, dict):
        return str(payload.get("text") or "").strip()
    return str(payload or "").strip()
