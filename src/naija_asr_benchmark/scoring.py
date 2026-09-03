"""Error rates over a set of transcriptions.

Milestone 1 computes **raw** WER and CER only — the text is compared as the
dataset and the model produced it. Normalisation is Milestone 2 and is a
separate decision with its own section in the plan (§5.3), because it changes
what the number means: raw asks "can this output be used directly?", normalised
asks "did the model recognise the word?". Publishing one without the other
overstates something either way, so neither is quietly applied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jiwer


@dataclass(frozen=True, slots=True)
class Utterance:
    """One reference/hypothesis pair with its own error rates."""

    index: int
    reference: str
    hypothesis: str
    latency_s: float
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int

    @property
    def reference_words(self) -> int:
        return self.substitutions + self.deletions + self.hits


@dataclass(frozen=True, slots=True)
class Score:
    """Corpus-level result. `wer` is computed over the pooled corpus.

    Pooled, not averaged over utterances. A per-utterance mean weights a
    three-word clip the same as a thirty-word one, which quietly inflates the
    influence of short utterances — and short utterances are where a model that
    emits nothing scores worst. jiwer pools when given lists, so the corpus
    figure comes from one call over every pair.
    """

    wer: float
    cer: float
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.utterances)

    @property
    def reference_words(self) -> int:
        return sum(u.reference_words for u in self.utterances)

    @property
    def empty_hypotheses(self) -> int:
        """Blank outputs. A model that returns nothing scores 100% WER, which is
        indistinguishable from one that returns confident nonsense — worth
        counting separately."""
        return sum(1 for u in self.utterances if not u.hypothesis.strip())


def score(pairs: list[tuple[str, str]], latencies: list[float] | None = None) -> Score:
    """Score reference/hypothesis pairs.

    Empty references are dropped rather than scored: WER divides by reference
    length, so a blank reference is undefined rather than perfect or terrible.
    """
    usable = [(i, r, h) for i, (r, h) in enumerate(pairs) if r.strip()]
    if not usable:
        raise ValueError("no usable pairs — every reference was empty")

    times = latencies or [0.0] * len(pairs)
    utterances: list[Utterance] = []
    for i, ref, hyp in usable:
        out = jiwer.process_words(ref, hyp)
        utterances.append(
            Utterance(
                index=i,
                reference=ref,
                hypothesis=hyp,
                latency_s=times[i] if i < len(times) else 0.0,
                wer=out.wer,
                cer=jiwer.cer(ref, hyp),
                substitutions=out.substitutions,
                deletions=out.deletions,
                insertions=out.insertions,
                hits=out.hits,
            )
        )

    refs = [r for _, r, _ in usable]
    hyps = [h for _, _, h in usable]
    pooled = jiwer.process_words(refs, hyps)
    return Score(wer=pooled.wer, cer=jiwer.cer(refs, hyps), utterances=utterances)
