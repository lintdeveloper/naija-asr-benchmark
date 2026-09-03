"""Scoring, including the two decisions that change what the number means."""

from __future__ import annotations

import pytest

from naija_asr_benchmark.scoring import score


def test_a_perfect_transcription_scores_zero() -> None:
    s = score([("an kwatanta faretin", "an kwatanta faretin")])
    assert s.wer == pytest.approx(0.0)
    assert s.cer == pytest.approx(0.0)
    assert s.count == 1


def test_counts_substitutions_deletions_and_hits() -> None:
    s = score([("a b c d", "a x c")])
    u = s.utterances[0]
    assert (u.substitutions, u.deletions, u.insertions, u.hits) == (1, 1, 0, 2)
    assert u.reference_words == 4
    assert u.wer == pytest.approx(0.5)


def test_the_corpus_figure_is_POOLED_not_averaged() -> None:
    # A 1-word clip fully wrong and a 9-word clip fully right.
    # Averaged per utterance: (100% + 0%) / 2 = 50%.
    # Pooled over 10 reference words: 1 error / 10 = 10%.
    # Pooled is correct — otherwise a three-word clip outweighs a thirty-word one.
    s = score([("x", "y"), ("a b c d e f g h i", "a b c d e f g h i")])
    assert s.reference_words == 10
    assert s.wer == pytest.approx(0.1)


def test_cer_below_wer_is_the_orthographic_signal() -> None:
    # The Hausa finding: every word wrong, most characters right.
    s = score([("ginshiki mai walkiya", "deginshiki mei valkiya")])
    assert s.wer == pytest.approx(1.0)
    assert s.cer < s.wer, "CER under WER is what distinguishes misspelling from mishearing"


def test_an_empty_hypothesis_is_counted_and_scores_100_percent() -> None:
    s = score([("a b c", "")])
    assert s.wer == pytest.approx(1.0)
    assert s.empty_hypotheses == 1


def test_blank_references_are_dropped_not_scored() -> None:
    # WER divides by reference length, so a blank reference is undefined —
    # neither perfect nor terrible. Scoring it would silently distort the corpus.
    s = score([("", "something"), ("a b", "a b")])
    assert s.count == 1
    assert s.wer == pytest.approx(0.0)


def test_all_blank_references_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError, match="every reference was empty"):
        score([("", "x"), ("   ", "y")])


def test_latencies_are_attached_per_utterance() -> None:
    s = score([("a b", "a b"), ("c d", "c d")], [1.5, 2.5])
    assert [u.latency_s for u in s.utterances] == [1.5, 2.5]


def test_a_repetition_collapse_is_flagged() -> None:
    # whisper-tiny returned 444 characters of "1,2,1,0,1,0,..." against a
    # 56-character Hausa reference containing 802.11n / 2.4ghz / 5.0ghz.
    ref = "kwatancin 802.11n na aiki duk akan mita 2.4ghz da 5.0ghz"
    s = score([(ref, "1,2,1,0," + "0," * 200)])
    assert s.utterances[0].degenerate is True
    assert s.degenerate_hypotheses == 1


def test_ordinary_misspelling_is_not_flagged_as_degenerate() -> None:
    # The orthographic failure mode keeps roughly the reference length.
    s = score([("ginshiki mai walkiya", "deginshiki mei valkiya")])
    assert s.utterances[0].degenerate is False
    assert s.degenerate_hypotheses == 0


def test_cer_excluding_collapses_is_reported_alongside_not_instead() -> None:
    # One collapse took corpus CER from ~38% to 108% on the first real run.
    # Both figures must survive: the full one says what the model does, the
    # filtered one says what it does when it does not collapse.
    ref = "kwatancin 802.11n na aiki duk akan mita"
    s = score([(ref, "1,2," + "0," * 200), ("an kwatanta faretin", "an kwatanta ferretun")])
    assert s.cer > 1.0, "the collapse must still show in the headline figure"
    excluded = s.cer_excluding_degenerate
    assert excluded is not None and excluded < s.cer


def test_no_filtered_figure_when_nothing_was_excluded() -> None:
    # Returning a number identical to the headline would imply a filter ran.
    s = score([("a b c", "a x c")])
    assert s.cer_excluding_degenerate is None
