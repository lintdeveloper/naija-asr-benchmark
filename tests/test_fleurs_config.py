"""Config resolution, with the hub's answer injected so no network is needed."""

from __future__ import annotations

import pytest

from naija_asr_benchmark.errors import SmokeError
from naija_asr_benchmark.fleurs import LANGUAGE_CONFIGS, resolve_config


@pytest.mark.parametrize(("language", "expected"), sorted(LANGUAGE_CONFIGS.items()))
def test_returns_the_recorded_config_when_the_hub_has_it(language: str, expected: str) -> None:
    assert resolve_config(language, configs=[expected, "zz_zz"]) == expected


def test_falls_back_to_a_prefix_match_when_the_recorded_name_is_gone() -> None:
    # The point of checking at runtime: a restructured dataset should produce a
    # usable answer, not a stack trace.
    assert resolve_config("ha", configs=["ha_ne", "yo_ng"]) == "ha_ne"


def test_prefers_the_first_candidate_deterministically() -> None:
    assert resolve_config("ha", configs=["ha_zz", "ha_ne"]) == "ha_ne"


def test_raises_with_a_hint_when_no_config_matches_the_language() -> None:
    with pytest.raises(SmokeError) as caught:
        resolve_config("ha", configs=["en_us", "fr_fr"])
    assert "no config starts with" in str(caught.value)
    assert caught.value.hint, "the operator needs to see what IS available"


def test_rejects_a_language_the_project_does_not_track() -> None:
    with pytest.raises(SmokeError) as caught:
        resolve_config("de")
    assert "unknown language" in str(caught.value)
