from __future__ import annotations

import pytest

from naija_asr_benchmark import cli
from naija_asr_benchmark.errors import SmokeError


def test_help_does_not_import_the_heavy_stack(capsys: pytest.CaptureFixture[str]) -> None:
    # --help must not pay for torch. If this starts failing, an import moved to
    # module scope.
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "smoke" in out and "evaluate" in out, "both milestones must be discoverable"


def test_a_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code == 2
    assert "required" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["smoke", "evaluate"])
def test_rejects_an_unsupported_language(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main([command, "--lang", "de"])
    assert caught.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_evaluate_takes_a_clip_count() -> None:
    args = cli._parse_args(["evaluate", "--clips", "5"])
    assert args.command == "evaluate"
    assert args.clips == 5


def test_evaluate_defaults_to_twenty_clips() -> None:
    # The plan scopes Milestone 1 at 20 clips.
    assert cli._parse_args(["evaluate"]).clips == 20


@pytest.mark.parametrize(
    ("command", "target"), [("smoke", "run_smoke"), ("evaluate", "run_evaluate")]
)
def test_each_command_routes_to_its_own_handler(
    command: str, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = []
    monkeypatch.setattr(cli, target, lambda _a: called.append(target))
    assert cli.main([command]) == 0
    assert called == [target]


def test_reports_a_smoke_error_as_exit_1(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_args: object) -> None:
        raise SmokeError("no reachable dataset", "check the network")

    monkeypatch.setattr(cli, "run_smoke", boom)
    assert cli.main(["smoke"]) == 1
