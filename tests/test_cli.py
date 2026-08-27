from __future__ import annotations

import pytest

from naija_asr_bench import cli


def test_help_does_not_import_the_heavy_stack(capsys: pytest.CaptureFixture[str]) -> None:
    # --help must not pay for torch. If this starts failing, an import moved to
    # module scope.
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "smoke test" in capsys.readouterr().out.lower()


def test_rejects_an_unsupported_language(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["--lang", "de"])
    assert caught.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_reports_a_smoke_error_as_exit_1(monkeypatch: pytest.MonkeyPatch) -> None:
    from naija_asr_bench.errors import SmokeError

    def boom(_args: object) -> None:
        raise SmokeError("no reachable dataset", "check the network")

    monkeypatch.setattr(cli, "run", boom)
    assert cli.main(["--lang", "ha"]) == 1
