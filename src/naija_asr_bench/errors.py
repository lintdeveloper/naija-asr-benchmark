"""The single failure type, so nothing below the CLI calls sys.exit."""

from __future__ import annotations


class SmokeError(Exception):
    """A failure the operator can act on.

    Carries a `hint` because every failure mode here has a concrete next step —
    a wrong config name, a missing audio backend, a stalled shard. The CLI is the
    only place that formats and exits; raising from library code keeps the
    functions testable, which `fail(); sys.exit(1)` did not.
    """

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint
