"""Presentation only. Kept separate so the logic can be tested without stdout."""

from __future__ import annotations

import textwrap

WIDTH = 72
_WRAP = 68


def rule(title: str) -> None:
    print(f"\n{'─' * WIDTH}\n{title}\n{'─' * WIDTH}")


def detail(label: str, value: str) -> None:
    print(f"  {label:<11} {value}")


def ok(message: str) -> None:
    print(f"  ✓ {message}")


def warn(message: str) -> None:
    print(f"  ✗ {message}")


def note(message: str) -> None:
    print(f"  → {message}")


def block(text: str, indent: str = "    ") -> None:
    print(textwrap.indent(textwrap.fill(text, _WRAP), indent))
