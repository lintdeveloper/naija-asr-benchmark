"""Report the toolchain and pick a compute device."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .errors import SmokeError


@dataclass(frozen=True, slots=True)
class Environment:
    python: str
    torch: str
    transformers: str
    datasets: str
    device: str
    device_note: str = ""


def describe() -> Environment:
    """Import the heavy stack and report it.

    torch and transformers are imported here rather than at module scope: they
    cost seconds, and `--help` should not pay for them.
    """
    try:
        import datasets
        import torch
        import transformers
    except ImportError as exc:
        raise SmokeError(
            f"missing dependency: {exc.name}",
            "Run `uv sync` (or `uv sync --extra dev` for the test tooling).",
        ) from exc

    note = ""
    # Apple Silicon exposes MPS. For one tiny-model clip CPU is fine and avoids a
    # class of MPS dtype quirks, so prefer CPU here and revisit for Milestone 1.
    if torch.backends.mps.is_available():
        note = "MPS available; using CPU for the smoke test"

    return Environment(
        python=sys.version.split()[0],
        torch=torch.__version__,
        transformers=transformers.__version__,
        datasets=datasets.__version__,
        device="cpu",
        device_note=note,
    )
