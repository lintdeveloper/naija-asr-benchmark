#!/usr/bin/env bash
# Milestone 0 environment setup.
#
# Creates a Python 3.11 virtualenv in .venv and installs the dependency set.
# Uses uv when available (much faster); falls back to venv + pip.
#
#   ./setup.sh

set -euo pipefail
cd "$(dirname "$0")"

PY=python3.11
command -v "$PY" >/dev/null || {
  echo "error: $PY not found."
  echo "The macOS system python (3.9) is too old for this stack — install 3.11:"
  echo "  brew install python@3.11"
  exit 1
}

if command -v uv >/dev/null; then
  echo "==> uv found — creating .venv with $PY"
  uv venv --python "$PY" .venv
  echo "==> installing dependencies"
  VIRTUAL_ENV=.venv uv pip install -r requirements.txt
else
  echo "==> uv not found — falling back to venv + pip"
  "$PY" -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

echo
echo "Done. Now run:"
echo "  ./.venv/bin/python milestone0.py"
