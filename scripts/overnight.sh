#!/usr/bin/env bash
# Play package: evolve the Blue network until stopped (Ctrl-C / systemctl stop).
# Re-running the same command resumes from the latest checkpoint.
# Usage: scripts/overnight.sh [-o runs/overnight] [--workers N] [--init mixed] ...
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
exec "$PY" -m stealth_tactics overnight "$@"
