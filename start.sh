#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
export MLX_ENABLE_TF32=0
export STUDIO_HOST="${STUDIO_HOST:-127.0.0.1}"
export STUDIO_PORT="${STUDIO_PORT:-8765}"

if [[ ! -x "engine/.venv/bin/python" ]]; then
  echo "Error: engine/.venv not found. Run: uv sync --directory engine --frozen --no-dev"
  exit 1
fi

echo "Starting YuE2 MLX Studio at http://${STUDIO_HOST}:${STUDIO_PORT}"
exec engine/.venv/bin/python -m studio.server
