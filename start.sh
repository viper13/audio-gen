#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
export MLX_ENABLE_TF32=0

if [[ "${1:-}" == "--lan" ]]; then
  export STUDIO_HOST="0.0.0.0"
else
  export STUDIO_HOST="${STUDIO_HOST:-127.0.0.1}"
fi
export STUDIO_PORT="${STUDIO_PORT:-8765}"

if [[ ! -x "engine/.venv/bin/python" ]]; then
  echo "Error: engine/.venv not found. Run: uv sync --directory engine --frozen --no-dev"
  exit 1
fi

if [[ "${STUDIO_HOST}" == "0.0.0.0" ]]; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
  LOCAL_HOST="$(scutil --get LocalHostName 2>/dev/null || true)"
  echo "Starting YuE2 MLX Studio on LAN:"
  echo "  Local : http://127.0.0.1:${STUDIO_PORT}"
  [[ -n "${LAN_IP}" ]] && echo "  Wi-Fi : http://${LAN_IP}:${STUDIO_PORT}"
  [[ -n "${LOCAL_HOST}" ]] && echo "  mDNS  : http://${LOCAL_HOST}.local:${STUDIO_PORT}"
else
  echo "Starting YuE2 MLX Studio at http://${STUDIO_HOST}:${STUDIO_PORT}"
fi
exec engine/.venv/bin/python -m studio.server
