#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

HOST="${RLCD_ROUTER_HOST:-127.0.0.1}"
PORT="${RLCD_ROUTER_PORT:-8001}"
PYTHON="${RLCD_PYTHON:-$DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "Python environment not found at $PYTHON" >&2
    echo "Create .venv or set RLCD_PYTHON to a Python executable." >&2
    exit 1
fi

echo "================================================================="
echo "  Starting RLCD Multi-Model Router (Local Apple Silicon)"
echo "  URL: http://$HOST:$PORT"
echo "================================================================="

export PYTHONPATH="$DIR:$PYTHONPATH"

# Use --reload in dev, plain in production
if [ "${RLCD_DEV:-0}" = "1" ]; then
    exec "$PYTHON" -m uvicorn server.app:app --host "$HOST" --port "$PORT" --reload
else
    exec "$PYTHON" -m uvicorn server.app:app --host "$HOST" --port "$PORT"
fi
