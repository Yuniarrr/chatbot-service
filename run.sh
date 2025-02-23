#!/usr/bin/env bash

# Exit on error
set -e

# Activate the virtual environment (Adjust path if needed)
if [ -d ".venv" ]; then
    # source .venv/bin/activate # Linux/Mac
    source .venv/Scripts/activate # Windows (Git Bash)
fi

# Set default values if not provided
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

# Run Uvicorn with the correct app path
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --forwarded-allow-ips='*'
