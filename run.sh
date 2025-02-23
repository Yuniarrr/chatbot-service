#!/usr/bin/env bash

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --forwarded-allow-ips '*'
