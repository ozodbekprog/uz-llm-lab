#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/bitnet"

MODEL="${MODEL:-models/Falcon3-7B-Instruct-1.58bit/ggml-model-i2_s.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-16384}"

exec ./build/bin/llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  -c "$CTX" \
  -t 8 \
  -fa auto \
  -ctk q8_0 \
  -ctv q8_0
