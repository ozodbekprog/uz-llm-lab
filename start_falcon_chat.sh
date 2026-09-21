#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/bitnet"
exec ./build/bin/llama-server \
  -m models/Falcon3-7B-Instruct-1.58bit/ggml-model-i2_s.gguf \
  --host 127.0.0.1 \
  --port "${PORT:-8080}" \
  -c 4096 \
  -t 8
