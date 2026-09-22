#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

BIN="${KRONOS_BIN:-tools/llama-vulkan/bin/llama-server}"
MODEL="${MODEL:-models/gguf/Falcon3-10B-Instruct-1.58bit-tq2_0.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"
NGL="${NGL:-40}"
DEV="${DEV:-Vulkan1}"

if [ ! -x "$BIN" ]; then
  BIN=/tmp/opencode/llama.cpp-main/build-vulkan/bin/llama-server
fi
if [ ! -f "$MODEL" ]; then
  MODEL=/tmp/opencode/falcon10b-tq2_0.gguf
fi

echo "KRONOS: $BIN"
echo "Model : $MODEL (ngl=$NGL, ctx=$CTX, dev=$DEV)"

exec "$BIN" -m "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" -t 8 \
  -ngl "$NGL" -dev "$DEV" \
  -ctk q8_0 -ctv q8_0 -fa auto
