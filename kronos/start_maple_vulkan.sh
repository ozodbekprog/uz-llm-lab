#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

BIN="${MAPLE_BIN:-tools/llama-vulkan/bin/llama-server}"
MODEL="${MODEL:-models/gguf/maple-preview-TQ2_0-head-Q4_K.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"
NGL="${NGL:-99}"
NCMOE="${NCMOE:-8}"
DEV="${DEV:-Vulkan1}"

echo "SPEED-50: Maple server"
echo "  model : $MODEL"
echo "  ngl=$NGL  ncmoe=$NCMOE  ctx=$CTX  dev=$DEV  port=$PORT"

exec "$BIN" -m "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" -t 8 --parallel 1 -b 512 -ub 512 \
  -ngl "$NGL" -dev "$DEV" --n-cpu-moe "$NCMOE" \
  -ctk q8_0 -ctv q8_0 -fa auto
