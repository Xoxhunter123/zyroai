#!/usr/bin/env bash
# ==============================================================================
# Zyro AI — Render 12 vCPU / 96 GB RAM Entrypoint
# Runs Qwen3-Coder-30B-A3B-Instruct (llama-server) + Zyro Agentic Orchestrator
# ==============================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
MODELS_DIR="${DATA_DIR}/models"
WORKSPACE_DIR="${DATA_DIR}/workspace"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-Qwen3-Coder-30B-A3B-Instruct}"
CTX_SIZE="${CTX_SIZE:-32768}"
THREADS="${THREADS:-$(nproc)}"
PORT="${PORT:-10000}"
LLAMA_PORT="8081"

mkdir -p "${MODELS_DIR}" "${WORKSPACE_DIR}"
MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"

# Launch Zyro Agentic HTTP API immediately so Render health check (/health) stays green
# while the 18.6 GB model downloads on first boot!
python3 /app/server.py \
  --port "${PORT}" \
  --upstream "http://127.0.0.1:${LLAMA_PORT}" \
  --workspace "${WORKSPACE_DIR}" \
  --model "${MODEL_ALIAS}" &
AGENT_PID=$!

# Download Qwen3-Coder-30B-A3B-Instruct GGUF into persistent /data/models if not cached
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[Zyro Render] Downloading ${MODEL_REPO}/${MODEL_FILE} into persistent disk..."
  MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
  curl -fL -C - --retry 5 --retry-delay 5 -o "${MODEL_PATH}.part" "${MODEL_URL}"
  mv "${MODEL_PATH}.part" "${MODEL_PATH}"
  echo "[Zyro Render] Download complete: ${MODEL_PATH}"
else
  echo "[Zyro Render] Using cached model: ${MODEL_PATH}"
fi

echo "[Zyro Render] Starting llama-server (${THREADS} threads, ctx=${CTX_SIZE})..."
/opt/llama.cpp/build/bin/llama-server \
  --model "${MODEL_PATH}" \
  --alias "${MODEL_ALIAS}" \
  --host 127.0.0.1 \
  --port "${LLAMA_PORT}" \
  --threads "${THREADS}" \
  --threads-batch "${THREADS}" \
  --ctx-size "${CTX_SIZE}" \
  --batch-size 1024 \
  --ubatch-size 512 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn \
  --jinja \
  --cont-batching \
  --metrics &
LLAMA_PID=$!

trap 'kill ${LLAMA_PID} ${AGENT_PID} 2>/dev/null || true' EXIT
wait -n "${AGENT_PID}" "${LLAMA_PID}"
