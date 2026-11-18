#!/usr/bin/env bash
# Benchmark one model on one GPU with vLLM.
#
# Usage:   benchmarks/run.sh <gpu-name> <hf-model-id> [extra vllm serve args...]
# Example: benchmarks/run.sh L40S Qwen/Qwen3-8B-FP8
#
# Starts the pinned vLLM image with the model, waits for /health, runs `vllm bench serve`
# at each concurrency level, and writes everything to
# benchmarks/results/<gpu>_<model>_<timestamp>/.
#
# Settings (environment variables):
#   VLLM_TAG       vLLM image tag (default v0.32.0, decision 004)
#   INPUT_LEN      prompt tokens per request (default 1024)
#   OUTPUT_LEN     generated tokens per request (default 256, EOS ignored so every request is exact)
#   CONCURRENCIES  concurrent users to test (default "1 8 32")
#   NUM_PROMPTS    requests per concurrency level (default 10 x concurrency, at least 32)
#   GPU_INDEX      which GPU on the machine to use (default 0)
#   HF_TOKEN       optional; Qwen3 isn't gated
set -euo pipefail

usage="usage: benchmarks/run.sh <gpu-name> <hf-model-id> [extra vllm serve args...]"
GPU_NAME=${1:?$usage}
MODEL=${2:?$usage}
shift 2
EXTRA_ARGS=("$@")

VLLM_TAG=${VLLM_TAG:-v0.32.0}
IMAGE="vllm/vllm-openai:${VLLM_TAG}"
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-256}
CONCURRENCIES=${CONCURRENCIES:-"1 8 32"}
GPU_INDEX=${GPU_INDEX:-0}
PORT=${PORT:-8000}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}   # seconds; the first run downloads 5-17 GB of weights
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
CONTAINER=privatecounsel-bench-vllm
API_KEY=bench-local-only                   # vLLM listens on 127.0.0.1 only

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RUN_ID="$(echo "${GPU_NAME}" | tr -c 'A-Za-z0-9.\n-' '-')_$(echo "${MODEL}" | tr '/' '_')_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULTS_ROOT:-$SCRIPT_DIR/results}/${RUN_ID}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# --- 1. Check the machine -------------------------------------------------------------------
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver on this machine"
command -v docker >/dev/null || die "docker not found"
mkdir -p "$RUN_DIR"
log "Results: $RUN_DIR"

nvidia-smi > "$RUN_DIR/nvidia-smi.txt"
DRIVER=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=driver_version --format=csv,noheader | tr -d ' ')
GPU_MODEL=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader)
GPU_MEMORY_TOTAL=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
log "GPU $GPU_INDEX: $GPU_MODEL, ${GPU_MEMORY_TOTAL} MiB, driver $DRIVER"

# The default vLLM image is built for CUDA 13.0, which needs driver 580+.
if [ "${DRIVER%%.*}" -lt 580 ] && [[ "$VLLM_TAG" != *cu12* ]]; then
  die "driver $DRIVER is older than 580; rerun with VLLM_TAG=${VLLM_TAG}-cu129"
fi

log "Pulling $IMAGE (first time only, about 10 GB)"
docker pull -q "$IMAGE" >/dev/null
docker run --rm --gpus "device=$GPU_INDEX" --entrypoint nvidia-smi "$IMAGE" -L >/dev/null \
  || die "containers can't see the GPU: install or fix the NVIDIA Container Toolkit"

# --- 2. Start vLLM --------------------------------------------------------------------------
MONITOR_PID=""
cleanup() {
  [ -n "$MONITOR_PID" ] && kill "$MONITOR_PID" 2>/dev/null || true
  docker logs "$CONTAINER" > "$RUN_DIR/vllm.log" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
mkdir -p "$HF_CACHE"
log "Starting vLLM with $MODEL"
# Same serving flags as docker-compose.yml. The reasoning flags are left out: they only affect
# the chat endpoint, and the benchmark uses /v1/completions.
docker run -d --name "$CONTAINER" \
  --gpus "device=$GPU_INDEX" --ipc=host \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$RUN_DIR:/results" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$IMAGE" \
  --model="$MODEL" \
  --api-key="$API_KEY" \
  --max-model-len=8192 \
  --enable-prefix-caching \
  --seed=0 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} >/dev/null

# --- 3. Wait until it's ready ---------------------------------------------------------------
log "Waiting for /health (timeout ${STARTUP_TIMEOUT}s; includes the model download)"
waited=0
until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
    docker logs --tail 40 "$CONTAINER" >&2 || true
    die "vLLM exited during startup; full log in $RUN_DIR/vllm.log"
  fi
  [ "$waited" -ge "$STARTUP_TIMEOUT" ] && die "vLLM not ready after ${STARTUP_TIMEOUT}s"
  sleep 5
  waited=$((waited + 5))
done
log "vLLM ready after ~${waited}s"

VLLM_VERSION=$(docker exec "$CONTAINER" python3 -c 'import vllm; print(vllm.__version__)')
log "vLLM version $VLLM_VERSION"
# vLLM reserves most of the GPU memory up front for the KV cache, so "after load" is close to the peak.
nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,memory.total --format=csv > "$RUN_DIR/gpu_memory_after_load.csv"
nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
  --format=csv,nounits -l 1 > "$RUN_DIR/gpu_monitor.csv" &
MONITOR_PID=$!

# --- 4. Benchmark ---------------------------------------------------------------------------
for c in $CONCURRENCIES; do
  n=${NUM_PROMPTS:-$(( c * 10 > 32 ? c * 10 : 32 ))}
  log "Concurrency $c: $n requests of ${INPUT_LEN} in / ${OUTPUT_LEN} out tokens"
  docker exec -e OPENAI_API_KEY="$API_KEY" "$CONTAINER" \
    vllm bench serve \
      --base-url "http://127.0.0.1:8000" \
      --model "$MODEL" \
      --dataset-name random \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --ignore-eos \
      --num-prompts "$n" \
      --max-concurrency "$c" \
      --request-rate inf \
      --num-warmups "$c" \
      --seed 0 \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,95,99 \
      --disable-tqdm \
      --save-result \
      --result-dir /results \
      --result-filename "c${c}.json" \
      --metadata "gpu=$GPU_NAME" "vllm_version=$VLLM_VERSION" \
    > "$RUN_DIR/c${c}.log" 2>&1 \
    || { tail -20 "$RUN_DIR/c${c}.log" >&2; die "benchmark failed at concurrency $c"; }
  grep -E "Output token throughput|Median TTFT|Median ITL" "$RUN_DIR/c${c}.log" | sed 's/^/    /' || true
done

# --- 5. Record the setup --------------------------------------------------------------------
kill "$MONITOR_PID" 2>/dev/null || true
MONITOR_PID=""
PEAK_MIB=$(awk -F', ' 'NR > 1 && $2 > max { max = $2 } END { print max + 0 }' "$RUN_DIR/gpu_monitor.csv")

GPU_NAME="$GPU_NAME" GPU_MODEL="$GPU_MODEL" DRIVER="$DRIVER" GPU_MEMORY_TOTAL="$GPU_MEMORY_TOTAL" \
PEAK_MIB="$PEAK_MIB" MODEL="$MODEL" IMAGE="$IMAGE" VLLM_VERSION="$VLLM_VERSION" \
INPUT_LEN="$INPUT_LEN" OUTPUT_LEN="$OUTPUT_LEN" CONCURRENCIES="$CONCURRENCIES" STARTED="$STARTED" \
EXTRA="${EXTRA_ARGS[*]:-}" python3 - > "$RUN_DIR/metadata.json" <<'EOF'
import json, os
from datetime import datetime, timezone
e = os.environ
print(json.dumps({
    "gpu": e["GPU_NAME"],
    "gpu_model": e["GPU_MODEL"],
    "driver_version": e["DRIVER"],
    "gpu_memory_total_mib": int(e["GPU_MEMORY_TOTAL"]),
    "gpu_memory_peak_mib": int(float(e["PEAK_MIB"])),
    "model": e["MODEL"],
    "vllm_image": e["IMAGE"],
    "vllm_version": e["VLLM_VERSION"],
    "input_len": int(e["INPUT_LEN"]),
    "output_len": int(e["OUTPUT_LEN"]),
    "concurrencies": [int(c) for c in e["CONCURRENCIES"].split()],
    "extra_vllm_args": e["EXTRA"],
    "started": e["STARTED"],
    "finished": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, indent=2))
EOF

log "Done: $RUN_DIR"
