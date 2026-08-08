#!/bin/bash
# 07_calibrate_under_batching.sh
#
# Tests whether concurrent batching changes a tenant's KV tensor bytes
# compared to solo calibration. Confirmed bug/finding from concurrent_run1:
# byte-exact fingerprints calibrated SOLO (06_calibrate_tenant_fingerprints.sh)
# did not transfer to a real concurrent capture, even with non-determinism
# ruled out (two solo runs of the identical prompt produced byte-identical
# dumps). The remaining explanation is that batching itself perturbs the
# computed KV values.
#
# This script tests that directly: ONE prefill-worker restart, ONE KVHOOK
# window, but with a TARGET tenant fired first (small head-start) and the
# other tenants fired shortly after -- so the target's KVHOOK dump reflects
# genuinely batched computation, while still being cleanly attributable to
# the target alone (KVHOOK's baseline/diff mechanism naturally isolates
# whichever blocks are newly-written during the whole window; since only
# the target's prompt is unique/fingerprintable here, and batch-mates are
# the OTHER real tenants from the same manifest, this captures the target
# tenant's blocks specifically, computed under real batched conditions).
#
# Usage: ./07_calibrate_under_batching.sh <decode_ip> <frontend_port> \
#            <tenant_manifest.tsv> <target_tenant_id> <out_dir> [head_start_ms]
#
# Example:
#   ./07_calibrate_under_batching.sh 10.42.5.114 8000 tenant_manifest.tsv \
#       tenant0 /tmp/batch_calib_tenant0 75

set -e

DECODE_IP="${1:?Usage: $0 <decode_ip> <frontend_port> <tenant_manifest.tsv> <target_tenant_id> <out_dir> [head_start_ms]}"
FRONTEND_PORT="${2:-8000}"
TENANT_MANIFEST="${3:?Must provide tenant_manifest.tsv path}"
TARGET_TENANT="${4:?Must specify which tenant_id is the calibration target}"
OUT_DIR="${5:-/tmp/batch_calib_$(date +%s)}"
HEAD_START_MS="${6:-75}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-8B-unsloth-bnb-4bit}"
VENV_PATH="${VENV_PATH:-/workspace/venv_dynamo_pd}"
KVHOOK_WARMUP_S="${KVHOOK_WARMUP_S:-60}"
KVHOOK_WINDOW_S="${KVHOOK_WINDOW_S:-30}"

if [ ! -f "$TENANT_MANIFEST" ]; then
  echo "ERROR: tenant manifest not found at $TENANT_MANIFEST"
  exit 1
fi

mkdir -p "$OUT_DIR"

# --- Load all tenants from manifest, separating target from batch-mates ---
declare -a OTHER_IDS
declare -a OTHER_PROMPTS
TARGET_PROMPT=""

i=0
while IFS=$'\t' read -r tenant_id marker prompt <&3; do
  [ -z "$tenant_id" ] && continue
  if [ "$tenant_id" == "$TARGET_TENANT" ]; then
    TARGET_PROMPT="$prompt"
  else
    OTHER_IDS[$i]="$tenant_id"
    OTHER_PROMPTS[$i]="$prompt"
    i=$((i + 1))
  fi
done 3< "$TENANT_MANIFEST"

if [ -z "$TARGET_PROMPT" ]; then
  echo "ERROR: target tenant '$TARGET_TENANT' not found in manifest"
  exit 1
fi

echo "============================================================"
echo " Batched Recalibration: $TARGET_TENANT"
echo "============================================================"
echo "Target tenant:    $TARGET_TENANT"
echo "Batch-mates:      ${OTHER_IDS[*]}"
echo "Head start:       ${HEAD_START_MS}ms"
echo "Output dir:       $OUT_DIR"
echo "============================================================"
echo ""

# --- Build JSON payload files up front (same fix as 06/04: avoid shell
#     quoting corruption on apostrophes/backticks in real prompt content) ---
TARGET_PAYLOAD="${OUT_DIR}/target_payload.json"
python3 -c "
import json, sys
payload = {'model': sys.argv[1], 'prompt': sys.argv[2], 'max_tokens': 8}
json.dump(payload, open(sys.argv[3], 'w'))
" "$MODEL_PATH" "$TARGET_PROMPT" "$TARGET_PAYLOAD"

declare -a OTHER_PAYLOADS
for idx in "${!OTHER_IDS[@]}"; do
  PFILE="${OUT_DIR}/other_${OTHER_IDS[$idx]}_payload.json"
  python3 -c "
import json, sys
payload = {'model': sys.argv[1], 'prompt': sys.argv[2], 'max_tokens': 8}
json.dump(payload, open(sys.argv[3], 'w'))
" "$MODEL_PATH" "${OTHER_PROMPTS[$idx]}" "$PFILE"
  OTHER_PAYLOADS[$idx]="$PFILE"
done

echo "Payload files written. Restoring connector + re-patching for a clean run..."

# --- Restore + re-patch (idempotent, matches 01_patch_connector.sh's own logic) ---
NIXL_FILE="$VENV_PATH/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py"
if [ -f "${NIXL_FILE}.bak" ]; then
  cp "${NIXL_FILE}.bak" "$NIXL_FILE"
fi
./01_patch_connector.sh "$VENV_PATH"

echo ""
echo "Launching prefill worker (warmup=${KVHOOK_WARMUP_S}s, window=${KVHOOK_WINDOW_S}s)..."
echo "This will run in the BACKGROUND of this script -- log at: ${OUT_DIR}/prefill.log"
echo ""

KVHOOK_WARMUP_S="$KVHOOK_WARMUP_S" \
KVHOOK_WINDOW_S="$KVHOOK_WINDOW_S" \
KVHOOK_OUT_BIN="${OUT_DIR}/kvhook_dump.bin" \
KVHOOK_OUT_MANIFEST="${OUT_DIR}/kvhook_manifest.txt" \
python3 -m dynamo.vllm \
  --model "$MODEL_PATH" \
  --tensor-parallel-size 1 \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --discovery-backend etcd \
  --request-plane tcp > "${OUT_DIR}/prefill.log" 2>&1 &
PREFILL_PID=$!

echo "Prefill worker started (PID $PREFILL_PID). Waiting for BASELINE..."

# Poll the log for the BASELINE line rather than sleeping blindly --
# more robust if warmup timing drifts from the configured default.
BASELINE_SEEN=0
for i in $(seq 1 $((KVHOOK_WARMUP_S + 30))); do
  if grep -q "BASELINE captured" "${OUT_DIR}/prefill.log" 2>/dev/null; then
    BASELINE_SEEN=1
    break
  fi
  sleep 1
done

if [ "$BASELINE_SEEN" -eq 0 ]; then
  echo "ERROR: BASELINE line never appeared within $((KVHOOK_WARMUP_S + 30))s."
  echo "Check ${OUT_DIR}/prefill.log for errors."
  kill "$PREFILL_PID" 2>/dev/null || true
  exit 1
fi

echo "BASELINE captured. Firing target ($TARGET_TENANT) now, then batch-mates after ${HEAD_START_MS}ms..."

# --- Fire target first ---
curl -s "http://localhost:${FRONTEND_PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "@${TARGET_PAYLOAD}" \
  > "${OUT_DIR}/target_response.json" &
TARGET_CURL_PID=$!

# --- Head start delay ---
python3 -c "import time; time.sleep($HEAD_START_MS / 1000.0)"

# --- Fire all batch-mates concurrently ---
declare -a OTHER_CURL_PIDS
for idx in "${!OTHER_IDS[@]}"; do
  RESP_FILE="${OUT_DIR}/other_${OTHER_IDS[$idx]}_response.json"
  curl -s "http://localhost:${FRONTEND_PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@${OTHER_PAYLOADS[$idx]}" \
    > "$RESP_FILE" &
  OTHER_CURL_PIDS[$idx]=$!
done

wait "$TARGET_CURL_PID"
for pid in "${OTHER_CURL_PIDS[@]}"; do
  wait "$pid"
done

echo "All requests completed. Verifying responses..."
ANY_FAILED=0
for f in "${OUT_DIR}"/target_response.json "${OUT_DIR}"/other_*_response.json; do
  if ! python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
assert 'choices' in d
" "$f" 2>/dev/null; then
    echo "  WARNING: $f does not look like a valid completion"
    ANY_FAILED=1
  fi
done
[ "$ANY_FAILED" -eq 0 ] && echo "  All responses OK."

echo ""
echo "Waiting for KVHOOK dump (up to ${KVHOOK_WINDOW_S}s remaining in window)..."
DUMP_SEEN=0
for i in $(seq 1 $((KVHOOK_WINDOW_S + 30))); do
  if grep -q "dumped .* bytes total" "${OUT_DIR}/prefill.log" 2>/dev/null; then
    DUMP_SEEN=1
    break
  fi
  sleep 1
done

if [ "$DUMP_SEEN" -eq 0 ]; then
  echo "ERROR: dump line never appeared. Check ${OUT_DIR}/prefill.log"
  kill "$PREFILL_PID" 2>/dev/null || true
  exit 1
fi

echo "Dump complete."
ls -la "${OUT_DIR}/kvhook_dump.bin" "${OUT_DIR}/kvhook_manifest.txt"

echo ""
echo "Stopping prefill worker (PID $PREFILL_PID)..."
kill "$PREFILL_PID" 2>/dev/null || true
wait "$PREFILL_PID" 2>/dev/null || true

echo ""
echo "============================================================"
echo " Batched recalibration for $TARGET_TENANT complete."
echo "============================================================"
echo "Output: ${OUT_DIR}/kvhook_dump.bin, ${OUT_DIR}/kvhook_manifest.txt"
echo ""
echo "Next: compare this dump against the SOLO calibration dump for the"
echo "same tenant (e.g. /tmp/calibration_run1/${TARGET_TENANT}/kvhook_dump.bin)"
echo "to check whether batching changed the computed bytes."