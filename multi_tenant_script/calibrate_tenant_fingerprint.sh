#!/bin/bash
# 06_calibrate_tenant_fingerprints.sh
#
# Generates per-tenant KV ground truth BEFORE the real concurrent test.
# This solves the attribution gap flagged in 05_analyze_multitenant.py:
# KVHOOK only gives ONE baseline/window/snapshot per prefill-worker
# lifetime, so it can't natively tell you "this block belongs to tenant
# 2" during a concurrent run. Instead, we learn each tenant's KV
# fingerprint SEPARATELY (sequentially, one prefill-worker restart per
# tenant), save it, then later match those known fingerprints against
# whatever the concurrent capture shows.
#
# This is an ORCHESTRATION SCRIPT, not a fire-and-forget automation --
# it follows the documented restart procedure from the toolkit README
# (01_patch_connector.sh -> start prefill -> wait for BASELINE line ->
# fire ONE request -> wait for dump -> repeat), so it prints clear
# step-by-step instructions and PAUSES for you to confirm each manual
# step (restarting the prefill worker process is not something this
# script can safely automate from the client side alone, since you're
# running this from home without direct server access right now).
#
# Usage: ./06_calibrate_tenant_fingerprints.sh <decode_node_ip> <frontend_port> \
#            <tenant_manifest.tsv> <out_dir>
#
# tenant_manifest.tsv format (same as produced by 04_concurrent_tenants.sh,
# or hand-write one ahead of time with your planned tenant prompts):
#   tenant0 <TAB> MARKER0 <TAB> full prompt text for tenant 0
#   tenant1 <TAB> MARKER1 <TAB> full prompt text for tenant 1
#   ...

set -e

DECODE_IP="${1:?Usage: $0 <decode_node_ip> <frontend_port> <tenant_manifest.tsv> <out_dir>}"
FRONTEND_PORT="${2:-8000}"
TENANT_MANIFEST="${3:?Must provide tenant_manifest.tsv path}"
OUT_DIR="${4:-/tmp/calibration_$(date +%s)}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-8B-unsloth-bnb-4bit}"
VENV_PATH="${VENV_PATH:-/workspace/venv_dynamo_pd}"

if [ ! -f "$TENANT_MANIFEST" ]; then
  echo "ERROR: tenant manifest not found at $TENANT_MANIFEST"
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "============================================================"
echo " Per-Tenant Fingerprint Calibration"
echo "============================================================"
echo "This will guide you through N sequential prefill-worker restarts,"
echo "one per tenant, to capture each tenant's KV ground truth in"
echo "isolation -- BEFORE running the real concurrent multi-tenant test."
echo ""
echo "Each iteration takes ~90s minimum (60s KVHOOK warmup + 30s window),"
echo "plus however long the prefill worker takes to actually restart."
echo "============================================================"
echo ""

N=0
while IFS=$'\t' read -r tenant_id marker prompt <&3; do
  [ -z "$tenant_id" ] && continue
  N=$((N + 1))

  TENANT_OUT="${OUT_DIR}/${tenant_id}"
  mkdir -p "$TENANT_OUT"
  echo "$prompt" > "${TENANT_OUT}/prompt.txt"
  echo "$marker" > "${TENANT_OUT}/marker.txt"

  # Build the JSON request payload as an actual file, using Python's
  # json.dumps for correct escaping -- NOT string-interpolated into a
  # single-quoted curl -d argument. Embedding raw prompt text directly
  # in a bash single-quoted string breaks the moment the prompt contains
  # an apostrophe, backtick, or other shell-special character (confirmed
  # bug: real system prompts and benchmark questions routinely contain
  # apostrophes -- e.g. "farmers' market", "Claude Code's" -- which
  # terminate the outer single-quote early and corrupt/truncate the JSON
  # the server receives, causing a silent parse failure server-side and
  # a 0-byte KVHOOK dump, with no obvious error unless you inspect the
  # response body).
  PAYLOAD_FILE="${TENANT_OUT}/request_payload.json"
  python3 -c "
import json, sys
payload = {
    'model': sys.argv[1],
    'prompt': sys.argv[2],
    'max_tokens': 8,
}
with open(sys.argv[3], 'w') as f:
    json.dump(payload, f)
" "$MODEL_PATH" "$prompt" "$PAYLOAD_FILE"

  echo ""
  echo "------------------------------------------------------------"
  echo " [$N] Calibrating: $tenant_id"
  echo "------------------------------------------------------------"
  echo "Marker: $marker"
  echo "Prompt: $prompt"
  echo "Payload file written to: $PAYLOAD_FILE"
  echo ""
  echo "MANUAL STEPS (run these on millaone, in order):"
  echo ""
  echo "  1. Patch + restart prefill worker with fresh dump targets:"
  echo "     ./01_patch_connector.sh $VENV_PATH"
  echo "     KVHOOK_OUT_BIN=${TENANT_OUT}/kvhook_dump.bin \\"
  echo "     KVHOOK_OUT_MANIFEST=${TENANT_OUT}/kvhook_manifest.txt \\"
  echo "     python3 -m dynamo.vllm \\"
  echo "       --model $MODEL_PATH \\"
  echo "       --tensor-parallel-size 1 \\"
  echo "       --disaggregation-mode prefill \\"
  echo "       --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \\"
  echo "       --discovery-backend etcd \\"
  echo "       --request-plane tcp 2>&1 | tee ${TENANT_OUT}/prefill.log"
  echo ""
  echo "  2. Wait for: [KVHOOK] BASELINE captured. You have 30s ..."
  echo ""
  echo "  3. In another terminal, IMMEDIATELY fire ONLY this tenant's request"
  echo "     (using the pre-built JSON file, NOT inline -d text -- avoids"
  echo "     shell-quoting corruption on prompts containing apostrophes,"
  echo "     backticks, or other shell-special characters):"
  echo "     curl -s http://localhost:${FRONTEND_PORT}/v1/completions \\"
  echo "       -H 'Content-Type: application/json' \\"
  echo "       --data-binary @${PAYLOAD_FILE} \\"
  echo "       > ${TENANT_OUT}/response.json"
  echo ""
  echo "  4. Wait for: [KVHOOK] dumped N bytes total to ..."
  echo ""
  echo "  CHECK BEFORE PROCEEDING: cat ${TENANT_OUT}/response.json"
  echo "  and confirm it's a real completion (has 'choices' field), NOT"
  echo "  an error like 'Failed to parse the request body as JSON'."
  echo ""
  read -p "Press ENTER once steps 1-4 are complete AND verified for $tenant_id (Ctrl+C to abort)... " < /dev/tty

  echo "Recorded calibration target dir: $TENANT_OUT"
  echo "Expecting files: kvhook_dump.bin, kvhook_manifest.txt, response.json"

done 3< "$TENANT_MANIFEST"

echo ""
echo "============================================================"
echo " Calibration loop complete: $N tenants processed."
echo "============================================================"
echo "Per-tenant ground truth saved under: $OUT_DIR/<tenant_id>/"
echo ""
echo "Next: run 04_concurrent_tenants.sh for the REAL concurrent test"
echo "(no KVHOOK needed for that run -- just the wire capture), then"
echo "match each tenant's calibrated fingerprints from here against"
echo "the concurrent capture to determine per-tenant attribution."