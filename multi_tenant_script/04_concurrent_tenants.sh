#!/bin/bash
# 04_concurrent_tenants.sh
#
# Fires N concurrent "tenant" requests at the frontend while a single
# tcpdump capture runs, to test whether their KV transfers are isolated
# (separate connections) or interleaved (shared connection) on the wire.
#
# Builds on the validated pattern from 02_capture_and_request.sh:
# same marker-based approach, same tcpdump invocation, just parallelized
# across N tenants instead of one sequential request.
#
# IMPORTANT — run this DURING the KVHOOK window if you also want
# source-level ground truth (see 01_patch_connector.sh). The patch takes
# ONE baseline + ONE final snapshot, so all N tenant requests below must
# fire within that single KVHOOK_WINDOW_S window. If you don't need
# ground truth for this run (e.g. just characterizing connection
# behavior first), you can run this standalone without the patch active.
#
# Usage: ./04_concurrent_tenants.sh <decode_node_ip> <frontend_port> <n_tenants> <out_dir> [existing_manifest.tsv]
#   e.g. ./04_concurrent_tenants.sh 10.42.5.114 8000 3 /tmp/multitenant_run1
#   e.g. ./04_concurrent_tenants.sh 10.42.5.114 8000 5 /tmp/multitenant_run1 /workspace/tenant_manifest.tsv
#
# If a 5th argument is given and points to an existing manifest (e.g. the
# same real-benchmark-dataset manifest used for calibration), its tenants
# and prompts are reused directly instead of generating fresh synthetic
# fillers -- recommended, so fingerprints match what was calibrated.

set -e

DECODE_IP="${1:?Usage: $0 <decode_node_ip> <frontend_port> <n_tenants> <out_dir> [existing_manifest.tsv]}"
FRONTEND_PORT="${2:-8000}"
N_TENANTS="${3:-3}"
OUT_DIR="${4:-/tmp/multitenant_$(date +%s)}"
EXISTING_MANIFEST="${5:-}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-8B-unsloth-bnb-4bit}"
IFACE="${IFACE:-eth0}"
MAX_TOKENS="${MAX_TOKENS:-8}"

mkdir -p "$OUT_DIR"
PCAP_OUT="${OUT_DIR}/capture.pcap"
MANIFEST="${OUT_DIR}/tenant_manifest.tsv"

echo "============================================================"
echo " Multi-Tenant Concurrent Capture"
echo "============================================================"
echo "Decode node:      $DECODE_IP"
echo "Frontend port:     $FRONTEND_PORT"
echo "Tenants:          $N_TENANTS"
echo "Output dir:        $OUT_DIR"
echo "============================================================"

declare -a MARKERS
declare -a PROMPTS

if [ -n "$EXISTING_MANIFEST" ] && [ -f "$EXISTING_MANIFEST" ]; then
  echo "Reusing existing manifest: $EXISTING_MANIFEST"
  echo "(recommended -- matches whatever was used for calibration)"
  cp "$EXISTING_MANIFEST" "$MANIFEST"

  i=0
  while IFS=$'\t' read -r tenant_id marker prompt <&3; do
    [ -z "$tenant_id" ] && continue
    MARKERS[$i]="$marker"
    PROMPTS[$i]="$prompt"
    i=$((i + 1))
  done 3< "$EXISTING_MANIFEST"

  if [ "$i" -ne "$N_TENANTS" ]; then
    echo "WARNING: existing manifest has $i tenant(s) but N_TENANTS=$N_TENANTS"
    echo "was requested. Using $i tenants from the manifest (actual count)."
    N_TENANTS=$i
  fi
else
  echo "No existing manifest provided -- generating synthetic filler prompts."
  echo "(Pass a 5th argument pointing to a real manifest, e.g. the one built"
  echo "by build_tenant_manifest.py, to reuse real benchmark content instead.)"

  # --- Generate one unique, fingerprintable marker per tenant ---
  # Each tenant gets a distinct marker string AND a distinct filler payload
  # so their prompts differ enough to produce genuinely different KV
  # content (not just a one-token difference), making later byte-level
  # fingerprint separation meaningful.
  TENANT_FILLERS=(
    "alpha quarterly revenue figures confidential review"
    "beta patient diagnosis history private record"
    "gamma legal contract clause negotiation draft"
    "delta source code proprietary algorithm details"
    "epsilon merger acquisition financial projection"
  )

  for i in $(seq 0 $((N_TENANTS - 1))); do
    MARK="TENANT${i}_$(date +%s%N | tail -c 9)"
    FILLER="${TENANT_FILLERS[$((i % ${#TENANT_FILLERS[@]}))]}"
    PROMPT="${MARK} ${FILLER} ${MARK}"
    MARKERS[$i]="$MARK"
    PROMPTS[$i]="$PROMPT"
    echo -e "tenant${i}\t${MARK}\t${PROMPT}" >> "$MANIFEST"
  done
fi

echo ""
echo "Tenant manifest written to: $MANIFEST"
cat "$MANIFEST"
echo ""

# --- Start capture ---
echo "Starting tcpdump on $IFACE, filtering host $DECODE_IP ..."
tcpdump -i "$IFACE" host "$DECODE_IP" -w "$PCAP_OUT" &
TCPDUMP_PID=$!
sleep 0.5

# --- Fire all N requests concurrently ---
echo "Firing $N_TENANTS concurrent requests at $(date +%s.%N) ..."
declare -a RESP_PIDS
declare -a RESP_FILES

for i in $(seq 0 $((N_TENANTS - 1))); do
  RESP_FILE="${OUT_DIR}/tenant${i}_response.json"
  RESP_FILES[$i]="$RESP_FILE"
  PAYLOAD_FILE="${OUT_DIR}/tenant${i}_payload.json"

  # Build the JSON payload via Python's json.dumps, NOT inline bash
  # string interpolation into a double-quoted curl -d argument. The
  # synthetic fillers above happen to contain no apostrophes/backticks,
  # so this was accidentally safe so far -- but the moment real prompt
  # content (system prompts, benchmark questions) is used instead, any
  # apostrophe corrupts the JSON the same way it did in calibration
  # (confirmed bug: see 06_calibrate_tenant_fingerprints.sh history).
  # Fixed proactively here for consistency and to avoid the same failure
  # mode if/when this script is pointed at real-content manifests.
  python3 -c "
import json, sys
payload = {
    'model': sys.argv[1],
    'prompt': sys.argv[2],
    'max_tokens': int(sys.argv[3]),
}
with open(sys.argv[4], 'w') as f:
    json.dump(payload, f)
" "$MODEL_PATH" "${PROMPTS[$i]}" "$MAX_TOKENS" "$PAYLOAD_FILE"

  (
    START_TS=$(date +%s.%N)
    curl -s "http://localhost:${FRONTEND_PORT}/v1/completions" \
      -H 'Content-Type: application/json' \
      --data-binary "@${PAYLOAD_FILE}" \
      > "$RESP_FILE"
    END_TS=$(date +%s.%N)
    echo "tenant${i} start=${START_TS} end=${END_TS}" >> "${OUT_DIR}/timing.log"
  ) &
  RESP_PIDS[$i]=$!
done

# Wait for all tenant requests to complete
for pid in "${RESP_PIDS[@]}"; do
  wait "$pid"
done

echo "All $N_TENANTS requests completed at $(date +%s.%N)"

# Sanity check: verify every response is a real completion, not a JSON
# parse error or other server-side rejection. A silently-failed request
# here would corrupt the capture's interpretation the same way a 0-byte
# KVHOOK dump did during calibration -- catch it immediately, not later
# during analysis.
echo ""
echo "Verifying all responses are real completions (not errors)..."
ANY_FAILED=0
for i in $(seq 0 $((N_TENANTS - 1))); do
  RESP_FILE="${OUT_DIR}/tenant${i}_response.json"
  if ! python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    assert 'choices' in d
except Exception as e:
    print(f'tenant${i}: INVALID response ({e})')
    sys.exit(1)
print(f'tenant${i}: OK')
" "$RESP_FILE"; then
    ANY_FAILED=1
  fi
done
if [ "$ANY_FAILED" -eq 1 ]; then
  echo ""
  echo "WARNING: at least one tenant's request failed or returned an"
  echo "error instead of a real completion. Inspect the response file(s)"
  echo "flagged above before trusting this capture's results -- a failed"
  echo "request means no real KV transfer happened for that tenant, which"
  echo "will silently skew the isolation/mixing analysis."
fi

# Small tail buffer to catch trailing packets, then stop capture
sleep 1
kill "$TCPDUMP_PID" 2>/dev/null || true
wait "$TCPDUMP_PID" 2>/dev/null || true

echo ""
echo "============================================================"
echo " Run complete"
echo "============================================================"
echo "Capture:   $PCAP_OUT"
echo "Manifest:  $MANIFEST"
echo "Timing:    ${OUT_DIR}/timing.log"
echo "Responses: ${OUT_DIR}/tenant*_response.json"
echo ""
echo "Next step: run 05_analyze_multitenant.py against this capture"
echo "to check (a) connection/stream count vs tenant count, and"
echo "(b) whether tenant fingerprints interleave within a stream."