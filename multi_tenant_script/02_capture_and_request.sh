#!/bin/bash
# 02_capture_and_request.sh
#
# Run this AFTER the patched prefill worker has printed:
#   "[KVHOOK] BASELINE captured. You have Ns to fire your ONE request now."
#
# It starts tcpdump, fires one completions request with a unique marker,
# then stops the capture. Run on the PREFILL node (millaone), since that's
# where the outbound NIXL traffic to the decode node originates.
#
# Usage: ./02_capture_and_request.sh <decode_node_ip> <frontend_port> <pcap_out>
#   e.g. ./02_capture_and_request.sh 10.42.5.114 8000 /tmp/nixl_capture_001.pcap

set -e

DECODE_IP="${1:?Usage: $0 <decode_node_ip> <frontend_port> <pcap_out>}"
FRONTEND_PORT="${2:-8000}"
PCAP_OUT="${3:-/tmp/nixl_capture_$(date +%s).pcap}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-8B-unsloth-bnb-4bit}"
IFACE="${IFACE:-eth0}"

# Unique marker per run so repeated captures never collide with cache hits.
MARKER="MARK_$(date +%s%N | tail -c 9)"
PROMPT="${MARKER} reproducible kv capture test request ${MARKER}"

echo "Capture file: $PCAP_OUT"
echo "Marker: $MARKER"

tcpdump -i "$IFACE" host "$DECODE_IP" -w "$PCAP_OUT" &
TCPDUMP_PID=$!
sleep 0.5

echo "Firing request at $(date)"
RESPONSE=$(curl -s "http://localhost:${FRONTEND_PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL_PATH}\", \"prompt\": \"${PROMPT}\", \"max_tokens\": 8}")
echo "Response: $RESPONSE"
echo "Done at $(date)"

sleep 1
kill "$TCPDUMP_PID" 2>/dev/null || true
wait "$TCPDUMP_PID" 2>/dev/null || true

echo ""
echo "Capture saved to $PCAP_OUT"
echo "Marker used: $MARKER (save this — needed for content verification later)"
echo "$MARKER" > "${PCAP_OUT%.pcap}.marker.txt"
