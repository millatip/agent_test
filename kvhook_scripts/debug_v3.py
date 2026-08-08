#!/usr/bin/env python3
"""Isolates the deframe_v3.py bug on the one block already hand-verified
to work with the fixed-period model (layer0/block29, Phase A original
capture) -- traces exactly where the self-synchronizing candidate
diverges from ground truth, if at all."""
import sys
import hashlib
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest
from deframe_and_match import SUBUNIT_BYTES
import deframe_v3
from deframe_v3 import deframe_from_anchor, HEADER_LEN
print("deframe_v3 module file:", deframe_v3.__file__)
print("HEADER_RE pattern:", deframe_v3.HEADER_RE.pattern)

PCAP = "/tmp/kv_phaseA_20260806_210137.pcap"
DUMP = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_dump_phaseA.bin"
MANIFEST = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_manifest_phaseA.tsv"

LAYER = sys.argv[1] if len(sys.argv) > 1 else "model.layers.0.self_attn.attn"
BLOCK_ID = sys.argv[2] if len(sys.argv) > 2 else "29"

rows = load_manifest(MANIFEST)
row = next(r for r in rows if r["layer_name"] == LAYER and r["block_id"] == BLOCK_ID)
with open(DUMP, "rb") as f:
    f.seek(int(row["offset"]))
    block_bytes = f.read(int(row["len"]))
assert hashlib.sha256(block_bytes).hexdigest() == row["content_hash"]
unit0 = block_bytes[:SUBUNIT_BYTES]

segments = list(parse_pcap(PCAP))
streams = reassemble_streams(segments)
bulk_key, buf = max(streams.items(), key=lambda kv: len(kv[1]))
anchor = buf.find(unit0)
print(f"anchor={anchor}, target_len={len(block_bytes)}")

expected_hdr_pos = anchor + 8176
print(f"expected header at anchor+8176={expected_hdr_pos}")
print(f"raw bytes there: {buf[expected_hdr_pos:expected_hdr_pos+16].hex()}")
direct = deframe_v3._find_header(buf, anchor, anchor + 8300)
print(f"_find_header(buf, anchor, anchor+8300) directly = {direct}")
if direct != -1:
    print(f"  offset from anchor: {direct - anchor}")

candidate, consumed = deframe_from_anchor(buf, anchor, len(block_bytes))
print(f"candidate is None: {candidate is None}")
if candidate is not None:
    print(f"candidate length: {len(candidate)}, consumed raw bytes: {consumed}")
    print(f"matches ground truth: {candidate == block_bytes}")
    if candidate != block_bytes:
        n = min(len(candidate), len(block_bytes))
        for i in range(n):
            if candidate[i] != block_bytes[i]:
                print(f"first divergence at byte {i}/{n}")
                print(f"  candidate around there: {candidate[max(0,i-8):i+16].hex()}")
                print(f"  truth around there:     {block_bytes[max(0,i-8):i+16].hex()}")
                break
        else:
            print("no divergence in overlapping region (length mismatch only)")
