#!/usr/bin/env python3
"""
The message-start-format header at the K/V boundary doesn't have the
expected V-half content shortly after it (not found within 4000 bytes).
Before assuming a longer header, check a more basic possibility: maybe
K and V for one block aren't transmitted back-to-back at all. Full-buffer
search for the V-half's expected first bytes, reporting wherever it's
actually found (or confirming it's absent from this stream entirely, in
which case it's likely in one of the other streams, or this "half split"
isn't actually a K/V split the way assumed).
"""
import sys
import hashlib
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest
from deframe_and_match import SUBUNIT_BYTES
from deframe_v3 import deframe_from_anchor

PCAP = "/tmp/kv_phaseA_clean_20260806_225414.pcap"
DUMP = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_dump_phaseA_clean.bin"
MANIFEST = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_manifest_phaseA_clean.tsv"

LAYER = "model.layers.0.self_attn.attn"
BLOCK_ID = "79"
KV_SPLIT_OFFSET = 32768


def main():
    rows = load_manifest(MANIFEST)
    row = next(r for r in rows if r["layer_name"] == LAYER and r["block_id"] == BLOCK_ID)
    with open(DUMP, "rb") as f:
        f.seek(int(row["offset"]))
        block_bytes = f.read(int(row["len"]))
    assert hashlib.sha256(block_bytes).hexdigest() == row["content_hash"]
    unit0 = block_bytes[:SUBUNIT_BYTES]

    print("parsing pcap + reassembling...")
    segments = list(parse_pcap(PCAP))
    streams = reassemble_streams(segments)

    v_half_target = block_bytes[KV_SPLIT_OFFSET:KV_SPLIT_OFFSET + 64]
    print(f"searching all {len(streams)} streams for the V-half's expected first 64 bytes...")
    for key, buf in streams.items():
        idx = buf.find(v_half_target)
        if idx != -1:
            print(f"  FOUND in stream {key} at offset {idx}")
        # also try smaller prefixes in case it's fragmented differently
        for n in (32, 16, 8):
            idx_n = buf.find(v_half_target[:n])
            if idx_n != -1:
                print(f"  {n}-byte prefix found in stream {key} at offset {idx_n}")
                break
        else:
            print(f"  not found at all (even 8-byte prefix) in stream {key} ({len(buf)} bytes)")

    # sanity: re-confirm K-half (first 64 bytes of block) location, should
    # match the already-known anchor
    k_half_target = block_bytes[:64]
    bulk_key, buf = max(streams.items(), key=lambda kv: len(kv[1]))
    k_idx = buf.find(k_half_target)
    print(f"\nsanity check -- K-half start (should be the known anchor): found at {k_idx}")


if __name__ == "__main__":
    main()
