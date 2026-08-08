#!/usr/bin/env python3
"""
Measures the message-start header's exact length at the K/V split
boundary (payload offset 32768), using the same divergence-sampling
method that characterized the continuation header in
test_framing_hypothesis.py: find the header's raw start position (via the
validated self-sync parser reaching payload offset 32768), then search
forward for where the true V-half content resumes, across several
independent blocks. If the gap is consistent, that's the header length.

Usage:
    python3 measure_kv_header.py
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

KV_SPLIT_OFFSET = 32768
SEARCH_WINDOW = 4000  # generous -- we don't know the header length yet
TARGET_LAYERS = [
    ("model.layers.0.self_attn.attn", "79"),
    ("model.layers.5.self_attn.attn", "79"),
    ("model.layers.10.self_attn.attn", "79"),
]


def main():
    rows = load_manifest(MANIFEST)
    row_by_key = {(r["layer_name"], r["block_id"]): r for r in rows}
    print("parsing pcap + reassembling...")
    segments = list(parse_pcap(PCAP))
    streams = reassemble_streams(segments)
    bulk_key, buf = max(streams.items(), key=lambda kv: len(kv[1]))
    print(f"bulk stream: {bulk_key}, {len(buf)} bytes\n")

    gaps = []
    with open(DUMP, "rb") as f:
        for layer, block_id in TARGET_LAYERS:
            row = row_by_key.get((layer, block_id))
            if row is None:
                print(f"[{layer} block_id={block_id}] not in manifest, skip")
                continue
            f.seek(int(row["offset"]))
            block_bytes = f.read(int(row["len"]))
            if hashlib.sha256(block_bytes).hexdigest() != row["content_hash"]:
                print(f"[{layer} block_id={block_id}] hash mismatch, skip")
                continue
            unit0 = block_bytes[:SUBUNIT_BYTES]
            anchor = buf.find(unit0)
            if anchor == -1:
                print(f"[{layer} block_id={block_id}] no anchor, skip")
                continue

            # true raw position where the K/V-split header starts
            _, consumed = deframe_from_anchor(buf, anchor, KV_SPLIT_OFFSET)
            if consumed is None:
                print(f"[{layer} block_id={block_id}] deframe_from_anchor could not "
                      f"reach payload offset {KV_SPLIT_OFFSET}, skip")
                continue
            header_start = anchor + consumed

            # true V-half content we're looking for
            v_half_start = block_bytes[KV_SPLIT_OFFSET:KV_SPLIT_OFFSET + 64]

            # search forward from header_start for where it resumes
            search_region = buf[header_start:header_start + SEARCH_WINDOW]
            idx = search_region.find(v_half_start[:32])  # match on a prefix in case tail also gets split
            if idx == -1:
                print(f"[{layer} block_id={block_id}] header_start={header_start}: "
                      f"V-half content not found within {SEARCH_WINDOW} bytes")
                print(f"    raw bytes at header_start: {buf[header_start:header_start+64].hex()}")
                continue
            gaps.append(idx)
            print(f"[{layer} block_id={block_id}] header_start={header_start}: "
                  f"gap to V-half content = {idx} bytes")
            print(f"    header bytes: {buf[header_start:header_start+idx].hex()}")

    print()
    if gaps:
        from collections import Counter
        c = Counter(gaps)
        print(f"gap distribution across {len(gaps)} samples: {dict(c)}")
        if len(c) == 1:
            print(f"CONSISTENT: message-start header length = {gaps[0]} bytes")
        else:
            print("NOT perfectly consistent -- may vary, or search matched a false "
                  "positive for some samples (32-byte prefix match could coincidentally "
                  "occur earlier in rare cases)")
    else:
        print("no gaps measured -- something else is wrong, check header_start computation")


if __name__ == "__main__":
    main()
