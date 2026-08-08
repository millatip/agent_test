#!/usr/bin/env python3
"""
Tests whether the byte-32768 boundary (exactly half of a 65,536-byte
block) is a message-start header for the V half of a K/V-split transfer,
given base_worker.py registers K and V as separate NIXL memory regions.

For several blocks in the clean 79MB capture:
  - dump the 64 bytes at the block's own true start (a known message-start
    header, from dump_header_bytes.py's earlier per-block anchor sampling)
  - dump the 64 bytes at offset 32768 within the block (the suspected K/V
    boundary)
  - dump the 64 bytes at offset 16384 (quarter-block, to rule out
    per-head-group splitting instead)
and compares structure across all three positions and across blocks.

Usage:
    python3 check_kv_split.py
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

WINDOW = 40


def main():
    rows = load_manifest(MANIFEST)
    print("parsing pcap + reassembling...")
    segments = list(parse_pcap(PCAP))
    streams = reassemble_streams(segments)
    bulk_key, buf = max(streams.items(), key=lambda kv: len(kv[1]))
    print(f"bulk stream: {bulk_key}, {len(buf)} bytes\n")

    # sample a handful of blocks spread across layers
    sample_layers = [
        "model.layers.0.self_attn.attn", "model.layers.5.self_attn.attn",
        "model.layers.10.self_attn.attn", "model.layers.20.self_attn.attn",
        "model.layers.30.self_attn.attn", "model.layers.35.self_attn.attn",
    ]

    with open(DUMP, "rb") as f:
        for layer in sample_layers:
            row = next((r for r in rows if r["layer_name"] == layer), None)
            if row is None:
                continue
            f.seek(int(row["offset"]))
            block_bytes = f.read(int(row["len"]))
            if hashlib.sha256(block_bytes).hexdigest() != row["content_hash"]:
                continue
            unit0 = block_bytes[:SUBUNIT_BYTES]
            anchor = buf.find(unit0)
            if anchor == -1:
                print(f"[{layer} block_id={row['block_id']}] no anchor, skip")
                continue

            print(f"=== {layer} block_id={row['block_id']} anchor={anchor} ===")
            print(f"  at block-start (message header, before anchor):")
            print(f"    {buf[anchor-WINDOW:anchor].hex()}")

            # use the validated self-sync parser to find the TRUE raw
            # position of each logical payload offset, accounting for any
            # continuation headers already crossed (naive anchor+offset
            # would be wrong by however many x21-byte headers preceded it)
            for label, payload_offset in (("16384 (quarter-block)", 16384), ("32768 (half-block, suspected K/V boundary)", 32768)):
                _, consumed = deframe_from_anchor(buf, anchor, payload_offset)
                if consumed is None:
                    print(f"  offset {label}: could not de-frame that far")
                    continue
                raw_pos = anchor + consumed
                print(f"  at logical payload offset {label}, TRUE raw position={raw_pos}:")
                print(f"    {buf[raw_pos-8:raw_pos+32].hex()}")
            print()


if __name__ == "__main__":
    main()
