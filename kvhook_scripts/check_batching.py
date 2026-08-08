#!/usr/bin/env python3
"""
Tests whether header_start (where block79's K-half ends) is actually the
start of the NEXT block's K-half, rather than a "K/V split" message for
the SAME block -- i.e. whether K-halves for many blocks are batched
together contiguously, with V-halves batched separately much later.
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
NEXT_BLOCK_ID = "80"
KV_SPLIT_OFFSET = 32768


def main():
    rows = load_manifest(MANIFEST)
    row79 = next(r for r in rows if r["layer_name"] == LAYER and r["block_id"] == BLOCK_ID)
    row80 = next(r for r in rows if r["layer_name"] == LAYER and r["block_id"] == NEXT_BLOCK_ID)

    with open(DUMP, "rb") as f:
        f.seek(int(row79["offset"]))
        block79 = f.read(int(row79["len"]))
        f.seek(int(row80["offset"]))
        block80 = f.read(int(row80["len"]))
    assert hashlib.sha256(block79).hexdigest() == row79["content_hash"]
    assert hashlib.sha256(block80).hexdigest() == row80["content_hash"]

    print("parsing pcap + reassembling...")
    segments = list(parse_pcap(PCAP))
    streams = reassemble_streams(segments)
    bulk_key, buf = max(streams.items(), key=lambda kv: len(kv[1]))

    anchor79 = buf.find(block79[:SUBUNIT_BYTES])
    print(f"block79 (K-half) anchor: {anchor79}")
    _, consumed = deframe_from_anchor(buf, anchor79, KV_SPLIT_OFFSET)
    header_start = anchor79 + consumed
    print(f"header_start (where block79's first 32768 bytes end): {header_start}")

    # is this the start of block80's K-half (first 4096 bytes)?
    block80_k_start = block80[:SUBUNIT_BYTES]
    idx_direct = buf.find(block80_k_start, header_start, header_start + 200)
    print(f"searching for block80's K-half start within 200 bytes of header_start: "
          f"{'FOUND at ' + str(idx_direct) if idx_direct != -1 else 'not found'}")

    # where IS block80's K-half, wherever it is?
    idx_anywhere = buf.find(block80_k_start)
    print(f"block80's K-half found anywhere in stream at: {idx_anywhere}")
    if idx_anywhere != -1:
        gap = idx_anywhere - header_start
        print(f"  gap from header_start: {gap} bytes")

    print(f"\nraw bytes at header_start: {buf[header_start:header_start+64].hex()}")


if __name__ == "__main__":
    main()
