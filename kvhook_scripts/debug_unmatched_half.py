#!/usr/bin/env python3
"""
Finds a specific unmatched half from extract_all_halves's output and
compares its extracted bytes against the true K-half or V-half of the
block it SHOULD be (inferred from position: it lies between two known
neighbors), to see exactly where/how the extraction went wrong.

Usage:
    python3 debug_unmatched_half.py <stream_index> <half_index>
Run diagnose_walker.py first to get stream/half indices of interest from
a reported desync run.
"""
import sys
import hashlib
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest
from reconstruct_v2 import extract_all_halves, HALF_SIZE
from deframe_v3 import HEADER_LEN

PCAP = "/tmp/kv_phaseA_clean_20260806_225414.pcap"
DUMP = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_dump_phaseA_clean.bin"
MANIFEST = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_manifest_phaseA_clean.tsv"


def main():
    # default: the layer17 desync run halves[76:81], stream index 0 (the
    # first productive stream in diagnose_walker's output)
    stream_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    half_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 77

    rows = load_manifest(MANIFEST)
    half_index = {}
    dump_cache = {}
    with open(DUMP, "rb") as fbin:
        for row in rows:
            offset = int(row["offset"])
            length = int(row["len"])
            fbin.seek(offset)
            block_bytes = fbin.read(length)
            if hashlib.sha256(block_bytes).hexdigest() != row["content_hash"]:
                continue
            k_half = block_bytes[:HALF_SIZE]
            v_half = block_bytes[HALF_SIZE:]
            half_index[hashlib.sha256(k_half).hexdigest()] = (row["layer_name"], row["block_id"], "K")
            half_index[hashlib.sha256(v_half).hexdigest()] = (row["layer_name"], row["block_id"], "V")
            dump_cache[(row["layer_name"], row["block_id"])] = block_bytes

    print("parsing pcap + reassembling...")
    segments = list(parse_pcap(PCAP))
    streams = reassemble_streams(segments)
    productive = [(k, b) for k, b in streams.items() if len(b) >= HALF_SIZE]
    key, buf = productive[stream_idx]
    print(f"stream: {key}, {len(buf)} bytes")

    halves = extract_all_halves(buf)
    print(f"{len(halves)} halves total\n")

    raw_start, raw_end, payload = halves[half_idx]
    h = hashlib.sha256(payload).hexdigest()
    hit = half_index.get(h)
    print(f"half[{half_idx}]: raw=[{raw_start}:{raw_end}] ({raw_end-raw_start} raw bytes, "
          f"{len(payload)} payload bytes)")
    print(f"match: {hit if hit else 'NONE'}")
    print(f"payload hash: {h}")
    print(f"first 32 bytes: {payload[:32].hex()}")
    print(f"last 32 bytes: {payload[-32:].hex()}")

    # show neighbors for context
    if half_idx > 0:
        pk = hashlib.sha256(halves[half_idx-1][2]).hexdigest()
        print(f"\nprev half match: {half_index.get(pk)}")
    if half_idx + 1 < len(halves):
        nk = hashlib.sha256(halves[half_idx+1][2]).hexdigest()
        print(f"next half match: {half_index.get(nk)}")

    # try to find this payload as a SUBSTRING of any known K/V half (partial garble?)
    print("\nchecking if this payload's content appears embedded in any known half...")
    found_partial = False
    for (layer, bid), block_bytes in dump_cache.items():
        if payload[:64] in block_bytes:
            idx = block_bytes.find(payload[:64])
            print(f"  first 64 bytes found within {layer} block_id={bid} at offset {idx}")
            found_partial = True
    if not found_partial:
        print("  not found as a substring of any known block either -- looks like genuinely "
              "garbled/misaligned extraction, not just a length mismatch")


if __name__ == "__main__":
    main()
