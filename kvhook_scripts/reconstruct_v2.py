#!/usr/bin/env python3
"""
Corrected reconstruction, built on the batching discovery: K-halves and
V-halves for a block are NOT transmitted contiguously once there are
enough blocks in a transfer -- NIXL batches many blocks' K-halves
together, then sends V-halves as a separate, much-later batch (confirmed:
block80's K-half starts exactly 21 bytes after block79's K-half ends;
block79's own V-half is found ~1,051,957 bytes later, matching roughly
32 blocks' worth of K-half traffic).

So instead of walking forward from a block's own anchor expecting its
full 65536 bytes to be contiguous (even with headers stripped), this
walks the ENTIRE stream once, treating every header as the start of a
32768-byte "half" (K or V, doesn't matter which yet), extracts each half
independently (de-framing across any internal continuation headers using
the already-validated logic), hashes it, and matches against every
block's K-half hash and V-half hash from the ground-truth dump. A block
counts as fully reconstructed only if BOTH its halves are found
(anywhere in the stream, regardless of position).

Usage:
    python3 reconstruct_v2.py <pcap> <kvhook_dump.bin> <kvhook_manifest.tsv>
"""
import sys
import hashlib
from collections import defaultdict
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest
from deframe_v3 import deframe_from_anchor, HEADER_LEN, _find_header

HALF_SIZE = 32768


def extract_all_halves(buf: bytes, max_halves=None):
    """Walks the whole buffer, treating every found header as the start
    of a HALF_SIZE-byte payload span. Returns list of (raw_header_pos,
    raw_end_pos, payload_bytes)."""
    halves = []
    pos = 0
    n = len(buf)
    while pos < n:
        hdr_idx = _find_header(buf, pos, n)
        if hdr_idx == -1:
            break
        payload_start = hdr_idx + HEADER_LEN
        payload, consumed = deframe_from_anchor(buf, payload_start, HALF_SIZE)
        if payload is None:
            # couldn't extract a clean half from here -- advance past this
            # header and keep scanning rather than getting stuck
            pos = hdr_idx + HEADER_LEN
            continue
        raw_end = payload_start + consumed
        halves.append((hdr_idx, raw_end, payload))
        pos = raw_end
        if max_halves and len(halves) >= max_halves:
            break
    return halves


def main():
    pcap_path, dump_path, manifest_path = sys.argv[1:4]

    print(f"=== parsing pcap: {pcap_path} ===")
    segments = list(parse_pcap(pcap_path))
    print(f"{len(segments)} TCP segments with nonzero payload")

    print("=== reassembling TCP streams by sequence number ===")
    streams = reassemble_streams(segments)
    for key, buf in streams.items():
        print(f"  {key}: {len(buf)} bytes")

    print(f"=== loading manifest + building K/V half hash index: {manifest_path} ===")
    rows = load_manifest(manifest_path)
    # hash -> (layer, block_id, 'K' or 'V')
    half_index = {}
    per_block_layer = defaultdict(int)
    with open(dump_path, "rb") as fbin:
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
            per_block_layer[row["layer_name"]] += 1
    print(f"{len(half_index)} K/V half-hashes indexed from {sum(per_block_layer.values())} blocks")

    print("=== walking bulk stream(s), extracting all halves, matching by hash ===")
    found = defaultdict(set)  # (layer, block_id) -> {'K', 'V'}
    total_halves = 0
    matched_halves = 0
    for key, buf in streams.items():
        if len(buf) < HALF_SIZE:
            continue
        halves = extract_all_halves(buf)
        total_halves += len(halves)
        for raw_start, raw_end, payload in halves:
            h = hashlib.sha256(payload).hexdigest()
            hit = half_index.get(h)
            if hit:
                layer, block_id, which = hit
                found[(layer, block_id)].add(which)
                matched_halves += 1
        print(f"  {key}: {len(halves)} halves extracted")

    print()
    print(f"=== overall: {matched_halves}/{total_halves} extracted halves matched a known K/V half ===")

    total_blocks = sum(per_block_layer.values())
    full_blocks = sum(1 for v in found.values() if v == {"K", "V"})
    partial_blocks = sum(1 for v in found.values() if len(v) == 1)
    print(f"=== blocks fully reconstructed (both K and V found): {full_blocks}/{total_blocks} "
          f"({100.0*full_blocks/total_blocks:.1f}%) ===")
    print(f"=== blocks with only one half found: {partial_blocks} ===")
    print(f"=== blocks with neither half found: {total_blocks - full_blocks - partial_blocks} ===")

    print()
    print("=== per-layer full-block reconstruction rate ===")
    per_layer_total = defaultdict(int)
    per_layer_full = defaultdict(int)
    for row in rows:
        per_layer_total[row["layer_name"]] += 1
    for (layer, block_id), halves in found.items():
        if halves == {"K", "V"}:
            per_layer_full[layer] += 1
    for layer in sorted(per_layer_total, key=lambda l: int(l.split(".")[2])):
        tot = per_layer_total[layer]
        full = per_layer_full[layer]
        print(f"  {layer}: {full}/{tot} ({100.0*full/tot:.1f}%)")


if __name__ == "__main__":
    main()
