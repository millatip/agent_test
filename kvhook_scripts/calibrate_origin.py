#!/usr/bin/env python3
"""
Searches for the correct phase/origin of the (16B header, 8192B segment)
grid empirically, using known-good anchors instead of assuming origin=0.

For each candidate phase p in [0, 8192), de-frames the stream as if
segment boundaries were at (p, p+8192, p+16384, ...) instead of
(0, 8192, 16384, ...), and checks whether a KNOWN block (found via its
raw, un-deframed anchor position) becomes a full exact match.

If a single phase works for multiple blocks spread across the stream,
it's a constant offset -- deframe_v2.py just needs that origin instead of
0. If the winning phase differs from block to block, the grid itself
shifts partway through the stream (e.g. from per-message overhead beyond
the regular 16-byte header), and a fixed-origin model can't work at all --
a sequential/adaptive de-framer would be needed instead.

Usage:
    python3 calibrate_origin.py <pcap> <dump.bin> <manifest.tsv> <layer_name> <block_id> [<layer_name2> <block_id2> ...]
"""
import sys
import hashlib
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest
from deframe_and_match import SUBUNIT_BYTES

SEG_SIZE = 8192
HEADER_SIZE = 16


def deframe_phased(buf: bytes, phase: int) -> bytes:
    """Same grid as deframe_v2 but with segment boundaries at
    (phase, phase+8192, ...) instead of (0, 8192, ...). Bytes before
    `phase` are left untouched (kept as-is, not part of any segment)."""
    out = bytearray(buf[:phase])
    n = len(buf)
    pos = phase
    while pos < n:
        seg_idx = (pos - phase) // SEG_SIZE
        seg_start = phase + seg_idx * SEG_SIZE
        offset_in_seg = pos - seg_start
        if offset_in_seg < HEADER_SIZE:
            pos = seg_start + HEADER_SIZE
            continue
        seg_end = seg_start + SEG_SIZE
        take_end = min(seg_end, n)
        out.extend(buf[pos:take_end])
        pos = take_end
    return bytes(out)


def find_winning_phases(buf: bytes, block_bytes: bytes, raw_anchor: int, label: str):
    """Try every phase, report which ones make block_bytes match exactly
    in the de-framed output. Searches phase in a window around raw_anchor
    (the block must start somewhere in [phase, phase+SEG_SIZE) relative
    terms don't matter -- we just brute force all 8192 phases; it's cheap
    per phase if we only de-frame a bounded region around the anchor)."""
    # Only need to de-frame a small region around the anchor for this test,
    # not the whole multi-MB buffer -- much faster.
    region_start = max(0, raw_anchor - SEG_SIZE)
    region_end = min(len(buf), raw_anchor + len(block_bytes) + 4 * SEG_SIZE)
    region = buf[region_start:region_end]
    local_anchor = raw_anchor - region_start

    winners = []
    for phase in range(SEG_SIZE):
        d = deframe_phased(region, phase)
        if d.find(block_bytes) != -1:
            winners.append(phase)
    print(f"[{label}] raw anchor={raw_anchor}, winning phases (mod {SEG_SIZE}) "
          f"in local region: {winners if winners else 'NONE'}")
    return winners


def main():
    pcap_path, dump_path, manifest_path = sys.argv[1:4]
    pairs = sys.argv[4:]
    targets = list(zip(pairs[0::2], pairs[1::2]))
    if not targets:
        targets = [("model.layers.0.self_attn.attn", "29"), ("model.layers.1.self_attn.attn", "29")]

    rows = load_manifest(manifest_path)
    print("parsing pcap + reassembling...")
    segments = list(parse_pcap(pcap_path))
    streams = reassemble_streams(segments)
    bulk_key, bulk_buf = max(streams.items(), key=lambda kv: len(kv[1]))
    print(f"bulk stream: {bulk_key}, {len(bulk_buf)} bytes")
    print()

    with open(dump_path, "rb") as f:
        for layer_name, block_id in targets:
            row = next(r for r in rows if r["layer_name"] == layer_name and r["block_id"] == block_id)
            f.seek(int(row["offset"]))
            block_bytes = f.read(int(row["len"]))
            assert hashlib.sha256(block_bytes).hexdigest() == row["content_hash"]
            unit0 = block_bytes[:SUBUNIT_BYTES]
            anchor = bulk_buf.find(unit0)
            if anchor == -1:
                print(f"[{layer_name} block_id={block_id}] no anchor found at all, skipping")
                continue
            find_winning_phases(bulk_buf, block_bytes, anchor, f"{layer_name} block_id={block_id}")


if __name__ == "__main__":
    main()
