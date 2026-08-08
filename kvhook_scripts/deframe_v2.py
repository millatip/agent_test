#!/usr/bin/env python3
"""
Corrected de-framer: UCX_TCP_TX_SEG_SIZE=8K means TCP segments on the
transmit side (spark-523e, prefill -> decode direction) are 8192 bytes
total: a 16-byte header + 8176 bytes of payload, at FIXED, ABSOLUTE stream
positions (k*8192 for k=0,1,2,...) -- not a counter that accumulates from
each block's own start. That's why the earlier version worked at 2.4MB
(small enough that drift from using the wrong header size, 21 instead of
16, and payload-relative rather than stream-relative counting, stayed
under a block's own span) and failed completely by ~79-95MB (drift
compounds over the whole stream).

This does ONE global de-framing pass using the correct fixed grid, then
plain whole-block search on the result -- no per-block anchoring/resetting
needed, because a correct fixed-origin model doesn't drift.

Residual message-start overhead (the 21 bytes measured at the very first
block, vs the regular 16-byte periodic header -- a 5-byte difference) is
NOT modeled explicitly here. It should show up as a few stray bytes
immediately before each block's true content in the de-framed stream,
which a content search (.find(), not an assumed fixed offset) naturally
skips over. If whole-block matches still fail after this fix, that
residual is the next thing to characterize.

Usage:
    python3 deframe_v2.py <pcap> <kvhook_dump.bin> <kvhook_manifest.tsv>
"""
import sys
import hashlib
from collections import defaultdict
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest

SEG_SIZE = 8192
HEADER_SIZE = 16


def deframe_global(buf: bytes) -> bytes:
    """Strip HEADER_SIZE bytes at the start of every SEG_SIZE-byte segment,
    at fixed absolute positions from buf[0] (assumed to be this stream's
    true origin -- the first captured byte of the connection's data)."""
    out = bytearray()
    n = len(buf)
    pos = 0
    while pos < n:
        seg_idx = pos // SEG_SIZE
        seg_start = seg_idx * SEG_SIZE
        offset_in_seg = pos - seg_start
        if offset_in_seg < HEADER_SIZE:
            pos = seg_start + HEADER_SIZE
            continue
        seg_end = seg_start + SEG_SIZE
        take_end = min(seg_end, n)
        out.extend(buf[pos:take_end])
        pos = take_end
    return bytes(out)


def main():
    pcap_path, dump_path, manifest_path = sys.argv[1:4]

    print(f"=== parsing pcap: {pcap_path} ===")
    segments = list(parse_pcap(pcap_path))
    print(f"{len(segments)} TCP segments with nonzero payload")

    print("=== reassembling TCP streams by sequence number ===")
    streams = reassemble_streams(segments)
    for key, buf in streams.items():
        direction = "TX (523e sending, 8K segments expected)" if key[0] == "192.168.200.12" else "RX (523e receiving, 64K segments expected -- rule below may not apply)"
        print(f"  {key}: {len(buf)} bytes -- {direction}")

    print("=== de-framing each stream on the fixed (16B header / 8192B segment) grid ===")
    deframed = {}
    for key, buf in streams.items():
        d = deframe_global(buf)
        deframed[key] = d
        print(f"  {key}: {len(buf)} -> {len(d)} bytes ({len(buf) - len(d)} header bytes removed, "
              f"{(len(buf) - len(d)) / max(1, len(buf) // SEG_SIZE):.1f} avg bytes/segment)")

    print(f"=== loading manifest: {manifest_path} ===")
    rows = load_manifest(manifest_path)
    print(f"{len(rows)} dumped blocks")

    print("=== matching WHOLE 64KB blocks against globally de-framed streams ===")
    per_layer_total = defaultdict(int)
    per_layer_matched = defaultdict(int)
    matches = []

    with open(dump_path, "rb") as fbin:
        for row in rows:
            layer = row["layer_name"]
            offset = int(row["offset"])
            length = int(row["len"])
            expected_hash = row["content_hash"]
            per_layer_total[layer] += 1

            fbin.seek(offset)
            block_bytes = fbin.read(length)
            actual_hash = hashlib.sha256(block_bytes).hexdigest()
            if actual_hash != expected_hash:
                print(f"  WARNING: manifest/dump mismatch for {layer} block_id={row['block_id']}")
                continue

            found_at = None
            for key, buf in deframed.items():
                idx = buf.find(block_bytes)
                if idx != -1:
                    found_at = (key, idx)
                    break

            if found_at is not None:
                per_layer_matched[layer] += 1
                matches.append((layer, row["block_id"], expected_hash, found_at))

    print()
    print("=== per-layer WHOLE-BLOCK match rate (globally de-framed) ===")
    for layer in sorted(per_layer_total, key=lambda l: int(l.split(".")[2])):
        tot = per_layer_total[layer]
        m = per_layer_matched[layer]
        print(f"  {layer}: {m}/{tot} ({100.0*m/tot:.1f}%)")

    total = len(rows)
    total_matched = len(matches)
    print()
    print(f"=== overall whole-block, globally de-framed: {total_matched}/{total} "
          f"({100.0*total_matched/total:.1f}%) ===")
    print(f"({total - total_matched} blocks dumped locally but never appeared on the wire -- "
          f"expected for blocks served entirely from prefix-cache)")

    if matches:
        print()
        print("first few matches (layer, block_id, stream, offset_in_deframed_stream):")
        for layer, block_id, h, (key, idx) in matches[:10]:
            print(f"  {layer} block_id={block_id} hash={h[:12]}... "
                  f"stream={key[0]}:{key[1]}->{key[2]}:{key[3]} offset={idx}")


if __name__ == "__main__":
    main()
