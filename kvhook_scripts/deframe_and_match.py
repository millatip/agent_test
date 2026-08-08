#!/usr/bin/env python3
"""
Corrected Stage 3 matcher: strips NIXL's periodic framing before matching,
instead of requiring naive whole-block contiguity.

test_framing_hypothesis.py established the local rule empirically, verified
against 8 independent boundary measurements on the Phase A capture (not
assumed): a 21-byte header is inserted every 8176 bytes of real payload.
A first attempt applied this as ONE continuous counter for the whole
reassembled stream and only recovered the very first block (layer 0) --
meaning the period does not stay in sync globally across a whole
connection's worth of separate block transfers (most likely: each
block/message gets its own fresh framing, not one running counter for the
life of the TCP stream).

This version anchors de-framing PER BLOCK instead: for each dumped block,
locate its first 4096-byte sub-unit via plain substring search (already
proven reliable across every layer, both phases, in the earlier
unit-granularity pass), treat the 21 bytes immediately before that as this
block's own leading header, and de-frame forward from there with a fresh
period counter. If the framing genuinely resets per block-transfer, this
should recover full 64KB blocks, not just isolated sub-units.

Usage:
    python3 deframe_and_match.py <pcap> <kvhook_dump.bin> <kvhook_manifest.tsv>
"""
import sys
import hashlib
from collections import defaultdict
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest

HEADER_SIZE = 21
PAYLOAD_PERIOD = 8176
SUBUNIT_BYTES = 4096


def deframe_from(buf: bytes, raw_start: int, max_payload_bytes: int) -> bytes:
    """De-frame starting fresh at raw_start, which must be the position of
    a header's first byte (i.e. payload begins at raw_start + HEADER_SIZE).
    Produces up to max_payload_bytes of payload, resetting the period
    counter at raw_start rather than at buffer position 0."""
    out = bytearray()
    pos = raw_start
    n = len(buf)
    while len(out) < max_payload_bytes and pos < n:
        pos += HEADER_SIZE
        remaining_needed = max_payload_bytes - len(out)
        take = min(PAYLOAD_PERIOD, remaining_needed, n - pos)
        if take <= 0:
            break
        out.extend(buf[pos:pos + take])
        pos += take
    return bytes(out)


def main():
    pcap_path, dump_path, manifest_path = sys.argv[1:4]

    print(f"=== parsing pcap: {pcap_path} ===")
    segments = list(parse_pcap(pcap_path))
    print(f"{len(segments)} TCP segments with nonzero payload")

    print("=== reassembling TCP streams by sequence number ===")
    streams = reassemble_streams(segments)
    for key, buf in streams.items():
        print(f"  {key}: {len(buf)} bytes")

    print(f"=== loading manifest: {manifest_path} ===")
    rows = load_manifest(manifest_path)
    print(f"{len(rows)} dumped blocks")

    print("=== per-block anchored de-framing + whole-block matching ===")
    per_layer_total = defaultdict(int)
    per_layer_matched = defaultdict(int)
    matches = []
    no_anchor = []
    anchor_found_but_deframe_failed = []

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

            unit0 = block_bytes[:SUBUNIT_BYTES]

            found = False
            for key, buf in streams.items():
                anchor = buf.find(unit0)
                if anchor == -1:
                    continue
                raw_start = anchor - HEADER_SIZE
                if raw_start < 0:
                    continue
                candidate = deframe_from(buf, raw_start, length)
                if candidate == block_bytes:
                    per_layer_matched[layer] += 1
                    matches.append((layer, row["block_id"], expected_hash, key, anchor))
                    found = True
                    break
                else:
                    anchor_found_but_deframe_failed.append(
                        (layer, row["block_id"], key, anchor, len(candidate))
                    )

            if not found and not any(
                m[0] == layer and m[1] == row["block_id"] for m in anchor_found_but_deframe_failed
            ):
                no_anchor.append((layer, row["block_id"]))

    print()
    print("=== per-layer WHOLE-BLOCK match rate (per-block anchored de-framing) ===")
    for layer in sorted(per_layer_total, key=lambda l: int(l.split(".")[2])):
        tot = per_layer_total[layer]
        m = per_layer_matched[layer]
        print(f"  {layer}: {m}/{tot} ({100.0*m/tot:.1f}%)")

    total = len(rows)
    total_matched = len(matches)
    print()
    print(f"=== overall whole-block, anchored de-framing: {total_matched}/{total} "
          f"({100.0*total_matched/total:.1f}%) ===")
    print(f"no anchor found at all (unit0 never seen on the wire): {len(no_anchor)}")
    print(f"anchor found but de-framed content didn't match block: {len(anchor_found_but_deframe_failed)}")

    if anchor_found_but_deframe_failed:
        print()
        print("first few anchor-found-but-mismatched cases (worth inspecting if this list is long):")
        for layer, block_id, key, anchor, cand_len in anchor_found_but_deframe_failed[:5]:
            print(f"  {layer} block_id={block_id} anchor={anchor} candidate_len={cand_len}")

    if matches:
        print()
        print("first few matches (layer, block_id, stream, anchor_offset):")
        for layer, block_id, h, key, anchor in matches[:10]:
            print(f"  {layer} block_id={block_id} hash={h[:12]}... "
                  f"stream={key[0]}:{key[1]}->{key[2]}:{key[3]} anchor={anchor}")


if __name__ == "__main__":
    main()
