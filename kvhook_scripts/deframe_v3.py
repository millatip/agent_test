#!/usr/bin/env python3
"""
Self-synchronizing de-framer: searches for the continuation-header magic
byte pattern directly instead of predicting where headers are by position
math. Confirmed via dump_header_bytes.py against 8 independent Phase A
boundaries: continuation headers are exactly

    13 00 20 00 00 db 00 00 00 00 00 00 00 <2B LE cumulative offset> 00 00 00 00 00 00

21 bytes total, with the 13-byte prefix constant and the 2-byte field at
offset 13 an exact cumulative-payload-offset counter (verified: read
8176, 16352, 24528, ... = n x 8176 across all 8 tested boundaries).

Given a block's start position (found by content search on its first
4096-byte sub-unit, already reliable), this walks forward: copy payload
until the next occurrence of the 13-byte magic, skip 21 bytes, repeat,
until len(block_bytes) worth of payload is collected. No period assumed,
no phase to calibrate, no drift possible -- headers are found where they
actually are.

Usage:
    python3 deframe_v3.py <pcap> <kvhook_dump.bin> <kvhook_manifest.tsv>
"""
import sys
import re
import hashlib
from collections import defaultdict
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest
from deframe_and_match import SUBUNIT_BYTES

HEADER_LEN = 21
MAX_LOOKAHEAD = 16000  # widened from 8300 -- found a real corruption case where
# a gap between headers exceeded the old margin, and the fallback logic
# below used to blindly treat "no header found in a too-short window" the
# same as "confirmed no header needed," silently producing corrupted
# (zero-padded) output that still returned as if it had succeeded.

# Byte positions confirmed to vary, empirically, across three separate
# captures:
#   - bytes 1-2: differ on the last fragment before a block ends (13 90 00
#     vs 13 00 20 elsewhere) -- likely a "last fragment" flag.
#   - bytes 5-6: a 2-byte (little-endian) tag/channel field, not a single
#     byte -- 0x00db (small capture, layer 0), 0x00dd (layer 1), 0x2273
#     (clean 79MB capture). The small capture's tag values happened to
#     have a zero high byte, which is why one byte looked constant there;
#     the clean capture's tag has a nonzero high byte, breaking that
#     assumption. Both bytes are wildcarded now.
# What's constant regardless: byte 0 (0x13), bytes 3-4 (0x00 0x00), and
# bytes 7-12 (six 0x00 bytes, not seven). 9 fixed bytes across the match
# is still specific enough that a false hit inside real tensor content is
# not a practical concern.
HEADER_RE = re.compile(rb"\x13..\x00\x00..\x00{6}", re.DOTALL)


def _find_header(buf: bytes, start: int, end: int):
    m = HEADER_RE.search(buf, start, end)
    return m.start() if m else -1


def deframe_from_anchor(buf: bytes, anchor: int, target_len: int):
    """Starting at `anchor` (a block's confirmed first content byte),
    collect target_len bytes of real payload, skipping any continuation
    header (found by magic-byte search, not position prediction) along
    the way. Returns (payload_bytes, consumed_raw_bytes) or (None, None)
    on any failure -- including when the search window can't confirm
    whether a header was missed, rather than guessing and returning
    corrupted-but-successful-looking output (a real bug found via
    debug_unmatched_half.py: a half whose first 64 bytes matched ground
    truth exactly but whose last 32 bytes were all zero, because this
    function used to treat 'no header found in a too-short window' the
    same as 'confirmed no header needed')."""
    out = bytearray()
    pos = anchor
    n = len(buf)
    while len(out) < target_len and pos < n:
        remaining_needed = target_len - len(out)
        search_end = min(n, pos + MAX_LOOKAHEAD)
        hdr_idx = _find_header(buf, pos, search_end)

        if hdr_idx != -1 and hdr_idx - pos < remaining_needed:
            # found a real header before we'd finish -- copy up to it, skip it
            out.extend(buf[pos:hdr_idx])
            pos = hdr_idx + HEADER_LEN
            continue

        # no header found before remaining_needed within the search window.
        # Only safe to conclude "no header needed" if the window covered
        # the ENTIRE remaining span -- otherwise a header could exist just
        # past where we stopped looking, and blindly copying up to
        # remaining_needed would silently include header bytes as if they
        # were payload.
        if remaining_needed <= MAX_LOOKAHEAD:
            take = min(remaining_needed, n - pos)
            out.extend(buf[pos:pos + take])
            pos += take
            break
        else:
            return None, None  # genuine failure -- don't guess
    if len(out) != target_len:
        return None, None
    return bytes(out), pos - anchor


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

    print("=== self-synchronizing de-frame + whole-block matching ===")
    per_layer_total = defaultdict(int)
    per_layer_matched = defaultdict(int)
    matches = []
    no_anchor = 0
    anchor_but_no_match = 0

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
                candidate, _ = deframe_from_anchor(buf, anchor, length)
                if candidate == block_bytes:
                    per_layer_matched[layer] += 1
                    matches.append((layer, row["block_id"], expected_hash, key, anchor))
                    found = True
                    break
                elif candidate is not None:
                    anchor_but_no_match += 1
            if not found:
                no_anchor += 1  # counts both "no anchor" and "anchor but wrong content"

    print()
    print("=== per-layer WHOLE-BLOCK match rate (self-synchronizing) ===")
    for layer in sorted(per_layer_total, key=lambda l: int(l.split(".")[2])):
        tot = per_layer_total[layer]
        m = per_layer_matched[layer]
        print(f"  {layer}: {m}/{tot} ({100.0*m/tot:.1f}%)")

    total = len(rows)
    total_matched = len(matches)
    print()
    print(f"=== overall whole-block, self-synchronizing: {total_matched}/{total} "
          f"({100.0*total_matched/total:.1f}%) ===")
    print(f"anchor found but de-framed content still didn't match: {anchor_but_no_match}")

    if matches:
        print()
        print("first few matches (layer, block_id, stream, anchor):")
        for layer, block_id, h, key, anchor in matches[:10]:
            print(f"  {layer} block_id={block_id} hash={h[:12]}... "
                  f"stream={key[0]}:{key[1]}->{key[2]}:{key[3]} anchor={anchor}")


if __name__ == "__main__":
    main()
