#!/usr/bin/env python3
"""
Quantifies how much of the ~50% unreconstructed-block gap is explained
by genuine TCP capture loss, using the presence bitmap
reassemble_streams_with_presence() now exposes.

For every half extract_all_halves() produces (matched or not), checks
whether its raw byte range contains any uncaptured position. Cross-tabs
match status against gap presence: if "unmatched" and "has a gap"
correlate strongly, capture loss explains most of the failure; if many
unmatched halves have NO gap in their range, something else is also
going on and shouldn't be attributed to capture loss.

Usage:
    python3 quantify_gap_impact.py <pcap> <kvhook_dump.bin> <kvhook_manifest.tsv>
"""
import sys
import hashlib
from collections import defaultdict
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams_with_presence, load_manifest
from reconstruct_v2 import extract_all_halves, HALF_SIZE


def main():
    pcap_path, dump_path, manifest_path = sys.argv[1:4]

    print(f"=== parsing pcap: {pcap_path} ===")
    segments = list(parse_pcap(pcap_path))
    print(f"{len(segments)} TCP segments with nonzero payload")

    print("=== reassembling with presence tracking ===")
    streams, presence = reassemble_streams_with_presence(segments)
    for key, buf in streams.items():
        pres = presence[key]
        missing = pres.count(0)
        print(f"  {key}: {len(buf)} bytes, {missing} uncaptured ({100.0*missing/len(buf):.4f}%)")

    print(f"=== loading manifest + building K/V half hash index: {manifest_path} ===")
    rows = load_manifest(manifest_path)
    half_index = {}
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

    print("=== walking streams, cross-tabbing match status against gap presence ===")
    counts = defaultdict(int)  # (matched: bool, has_gap: bool) -> count
    total_extracted = 0
    for key, buf in streams.items():
        if len(buf) < HALF_SIZE:
            continue
        pres = presence[key]
        halves = extract_all_halves(buf)
        for raw_start, raw_end, payload in halves:
            total_extracted += 1
            h = hashlib.sha256(payload).hexdigest()
            matched = half_index.get(h) is not None
            has_gap = 0 in pres[raw_start:raw_end]
            counts[(matched, has_gap)] += 1

    print()
    print(f"total halves extracted: {total_extracted}")
    print(f"  matched,   no gap in range: {counts[(True, False)]}")
    print(f"  matched,   HAS gap in range: {counts[(True, True)]}  "
          f"(matched despite a gap -- gap must be outside the hashed payload, "
          f"e.g. in a header region that got skipped)")
    print(f"  unmatched, no gap in range: {counts[(False, False)]}  "
          f"(NOT explained by capture loss -- some other cause)")
    print(f"  unmatched, HAS gap in range: {counts[(False, True)]}  "
          f"(explained by capture loss)")

    unmatched_total = counts[(False, False)] + counts[(False, True)]
    if unmatched_total:
        pct_gap_explained = 100.0 * counts[(False, True)] / unmatched_total
        print()
        print(f"Of {unmatched_total} unmatched halves, {counts[(False, True)]} "
              f"({pct_gap_explained:.1f}%) have a capture gap in their byte range.")
        print(f"The remaining {counts[(False, False)]} ({100-pct_gap_explained:.1f}%) "
              f"do NOT -- capture loss alone doesn't explain them.")


if __name__ == "__main__":
    main()
