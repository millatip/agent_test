#!/usr/bin/env python3
"""
Diagnoses where extract_all_halves loses sync in the clean 79MB capture,
by logging every extracted half's raw byte range and match status per
stream, and inspecting the first unmatched run in detail.

Usage:
    python3 diagnose_walker.py
"""
import sys
import hashlib
from collections import defaultdict
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest
from reconstruct_v2 import extract_all_halves, HALF_SIZE
from deframe_v3 import HEADER_LEN, _find_header

PCAP = "/tmp/kv_phaseA_clean_20260806_225414.pcap"
DUMP = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_dump_phaseA_clean.bin"
MANIFEST = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_manifest_phaseA_clean.tsv"


def main():
    rows = load_manifest(MANIFEST)
    half_index = {}
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

    print("parsing pcap + reassembling...")
    segments = list(parse_pcap(PCAP))
    streams = reassemble_streams(segments)

    for key, buf in streams.items():
        if len(buf) < HALF_SIZE:
            continue
        halves = extract_all_halves(buf)
        print(f"\n=== stream {key}: {len(halves)} halves, {len(buf)} bytes ===")

        # find the first run of consecutive unmatched halves, and the
        # matched half immediately before it (context for what "before
        # desync" looks like)
        prev_matched = None
        run_start_idx = None
        for i, (raw_start, raw_end, payload) in enumerate(halves):
            h = hashlib.sha256(payload).hexdigest()
            hit = half_index.get(h)
            expected_len = raw_end - raw_start  # includes internal headers
            if hit is None:
                if run_start_idx is None:
                    run_start_idx = i
            else:
                if run_start_idx is not None and i - run_start_idx >= 5:
                    # a real desync run just ended -- report it
                    print(f"  desync run: halves[{run_start_idx}:{i}] "
                          f"({i - run_start_idx} unmatched), "
                          f"raw byte range [{halves[run_start_idx][0]}:{halves[i-1][1]}]")
                    if prev_matched is not None:
                        pl, pbid, pw = prev_matched
                        print(f"    last matched before desync: {pl} block_id={pbid} half={pw} "
                              f"at raw [{halves[run_start_idx-1][0]}:{halves[run_start_idx-1][1]}]")
                    l, bid, w = hit
                    print(f"    first matched after desync: {l} block_id={bid} half={w}")
                run_start_idx = None
                prev_matched = hit

        # summary of gaps between consecutive halves (should be 0 if the
        # walker is perfectly tracking header-to-header; large gaps or
        # negative "gaps" indicate something skipped or overlapped)
        gaps = []
        for i in range(1, len(halves)):
            gap = halves[i][0] - halves[i - 1][1]
            gaps.append(gap)
        if gaps:
            from collections import Counter
            c = Counter(gaps)
            print(f"  gap-between-halves distribution (top 5): {c.most_common(5)}")


if __name__ == "__main__":
    main()
