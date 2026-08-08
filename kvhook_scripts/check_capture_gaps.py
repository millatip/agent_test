#!/usr/bin/env python3
"""
reassemble_streams() computes a `present` bitmap (which byte positions
were actually captured, vs. defaulted to zero because no segment covered
them) but discards it. This checks whether the previously-found
corrupted extraction (half[77] in stream 0 of the clean capture, raw
range [2531221:2564073], trailing 32 bytes all zero) corresponds to a
genuine capture gap rather than legitimate zero-valued tensor data.
"""
import sys
from collections import defaultdict
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap

PCAP = "/tmp/kv_phaseA_clean_20260806_225414.pcap"
TARGET_STREAM_HINT = "49085"  # dst port from stream 0 in earlier output
CHECK_RANGE = (2531221, 2564073)  # raw offsets within that stream's buffer


def main():
    print("parsing pcap...")
    segments = list(parse_pcap(PCAP))

    by_stream = defaultdict(list)
    for ts, src_ip, src_port, dst_ip, dst_port, seq, payload in segments:
        key = (src_ip, src_port, dst_ip, dst_port)
        by_stream[key].append((seq, payload))

    # find the stream matching our hint (the big one to 49085)
    target_key = None
    for key in by_stream:
        if str(key[3]) == TARGET_STREAM_HINT and key[0] == "192.168.200.12":
            target_key = key
            break
    if target_key is None:
        print("stream not found, available streams:", list(by_stream.keys()))
        return
    print(f"target stream: {target_key}")

    pieces = by_stream[target_key]
    byte_map = {}
    for seq, payload in pieces:
        for i, b in enumerate(payload):
            pos = seq + i
            if pos not in byte_map:
                byte_map[pos] = b
    lo, hi = min(byte_map), max(byte_map)
    print(f"stream spans seq [{lo}:{hi}], {hi-lo+1} bytes total, {len(byte_map)} bytes actually captured")
    print(f"missing bytes overall: {(hi - lo + 1) - len(byte_map)}")

    # check coverage specifically in the problematic range (raw offsets
    # are relative to `lo`, so absolute seq = lo + raw_offset)
    start_abs = lo + CHECK_RANGE[0]
    end_abs = lo + CHECK_RANGE[1]
    missing_in_range = [pos for pos in range(start_abs, end_abs) if pos not in byte_map]
    print(f"\nchecking raw range {CHECK_RANGE} (abs seq [{start_abs}:{end_abs}]):")
    print(f"  missing positions in this range: {len(missing_in_range)}")
    if missing_in_range:
        print(f"  first missing (relative to range start): {missing_in_range[0] - start_abs}")
        print(f"  last missing (relative to range start): {missing_in_range[-1] - start_abs}")
        # show as contiguous runs
        runs = []
        run_start = missing_in_range[0]
        prev = missing_in_range[0]
        for p in missing_in_range[1:]:
            if p != prev + 1:
                runs.append((run_start - start_abs, prev - start_abs))
                run_start = p
            prev = p
        runs.append((run_start - start_abs, prev - start_abs))
        print(f"  missing runs (relative offsets): {runs[:10]}{'...' if len(runs) > 10 else ''}")


if __name__ == "__main__":
    main()
