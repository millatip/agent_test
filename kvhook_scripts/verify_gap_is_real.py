#!/usr/bin/env python3
"""
Before concluding the missing bytes are genuine capture loss: check
whether the missing absolute TCP sequence range is truly absent from
EVERY segment in the pcap (any stream, any 4-tuple), or whether it's
present somewhere but reassemble_streams() failed to place it correctly
(wrong stream key, retransmission/overlap handling, off-by-one, etc.).
If the range shows up anywhere, this is a reassembly bug, not capture
loss, and recapturing would not fix anything.
"""
import sys
from collections import defaultdict
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap

PCAP = "/tmp/kv_phaseA_clean_20260806_225414.pcap"
TARGET_KEY = ("192.168.200.12", 49717, "192.168.200.13", 49085)
MISSING_ABS_RANGE = (3106204111, 3106204175)  # lo + [32788, 32851] from earlier check


def main():
    print("parsing pcap (all segments, all streams)...")
    segments = list(parse_pcap(PCAP))
    print(f"{len(segments)} total TCP segments with nonzero payload, across all streams\n")

    start, end = MISSING_ABS_RANGE
    print(f"searching ALL segments (any 4-tuple) for coverage of abs seq range [{start}:{end})...")

    hits = []
    for ts, src_ip, src_port, dst_ip, dst_port, seq, payload in segments:
        seg_end = seq + len(payload)
        # does this segment's [seq, seg_end) overlap [start, end)?
        if seg_end > start and seq < end:
            hits.append((ts, src_ip, src_port, dst_ip, dst_port, seq, seg_end, len(payload)))

    if not hits:
        print("NO segment anywhere in the pcap covers this range, at all.")
        print("This supports genuine capture loss (the packet was never captured),")
        print("not a reassembly bug.")
    else:
        print(f"FOUND {len(hits)} segment(s) covering this range:")
        for ts, src_ip, src_port, dst_ip, dst_port, seq, seg_end, plen in hits:
            key = (src_ip, src_port, dst_ip, dst_port)
            same_stream = "SAME stream" if key == TARGET_KEY else f"DIFFERENT stream {key}"
            print(f"  ts={ts:.6f} seq=[{seq}:{seg_end}) len={plen} -- {same_stream}")
        print("\nThis means the data WAS captured -- if it's not showing up in "
              "reassemble_streams()'s output, that's a reassembly bug (wrong "
              "stream classification, retransmission/overlap handling, etc.), "
              "not capture loss. Recapturing would not fix this.")

    # also check: total segment count assigned to each 4-tuple, to catch a
    # stream key mismatch (segments existing under a slightly different key)
    print("\n--- sanity: all distinct 4-tuples seen in the pcap ---")
    by_key = defaultdict(int)
    for ts, src_ip, src_port, dst_ip, dst_port, seq, payload in segments:
        by_key[(src_ip, src_port, dst_ip, dst_port)] += 1
    for k, n in sorted(by_key.items(), key=lambda kv: -kv[1]):
        marker = " <-- target" if k == TARGET_KEY else ""
        print(f"  {k}: {n} segments{marker}")


if __name__ == "__main__":
    main()
