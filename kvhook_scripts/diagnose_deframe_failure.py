#!/usr/bin/env python3
"""
For a block where the anchor (unit0) was found but de-framed content didn't
match the dump, finds exactly where the de-framed candidate diverges from
the true block bytes -- to distinguish "framing parameters drift slightly
after some point" from "something structurally different happens here."

Usage:
    python3 diagnose_deframe_failure.py <pcap> <dump.bin> <manifest.tsv> <layer_name> <block_id>
"""
import sys
import hashlib
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest
from deframe_and_match import deframe_from, HEADER_SIZE, PAYLOAD_PERIOD, SUBUNIT_BYTES


def main():
    pcap_path, dump_path, manifest_path, layer_name, block_id = sys.argv[1:6]

    rows = load_manifest(manifest_path)
    row = next(r for r in rows if r["layer_name"] == layer_name and r["block_id"] == block_id)
    with open(dump_path, "rb") as f:
        f.seek(int(row["offset"]))
        block_bytes = f.read(int(row["len"]))
    assert hashlib.sha256(block_bytes).hexdigest() == row["content_hash"]
    unit0 = block_bytes[:SUBUNIT_BYTES]

    print("parsing pcap + reassembling...")
    segments = list(parse_pcap(pcap_path))
    streams = reassemble_streams(segments)
    bulk_key, bulk_buf = max(streams.items(), key=lambda kv: len(kv[1]))

    anchor = bulk_buf.find(unit0)
    print(f"anchor (unit0 match) at {anchor}")
    raw_start = anchor - HEADER_SIZE
    candidate = deframe_from(bulk_buf, raw_start, len(block_bytes))
    print(f"de-framed candidate length: {len(candidate)} (expected {len(block_bytes)})")

    # find first byte of divergence
    n = min(len(candidate), len(block_bytes))
    div = n
    for i in range(n):
        if candidate[i] != block_bytes[i]:
            div = i
            break
    print(f"first divergence at payload byte {div}/{len(block_bytes)} "
          f"({100.0*div/len(block_bytes):.1f}% of the way through)")

    if div == n:
        print("no divergence found in the overlapping region -- likely just a length mismatch")
        return

    # How many periods have we consumed by that point? Under the (21, 8176)
    # model, divergence should always fall near a period boundary (position
    # near a multiple of 8176) if it's a clean parameter-drift issue.
    period_index = div // PAYLOAD_PERIOD
    offset_within_period = div % PAYLOAD_PERIOD
    print(f"under the (header={HEADER_SIZE}, period={PAYLOAD_PERIOD}) model, this is "
          f"period #{period_index}, {offset_within_period} bytes into that period")
    print(f"(if this offset is near 0 or near {PAYLOAD_PERIOD}, the header/period "
          f"size itself is probably slightly off starting around here; if it's "
          f"somewhere in the middle, something else is going on)")

    # Try to find where the REST of the block (from the divergence point
    # onward) actually lives in the raw stream, searching nearby.
    remaining = block_bytes[div:div + 256]  # a chunk starting right at the divergence
    search_region_start = raw_start + HEADER_SIZE * (period_index + 1) + div  # rough guess
    for window_start in range(max(0, search_region_start - 500), search_region_start + 2000):
        if bulk_buf[window_start:window_start + len(remaining)] == remaining:
            print(f"found the continuation at raw offset {window_start} "
                  f"(vs. where the (21,8176) model predicted it should resume)")
            break
    else:
        print(f"could not find the continuation bytes within +/-2000 of the naive "
              f"prediction -- searching the whole buffer instead (slower)...")
        idx = bulk_buf.find(remaining)
        if idx == -1:
            print("not found anywhere in this stream at all")
        else:
            print(f"found at raw offset {idx}")


if __name__ == "__main__":
    main()
