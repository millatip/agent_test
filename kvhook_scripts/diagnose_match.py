#!/usr/bin/env python3
"""
Diagnostic follow-up to reconstruct.py's 0/1044 result. Checks whether
dumped block content is present in the reassembled bulk stream but
fragmented (e.g. small protocol headers interleaved every N bytes) rather
than genuinely absent, by sliding smaller windows across one block and
searching for each window independently.

Usage:
    python3 diagnose_match.py <pcap> <kvhook_dump.bin> <kvhook_manifest.tsv> [layer_name] [block_id]
If layer_name/block_id omitted, uses the last dumped block of the first
layer in the manifest (the tail block of a request is the most likely to
have actually crossed the wire rather than being prefix-cache-served).
"""
import sys
import hashlib
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest


def main():
    pcap_path, dump_path, manifest_path = sys.argv[1:4]
    rows = load_manifest(manifest_path)

    if len(sys.argv) >= 6:
        layer_name, block_id = sys.argv[4], sys.argv[5]
        row = next(r for r in rows if r["layer_name"] == layer_name and r["block_id"] == block_id)
    else:
        # last block_id of the first layer present
        first_layer = rows[0]["layer_name"]
        same_layer = [r for r in rows if r["layer_name"] == first_layer]
        row = max(same_layer, key=lambda r: int(r["block_id"]))

    print(f"probing layer={row['layer_name']} block_id={row['block_id']} "
          f"offset={row['offset']} len={row['len']} hash={row['content_hash'][:16]}...")

    with open(dump_path, "rb") as f:
        f.seek(int(row["offset"]))
        block_bytes = f.read(int(row["len"]))
    assert hashlib.sha256(block_bytes).hexdigest() == row["content_hash"], "dump/manifest mismatch"

    print("parsing pcap + reassembling streams...")
    segments = list(parse_pcap(pcap_path))
    streams = reassemble_streams(segments)

    # pick the largest stream (the bulk data direction)
    bulk_key, bulk_buf = max(streams.items(), key=lambda kv: len(kv[1]))
    print(f"bulk stream: {bulk_key}, {len(bulk_buf)} bytes")

    print()
    print("=== whole-block search (should be 0, confirms reconstruct.py result) ===")
    print("found:", bulk_buf.find(block_bytes) != -1)

    print()
    print("=== sliding-window search at decreasing granularity ===")
    for win in (32768, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16):
        n_windows = len(block_bytes) // win
        found = 0
        first_found_at = None
        for i in range(n_windows):
            w = block_bytes[i * win:(i + 1) * win]
            idx = bulk_buf.find(w)
            if idx != -1:
                found += 1
                if first_found_at is None:
                    first_found_at = (i * win, idx)
        print(f"  window={win:6d}B: {found}/{n_windows} windows found in bulk stream"
              f"{'  first hit: block_offset=' + str(first_found_at[0]) + ' stream_offset=' + str(first_found_at[1]) if first_found_at else ''}")
        if found > 0:
            break  # no need to go finer once we've located *something*

    print()
    print("=== reverse check: does the bulk stream contain long runs NOT in the block? ===")
    print("(sanity: are we even looking at plausible tensor-like data, or something else "
          "entirely -- e.g. metadata/control -- masquerading as bulk?)")
    print("first 64 bytes of bulk stream (hex):", bulk_buf[:64].hex())
    print("first 64 bytes of dumped block (hex):", block_bytes[:64].hex())

    print()
    print("=== byte-value histogram sanity check ===")
    print("(bf16 tensor content should look high-entropy/random-ish, not text/structured)")
    import collections
    bc = collections.Counter(bulk_buf[:65536])
    most_common = bc.most_common(5)
    print("bulk stream most common bytes (byte_value, count) in first 64KB:", most_common)
    dc = collections.Counter(block_bytes)
    print("dumped block most common bytes (byte_value, count):", dc.most_common(5))


if __name__ == "__main__":
    main()
