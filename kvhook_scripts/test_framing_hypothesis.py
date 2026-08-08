#!/usr/bin/env python3
"""
Tests a specific hypothesis about why only even-indexed (K/V, head)
4096-byte sub-units matched in reconstruct.py's naive contiguous search:

    NIXL fragments the wire transfer at some fixed size F (header + payload).
    If F's payload portion is close to 8192 bytes (= two 4096-byte
    sub-units), then sub-unit 2k lands contiguously at a fragment start, but
    sub-unit 2k+1 straddles the boundary: its first ~4075 bytes are still in
    the same fragment as 2k, but its last ~21 bytes land after the NEXT
    fragment's header. A naive whole-4096-byte search for sub-unit 2k+1
    would find nothing even though every single byte of it is present on
    the wire, just not contiguous.

Strategy: for a block already known to have its even sub-units matched
(from reconstruct.py's output), look at the EXACT byte position right after
where an even sub-unit matched, and do a byte-by-byte comparison against the
following odd sub-unit to find exactly where it diverges (if it diverges at
all -- full mismatch from byte 0 would falsify the hypothesis). Then search
a small window past the divergence point for the remaining suffix bytes, to
measure the actual header size directly from the capture rather than
assuming one.

Usage:
    python3 test_framing_hypothesis.py <pcap> <dump.bin> <manifest.tsv> <layer_name> <block_id>
"""
import sys
import hashlib
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest

SUBUNIT_BYTES = 4096


def longest_common_prefix(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    pcap_path, dump_path, manifest_path, layer_name, block_id = sys.argv[1:6]

    rows = load_manifest(manifest_path)
    row = next(r for r in rows if r["layer_name"] == layer_name and r["block_id"] == block_id)
    with open(dump_path, "rb") as f:
        f.seek(int(row["offset"]))
        block_bytes = f.read(int(row["len"]))
    assert hashlib.sha256(block_bytes).hexdigest() == row["content_hash"]
    n_units = len(block_bytes) // SUBUNIT_BYTES
    subunits = [block_bytes[i * SUBUNIT_BYTES:(i + 1) * SUBUNIT_BYTES] for i in range(n_units)]

    print(f"probing {layer_name} block_id={block_id}: {n_units} sub-units of {SUBUNIT_BYTES}B")
    print("parsing pcap + reassembling streams...")
    segments = list(parse_pcap(pcap_path))
    streams = reassemble_streams(segments)
    bulk_key, bulk_buf = max(streams.items(), key=lambda kv: len(kv[1]))
    print(f"bulk stream: {bulk_key}, {len(bulk_buf)} bytes")
    print()

    # Locate every even sub-unit's exact offset (should all be found).
    even_offsets = {}
    for i in range(0, n_units, 2):
        idx = bulk_buf.find(subunits[i])
        even_offsets[i] = idx
        status = f"found at {idx}" if idx != -1 else "NOT FOUND (hypothesis needs a different anchor block)"
        print(f"  unit {i:2d} (even): {status}")
    print()

    fragment_sizes = []
    header_sizes = []

    for i in range(0, n_units, 2):
        even_off = even_offsets[i]
        if even_off == -1 or i + 1 >= n_units:
            continue
        odd_unit = subunits[i + 1]
        expected_start = even_off + SUBUNIT_BYTES  # where the odd unit WOULD start if fully contiguous

        window = bulk_buf[expected_start:expected_start + SUBUNIT_BYTES]
        lcp = longest_common_prefix(window, odd_unit)

        print(f"unit {i+1:2d} (odd): contiguous-prefix match with unit {i} = {lcp}/{SUBUNIT_BYTES} bytes")

        if lcp == SUBUNIT_BYTES:
            print(f"    -> fully contiguous, no header found here (fragment boundary is elsewhere)")
            continue
        if lcp == 0:
            print(f"    -> ZERO byte match at the expected contiguous position -- hypothesis "
                  f"falsified for this unit, or wrong anchor offset")
            continue

        # Search a small window past the divergence point for the tail bytes
        # of the odd unit (odd_unit[lcp:]), to measure the actual gap size.
        tail = odd_unit[lcp:]
        search_start = expected_start + lcp
        search_region = bulk_buf[search_start:search_start + 200]  # generous window
        tail_idx_in_region = search_region.find(tail[:min(len(tail), 64)])  # match on a prefix of the tail, in case tail itself spans another boundary

        if tail_idx_in_region == -1:
            print(f"    -> divergence at byte {lcp}, but could not find the remaining "
                  f"{len(tail)} tail bytes within the next 200 bytes of the stream. "
                  f"Hypothesis needs refinement for this unit.")
            continue

        gap = tail_idx_in_region  # bytes between divergence point and where the tail resumes
        print(f"    -> divergence at byte {lcp}/{SUBUNIT_BYTES}. Tail resumes {gap} bytes later "
              f"(that gap is the inferred header/framing size).")

        fragment_payload = lcp + (SUBUNIT_BYTES - lcp)  # = SUBUNIT_BYTES always, just for clarity
        implied_fragment_total = (SUBUNIT_BYTES + lcp) + gap  # bytes from even_off to where odd unit's tail resumes, i.e. one full fragment span
        fragment_sizes.append(SUBUNIT_BYTES + lcp + gap)
        header_sizes.append(gap)

    print()
    if header_sizes:
        print(f"=== summary across {len(header_sizes)} tested boundaries ===")
        print(f"inferred header/gap sizes: {header_sizes}")
        print(f"implied fragment total sizes (payload-before-header + header): {fragment_sizes}")
        print()
        print("If these are consistent, the even/odd pattern is a fragmentation "
              "artifact, not missing data -- every byte of both even AND odd "
              "sub-units is present on the wire, just not contiguous at the "
              "4096B granularity a naive .find() requires.")
    else:
        print("No boundary could be fully characterized -- hypothesis not confirmed "
              "with this anchor block. Try a different block_id or layer.")


if __name__ == "__main__":
    main()
