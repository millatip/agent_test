#!/usr/bin/env python3
"""
Samples raw header bytes at many known boundary positions, looking for a
self-describing structure (constant magic/type bytes + a length field)
instead of assuming a fixed period. Two sample sources:

  1. Phase A's 8 known odd-sub-unit-divergence boundaries (small capture,
     precisely characterized already by test_framing_hypothesis.py).
  2. The clean rerun's 1,152 per-block anchors -- the bytes immediately
     before each anchor are presumed to be that block-transfer's own
     leading header.

For each byte position in a fixed window, reports whether the value is
constant across all samples (a candidate magic/type byte) or varies (a
candidate length/sequence field) -- and for varying positions, checks
1/2/4-byte little- and big-endian interpretations against the actual
distance to the next boundary, in case that's literally what's encoded.

Usage:
    python3 dump_header_bytes.py
(paths are hardcoded below to this session's known-good captures)
"""
import sys
import hashlib
import struct
sys.path.insert(0, "/home/s3lab-spark/LG2026/KVHOOK")
from reconstruct import parse_pcap, reassemble_streams, load_manifest
from deframe_and_match import SUBUNIT_BYTES

PHASE_A_PCAP = "/tmp/kv_phaseA_20260806_210137.pcap"
PHASE_A_DUMP = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_dump_phaseA.bin"
PHASE_A_MANIFEST = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_manifest_phaseA.tsv"

CLEAN_PCAP = "/tmp/kv_phaseA_clean_20260806_225414.pcap"
CLEAN_DUMP = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_dump_phaseA_clean.bin"
CLEAN_MANIFEST = "/home/s3lab-spark/LG2026/KVHOOK/dumps/kvhook_manifest_phaseA_clean.tsv"

WINDOW = 40  # bytes to dump per sample


def longest_common_prefix(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def get_phase_a_boundary_headers():
    """Re-derives the 8 known boundaries in Phase A's layer0/block29, this
    time returning the raw header-window bytes and the gap-to-next-content
    distance, instead of just the gap size."""
    rows = load_manifest(PHASE_A_MANIFEST)
    row = next(r for r in rows if r["layer_name"] == "model.layers.0.self_attn.attn" and r["block_id"] == "29")
    with open(PHASE_A_DUMP, "rb") as f:
        f.seek(int(row["offset"]))
        block_bytes = f.read(int(row["len"]))
    assert hashlib.sha256(block_bytes).hexdigest() == row["content_hash"]
    n_units = len(block_bytes) // SUBUNIT_BYTES
    subunits = [block_bytes[i * SUBUNIT_BYTES:(i + 1) * SUBUNIT_BYTES] for i in range(n_units)]

    segments = list(parse_pcap(PHASE_A_PCAP))
    streams = reassemble_streams(segments)
    bulk_key, buf = max(streams.items(), key=lambda kv: len(kv[1]))

    samples = []
    for i in range(0, n_units, 2):
        even_off = buf.find(subunits[i])
        if even_off == -1 or i + 1 >= n_units:
            continue
        odd_unit = subunits[i + 1]
        expected_start = even_off + SUBUNIT_BYTES
        window = buf[expected_start:expected_start + SUBUNIT_BYTES]
        lcp = longest_common_prefix(window, odd_unit)
        if lcp == 0 or lcp == SUBUNIT_BYTES:
            continue
        header_pos = expected_start + lcp
        tail = odd_unit[lcp:]
        search_region = buf[header_pos:header_pos + 200]
        tail_idx = search_region.find(tail[:min(len(tail), 64)])
        if tail_idx == -1:
            continue
        content_resumes_at = header_pos + tail_idx
        samples.append({
            "label": f"unit{i}/unit{i+1} boundary",
            "header_bytes": buf[header_pos:header_pos + WINDOW],
            "gap_to_content": tail_idx,
        })
    return samples


def get_clean_anchor_headers(max_samples=60):
    """The bytes immediately before each block's anchor (start of unit0),
    for a sample spread across the whole clean-rerun capture."""
    rows = load_manifest(CLEAN_MANIFEST)
    segments = list(parse_pcap(CLEAN_PCAP))
    streams = reassemble_streams(segments)

    samples = []
    step = max(1, len(rows) // max_samples)
    with open(CLEAN_DUMP, "rb") as f:
        for row in rows[::step]:
            f.seek(int(row["offset"]))
            block_bytes = f.read(SUBUNIT_BYTES)  # just need unit0
            for key, buf in streams.items():
                anchor = buf.find(block_bytes)
                if anchor != -1 and anchor >= WINDOW:
                    samples.append({
                        "label": f"{row['layer_name']} block_id={row['block_id']}",
                        "header_bytes": buf[anchor - WINDOW:anchor],
                        "anchor": anchor,
                    })
                    break
    return samples


def analyze(samples, name):
    print(f"\n{'='*70}\n{name}: {len(samples)} samples\n{'='*70}")
    if not samples:
        print("no samples")
        return

    for s in samples[:20]:
        print(f"  [{s['label']}] {s['header_bytes'].hex()}")

    # constant-byte-position analysis
    min_len = min(len(s["header_bytes"]) for s in samples)
    print(f"\nper-position constancy (first {min_len} bytes, '.' = varies):")
    const_line = ""
    for pos in range(min_len):
        vals = set(s["header_bytes"][pos] for s in samples)
        const_line += f"{next(iter(vals)):02x}" if len(vals) == 1 else ".."
    print(f"  {const_line}")

    # check varying positions against known gap/distance fields, if present
    if samples and "gap_to_content" in samples[0]:
        print("\nchecking if any 1/2/4-byte field at any offset encodes gap_to_content:")
        for pos in range(0, min_len - 4):
            for width, fmt_le, fmt_be in ((1, "<B", ">B"), (2, "<H", ">H"), (4, "<I", ">I")):
                matches = 0
                for s in samples:
                    raw = s["header_bytes"][pos:pos + width]
                    if len(raw) < width:
                        continue
                    le_val = struct.unpack(fmt_le, raw)[0]
                    be_val = struct.unpack(fmt_be, raw)[0]
                    if le_val == s["gap_to_content"] or be_val == s["gap_to_content"]:
                        matches += 1
                if matches == len(samples) and matches > 0:
                    print(f"  offset={pos} width={width}: matches gap_to_content for ALL {matches} samples!")
                elif matches > len(samples) // 2:
                    print(f"  offset={pos} width={width}: matches for {matches}/{len(samples)} samples (partial)")


def main():
    print("Sampling Phase A's 8 known divergence boundaries...")
    pa_samples = get_phase_a_boundary_headers()
    analyze(pa_samples, "Phase A boundary headers (small capture, precise)")

    print("\n\nSampling clean-rerun per-block anchor headers...")
    clean_samples = get_clean_anchor_headers()
    analyze(clean_samples, "Clean rerun per-block leading headers")


if __name__ == "__main__":
    main()
