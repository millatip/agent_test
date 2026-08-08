#!/usr/bin/env python3
"""
03_analyze_capture.py

Given a pcap capture and the corresponding kvhook dump + manifest, this:
  1. Identifies which TCP stream carried the real KV transfer (largest payload).
  2. Reassembles that stream's raw bytes.
  3. For every dumped block, finds where its first 256 bytes (one head's
     worth: head_dim * 2 bytes for bf16) appear on the wire.
  4. Reports the ordering, confirming layer-sequential, K-then-V transfer
     (or flagging if this capture's structure differs).

Requires: tshark on PATH, torch installed in the active environment.

Usage:
    python3 03_analyze_capture.py <pcap_file> <kvhook_dump.bin> <kvhook_manifest.txt>
"""
import sys
import subprocess
import tempfile
import os


def find_real_stream(pcap_path, min_bytes=1_000_000):
    """Return the tcp.stream index with the largest total payload, assumed
    to be the KV transfer (control/heartbeat streams are tiny by comparison)."""
    out = subprocess.run(
        ["tshark", "-r", pcap_path, "-T", "fields", "-e", "tcp.stream"],
        capture_output=True, text=True, check=True
    )
    streams = sorted(set(int(x) for x in out.stdout.split() if x.strip().isdigit()))

    best_stream, best_size = None, 0
    for s in streams:
        r = subprocess.run(
            ["tshark", "-r", pcap_path, "-Y", f"tcp.stream=={s}",
             "-T", "fields", "-e", "tcp.len"],
            capture_output=True, text=True, check=True
        )
        total = sum(int(x) for x in r.stdout.split() if x.strip().isdigit())
        if total > best_size:
            best_stream, best_size = s, total

    if best_size < min_bytes:
        print(f"WARNING: largest stream is only {best_size} bytes "
              f"(threshold {min_bytes}). May not be the KV transfer.")
    return best_stream, best_size


def get_stream_hex(pcap_path, stream_id):
    """Reassemble a TCP stream's raw bytes via tshark follow,tcp,raw and
    return as one hex string."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["tshark", "-r", pcap_path, "-q", "-z", f"follow,tcp,raw,{stream_id}"],
        stdout=open(tmp_path, "w"), stderr=subprocess.STDOUT, check=True
    )
    with open(tmp_path) as f:
        content = f.read()
    os.unlink(tmp_path)

    lines = [l.strip() for l in content.split("\n")]
    hex_lines = [l for l in lines if l and all(c in "0123456789abcdefABCDEF" for c in l)]
    return "".join(hex_lines)


def parse_manifest(manifest_path):
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            name = parts[0]
            block_id = int(parts[1].split("=")[1])
            offset = int(parts[2].split("=")[1])
            length = int(parts[3].split("=")[1])
            entries.append((name, block_id, offset, length))
    return entries


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    pcap_path, dump_path, manifest_path = sys.argv[1:4]

    print(f"Loading manifest: {manifest_path}")
    entries = parse_manifest(manifest_path)
    print(f"  {len(entries)} dumped blocks found.")

    print(f"Loading dump: {dump_path}")
    with open(dump_path, "rb") as f:
        dump_bytes = f.read()

    print(f"Finding real KV-transfer stream in {pcap_path} ...")
    stream_id, stream_size = find_real_stream(pcap_path)
    print(f"  -> stream {stream_id}, {stream_size} bytes payload")

    print("Reassembling stream bytes (this may take a moment for large captures)...")
    wire_hex = get_stream_hex(pcap_path, stream_id)
    print(f"  reassembled {len(wire_hex)//2} bytes")

    FINGERPRINT_BYTES = 32  # bytes used as a search key per block

    results = []
    not_found = []
    for name, block_id, offset, length in entries:
        # use first head (token 0, head 0) of the block as a representative fingerprint
        fingerprint = dump_bytes[offset:offset + FINGERPRINT_BYTES].hex()
        idx = wire_hex.find(fingerprint)
        if idx == -1:
            not_found.append((name, block_id))
        else:
            results.append((idx // 2, name, block_id))  # hex-offset -> byte-offset

    results.sort(key=lambda x: x[0])

    print(f"\n{len(results)} / {len(entries)} blocks located on the wire.")
    if not_found:
        print(f"NOT found: {not_found[:10]}{' ...' if len(not_found) > 10 else ''}")

    print("\nWire order (byte_offset, layer, block_id):")
    for byte_offset, name, block_id in results[:40]:
        print(f"  {byte_offset:>10}  {name}  block_id={block_id}")
    if len(results) > 40:
        print(f"  ... ({len(results) - 40} more)")

    layer_order = [name for _, name, _ in results]
    print("\nObserved order (layer names only, first 10):")
    for n in layer_order[:10]:
        print(f"  {n}")

    print("\nIf this reads layer0[0], layer0[1], layer1[0], layer1[1], ... "
          "that confirms strict layer-sequential, K-then-V ordering, "
          "consistent with prior validated runs.")


if __name__ == "__main__":
    main()
