#!/usr/bin/env python3
"""
Stage 3 reconstruction: match KVHOOK-dumped KV blocks (ground truth, from
the patched prefill worker's GPU memory) against a wire capture (pcap),
by exact content + SHA-256, never by block_id/slot index.

Stdlib only (no scapy/dpkt, no torch/vllm) -- deliberately runnable with
plain system python3, not the pinned venv_dynamo_pd.

Usage:
    python3 reconstruct.py <pcap_path> <kvhook_dump.bin> <kvhook_manifest.tsv> [--host-pair IP1 IP2]

What it does:
  1. Parses the pcap (classic libpcap format -- what tcpdump writes by
     default) into individual TCP segments: (src_ip, src_port, dst_ip,
     dst_port, seq, payload_bytes).
  2. Groups segments into TCP streams by 4-tuple (both directions kept
     separate -- a "stream" here is one direction of one connection).
  3. Reassembles each stream's payload in TCP SEQUENCE order, not capture
     order (retransmits/out-of-order capture are collapsed by seq
     position, not by when tcpdump happened to see them).
  4. For each block in the manifest, reads its exact bytes out of the
     KVHOOK .bin dump at the recorded offset, and searches for that exact
     byte string across every reassembled stream. A hit is scored only on
     byte-exact match (this doubles as free confirmation of the block's
     recorded content_hash, since we search using bytes taken directly
     from a manifest row whose hash was computed off those same bytes).
  5. Reports per-layer match counts and an overall summary.

This does NOT decrypt or interpret anything -- it is a plaintext
substring search, appropriate here specifically because Stage 1 already
established the wire is unencrypted (UCX_TLS=tcp with no TLS layer).
"""
import struct
import sys
import hashlib
from collections import defaultdict


def parse_pcap(path):
    """Yields (ts, src_ip, src_port, dst_ip, dst_port, seq, payload) for
    each TCP segment carrying a nonzero payload. Classic pcap format only."""
    with open(path, "rb") as f:
        global_hdr = f.read(24)
        if len(global_hdr) < 24:
            raise ValueError("file too short to be a pcap")
        magic = struct.unpack("<I", global_hdr[:4])[0]
        if magic == 0xA1B2C3D4:
            endian = "<"
        elif magic == 0xD4C3B2A1:
            endian = ">"
        elif magic in (0x0A0D0D0A,):
            raise ValueError(
                "this is pcapng, not classic pcap -- convert first with: "
                "tshark -F pcap -r <in> -w <out>"
            )
        else:
            raise ValueError(f"unrecognized pcap magic: {magic:#x}")

        while True:
            rec_hdr = f.read(16)
            if len(rec_hdr) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + "IIII", rec_hdr)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            parsed = _parse_eth_ip_tcp(data)
            if parsed is not None:
                src_ip, src_port, dst_ip, dst_port, seq, payload = parsed
                if payload:
                    yield (ts_sec + ts_usec / 1e6, src_ip, src_port, dst_ip, dst_port, seq, payload)


def _parse_eth_ip_tcp(data):
    if len(data) < 14:
        return None
    eth_type = struct.unpack(">H", data[12:14])[0]
    offset = 14
    if eth_type == 0x8100:  # 802.1Q VLAN tag
        if len(data) < 18:
            return None
        eth_type = struct.unpack(">H", data[16:18])[0]
        offset = 18
    if eth_type != 0x0800:  # not IPv4
        return None
    if len(data) < offset + 20:
        return None
    ip_hdr = data[offset:offset + 20]
    ver_ihl = ip_hdr[0]
    ihl = (ver_ihl & 0x0F) * 4
    proto = ip_hdr[9]
    if proto != 6:  # not TCP
        return None
    src_ip = ".".join(str(b) for b in ip_hdr[12:16])
    dst_ip = ".".join(str(b) for b in ip_hdr[16:20])
    ip_total_len = struct.unpack(">H", ip_hdr[2:4])[0]
    tcp_start = offset + ihl
    if len(data) < tcp_start + 20:
        return None
    tcp_hdr = data[tcp_start:tcp_start + 20]
    src_port, dst_port = struct.unpack(">HH", tcp_hdr[0:4])
    seq = struct.unpack(">I", tcp_hdr[4:8])[0]
    data_offset = (tcp_hdr[12] >> 4) * 4
    payload_start = tcp_start + data_offset
    ip_payload_end = offset + ip_total_len
    payload_end = min(len(data), ip_payload_end) if ip_total_len > 0 else len(data)
    payload = data[payload_start:payload_end] if payload_end > payload_start else b""
    return (src_ip, src_port, dst_ip, dst_port, seq, payload)


def reassemble_streams(segments):
    """Returns {(src_ip,src_port,dst_ip,dst_port): reassembled_bytes}.
    Reassembly is by TCP sequence number (relative to that stream's first
    seen seq), not capture order. Overlapping retransmits: first-seen
    bytes at a given absolute seq position win (doesn't matter for
    identical retransmitted content anyway)."""
    by_stream = defaultdict(list)
    for ts, src_ip, src_port, dst_ip, dst_port, seq, payload in segments:
        key = (src_ip, src_port, dst_ip, dst_port)
        by_stream[key].append((seq, payload))

    reassembled = {}
    for key, pieces in by_stream.items():
        # Build a sparse byte map keyed by absolute seq (uint32, may wrap;
        # not handling wraparound -- these captures are far too small for
        # 4GB of sequence space to matter).
        byte_map = {}
        for seq, payload in pieces:
            for i, b in enumerate(payload):
                pos = seq + i
                if pos not in byte_map:
                    byte_map[pos] = b
        if not byte_map:
            continue
        lo, hi = min(byte_map), max(byte_map)
        buf = bytearray(hi - lo + 1)
        present = bytearray(hi - lo + 1)
        for pos, b in byte_map.items():
            buf[pos - lo] = b
            present[pos - lo] = 1
        reassembled[key] = bytes(buf)
    return reassembled


def reassemble_streams_with_presence(segments):
    """Same as reassemble_streams(), but also returns the presence bitmap
    that function computes and discards -- 1 where a byte was actually
    captured, 0 where the position was never covered by any segment (and
    so defaults to a zero byte, indistinguishable from legitimate
    zero-valued content without this). Added to investigate/quantify
    capture-gap-driven reconstruction failures; see REPORT.md."""
    by_stream = defaultdict(list)
    for ts, src_ip, src_port, dst_ip, dst_port, seq, payload in segments:
        key = (src_ip, src_port, dst_ip, dst_port)
        by_stream[key].append((seq, payload))

    reassembled = {}
    presence = {}
    for key, pieces in by_stream.items():
        byte_map = {}
        for seq, payload in pieces:
            for i, b in enumerate(payload):
                pos = seq + i
                if pos not in byte_map:
                    byte_map[pos] = b
        if not byte_map:
            continue
        lo, hi = min(byte_map), max(byte_map)
        buf = bytearray(hi - lo + 1)
        present = bytearray(hi - lo + 1)
        for pos, b in byte_map.items():
            buf[pos - lo] = b
            present[pos - lo] = 1
        reassembled[key] = bytes(buf)
        presence[key] = bytes(present)
    return reassembled, presence


def load_manifest(manifest_path):
    rows = []
    with open(manifest_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = dict(zip(header, line.split("\t")))
            rows.append(fields)
    return rows


def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <pcap> <kvhook_dump.bin> <kvhook_manifest.tsv>")
        sys.exit(1)
    pcap_path, dump_path, manifest_path = sys.argv[1:4]

    print(f"=== parsing pcap: {pcap_path} ===")
    segments = list(parse_pcap(pcap_path))
    print(f"{len(segments)} TCP segments with nonzero payload")

    stream_endpoints = sorted(set((s[1], s[2], s[3], s[4]) for s in segments))
    print(f"{len(stream_endpoints)} distinct (src_ip,src_port,dst_ip,dst_port) directions:")
    for ep in stream_endpoints:
        n = sum(1 for s in segments if (s[1], s[2], s[3], s[4]) == ep)
        total_bytes = sum(len(s[6]) for s in segments if (s[1], s[2], s[3], s[4]) == ep)
        print(f"  {ep[0]}:{ep[1]} -> {ep[2]}:{ep[3]}  segments={n}  bytes={total_bytes}")

    print("=== reassembling TCP streams by sequence number ===")
    streams = reassemble_streams(segments)
    for key, buf in streams.items():
        print(f"  {key}: reassembled {len(buf)} bytes")

    print(f"=== loading manifest: {manifest_path} ===")
    rows = load_manifest(manifest_path)
    print(f"{len(rows)} dumped blocks")

    # Sub-block granularity: NIXL frames its wire transfers such that a
    # whole 64KB block is rarely contiguous end-to-end (small per-fragment
    # protocol headers break it up), but one (K-or-V, head) unit --
    # block_size(16 tokens) x head_dim(128) x 2 bytes(bf16) = 4096 bytes --
    # survives intact (confirmed empirically via diagnose_match.py: 4096B
    # windows matched, nothing coarser did, once block bytes were dumped in
    # physical stride order rather than logical shape order). This is the
    # real recovery unit, not the full block.
    SUBUNIT_BYTES = 4096

    print(f"=== matching against dump: {dump_path} (whole-block AND "
          f"{SUBUNIT_BYTES}B sub-unit granularity) ===")
    per_layer_total = defaultdict(int)
    per_layer_whole_matched = defaultdict(int)
    per_layer_subunits_total = defaultdict(int)
    per_layer_subunits_matched = defaultdict(int)
    whole_matches = []
    subunit_matches = []

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
                print(
                    f"  WARNING: manifest/dump mismatch for {layer} block_id="
                    f"{row['block_id']} -- dump file may be truncated or corrupted"
                )
                continue

            for key, buf in streams.items():
                idx = buf.find(block_bytes)
                if idx != -1:
                    per_layer_whole_matched[layer] += 1
                    whole_matches.append((layer, row["block_id"], expected_hash, (key, idx)))
                    break

            n_sub = length // SUBUNIT_BYTES
            for i in range(n_sub):
                sub = block_bytes[i * SUBUNIT_BYTES:(i + 1) * SUBUNIT_BYTES]
                per_layer_subunits_total[layer] += 1
                for key, buf in streams.items():
                    idx = buf.find(sub)
                    if idx != -1:
                        per_layer_subunits_matched[layer] += 1
                        subunit_matches.append(
                            (layer, row["block_id"], i, hashlib.sha256(sub).hexdigest(), (key, idx))
                        )
                        break

    print()
    print("=== per-layer match rate (whole 64KB block, contiguous) ===")
    for layer in sorted(per_layer_total, key=lambda l: int(l.split(".")[2])):
        tot = per_layer_total[layer]
        m = per_layer_whole_matched[layer]
        print(f"  {layer}: {m}/{tot} ({100.0*m/tot:.1f}%)")

    print()
    print(f"=== per-layer match rate ({SUBUNIT_BYTES}B sub-unit = one "
          f"K-or-V head's worth of one block) ===")
    for layer in sorted(per_layer_total, key=lambda l: int(l.split(".")[2])):
        tot = per_layer_subunits_total[layer]
        m = per_layer_subunits_matched[layer]
        pct = 100.0 * m / tot if tot else 0.0
        print(f"  {layer}: {m}/{tot} ({pct:.1f}%)")

    total = len(rows)
    total_whole = len(whole_matches)
    total_sub = sum(per_layer_subunits_total.values())
    total_sub_matched = len(subunit_matches)
    print()
    print(f"=== overall whole-block: {total_whole}/{total} "
          f"({100.0*total_whole/total:.1f}%) ===")
    print(f"=== overall sub-unit ({SUBUNIT_BYTES}B): {total_sub_matched}/{total_sub} "
          f"({100.0*total_sub_matched/total_sub:.1f}%) ===")
    print(
        f"(blocks/sub-units dumped locally but never found on the wire are "
        f"expected here -- most of this request's 464 prompt tokens were "
        f"served from an existing prefix-cache hit at the decode side, from "
        f"repeated capture attempts against this same fixed payload earlier "
        f"in this session; only the tail that wasn't already cached needed a "
        f"fresh transfer)"
    )

    if subunit_matches:
        print()
        print("first few sub-unit matches (layer, block_id, kv/head-unit-index, stream, offset_in_stream):")
        for layer, block_id, unit_i, h, (key, idx) in subunit_matches[:10]:
            print(f"  {layer} block_id={block_id} unit={unit_i} hash={h[:12]}... "
                  f"stream={key[0]}:{key[1]}->{key[2]}:{key[3]} offset={idx}")


if __name__ == "__main__":
    main()
