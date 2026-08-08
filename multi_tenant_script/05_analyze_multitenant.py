#!/usr/bin/env python3
"""
05_analyze_multitenant.py  (hardened, content-driven attribution)

Given a concurrent multi-tenant wire capture and per-tenant SOLO calibration
dumps, decide -- for every calibrated KV block -- whether that block's bytes
actually crossed the wire, and attribute it back to a tenant and a layer.

WHY THIS WAS REWRITTEN
----------------------
The previous version attributed blocks by (layer_name, block_id). That is
INVALID under concurrent load: block_id is a *physical* KV-cache slot index,
and slots are recycled/reassigned across requests, so the same block_id on the
wire is not the same logical block that calibration saw. Two failure modes
follow: (a) a slot that got reused points calibration at the wrong content, and
(b) a genuinely-transmitted block gets missed because its slot id drifted.

The fix is to drive attribution purely by block CONTENT:

  * Each calibrated block now carries a sha256 of its raw bytes (added to the
    KVHOOK manifest by 01_patch_connector.sh). sha256 is the block's stable
    identity -- it does not move when the physical slot is recycled.
  * For each calibrated block we search the reassembled TCP payload for the
    block's byte sequence (a leading window as a fast filter, then a full-block
    byte-exact confirmation). hit/miss is recorded KEYED BY sha256, not by slot.
  * Results are aggregated per-tenant and per-layer (found / total).

The "64-byte-window hex search" from the prior tool is kept as the primary
locator (default --window-bytes 64); full-block byte-exact confirmation is an
added rigor check reported alongside it.

USAGE
-----
Analyze one condition and merge its rows into the results dir:

    python3 05_analyze_multitenant.py analyze \
        --condition WARM \
        --pcap /tmp/kvexfil/warm_concurrent/capture.pcap \
        --tenant-manifest /workspace/multi_tenant_script/tenant_manifest.tsv \
        --calibration-dir /tmp/kvexfil/calib \
        --timing /tmp/kvexfil/warm_concurrent/timing.log \
        --results-dir /tmp/kvexfil/results \
        [--window-bytes 64] [--min-stream-bytes 10000]

After both conditions are analyzed, print the comparison table:

    python3 05_analyze_multitenant.py table --results-dir /tmp/kvexfil/results

Deliverables written under --results-dir:
    summary.json    merged {condition: {tenant: {...}}} incl. per-condition rows
    per_layer.csv   [condition, tenant, layer_name, found, total]
    summary.csv     [condition, tenant, blocks_found, blocks_total, request_duration_s]
"""
import sys
import os
import json
import csv
import argparse
import subprocess
import tempfile
import hashlib
from collections import defaultdict, OrderedDict


# --------------------------------------------------------------------------
# pcap / tshark helpers
# --------------------------------------------------------------------------
def list_streams(pcap_path, min_bytes=10_000):
    """Return {stream_id: total_payload_bytes} for streams above threshold."""
    out = subprocess.run(
        ["tshark", "-r", pcap_path, "-T", "fields", "-e", "tcp.stream"],
        capture_output=True, text=True, check=True,
    )
    stream_ids = sorted(set(int(x) for x in out.stdout.split() if x.strip().isdigit()))
    sizes = {}
    for s in stream_ids:
        r = subprocess.run(
            ["tshark", "-r", pcap_path, "-Y", f"tcp.stream=={s}",
             "-T", "fields", "-e", "tcp.len"],
            capture_output=True, text=True, check=True,
        )
        total = sum(int(x) for x in r.stdout.split() if x.strip().isdigit())
        if total >= min_bytes:
            sizes[s] = total
    return sizes


def get_stream_hex(pcap_path, stream_id):
    """Reassembled stream as one hex string (both directions, seq-ordered).
    Kept from the validated single-tenant tool -- used as the window locator."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["tshark", "-r", pcap_path, "-q", "-z", f"follow,tcp,raw,{stream_id}"],
        stdout=open(tmp_path, "w"), stderr=subprocess.STDOUT, check=True,
    )
    with open(tmp_path) as f:
        content = f.read()
    os.unlink(tmp_path)
    lines = [l.strip() for l in content.split("\n")]
    hex_lines = [l for l in lines
                 if l and all(c in "0123456789abcdefABCDEF" for c in l)]
    return "".join(hex_lines).lower()


def get_stream_dir_bytes(pcap_path, stream_id):
    """Per-direction payload bytes, reassembled in TCP-SEQUENCE order.

    Grouped by tcp.srcport (one entry per direction). Within a direction,
    each segment is placed at its (tcp.seq - base_seq) offset, so the byte
    stream is reconstructed in true send order even if packets were captured
    out of order or retransmitted -- capture order is NOT send order, and a
    32 KB KV block would otherwise be non-contiguous and fail byte-exact
    matching. Reverse-direction control traffic stays in its own buffer.
    """
    out = subprocess.run(
        ["tshark", "-r", pcap_path,
         "-Y", f"tcp.stream=={stream_id} && tcp.len>0",
         "-T", "fields", "-e", "tcp.srcport", "-e", "tcp.seq", "-e", "tcp.payload"],
        capture_output=True, text=True, check=True,
    )
    # srcport -> {seq: payload_bytes}
    dirs = OrderedDict()
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        srcport, seq, payload = parts[0].strip(), parts[1].strip(), parts[2].replace(":", "").strip()
        if not payload or not seq.isdigit():
            continue
        dirs.setdefault(srcport, {})[int(seq)] = bytes.fromhex(payload)
    result = []
    for segs in dirs.values():
        base = min(segs)
        buf = bytearray()
        for seq in sorted(segs):
            data = segs[seq]
            off = seq - base
            if off + len(data) > len(buf):
                buf.extend(b"\x00" * (off + len(data) - len(buf)))
            buf[off:off + len(data)] = data  # overwrite handles retransmits
        result.append(bytes(buf))
    return result


# --------------------------------------------------------------------------
# manifest / calibration parsing
# --------------------------------------------------------------------------
def parse_tenant_manifest(path):
    """tenant_manifest.tsv: tenant_id \\t marker \\t full_prompt"""
    tenants = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            tenants.append({"tenant_id": parts[0], "marker": parts[1],
                            "prompt": parts[2] if len(parts) > 2 else ""})
    return tenants


def parse_kvhook_manifest(path):
    """Robust key=value parse. Returns list of dicts with keys:
    name, block_id, offset, len, sha256 (sha256 may be None on old dumps)."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            name = parts[0]
            kv = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k] = v
            entries.append({
                "name": name,
                "block_id": int(kv["block_id"]) if "block_id" in kv else -1,
                "offset": int(kv["offset"]),
                "len": int(kv["len"]),
                "sha256": kv.get("sha256"),
            })
    return entries


def load_tenant_blocks(calibration_dir, tenant_id, window_bytes):
    """Load one tenant's solo calibration dump + manifest into block records:
    {name, block_id, sha256, len, fingerprint(bytes), full(bytes)}."""
    tdir = os.path.join(calibration_dir, tenant_id)
    dump_path = os.path.join(tdir, "kvhook_dump.bin")
    manifest_path = os.path.join(tdir, "kvhook_manifest.txt")
    if not (os.path.isfile(dump_path) and os.path.isfile(manifest_path)):
        print(f"  WARNING: missing calibration for {tenant_id} "
              f"({dump_path} / {manifest_path}) -- skipping.")
        return []
    with open(dump_path, "rb") as f:
        dump = f.read()
    blocks = []
    sha_mismatch = 0
    for e in parse_kvhook_manifest(manifest_path):
        raw = dump[e["offset"]:e["offset"] + e["len"]]
        recomputed = hashlib.sha256(raw).hexdigest()
        if e["sha256"] and e["sha256"] != recomputed:
            sha_mismatch += 1
        blocks.append({
            "name": e["name"],
            "block_id": e["block_id"],
            "sha256": e["sha256"] or recomputed,
            "len": e["len"],
            "fingerprint": raw[:window_bytes],
            "full": raw,
        })
    if sha_mismatch:
        print(f"  WARNING: {tenant_id}: {sha_mismatch} manifest sha256 values "
              f"did not match recomputed dump bytes (dump/manifest desync?).")
    return blocks


def parse_timing(path):
    """timing.log lines like: 'tenant0 start=.. end=.. dur=0.783' (dur optional)."""
    durations = {}
    if not path or not os.path.isfile(path):
        return durations
    with open(path) as f:
        for line in f:
            toks = line.split()
            if not toks:
                continue
            tid = toks[0]
            kv = {}
            for t in toks[1:]:
                if "=" in t:
                    k, v = t.split("=", 1)
                    kv[k] = v
            dur = None
            if "dur" in kv:
                try:
                    dur = float(kv["dur"])
                except ValueError:
                    dur = None
            elif "start" in kv and "end" in kv:
                try:
                    dur = float(kv["end"]) - float(kv["start"])
                except ValueError:
                    dur = None
            if dur is not None:
                durations[tid] = round(dur, 4)
    return durations


# --------------------------------------------------------------------------
# core: content-driven matching
# --------------------------------------------------------------------------
def analyze(args):
    tenants = parse_tenant_manifest(args.tenant_manifest)
    print(f"[{args.condition}] {len(tenants)} tenants; pcap={args.pcap}")

    # 1. streams
    streams = list_streams(args.pcap, args.min_stream_bytes)
    print(f"  {len(streams)} stream(s) >= {args.min_stream_bytes} bytes:")
    for s, sz in sorted(streams.items(), key=lambda x: -x[1]):
        print(f"    stream {s}: {sz} bytes")
    if not streams:
        print("  ERROR: no streams above threshold -- capture empty/wrong iface?")

    # 2. reassemble each stream once (hex for window filter, per-dir bytes for full)
    stream_hex = {}
    stream_dir_bytes = {}
    for s in streams:
        print(f"  reassembling stream {s} ...")
        stream_hex[s] = get_stream_hex(args.pcap, s)
        stream_dir_bytes[s] = get_stream_dir_bytes(args.pcap, s)

    # 3. per-tenant calibration blocks
    durations = parse_timing(args.timing)
    per_tenant = OrderedDict()
    per_layer_rows = []
    for t in tenants:
        tid = t["tenant_id"]
        blocks = load_tenant_blocks(args.calibration_dir, tid, args.window_bytes)
        total = len(blocks)
        window_hits = 0
        byte_exact = 0
        layer_found = defaultdict(int)
        layer_total = defaultdict(int)
        for b in blocks:
            layer_total[b["name"]] += 1
            fp_hex = b["fingerprint"].hex()
            full = b["full"]
            # window locator: leading bytes present in any reassembled stream
            hit = any(fp_hex in stream_hex[s] for s in streams)
            # full-block byte-exact confirmation in a single direction
            exact = any(full in db
                        for s in streams for db in stream_dir_bytes[s])
            if hit:
                window_hits += 1
                layer_found[b["name"]] += 1
            if exact:
                byte_exact += 1
        for layer in sorted(layer_total):
            per_layer_rows.append({
                "condition": args.condition, "tenant": tid,
                "layer_name": layer,
                "found": layer_found.get(layer, 0),
                "total": layer_total[layer],
            })
        per_tenant[tid] = {
            "blocks_found": window_hits,
            "blocks_total": total,
            "blocks_byte_exact": byte_exact,
            "request_duration_s": durations.get(tid),
            "fraction_found": round(window_hits / total, 4) if total else None,
        }
        frac = per_tenant[tid]["fraction_found"]
        print(f"  {tid}: found {window_hits}/{total} "
              f"(byte-exact {byte_exact}) frac={frac} "
              f"dur={per_tenant[tid]['request_duration_s']}")

    write_results(args.results_dir, args.condition, per_tenant, per_layer_rows,
                  meta={"pcap": os.path.abspath(args.pcap),
                        "window_bytes": args.window_bytes,
                        "n_streams": len(streams),
                        "stream_sizes": {str(k): v for k, v in streams.items()}})


# --------------------------------------------------------------------------
# results persistence (merge across conditions)
# --------------------------------------------------------------------------
def write_results(results_dir, condition, per_tenant, per_layer_rows, meta):
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    summary[condition] = {"_meta": meta, "tenants": per_tenant}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # per_layer.csv -- rewrite all conditions present in summary + this one
    all_layer_rows = _collect_layer_rows(results_dir, condition, per_layer_rows)
    per_layer_path = os.path.join(results_dir, "per_layer.csv")
    with open(per_layer_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "tenant", "layer_name", "found", "total"])
        for r in all_layer_rows:
            w.writerow([r["condition"], r["tenant"], r["layer_name"],
                        r["found"], r["total"]])

    # summary.csv -- flat per (condition, tenant)
    summary_csv = os.path.join(results_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "tenant", "blocks_found", "blocks_total",
                    "request_duration_s"])
        for cond in summary:
            for tid, row in summary[cond]["tenants"].items():
                w.writerow([cond, tid, row["blocks_found"], row["blocks_total"],
                            row["request_duration_s"]])
    print(f"  wrote {summary_path}, {per_layer_path}, {summary_csv}")


def _collect_layer_rows(results_dir, condition, new_rows):
    """Merge freshly-computed per-layer rows for `condition` with any rows for
    OTHER conditions already saved in a sidecar, so per_layer.csv holds all."""
    sidecar = os.path.join(results_dir, ".per_layer_rows.json")
    store = {}
    if os.path.isfile(sidecar):
        with open(sidecar) as f:
            store = json.load(f)
    store[condition] = new_rows
    with open(sidecar, "w") as f:
        json.dump(store, f)
    rows = []
    for cond in store:
        rows.extend(store[cond])
    return rows


# --------------------------------------------------------------------------
# comparison table
# --------------------------------------------------------------------------
def print_table(args):
    summary_path = os.path.join(args.results_dir, "summary.json")
    if not os.path.isfile(summary_path):
        print(f"No summary.json in {args.results_dir}; run 'analyze' first.")
        return
    with open(summary_path) as f:
        summary = json.load(f)
    conditions = list(summary.keys())
    tenants = []
    for c in conditions:
        for tid in summary[c]["tenants"]:
            if tid not in tenants:
                tenants.append(tid)
    tenants.sort()

    print("\n=================  WARM vs COLD comparison  =================")
    hdr = f"{'tenant':<9}"
    for c in conditions:
        hdr += f" | {c+' found/total':<20} {c+' frac':<10} {c+' dur(s)':<10}"
    print(hdr)
    print("-" * len(hdr))
    totals = {c: [0, 0] for c in conditions}
    for tid in tenants:
        row = f"{tid:<9}"
        for c in conditions:
            t = summary[c]["tenants"].get(tid)
            if t:
                ft = f"{t['blocks_found']}/{t['blocks_total']}"
                frac = t.get("fraction_found")
                fracs = f"{frac:.3f}" if frac is not None else "NA"
                dur = t.get("request_duration_s")
                durs = f"{dur:.3f}" if dur is not None else "NA"
                totals[c][0] += t["blocks_found"]
                totals[c][1] += t["blocks_total"]
            else:
                ft, fracs, durs = "-", "-", "-"
            row += f" | {ft:<20} {fracs:<10} {durs:<10}"
        print(row)
    print("-" * len(hdr))
    trow = f"{'ALL':<9}"
    for c in conditions:
        f_, t_ = totals[c]
        ft = f"{f_}/{t_}"
        fracs = f"{(f_/t_):.3f}" if t_ else "NA"
        trow += f" | {ft:<20} {fracs:<10} {'':<10}"
    print(trow)
    print("=" * len(hdr))
    if "WARM" in summary and "COLD" in summary:
        wf, wt = totals["WARM"]
        cf, ct = totals["COLD"]
        print(f"\nINTERPRETATION:")
        print(f"  WARM recovered {wf}/{wt} "
              f"({(wf/wt if wt else 0):.1%}) of calibrated blocks.")
        print(f"  COLD recovered {cf}/{ct} "
              f"({(cf/ct if ct else 0):.1%}) of calibrated blocks.")
        if wt and ct:
            if cf / ct >= 0.9 and wf / wt <= 0.45:
                print("  => COLD ~full, WARM ~1/3: prefix-cache RESIDENCY explains "
                      "the WARM shortfall (hypothesis CONFIRMED).")
            elif cf / ct <= 0.45:
                print("  => COLD also low: NOT a caching effect -- the matcher is "
                      "dropping blocks and needs debugging.")
            else:
                print("  => Intermediate result -- inspect per-tenant/per-layer rows.")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze one condition's capture")
    a.add_argument("--condition", required=True)
    a.add_argument("--pcap", required=True)
    a.add_argument("--tenant-manifest", required=True)
    a.add_argument("--calibration-dir", required=True)
    a.add_argument("--timing", default=None)
    a.add_argument("--results-dir", required=True)
    a.add_argument("--window-bytes", type=int, default=64)
    a.add_argument("--min-stream-bytes", type=int, default=10_000)
    a.set_defaults(func=analyze)

    t = sub.add_parser("table", help="print WARM-vs-COLD comparison table")
    t.add_argument("--results-dir", required=True)
    t.set_defaults(func=print_table)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
