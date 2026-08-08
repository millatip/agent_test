#!/bin/bash
# 01_patch_connector.sh
#
# Patches vLLM's NixlConnector to dump newly-written KV cache blocks to disk
# after one isolated request, for ground-truth comparison against network
# captures. Run this ONCE per prefill-worker restart cycle.
#
# Usage: ./01_patch_connector.sh /path/to/venv
#   e.g. ./01_patch_connector.sh /workspace/venv_dynamo_pd

set -e

VENV_PATH="${1:-/workspace/venv_dynamo_pd}"
FILE="$VENV_PATH/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py"

if [ ! -f "$FILE" ]; then
    echo "ERROR: connector file not found at $FILE"
    echo "Check VENV_PATH argument or your python version (assumed 3.12)."
    exit 1
fi

# Always restore from backup first to avoid double-patching.
if [ -f "$FILE.bak" ]; then
    cp "$FILE.bak" "$FILE"
    echo "Restored $FILE from backup."
else
    cp "$FILE" "$FILE.bak"
    echo "Created backup at $FILE.bak"
fi

if grep -q "KVHOOK" "$FILE"; then
    echo "ERROR: KVHOOK marker still present after restore. Aborting."
    exit 1
fi

python3 << PYEOF
path = "$FILE"

with open(path) as f:
    content = f.read()

old = '''        self.device_kv_caches = kv_caches'''

new = '''        self.device_kv_caches = kv_caches

        import os as _os
        import threading as _th
        import time as _time

        def _kvhook_oneshot(cache_refs):
            warmup_s = float(_os.environ.get("KVHOOK_WARMUP_S", "60"))
            window_s = float(_os.environ.get("KVHOOK_WINDOW_S", "30"))
            out_bin = _os.environ.get("KVHOOK_OUT_BIN", "/tmp/kvhook_dump.bin")
            out_manifest = _os.environ.get("KVHOOK_OUT_MANIFEST", "/tmp/kvhook_dump_manifest.txt")

            print(f"[KVHOOK] waiting {warmup_s}s for cudagraph capture / warmup to finish", flush=True)
            _time.sleep(warmup_s)

            print("[KVHOOK] taking BASELINE: recording nonzero block ids per layer", flush=True)
            def _nonzero_block_ids(t):
                # t shape: (num_blocks, block_size, num_kv_heads, head_dim)
                flat = t.reshape(t.shape[0], -1)
                nz = (flat != 0).any(dim=1)
                return set(torch.nonzero(nz, as_tuple=True)[0].tolist())

            baseline = {name: _nonzero_block_ids(t) for name, t in cache_refs}
            print(f"[KVHOOK] BASELINE captured. You have {window_s}s to fire your ONE request now.", flush=True)

            _time.sleep(window_s)

            print("[KVHOOK] taking FINAL snapshot, diffing block ids", flush=True)
            import hashlib as _hashlib
            with open(out_bin, "wb") as fdump, open(out_manifest, "w") as fman:
                offset = 0
                for name, t in cache_refs:
                    final_ids = _nonzero_block_ids(t)
                    new_ids = sorted(final_ids - baseline[name])
                    if not new_ids:
                        continue
                    for bid in new_ids:
                        block = t[bid].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                        fdump.write(block)
                        # sha256 over the block's raw bytes: a content identity
                        # that survives physical KV-slot recycling under concurrent
                        # load (block_id is NOT a stable identity, so downstream
                        # attribution keys on this hash, not on layer_name+block_id).
                        sha = _hashlib.sha256(block).hexdigest()
                        _TAB = chr(9)
                        _NL = chr(10)
                        manifest_line = (
                            name + _TAB +
                            "block_id=" + str(bid) + _TAB +
                            "offset=" + str(offset) + _TAB +
                            "len=" + str(len(block)) + _TAB +
                            "sha256=" + sha + _TAB +
                            "block_shape=" + str(tuple(t.shape[1:])) + _TAB +
                            "dtype=" + str(t.dtype) + _NL
                        )
                        fman.write(manifest_line)
                        offset += len(block)
            print(f"[KVHOOK] dumped {offset} bytes total to {out_bin}", flush=True)
            print(f"[KVHOOK] manifest at {out_manifest}", flush=True)

        _cache_refs = []
        for _layer_name, _cache_or_caches in kv_caches.items():
            _cache_list = (
                _cache_or_caches if self.kv_topo.split_k_and_v else [_cache_or_caches]
            )
            for _idx, _c in enumerate(_cache_list):
                _cache_refs.append((f"{_layer_name}[{_idx}]", _c))

        _oneshot_thread = _th.Thread(
            target=_kvhook_oneshot, args=(_cache_refs,), daemon=True
        )
        _oneshot_thread.start()
        print(f"[KVHOOK] one-shot thread started, tracking {len(_cache_refs)} tensors", flush=True)'''

assert content.count(old) == 1, f"expected exactly 1 match, found {content.count(old)}"
content = content.replace(old, new)

with open(path, 'w') as f:
    f.write(content)

print("Patched successfully.")
PYEOF

echo ""
echo "Done. Tunable via environment variables before launching the worker:"
echo "  KVHOOK_WARMUP_S      (default 60)  - seconds to wait for cudagraph capture to clear"
echo "  KVHOOK_WINDOW_S      (default 30)  - seconds you have to fire your request after baseline"
echo "  KVHOOK_OUT_BIN       (default /tmp/kvhook_dump.bin)"
echo "  KVHOOK_OUT_MANIFEST  (default /tmp/kvhook_dump_manifest.txt)"
echo ""
echo "Restore original file anytime with:"
echo "  cp $FILE.bak $FILE"