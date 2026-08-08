"""
KVHOOK reconstruction — based on the documented spec in your project files:
  1. Wait KVHOOK_WARMUP_S (default 60s) for CUDA graph capture to clear.
  2. Baseline snapshot: which KV cache blocks are nonzero, per layer.
  3. Open KVHOOK_WINDOW_S (default 30s) window for exactly one request.
  4. Final snapshot, diff against baseline, dump newly-nonzero blocks + manifest.

CAVEAT: this is NOT a copy of your original 01_patch_connector.sh — I don't have
those bytes. This is rebuilt from the documented behavior only. The tensor
shape / dict-key assumptions below (kv_caches as dict[layer_name] -> tensor of
shape [2, num_blocks, block_size, num_kv_heads, head_dim]) match the common
vLLM KVConnector convention, but you MUST check this against your actual
installed nixl_connector.py's register_kv_caches signature before trusting it.
If your version differs, this silently produces wrong or empty dumps —
exactly like bug #3 in your own report, just a different cause.

Includes the content-hash fix already decided on: sha256 per block at dump
time, since block_id (physical slot from torch.nonzero) is not stable across
runs under concurrent load.
"""

import os
import time
import hashlib
import torch

KVHOOK_WARMUP_S = float(os.environ.get("KVHOOK_WARMUP_S", 60))
KVHOOK_WINDOW_S = float(os.environ.get("KVHOOK_WINDOW_S", 30))
KVHOOK_OUT_BIN = os.environ.get("KVHOOK_OUT_BIN", "/tmp/kvhook_dump.bin")
KVHOOK_OUT_MANIFEST = os.environ.get("KVHOOK_OUT_MANIFEST", "/tmp/kvhook_manifest.txt")

_original_register_kv_caches = None  # set by patch_connector() below


def _block_nonzero_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns a 1D bool mask over the block dimension: True where that block
    has any nonzero element. ASSUMES block dimension is dim=1 of a tensor
    shaped [2 (K/V), num_blocks, block_size, num_kv_heads, head_dim].
    VERIFY this matches your actual cache tensor layout.
    """
    flat = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
    return (flat != 0).any(dim=-1).any(dim=0)


def _snapshot(kv_caches: dict) -> dict:
    """layer_name -> bool tensor of which blocks are currently nonzero."""
    return {name: _block_nonzero_mask(t).clone() for name, t in kv_caches.items()}


def _dump_diff(kv_caches: dict, baseline: dict, final: dict):
    manifest_lines = []
    with open(KVHOOK_OUT_BIN, "wb") as fbin:
        offset = 0
        for layer_name, tensor in kv_caches.items():
            newly_nonzero = final[layer_name] & (~baseline[layer_name])
            block_ids = torch.nonzero(newly_nonzero, as_tuple=True)[0].tolist()
            for block_id in block_ids:
                # ASSUMES block dim is dim=1; adjust to your actual layout.
                block = tensor[:, block_id].contiguous()
                block_bytes = block.cpu().numpy().tobytes()
                content_hash = hashlib.sha256(block_bytes).hexdigest()
                fbin.write(block_bytes)
                length = len(block_bytes)
                shape_str = "x".join(str(d) for d in block.shape)
                # chr(9)/chr(10) instead of \t/\n literals — survives heredoc
                # escaping issues per your bug #1.
                line = chr(9).join([
                    layer_name,
                    str(block_id),
                    str(offset),
                    str(length),
                    shape_str,
                    str(block.dtype),
                    content_hash,          # <-- the fix: stable across runs
                ])
                manifest_lines.append(line)
                offset += length
    with open(KVHOOK_OUT_MANIFEST, "w") as fman:
        fman.write(chr(9).join([
            "layer_name", "block_id", "offset", "len",
            "block_shape", "dtype", "content_hash"
        ]) + chr(10))
        fman.write(chr(10).join(manifest_lines) + chr(10))


def patched_register_kv_caches(self, kv_caches: dict, *args, **kwargs):
    result = _original_register_kv_caches(self, kv_caches, *args, **kwargs)

    print(f"[KVHOOK] waiting {KVHOOK_WARMUP_S}s for CUDA graph capture...")
    time.sleep(KVHOOK_WARMUP_S)

    baseline = _snapshot(kv_caches)
    print(f"[KVHOOK] baseline captured. window open for {KVHOOK_WINDOW_S}s — fire the request now.")
    time.sleep(KVHOOK_WINDOW_S)

    final = _snapshot(kv_caches)
    _dump_diff(kv_caches, baseline, final)
    print(f"[KVHOOK] dumped to {KVHOOK_OUT_BIN}, manifest at {KVHOOK_OUT_MANIFEST}")

    return result


def patch_connector():
    """Call this before the connector class is instantiated."""
    global _original_register_kv_caches
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        NixlConnector,  # ADJUST import path to match your installed version
    )
    _original_register_kv_caches = NixlConnector.register_kv_caches
    NixlConnector.register_kv_caches = patched_register_kv_caches
    print("[KVHOOK] patched NixlConnector.register_kv_caches")


if __name__ == "__main__":
    patch_connector()
