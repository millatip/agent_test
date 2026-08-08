"""
KVHOOK, ported from the 0.16.0 single-file nixl_connector.py version to
vLLM 0.25.1, where the connector is split across:

    kv_connector/v1/nixl/connector.py     - NixlBaseConnector / NixlPullConnector /
                                             NixlPushConnector facades
    kv_connector/v1/nixl/base_worker.py   - NixlBaseConnectorWorker.register_kv_caches
                                             (~line 933; the packed-storage branch
                                             returns at ~line 948, the general path
                                             calls nixl_wrapper.register_memory at
                                             ~line 1138)
    kv_connector/v1/nixl/pull_worker.py   - NixlPullConnectorWorker, does NOT
                                             override register_kv_caches
    kv_connector/v1/nixl/push_worker.py   - NixlPushConnectorWorker DOES override
                                             register_kv_caches (calls super() then
                                             does extra push-mode setup)

On this host the launch config is:
    --kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_both"}
`NixlConnector` is a backward-compat alias for `NixlPullConnector`
(connector.py: `NixlConnector = NixlPullConnector`), and NixlPullConnectorWorker
does not override register_kv_caches, so the live method resolves to
`NixlBaseConnectorWorker.register_kv_caches`. That is the hook target below.
If this host is ever reconfigured to push mode, patch
NixlPushConnectorWorker.register_kv_caches instead (or in addition) since it
shadows the base implementation.

Differences from the 0.16.0 version this replaces:

  - kv_caches is dict[layer_name] -> torch.Tensor with WHATEVER shape this
    build's TransferTopology decided on (K/V may or may not be split into
    separate dict entries depending on backend / kv_cache_layout — see
    utils.py:TransferTopology.get_transfer_cache_regions). The old code
    hardcoded dim=1 as the block dimension and a fixed 5-D
    [2, num_blocks, block_size, num_kv_heads, head_dim] shape. That
    assumption is NOT re-used here. Instead, the block dimension is
    detected per-layer at runtime by matching tensor.shape[d] against
    self.num_blocks (the authoritative block count NixlBaseConnectorWorker
    itself computed to register NIXL memory).
  - block_id is recorded as metadata only. Identity is the SHA-256 content
    hash of the dumped block bytes, since physical slot indices are
    recycled across requests under the block allocator and are not a
    stable cross-run key (this was bug-class relevant on the 0.16.0 A6000
    work too, just fixed there by convention rather than being enforced).
  - A companion layout JSON is written on every register_kv_caches call
    (cheap, always-on) recording the REAL per-layer shape/dtype/stride/
    nbytes plus self.kv_cache_layout, self.block_size, self.num_blocks,
    self.block_len_per_layer, self.use_mla. This is what "re-derive rather
    than carry over" means in practice: don't trust the A6000 numbers,
    read these back after the hook fires.

Activation: this module does nothing on import. Call patch_connector()
explicitly, or set KVHOOK_ENABLE=1 and let sitecustomize.py in the venv
call it (see venv sitecustomize.py). Gating behind an explicit call/env
var means importing this file is always safe even if something below is
wrong for a future vLLM version — it just won't patch anything.
"""

import hashlib
import json
import os
import threading
import time

import torch

KVHOOK_WARMUP_S = float(os.environ.get("KVHOOK_WARMUP_S", 60))
KVHOOK_WINDOW_S = float(os.environ.get("KVHOOK_WINDOW_S", 30))
KVHOOK_OUT_DIR = os.environ.get(
    "KVHOOK_OUT_DIR", os.path.expanduser("~/LG2026/KVHOOK/dumps")
)
KVHOOK_LAYOUT_JSON = os.environ.get(
    "KVHOOK_LAYOUT_JSON", os.path.join(KVHOOK_OUT_DIR, "kvhook_layout.json")
)
KVHOOK_LOG = os.environ.get(
    "KVHOOK_LOG", os.path.join(KVHOOK_OUT_DIR, "kvhook_patch.log")
)

_original_register_kv_caches = None
_patched_class = None
_already_patched = False


def _log(msg: str) -> None:
    line = f"[KVHOOK {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(KVHOOK_OUT_DIR, exist_ok=True)
        with open(KVHOOK_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _find_block_dim(tensor: torch.Tensor, num_blocks: int) -> int | None:
    """Which dim of this real (not assumed) tensor is the block dimension.

    Prefers dim 0 if it matches (the common case for HND / blocks-first
    layouts per TransferTopology.is_kv_layout_blocks_first), but checks all
    dims and warns if dim 0 doesn't match or if the match is ambiguous.
    """
    matches = [d for d, size in enumerate(tensor.shape) if size == num_blocks]
    if not matches:
        return None
    if 0 in matches:
        return 0
    return matches[0]


def _dump_layout(kv_caches: dict, worker) -> dict:
    """Record REAL per-layer tensor metadata. This is the empirical
    ground truth that supersedes any carried-over A6000 numbers."""
    layers = {}
    for name, tensor in kv_caches.items():
        if isinstance(tensor, (tuple, list)):
            # e.g. mamba (conv, ssm) pairs
            layers[name] = [
                {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "stride": list(t.stride()),
                    "nbytes": t.numel() * t.element_size(),
                }
                for t in tensor
            ]
            continue
        layers[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()),
            "nbytes": tensor.numel() * tensor.element_size(),
            "block_dim": _find_block_dim(tensor, getattr(worker, "num_blocks", -1)),
        }

    info = {
        "engine_id": getattr(worker, "engine_id", None),
        "tp_rank": getattr(worker, "tp_rank", None),
        "kv_cache_layout": getattr(worker, "kv_cache_layout", None),
        "block_size": getattr(worker, "block_size", None),
        "num_blocks": getattr(worker, "num_blocks", None),
        "block_len_per_layer": list(getattr(worker, "block_len_per_layer", []) or []),
        "use_mla": getattr(worker, "use_mla", None),
        "use_host_buffer": getattr(worker, "use_host_buffer", None),
        "num_layers": len(kv_caches),
        "layers": layers,
    }
    os.makedirs(KVHOOK_OUT_DIR, exist_ok=True)
    with open(KVHOOK_LAYOUT_JSON, "w") as f:
        json.dump(info, f, indent=2)
    _log(
        f"layout dumped: {len(layers)} layers, kv_cache_layout="
        f"{info['kv_cache_layout']}, block_size={info['block_size']}, "
        f"num_blocks={info['num_blocks']} -> {KVHOOK_LAYOUT_JSON}"
    )
    return info


def _reduce_all_but(tensor: torch.Tensor, keep_dim: int) -> torch.Tensor:
    """1D bool mask over `keep_dim`: True where that block has any nonzero
    element (collapsing every other dim)."""
    moved = tensor.movedim(keep_dim, 0)
    flat = moved.reshape(moved.shape[0], -1)
    return (flat != 0).any(dim=-1)


def _extract_block(tensor: torch.Tensor, block_dim: int, block_id: int) -> torch.Tensor:
    """Extract one block in TRUE PHYSICAL memory order (dims sorted by
    stride, descending), not the logical shape-declared dim order.

    Calling .contiguous() directly on the selected block reorders it to
    match the LOGICAL shape order, which is not the same as physical
    memory order for a layout like HND: the shape lists block_size (N)
    before heads (H), but "HND" means heads actually come before N in
    physical memory. Confirmed empirically -- dumping in logical order
    made only 256-byte windows (one head_dim run, the one dim untouched
    by the N/H swap) match the wire; nothing larger ever matched, because
    N and H were transposed relative to the real byte order NIXL
    registers and sends. Permuting to stride-descending order before
    flattening reproduces the actual physical byte sequence.
    """
    block = tensor.select(block_dim, block_id)
    phys_order = sorted(range(block.dim()), key=lambda d: -block.stride(d))
    return block.permute(*phys_order).contiguous()


def _snapshot(kv_caches: dict, block_dims: dict) -> dict:
    snap = {}
    for name, tensor in kv_caches.items():
        if name not in block_dims:
            continue
        snap[name] = _reduce_all_but(tensor, block_dims[name]).clone()
    return snap


def _dump_diff(
    kv_caches: dict,
    block_dims: dict,
    baseline: dict,
    final: dict,
    out_bin: str,
    out_manifest: str,
):
    manifest_lines = []
    n_blocks_dumped = 0
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    with open(out_bin, "wb") as fbin:
        offset = 0
        for layer_name, tensor in kv_caches.items():
            if layer_name not in block_dims:
                continue
            block_dim = block_dims[layer_name]
            newly_nonzero = final[layer_name] & (~baseline[layer_name])
            block_ids = torch.nonzero(newly_nonzero, as_tuple=True)[0].tolist()
            for block_id in block_ids:
                block = _extract_block(tensor, block_dim, block_id)
                # .numpy() doesn't support bfloat16 -- reinterpret as raw
                # bytes via uint8 first. Byte content is unaffected, this
                # is not a dtype conversion, just a reinterpretation.
                block_bytes = block.cpu().view(torch.uint8).numpy().tobytes()
                content_hash = hashlib.sha256(block_bytes).hexdigest()
                fbin.write(block_bytes)
                length = len(block_bytes)
                shape_str = "x".join(str(d) for d in block.shape)
                line = "\t".join(
                    [
                        layer_name,
                        str(block_id),
                        str(offset),
                        str(length),
                        shape_str,
                        str(block.dtype),
                        content_hash,
                    ]
                )
                manifest_lines.append(line)
                offset += length
                n_blocks_dumped += 1
    with open(out_manifest, "w") as fman:
        fman.write(
            "\t".join(
                [
                    "layer_name",
                    "block_id",
                    "offset",
                    "len",
                    "block_shape",
                    "dtype",
                    "content_hash",
                ]
            )
            + "\n"
        )
        fman.write("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))
    return n_blocks_dumped


KVHOOK_TRIGGER_DIR = os.environ.get(
    "KVHOOK_TRIGGER_DIR", os.path.join(KVHOOK_OUT_DIR, "triggers")
)
KVHOOK_POLL_S = float(os.environ.get("KVHOOK_POLL_S", 1.0))
# Fallback auto-stop if no .stop trigger shows up (safety net, not the
# primary mechanism): same default as the old fixed WINDOW_S.
KVHOOK_MAX_WINDOW_S = float(os.environ.get("KVHOOK_MAX_WINDOW_S", KVHOOK_WINDOW_S))


def _wait_for_file_with_prefix(directory: str, suffix: str, timeout: float | None = None):
    """Poll `directory` for the first file matching *<suffix>. Returns the
    matched filename's label (basename minus suffix), or None on timeout."""
    start = time.monotonic()
    while timeout is None or (time.monotonic() - start) < timeout:
        try:
            for fname in os.listdir(directory):
                if fname.endswith(suffix):
                    return fname[: -len(suffix)]
        except FileNotFoundError:
            os.makedirs(directory, exist_ok=True)
        time.sleep(KVHOOK_POLL_S)
    return None


def _capture_worker(kv_caches: dict, worker):
    """Repeatable, trigger-file-driven capture loop -- runs for the life of
    the process so multiple phases (e.g. Phase A fixed payload, Phase B
    agentic run) can each get their own baseline/window/dump without
    restarting the server in between.

    Usage from the shell, per phase, with label being any string with no
    slashes (e.g. "phaseA", "phaseB_run1"):
        touch KVHOOK_TRIGGER_DIR/<label>.start   # snapshots baseline now
        # ... fire the request ...
        touch KVHOOK_TRIGGER_DIR/<label>.stop    # snapshots final, dumps diff
    Output for each label goes to kvhook_dump_<label>.bin /
    kvhook_manifest_<label>.tsv in KVHOOK_OUT_DIR.
    If no .stop shows up within KVHOOK_MAX_WINDOW_S of the .start, dumps
    anyway using whatever changed by then (safety net, logged as such).
    """
    try:
        block_dims = {}
        for name, tensor in kv_caches.items():
            if isinstance(tensor, (tuple, list)):
                _log(f"skipping {name}: mamba (conv, ssm) tuple, not handled by capture")
                continue
            bd = _find_block_dim(tensor, getattr(worker, "num_blocks", -1))
            if bd is None:
                _log(
                    f"skipping {name}: no dim matches num_blocks="
                    f"{getattr(worker, 'num_blocks', -1)} in shape {tuple(tensor.shape)}"
                )
                continue
            block_dims[name] = bd

        os.makedirs(KVHOOK_TRIGGER_DIR, exist_ok=True)
        _log(f"waiting KVHOOK_WARMUP_S={KVHOOK_WARMUP_S}s for CUDA graph capture...")
        time.sleep(KVHOOK_WARMUP_S)
        _log(
            f"ready. watching {KVHOOK_TRIGGER_DIR} for <label>.start / <label>.stop "
            f"files, one capture per label, repeatable without restart."
        )

        while True:
            try:
                _capture_one_window(kv_caches, block_dims)
            except Exception:
                import traceback

                _log(
                    "capture window crashed (thread stays alive, next "
                    "<label>.start will still be picked up):\n"
                    + traceback.format_exc()
                )
    except Exception:
        import traceback

        _log("capture thread crashed:\n" + traceback.format_exc())


def _capture_one_window(kv_caches: dict, block_dims: dict):
    label = _wait_for_file_with_prefix(KVHOOK_TRIGGER_DIR, ".start")
    if label is None:
        return
    start_path = os.path.join(KVHOOK_TRIGGER_DIR, f"{label}.start")

    if os.environ.get("KVHOOK_ZERO_ON_START", "1") == "1":
        # Re-zero right before baseline, not just once at registration.
        # The block allocator can (and empirically does, via a LIFO
        # free-list) hand a real request blocks that CUDA-graph warmup
        # already dirtied and freed. The once-at-init zero only covers
        # memory that's never been touched again since; it does not stop
        # the allocator recycling those exact blocks into the first real
        # request, which then writes into an already-nonzero block and is
        # invisible to a zero->nonzero diff. Re-zeroing here trades one
        # full device memset per capture window (~sub-second) for a
        # baseline that's actually clean regardless of allocator history.
        # Only safe because captures are one-request-at-a-time by design
        # here; concurrent unrelated traffic during a capture window would
        # get its KV wiped too.
        _zero_kv_caches(kv_caches)

    baseline = _snapshot(kv_caches, block_dims)
    try:
        os.remove(start_path)
    except OSError:
        pass
    baseline_nonzero = sum(int(m.sum()) for m in baseline.values())
    baseline_total = sum(m.numel() for m in baseline.values())
    _log(
        f"[{label}] baseline captured over {len(baseline)} layers. "
        f"baseline nonzero blocks: {baseline_nonzero}/{baseline_total} "
        f"({100.0 * baseline_nonzero / baseline_total:.1f}%) -- if this is "
        f"already near 100%, the zero->nonzero diff will be blind to writes "
        f"into already-dirty blocks. "
        f"waiting for {label}.stop (or {KVHOOK_MAX_WINDOW_S}s safety timeout)."
    )

    stop_label = _wait_for_file_with_prefix(
        KVHOOK_TRIGGER_DIR, ".stop", timeout=KVHOOK_MAX_WINDOW_S
    )
    timed_out = stop_label is None
    if not timed_out:
        stop_path = os.path.join(KVHOOK_TRIGGER_DIR, f"{stop_label}.stop")
        try:
            os.remove(stop_path)
        except OSError:
            pass
        if stop_label != label:
            _log(
                f"[{label}] WARNING: stop trigger label '{stop_label}' != "
                f"start label '{label}'; dumping under '{label}' anyway"
            )

    final = _snapshot(kv_caches, block_dims)
    final_nonzero = sum(int(m.sum()) for m in final.values())
    final_total = sum(m.numel() for m in final.values())
    out_bin = os.path.join(KVHOOK_OUT_DIR, f"kvhook_dump_{label}.bin")
    out_manifest = os.path.join(KVHOOK_OUT_DIR, f"kvhook_manifest_{label}.tsv")
    n = _dump_diff(kv_caches, block_dims, baseline, final, out_bin, out_manifest)
    note = " (safety-timeout, no .stop seen)" if timed_out else ""
    _log(
        f"[{label}] final nonzero blocks: {final_nonzero}/{final_total} "
        f"({100.0 * final_nonzero / final_total:.1f}%). "
        f"dumped {n} newly-nonzero blocks to {out_bin}, "
        f"manifest at {out_manifest}{note}"
    )


def _zero_kv_caches(kv_caches: dict):
    """Explicitly zero the whole KV cache once, right after registration.

    Freshly allocated GPU memory (torch.empty-style) is not guaranteed to
    start at zero -- it can carry over whatever was previously in that
    device memory from earlier allocations in this process (weights,
    scratch buffers, etc). Without this, the zero->nonzero diff used by
    the capture loop can be blind from the very first request: if a block
    a real request writes into was already nonzero at baseline time (stale
    garbage, not stale *data* from a prior request -- this runs once,
    before any request is served), the diff never sees the write.
    Zeroing here happens before any inference traffic and before CUDA
    graph capture, so it's a one-time, safe, content-only mutation (NIXL
    already registered these regions by address, not content, so this
    does not invalidate that registration).
    """
    for name, tensor in kv_caches.items():
        if isinstance(tensor, (tuple, list)):
            for t in tensor:
                t.zero_()
        else:
            tensor.zero_()


def patched_register_kv_caches(self, kv_caches: dict, *args, **kwargs):
    result = _original_register_kv_caches(self, kv_caches, *args, **kwargs)

    try:
        _dump_layout(kv_caches, self)
    except Exception:
        import traceback

        _log("layout dump failed:\n" + traceback.format_exc())

    if os.environ.get("KVHOOK_ENABLE") == "1":
        if os.environ.get("KVHOOK_ZERO_ON_INIT", "1") == "1":
            try:
                _zero_kv_caches(kv_caches)
                _log("zeroed KV cache tensors post-registration (KVHOOK_ZERO_ON_INIT=1)")
            except Exception:
                import traceback

                _log("zeroing KV cache tensors failed:\n" + traceback.format_exc())
        t = threading.Thread(
            target=_capture_worker, args=(kv_caches, self), daemon=True
        )
        t.start()
        _log("capture thread started")
    else:
        _log("KVHOOK_ENABLE != 1, layout logged only, no capture")

    return result


def patch_connector():
    """Idempotent. Call before/at worker init, before register_kv_caches runs."""
    global _original_register_kv_caches, _patched_class, _already_patched
    if _already_patched:
        _log("patch_connector() called again, already patched, skipping")
        return

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )

    _original_register_kv_caches = NixlBaseConnectorWorker.register_kv_caches
    NixlBaseConnectorWorker.register_kv_caches = patched_register_kv_caches
    _patched_class = NixlBaseConnectorWorker
    _already_patched = True
    _log(
        "patched NixlBaseConnectorWorker.register_kv_caches "
        f"(covers NixlPullConnectorWorker; NixlPushConnectorWorker overrides "
        f"this method itself and is NOT covered by this patch)"
    )


if __name__ == "__main__":
    patch_connector()
