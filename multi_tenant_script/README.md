# KV Cache Extraction & Verification Toolkit

Validated methodology for confirming that KV cache tensors transferred between
NVIDIA Dynamo + SGLang/vLLM prefill and decode nodes (NixlConnector over TCP)
are recoverable from a passive network capture, with byte-exact ground truth.

Established this session (2026-06-17) on Qwen3-8B-unsloth-bnb-4bit,
36 layers, 8 KV heads, head_dim=128, block_size=16, bf16 KV cache.

## What this confirms

- KV cache tensors cross the wire in plaintext (no encryption observed at
  the connector level or in linked binary symbols).
- Transfer order is strictly layer-sequential, K immediately followed by V
  within each layer (layer0 K, layer0 V, layer1 K, layer1 V, ...). No
  cross-layer interleaving, no per-head or per-token reordering observed.
- Captured bytes match known-content tensors exactly at the byte level
  (not just statistically), verified via source-level instrumentation
  rather than blind offset-guessing on the wire capture.

## Why source-level instrumentation, not just tcpdump

Blind reverse-engineering of wire byte alignment (guessing offsets, anchoring
on zero-runs, etc.) repeatedly failed to produce sane decoded float values.
The reliable method is: patch the connector to dump real, known-content
tensor data at the moment it's written, with exact byte offsets and shapes
recorded in a manifest — then search for those known bytes in the capture.
This is a known-plaintext approach, not inference.

## Known pitfalls already worked around in these scripts

- **numpy can't represent bfloat16.** Any `.numpy()` call on a bf16 tensor
  raises `TypeError: Got unsupported ScalarType BFloat16`. Fixed by
  `.view(torch.uint8)` before `.numpy()` for byte-level access.
- **Touching the GPU during CUDA graph capture crashes the engine**
  (`CUDA error: operation not permitted when stream is capturing`).
  Fixed by a fixed warmup delay (default 60s) before any tensor read.
- **Dumping whole tensors is enormous and useless** (~36GB for one request,
  almost entirely unused cache slots). Fixed by checking per-block
  "any nonzero" status and only dumping blocks that changed between a
  baseline and final snapshot.
- **TCP stream index is not stable across capture files.** Always
  re-identify the real data stream per-capture (largest payload), don't
  assume it's the same index as a previous run.

## Environment setup (per pod recreation)

The pod's `/workspace` (and venvs in it) survive pod recreation, but running
processes, IPs, and infra (etcd/NATS) do not. Check before assuming any of
this is already up.

### 0a. Confirm IPs (pod recreation reassigns these)

On millaone:
```bash
hostname -I
```
On millatwo:
```bash
hostname -I
```
Update every command below that references an IP if these have changed
from your last session.

### 0b. Confirm the venv and tools exist

```bash
ls /workspace/venv_dynamo_pd/bin/activate   # should exist
source /workspace/venv_dynamo_pd/bin/activate
pip show ai-dynamo                          # confirm version (this session: 1.0.2)
python3 -c "import vllm; print(vllm.__version__)"   # this session: 0.16.0
which tcpdump tshark                        # install via apt-get if missing
```

If `tcpdump`/`tshark` are missing:
```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y tcpdump tshark
```
(`tshark` may prompt about non-root packet capture; `DEBIAN_FRONTEND=noninteractive`
avoids the hang, defaulting to deny — re-run interactively once if you need
to answer "yes".)

### 0c. Start etcd and NATS (millaone only, once per pod lifetime)

```bash
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://<MILLAONE_IP>:2379 \
     --listen-peer-urls http://0.0.0.0:2380 \
     --initial-advertise-peer-urls http://<MILLAONE_IP>:2380 \
     --initial-cluster default=http://<MILLAONE_IP>:2380 \
     --data-dir /tmp/etcd-data \
     > /tmp/etcd.log 2>&1 &

nats-server -p 4222 > /tmp/nats.log 2>&1 &

sleep 2
curl -s http://127.0.0.1:2379/health   # expect {"health":"true",...}
```

If etcd fails with an `--initial-cluster` mismatch error, it's almost always
because `<MILLAONE_IP>` doesn't match across `--advertise-client-urls`,
`--initial-advertise-peer-urls`, and `--initial-cluster` — all three must
use the same current IP.

## Starting the serving stack

Bring up in this order: etcd/NATS (above) -> prefill worker -> frontend ->
decode worker. If decode comes up before prefill has registered, restart it.

### 1. Patch the connector (once per worker restart)

```bash
./01_patch_connector.sh /workspace/venv_dynamo_pd
```

Restores from `.bak` first if a previous patch is present, so safe to re-run.
To remove the patch entirely: `cp nixl_connector.py.bak nixl_connector.py`
(path printed by the script).

### 2. Start the prefill worker (millaone)

```bash
source /workspace/venv_dynamo_pd/bin/activate
python3 -m dynamo.vllm \
  --model /workspace/models/Qwen3-8B-unsloth-bnb-4bit \
  --tensor-parallel-size 1 \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --discovery-backend etcd \
  --request-plane tcp 2>&1 | tee /tmp/prefill_with_hook.log
```

Wait for the line:
```
[KVHOOK] BASELINE captured. You have 30s to fire your ONE request now.
```

### 2b. Start the frontend (millaone, separate terminal from prefill)

Only needed if not already running — the frontend doesn't touch
`nixl_connector.py`, so it does NOT need restarting when you re-patch or
restart the prefill worker.

```bash
source /workspace/venv_dynamo_pd/bin/activate
export ETCD_HOST=127.0.0.1
export ETCD_PORT=2379
export DYN_NATS_URL="nats://127.0.0.1:4222"

python3 -m dynamo.frontend \
  --http-host 0.0.0.0 \
  --http-port 8000 \
  --model-name /workspace/models/Qwen3-8B-unsloth-bnb-4bit \
  --discovery-backend etcd \
  --request-plane tcp
```

### 2c. Start the decode worker (millatwo)

Unpatched — the dump hook only needs to run on the prefill side, since
that's where the KV cache is computed and registered first. Restart only
if it's not already running or if millaone's IP changed.

```bash
source /workspace/venv_dynamo_pd/bin/activate
export PREFILL_IP=<MILLAONE_IP>
export ETCD_HOST=$PREFILL_IP
export ETCD_PORT=2379
export ETCD_ENDPOINTS="http://$PREFILL_IP:2379"
export DYN_NATS_URL="nats://$PREFILL_IP:4222"

python3 -m dynamo.vllm \
  --model /workspace/models/Qwen3-8B-unsloth-bnb-4bit \
  --tensor-parallel-size 1 \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --discovery-backend etcd \
  --request-plane tcp
```

### 2d. Sanity check before running the capture toolkit

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "/workspace/models/Qwen3-8B-unsloth-bnb-4bit", "prompt": "test", "max_tokens": 10}'
```
Expect a real completion with no errors before proceeding to step 3 below.

### 3. In a second terminal, capture + fire the request

```bash
./02_capture_and_request.sh <decode_node_ip> 8000 /tmp/nixl_capture_001.pcap
```

This starts tcpdump, fires one completions request with a fresh unique
marker (auto-generated, no risk of cache-hit contamination), and stops the
capture. The marker is saved alongside the pcap as `<name>.marker.txt`.

### 4. Wait for the dump

Back in the prefill terminal, wait for:
```
[KVHOOK] dumped N bytes total to /tmp/kvhook_dump.bin
```

### 5. Analyze

```bash
python3 03_analyze_capture.py /tmp/nixl_capture_001.pcap \
    /tmp/kvhook_dump.bin /tmp/kvhook_dump_manifest.txt
```

Reports which TCP stream carried the transfer, which dumped blocks were
found on the wire, and their order — should reproduce the layer-sequential,
K-then-V pattern. Any deviation is itself worth investigating, not assumed
to be a bug.

## Repeating for a dataset (e.g. inversion MLP training data)

For each sample: restart the prefill worker (clears KV cache state, ensures
a clean baseline), repeat steps 2-5 with a new pcap filename. The 60s warmup
+ 30s window per run means each sample takes ~90s minimum — budget
accordingly for dataset size. Consider lowering `KVHOOK_WARMUP_S` once you've
confirmed cudagraph capture timing is consistent on this hardware (was ~11s
in observed runs; 60s default is conservative headroom, not a measured
requirement).

## Known open question

Whether the ~131KB region size observed between per-layer K/V boundaries in
this session's capture corresponds exactly to one block's worth of transfer
or includes additional in-flight blocks/handshake overhead was not fully
resolved — the first K-to-V gap (136,584 bytes) was larger than subsequent
steady-state gaps (~131,450 bytes), suggesting connection-setup overhead in
the first transition. Worth re-deriving precisely if exact per-region byte
accounting matters for the paper.

## Multi-tenant isolation test (new, 2026-06-18)

See `README_MULTITENANT.md` for a separate set of scripts (04-06) that
extend this toolkit to test whether concurrent multi-tenant requests are
isolated onto separate connections or multiplexed onto a shared one —
an open question not addressed by prior published work (Shadow-in-the-
Cache/KV-Cloak included). Builds directly on the validated 01-03 scripts
above; read that file before running 04-06.
