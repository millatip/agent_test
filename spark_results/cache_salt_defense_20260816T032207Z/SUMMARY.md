# cache_salt defense evaluation -- eviction channel

Date: 2026-08-16 (03:22-03:49 UTC). Testbed: spark-523e, flat single-process
`vllm serve` (no dynamo), model `Qwen3-8B-unsloth-bnb-4bit`, vLLM 0.25.1,
`--gpu-memory-utilization 0.4`. GPU KV cache size (from worker startup log,
`worker_startup.log` in this directory): **296,880 tokens**. Flood sized to
1.25x capacity = 371,100 tokens per flood (the fraction that reliably
produced the eviction jump across all four prior flat E0 runs in
`results/flat_*/e0.json`). n=5 trials per arm, seed=20260816.

## Bottom line

**`cache_salt` does not close the eviction channel on this testbed, at this
cache size, under this flood pattern.** An attacker who floods the shared KV
cache with unrelated, unsalted-or-differently-salted content evicts a
salted victim's cached prefix just as effectively as an attacker who floods
with no salting scheme in play at all. Salting changes which blocks are
*reusable*; it does not change how many blocks exist, and eviction only
depends on the latter.

## Step 1: is cache_salt honored on this testbed?

Confirmed from source (`vllm/entrypoints/openai/completion/protocol.py` and
`chat_completion/protocol.py` in the pinned venv): `cache_salt` is a
top-level field on both `/v1/completions` and `/v1/chat/completions`,
injected into the prefix-cache hash key only for the first block
(`kv_cache_utils.py`). Live-verified in `step1_verify_cache_salt.json`:
different salts on the identical prompt -> second probe stayed at
miss-level latency; same salt twice -> second probe dropped to hit-level
latency. `"honored": true`.

Note: this testbed's `ai-dynamo` 1.2.1 frontend (the disaggregated
prefill/decode deployment normally running on this node) rejects any
request containing `cache_salt` with `400 Unsupported parameter(s):
cache_salt` -- traced to the compiled Rust core (`dynamo/_core.abi3.so`),
not to vLLM (vLLM's own Python request models and dynamo's own
`vllm_processor.py` both already support and forward the field). That
deployment was stopped for this experiment (idle, 13 total requests logged
since Aug 6; exact prior launch commands preserved in
`../cache_salt_defense_launch_notes/prior_disagg_deployment_cmdlines.txt`)
and a native flat `vllm serve` instance was started in its place, which is
the only path on this testbed that lets `cache_salt` reach vLLM at all. This
is a testbed/proxy-layer finding about ai-dynamo 1.2.1, separate from and
not a substitute for the eviction-channel result below.

## Step 2/3: four-arm result

| Arm | Victim salt | Attacker salt | Flood | n | Mean delta (post-flood TTFT − pre-flood hit TTFT) | stdev | Paired t-test vs D | Self-hit intact? |
|---|---|---|---|---|---|---|---|---|
| A | none | none | yes | 5 | **+351.2 ms** | 3.4 | t=281.0, df=4, p=9.6e-10 | yes (5/5) |
| B | victim-\<random\> | none | yes | 5 | **+350.6 ms** | 2.4 | t=291.3, df=4, p=8.3e-10 | yes (5/5) |
| C | victim-\<random\> | attacker-\<random\> | yes | 5 | **+344.6 ms** | 6.4 | t=121.5, df=4, p=2.8e-08 | yes (5/5) |
| D | victim-\<random\> | -- | no | 5 | −0.16 ms | 2.1 | -- (control) | yes (5/5) |

- **A reproduces the known effect**: unsalted flood evicts an unsalted
  victim, +351 ms, consistent with the earlier +64.8 ms energy-channel
  result in direction and now measured directly against a same-arm
  no-flood baseline rather than an external control.
- **B (victim salts, attacker floods unsalted) shows eviction persists**,
  statistically indistinguishable from A (350.6 ms vs 351.2 ms, well within
  each other's noise) and hugely different from D.
- **C (victim and attacker use different, unrelated salts) shows the same
  thing**: 344.6 ms, again indistinguishable from A/B. This is the cleanest
  arm -- attacker cache-salt is genuinely irrelevant to the victim's cache
  entries, and the victim's own salt still doesn't protect it.
- **D confirms the baseline**: with no flood, the victim's own re-probe
  after the same wall-clock gap shows no delay (mean -0.16 ms), so A/B/C's
  +345-351 ms are attributable to the flood, not to elapsed time or
  measurement drift.
- **Self-hit sanity check passed in every trial of every arm** (20/20):
  the victim's own re-probe with its own salt, immediately after
  establishing the anchor and before any flooding, was always classified
  as a cache HIT (rewarm TTFT ~145-150 ms, well under tau=320 ms). This
  rules out the "salting is just broken" explanation for B/C -- the salt
  mechanism itself works exactly as documented; it simply doesn't gate
  eviction, only reuse-matching.

## Scope

This result is narrow by design: one documented defense (`cache_salt`),
tested against one channel (capacity-driven eviction, not cache-presence
detection, which `cache_salt` genuinely does defeat), on this testbed
(single GB10 node, `--gpu-memory-utilization 0.4`, ~296,880-token cache),
at one flood pattern (1.25x capacity, 2000-token unique filler prompts). It
does not claim to characterize `cache_salt` against eviction at other cache
sizes, other flood shapes, or with a resource-quota-aware scheduler in
front of vLLM (which could, in principle, throttle a single attacker's
prompt-token budget and indirectly limit flood volume -- a mitigation
outside `cache_salt` itself, not evaluated here).

## Files in this directory

- `step1_verify_cache_salt.json` -- Step 1 smoke test (field honored, both endpoints).
- `calibration.json` -- hit/miss battery used to derive tau=320.0 ms.
- `arm_A.json`, `arm_B.json`, `arm_C.json`, `arm_D.json` -- full per-trial raw data.
- `significance.json` -- paired t-test results (A/B/C vs D).
- `run_metadata.json` -- serving mode, capacity, gpu-memory-utilization, vLLM version, nvidia-smi snapshots at start/end.
- `worker_startup.log` -- the flat server's own startup log (source of the 296,880-token capacity figure).
