# Reuse-aware (segmented) eviction as a defense against the eviction channel

Date: 2026-08-17. Testbed: spark-523e, flat single-process `vllm serve` (no
dynamo), model `Qwen3-8B-unsloth-bnb-4bit`, vLLM 0.25.1,
`--gpu-memory-utilization 0.4`. GPU KV cache size: 298,880 tokens (Step 3/5),
296,128 tokens (Step 4 undefended re-launch) — both from the worker's own
startup log, ~0.9% apart run-to-run as expected for this hardware. Flood
sized to 1.25x capacity (373,600 tokens) throughout, matching the undefended
baseline in `results/cache_salt_defense_20260816T032207Z/`. n=5 trials for
the eviction arms (Step 3), n=3 for the adaptive-attacker measurement (Step
5, reduced from 5 to fit the Aug 19 time box — see note there).

## Bottom line

**The reuse-aware eviction defense closes the eviction channel against a
naive (single-pass) flooding attacker — the exact attack that produced the
project's +351 ms undefended result drops to statistical noise (mean +0.24
to -0.98 ms across all four arms, none distinguishable from a true no-flood
control). It does NOT close the channel against an adaptive attacker who
simply re-reads each flood prompt once: that trivial change fully restores
the +345 ms effect at a measured 2.0x request cost.** The defense is real
and correctly implemented (verified against undefended on both cache hit
rate and task accuracy — see Step 4), but as a security boundary it raises
attacker cost by a factor of two, not by an order of magnitude. Report it
as a cost-raising measure, not a channel closure, once an adaptive attacker
is in scope.

## Step 1: where vLLM selects blocks for eviction and records hits

Read before writing any code (`vllm/v1/core/block_pool.py`,
`kv_cache_utils.py`, `kv_cache_manager.py`, `single_type_kv_cache_manager.py`,
`kv_cache_coordinator.py` in the pinned venv):

- `KVCacheBlock` (`kv_cache_utils.py:118`) has **no reuse/hit counter** —
  only `block_id`, `ref_cnt`, `_block_hash`, and free-list pointers. The
  brief's assumption was correct.
- **Hit recording**: `BlockPool.touch(blocks)` (`block_pool.py:597`), called
  from `single_type_kv_cache_manager.py:220` on `new_computed_blocks` — the
  blocks that "just hit the prefix cache." Exactly the point needed.
- **Eviction selection**: `BlockPool.get_new_blocks()` (`block_pool.py:542`)
  calls `free_block_queue.popleft_n()` on a single doubly-linked FIFO
  (`FreeKVCacheBlockQueue`) that is strict LRU — least-recently-freed at the
  head. `.remove(block)` gives O(1) removal from the middle of that list, a
  primitive already used by `touch()` and reused by this patch instead of
  reimplementing the linked list.
- One `BlockPool` instance total (`kv_cache_coordinator.py:91`), shared
  across KV-cache groups on this (non-hybrid) model — one place to patch.

## Step 2: the patch

Segmented LRU (SLRU), two segments: **probation** (reuse_count == 0) and
**protected** (reuse_count >= 1) — the shape the brief suggested, and the
same shape as MySQL InnoDB's buffer-pool young/old sublist split. New
content always enters probation; a block only earns protected status (and
therefore later eviction priority) by actually being reused. On eviction,
probation is drained before protected is ever touched.

Implementation keeps vLLM's own `free_block_queue` as the sole physical
store of free blocks (so `get_num_free_blocks()`/`get_usage()` stay exactly
correct with zero shadow bookkeeping); a lightweight `OrderedDict` of
block_ids tags which of those free blocks are probation-eligible, and
eviction drains that index first via `.remove()`, falling back to native
`.popleft()` only once probation is empty. Tier-0 (never-hashed) blocks keep
vLLM's original "evicted before anything" priority — verified this doesn't
get inverted by probation (a bug caught and fixed before the first live
run; see `kvdefense_patch.py`'s tier-0 head-draining loop).

Patched via `~/LG2026/KVDEFENSE/{kvdefense_bootstrap.py,kvdefense_patch.py}`
plus a `.pth` file in `venv_dynamo_pd`'s site-packages
(`001_kvdefense.pth`), following the exact mechanism KVHOOK already uses on
this host. Gated behind `KVDEFENSE_ENABLE=1`; unset, the .pth no-ops and the
identical binary runs completely unpatched (verified: `hasattr(pool,
'_kvdefense_reuse_count')` is `False` with the var unset). No file under
`site-packages/vllm/` was touched. Unit-tested in isolation before any live
run (`test_kvdefense.py`, ALL ASSERTIONS PASSED) — a synthetic 6-block pool
confirms a once-touched block is evicted strictly after four never-touched
ones, reversing raw recency order exactly as designed.

## Step 3: security evaluation (defended vs. +351 ms undefended baseline)

Same 4-arm harness (`cache_salt_defense.py`), same capacity fraction, same
n=5, same flat mode as the undefended run.

| Arm | Config | Undefended (2026-08-16) | **Defended** | vs. no-flood control (defended) |
|---|---|---|---|---|
| A | no salt anywhere | +351.2 ms | **+0.24 ms** | p=0.30 |
| B | victim salted, attacker not | +350.6 ms | **-0.60 ms** | p=0.79 |
| C | victim + attacker salted differently | +344.6 ms | **-0.33 ms** | p=0.22 |
| D | no flood (control) | -0.16 ms | -0.98 ms | — |

All three defended flood arms are statistically indistinguishable from a
true no-flood control (p=0.22-0.79) and from each other — the channel is
closed for this attack shape. `self_hit_all_ok=True` in all 20 trials: the
victim's own salted/unsalted re-probe was always a hit, so this is not a
functional break, the same clean result E4's original evaluation had.

Mechanistically this is exactly the intended effect: the protocol's own
"rewarm" step (a genuine hit on the victim's anchor, right before the flood
starts) graduates the victim's blocks to protected before a single flood
request is sent. The flood — by construction all single-use nonce content —
never leaves probation, so it evicts itself before it can reach the
victim's protected blocks.

## Step 4: cost evaluation (does the defense hurt normal serving?)

**banking_sim** (12 sessions, concurrency 4, `--max-tokens 100`, unmodified
per project convention):

| | Defended | Undefended |
|---|---|---|
| Server-side prefix-cache hit rate | 93.69% | 93.68% |
| Mean per-turn latency (non-streaming, capped at 100 tokens — **not** TTFT, banking_sim's LangChain client doesn't stream) | 13,041 ms | 12,968 ms |

Hit rate within 0.01 points, latency within 0.6% (well inside each run's
own ~40-47 ms stdev on a ~13 s mean). The ~13 s figure itself reflects real
contention on this shared, concurrently-used GPU and Qwen3's default
"thinking" mode consuming the full 100-token budget on `<think>` reasoning
before any answer — confirmed via an isolated uncontended single-request
probe (3.3 s for the same 100 tokens) and reproduced identically on a
freshly-launched undefended server, so it is not an artifact of the defense
or of residual load from Step 3.

**GSM8K** (n=100, temperature 0, seed fixed, `max_tokens=512`):

| | Defended | Undefended |
|---|---|---|
| Accuracy | 21/100 (21.0%) | 21/100 (21.0%) |

**Exact match**, including identical running-accuracy checkpoints at every
20-sample interval throughout the run. 99/100 responses were byte-identical
between conditions; the one divergence (idx 61) was wrong under both
conditions regardless, consistent with ordinary batch-composition
floating-point nondeterminism in fused attention kernels — a known,
well-documented phenomenon unrelated to which physical block index a KV
entry happens to live at, not a defect in the patch. Task utility did not
move, as predicted: the defense only changes which idle blocks get evicted,
never what gets computed for a live sequence.

(21% is low for an 8B model on GSM8K in absolute terms — almost certainly
Qwen3's reasoning mode eating the 512-token budget before reaching the
`#### <answer>` format, a prompting artifact, not a defense-utility
question. Since both conditions are affected identically, it doesn't
compromise the equivalence claim this step exists to test.)

## Step 5: the adaptive attacker

The obvious bypass, quantified: an attacker re-reads each flood prompt once
(1 write + 1 immediate re-read) before moving to the next, inflating every
flood block's reuse count to >= 1 and graduating it into the same protected
segment the victim's blocks live in — at which point protected-tier
eviction is plain LRU again, identical to undefended.

| | Naive flood (undefended-equivalent) | Adaptive flood (reads_per_prompt=2) |
|---|---|---|
| Mean delta | +351.2 ms (undefended) / +0.24 ms (defended) | **+345.1 ms** |
| Requests for the same 373,600-token target | 216 | **432** |
| Cost multiplier | 1x | **2.0x** |

n=3 trials (deltas: 347.5, 348.5, 339.3 ms — tight and consistent),
`self_hit_all_ok=True` throughout. Reduced from the planned n=5 to fit the
Aug 19 time box; each adaptive trial floods ~432 requests vs. 216 for the
undefended/defended-naive arms, so this step alone cost roughly what all of
Step 3 did. The three trials already agree to within 9 ms of each other, so
the reduced n does not put the headline number in doubt, but it is a
smaller sample than the rest of this evaluation and is reported as such.

**This measured multiplier is a straightforward, worst-case-for-the-attacker
instantiation of the stated bypass (touch every flood block once, no
attempt to find the minimum), not a search for the cheapest possible
defeat.** A smarter attacker likely needs to protected-ify only enough
flood content to outweigh the victim's position in the protected segment,
not all of it — the true minimal multiplier could be lower than 2.0x. That
characterization is out of scope for the time box available here and is
flagged as follow-up work rather than assumed.

**Per the brief's own framing: a multiplier of 2.0x is a real but modest
cost increase, not a strong barrier.** The defense should be reported as
raising attacker cost by roughly 2x against the specific bypass tested here
— not as closing the channel — once an adaptive attacker is in the threat
model. Against a naive attacker (Step 3), it does close the channel.

## Files in this directory

- `step3_eviction_arms/` — 4-arm re-run under the defense: `arm_{A,B,C,D}.json`,
  `calibration.json`, `significance.json`, `run_metadata.json` (serving mode,
  cache size, gpu-memory-utilization, vLLM version, nvidia-smi snapshots).
- `step4_cost_eval/` — `banking_sim_{defended,undefended}.jsonl` (raw
  per-request records), `banking_sim_summary_{defended,undefended}.json`,
  `gsm8k_{defended,undefended}.json` (summary + all 100 raw responses each).
- `step5_adaptive_attacker/adaptive_attacker_*/` — `calibration.json`,
  `adaptive_attacker_result.json`, `run_metadata.json`.
- Patch source: `~/LG2026/KVDEFENSE/kvdefense_patch.py`,
  `kvdefense_bootstrap.py` (not copied into this results dir; referenced by
  path since it lives alongside KVHOOK per project convention).
