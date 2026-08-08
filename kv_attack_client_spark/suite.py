#!/usr/bin/env python3
"""KV-cache timing attack suite (E0-E5) for the DGX Spark testbed.

Self-contained for E0-E4: no import from attacker_probe/ or any sibling
kv_attack_client/ (A6000) config or calibrated constant. Every threshold
used here (tau, D, midpoints) is derived at runtime from this run's own
measurements -- never hardcoded from a previous testbed.

E5 is the one exception to self-containment: it explicitly reuses
banking_sim/'s existing ReAct banking agent as the victim (per the
change request that added it -- "do not rewrite it"), so it depends on
banking_sim/, langgraph, and langchain-openai being importable. If that
import fails (e.g. this file was copied somewhere without banking_sim/
alongside it), E5 is skipped and recorded as not-evaluated; E0-E4 still
run and save normally.

Threat model: API-only client, no server-side access. Endpoint is
identical across all three serving arms (flat / disagg_tcp / disagg_rdma)
-- this script does not change behavior based on --arm beyond recording
it in the output; the arm only changes what's running behind the URL.

Scope limit: no tensor interception/wire-capture/co-tenant-memory access
anywhere in this file. Not available on this testbed (DAC topology,
ptrace_scope=1, GB10 unified memory with no GPUDirect) and out of scope
for this suite regardless -- E5 is timing-based session tracking against
an agentic workload, not tensor recovery.

Usage:
    python suite.py --arm flat --base-url http://localhost:8000/v1
    python suite.py --arm disagg_rdma --base-url http://10.126.36.140:8000/v1

Style notes (matching the existing suite's conventions):
  - Synchronous throughout (httpx.Client, not async) for E0-E4. E5 bridges
    into banking_sim's async LangGraph agent via asyncio.run() per turn,
    since reusing it as-is (not rewriting it) means reusing its async API.
  - All tunables live in Config below -- no magic numbers inline.
  - A failed probe is recorded as {"ttft": None, "ok": False, ...} and
    the run continues; nothing raises out of an experiment function.
"""

import argparse
import glob
import json
import os
import random
import re
import statistics
import string
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


# --------------------------------------------------------------------------
# Config -- every tunable lives here.
# --------------------------------------------------------------------------

@dataclass
class Config:
    # Target
    base_url: str = "http://localhost:8000/v1"
    model: str = "/home/s3lab-spark/LG2026/models/Qwen3-8B-unsloth-bnb-4bit"
    api_key: str = "EMPTY"
    arm: str = "flat"  # flat | disagg_tcp | disagg_rdma
    request_timeout: float = 60.0

    # Warmup (before E0, samples discarded)
    warmup_requests: int = 5

    # E0 -- cache capacity reconnaissance
    e0_anchor_words: int = 150
    e0_warm_probe_repeats: int = 5
    # Sweep is in FRACTIONS OF MEASURED CACHE CAPACITY (tokens), not a
    # fixed prompt count -- a fixed count silently under-floods once
    # capacity is known (300 prompts of ~195 tokens is ~20% of a
    # 289,968-token cache; eviction was never going to trigger).
    e0_flood_fraction_sweep: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    # Long flood prompts so each fraction's token target is reached in a
    # manageable request count (~217 requests at 1.5x a ~290K-token cache
    # with ~2000-token prompts, vs. thousands at short-prompt granularity).
    e0_flood_target_tokens_per_prompt: int = 2000
    e0_max_flood_requests_per_point: int = 5000  # safety cap, independent of the token target
    e0_flood_progress_every: int = 25  # print progress every N flood requests within a sweep point
    # KV cache capacity in tokens. Auto-detected from the worker log if
    # available (see parse_worker_log); otherwise must come from
    # --kv-cache-tokens. No default guess -- an unknown capacity means
    # the sweep can't be sized correctly, so this stays None until set.
    e0_capacity_tokens: Optional[int] = None

    # E1 -- timing characterization. ~2000 tokens, not ~195: at 195 tokens
    # prefill is small relative to the ~100ms fixed floor (measured
    # intercept), so the hit/miss gap is mostly overhead, not cached work
    # -- technically real but not representative of the regime the attack
    # targets, and not comparable to results at realistic prompt lengths.
    e1_n_hit: int = 10
    e1_n_miss: int = 10
    e1_hit_target_tokens: int = 2000
    e1_miss_target_tokens: int = 2000
    e1_length_sweep_tokens: Tuple[int, ...] = (500, 1000, 2000, 4000)
    e1_length_sweep_repeats: int = 3
    # Rough English words-per-token calibration for sizing prompts toward a
    # nominal token target. Only affects how close we land to the label;
    # the ms/token fit itself uses the REAL measured prompt_tokens, so
    # miscalibration here doesn't bias the fit, only the sweep's x-spacing.
    words_per_token_estimate: float = 0.75

    # E2 -- cache presence detection. Must match E1's prompt length: tau
    # is length-dependent (prefill time scales with tokens), so a tau
    # calibrated at ~2000 tokens is meaningless against short prompts --
    # both hit and miss would land far under it, since even a genuine
    # miss at ~195 tokens is faster than a hit at ~2000 tokens.
    e2_n_hit: int = 15
    e2_n_miss: int = 15
    e2_target_tokens: int = 2000

    # E3 -- prompt content inference
    e3_n_templates: int = 20
    e3_template_min_words: int = 20
    e3_template_max_words: int = 1500
    e3_n_live_observations: int = 20

    # E4 -- defense resistance (post-hoc perturbation of E2's raw data;
    # no new live requests -- see run_e4 docstring).
    e4_jitter_ms: Tuple[float, ...] = (0, 10, 25, 50, 100, 200)
    e4_threshold_shift: Tuple[float, ...] = (-0.15, 0.0, 0.15)
    e4_rate_limit_delay_s: Tuple[float, ...] = (0.0, 0.5, 1.0)
    e4_jitter_trials: int = 200  # resamples per jitter level (cheap: reuses E2 data, no HTTP)

    # E5 -- agentic WORM workload (victim = banking_sim's real ReAct agent)
    e5_n_turns: int = 10
    e5_victim_max_tokens: int = 100  # matches banking_sim.main's own default cap
    e5_victim_chk_account: str = "CHK-9001"
    e5_victim_sav_account: str = "SAV-9002"

    # Output
    output_root: str = "results"

    # Optional best-effort metadata sources (only meaningful when run on
    # the actual serving host(s); null/empty when run remotely).
    worker_log_glob: str = "/tmp/*with_hook*.log"


# --------------------------------------------------------------------------
# Unique-content generator (nonce + random words -- guarantees no two
# prompts anywhere in a run share a prefix, and nothing is ever reused).
# --------------------------------------------------------------------------

_WORD_POOL = (
    "system network protocol latency throughput packet router switch "
    "gateway firmware kernel process thread memory cache buffer stack "
    "queue socket stream cipher token session cookie header payload "
    "endpoint cluster shard replica quorum ledger audit invoice receipt "
    "vendor supplier contract clause statute regulation compliance policy "
    "quarter forecast budget revenue expense margin equity asset liability "
    "harbor voyage cargo manifest customs tariff border checkpoint patrol "
    "orchard harvest irrigation drought reservoir aquifer sediment erosion "
    "canyon plateau ridge summit glacier tundra prairie wetland estuary "
    "reactor turbine generator transformer capacitor resistor inductor "
    "circuit voltage current resistance frequency amplitude wavelength "
    "spectrum isotope neutron proton electron molecule enzyme protein "
    "genome mutation allele chromosome organism habitat ecosystem biome "
    "sonnet ballad prologue epilogue chapter verse stanza metaphor irony "
    "quartet symphony concerto sonata rehearsal ensemble conductor tempo"
).split()


def unique_prompt(rng: random.Random, target_words: int) -> str:
    nonce = uuid.uuid4().hex
    words = rng.choices(_WORD_POOL, k=max(1, target_words))
    return f"nonce-{nonce} " + " ".join(words)


def words_for_token_target(cfg: Config, target_tokens: int) -> int:
    return max(1, round(target_tokens * cfg.words_per_token_estimate))


# --------------------------------------------------------------------------
# Core probe primitive
# --------------------------------------------------------------------------

def probe(client: httpx.Client, cfg: Config, prompt_text: str, max_tokens: int = 1) -> Dict[str, Any]:
    """One streamed /v1/completions probe. TTFT = wall-clock to first
    streamed data chunk (not full completion). Never raises -- failures
    come back as {"ttft": None, "ok": False, "error": ...} and the caller
    just continues."""
    payload = {
        "model": cfg.model,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    start = time.perf_counter()
    ttft_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None

    try:
        with client.stream(
            "POST", f"{cfg.base_url}/completions", json=payload, headers=headers, timeout=cfg.request_timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
    except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001 -- probes must never crash the run
        return {"ttft": None, "prompt_tokens": None, "ok": False, "error": str(exc)}

    if ttft_ms is None:
        return {"ttft": None, "prompt_tokens": prompt_tokens, "ok": False, "error": "no data chunks received"}
    return {"ttft": round(ttft_ms, 3), "prompt_tokens": prompt_tokens, "ok": True}


# --------------------------------------------------------------------------
# E0 -- cache capacity reconnaissance
# --------------------------------------------------------------------------

def resolve_capacity_tokens(cfg: Config, cli_kv_cache_tokens: Optional[int]) -> int:
    """Cache size if available from the worker log; otherwise the CLI
    --kv-cache-tokens value. Raises if neither is available -- E0's sweep
    can't be sized correctly against an unknown capacity, so this is a
    hard failure rather than a silent guess."""
    detected = parse_worker_log(cfg).get("gpu_kv_cache_size_tokens")
    if detected:
        return detected
    if cli_kv_cache_tokens:
        return cli_kv_cache_tokens
    raise RuntimeError(
        "Could not determine KV cache capacity: not found in worker log "
        f"(glob={cfg.worker_log_glob}) and --kv-cache-tokens not given. "
        "E0's sweep requires a known capacity to size fractions against."
    )


def run_e0(client: httpx.Client, cfg: Config, rng: random.Random, capacity_tokens: int) -> Dict[str, Any]:
    """Anchor established once. Hit signal confirmed (cold, then repeated
    warm). Then, for each fraction of measured cache capacity, independently:
    re-warm anchor -> flood fraction*capacity_tokens worth of fresh
    never-reused long prompts with ZERO anchor touches -> single post-flood
    anchor probe. Each fraction is a fresh trial, not cumulative -- avoids
    the interim-recheck-masks-eviction failure mode."""
    anchor = unique_prompt(rng, cfg.e0_anchor_words)

    cold = probe(client, cfg, anchor)
    warm_probes = [probe(client, cfg, anchor) for _ in range(cfg.e0_warm_probe_repeats)]
    warm_ttfts = [p["ttft"] for p in warm_probes if p["ok"]]
    warm_mean = statistics.mean(warm_ttfts) if warm_ttfts else None
    speedup = (cold["ttft"] / warm_mean) if (cold["ok"] and warm_mean) else None

    # Reference band for "TTFT jumps" -- derived from THIS run's own cold
    # vs warm measurements, not any prior testbed's constant.
    jump_midpoint = ((cold["ttft"] + warm_mean) / 2.0) if (cold["ok"] and warm_mean) else None

    flood_words = words_for_token_target(cfg, cfg.e0_flood_target_tokens_per_prompt)

    sweep = []
    jump_at_fraction = None
    jump_at_tokens = None
    for fraction in cfg.e0_flood_fraction_sweep:
        target_tokens = round(fraction * capacity_tokens)
        print(f"[E0] fraction={fraction} (target={target_tokens} tokens) -- re-warming anchor, then flooding...")
        rewarm = probe(client, cfg, anchor)

        flood_tokens_actual = 0
        flood_request_count = 0
        flood_requests_ok = 0
        while flood_tokens_actual < target_tokens and flood_request_count < cfg.e0_max_flood_requests_per_point:
            flood_prompt = unique_prompt(rng, flood_words)
            result = probe(client, cfg, flood_prompt)
            flood_request_count += 1
            if result["ok"]:
                flood_requests_ok += 1
                flood_tokens_actual += result.get("prompt_tokens") or 0
            if flood_request_count % cfg.e0_flood_progress_every == 0:
                print(f"[E0]   ...{flood_request_count} flood requests, {flood_tokens_actual}/{target_tokens} tokens")

        capped = flood_request_count >= cfg.e0_max_flood_requests_per_point and flood_tokens_actual < target_tokens
        if capped:
            print(f"[E0]   WARNING: hit e0_max_flood_requests_per_point={cfg.e0_max_flood_requests_per_point} before reaching target_tokens")

        post = probe(client, cfg, anchor)

        point = {
            "fraction_of_capacity": fraction,
            "target_flood_tokens": target_tokens,
            "actual_flood_tokens": flood_tokens_actual,
            "flood_request_count": flood_request_count,
            "flood_requests_ok": flood_requests_ok,
            "capped_before_target": capped,
            "rewarm_ttft": rewarm.get("ttft"),
            "rewarm_ok": rewarm["ok"],
            "post_flood_ttft": post.get("ttft"),
            "post_flood_ok": post["ok"],
        }
        sweep.append(point)
        print(f"[E0] fraction={fraction} done: {flood_request_count} requests / {flood_tokens_actual} tokens -> "
              f"post_flood_ttft={post.get('ttft')} (jump_midpoint={jump_midpoint})")

        if jump_at_fraction is None and jump_midpoint is not None and post["ok"] and post["ttft"] >= jump_midpoint:
            jump_at_fraction = fraction
            jump_at_tokens = flood_tokens_actual

    return {
        "capacity_tokens_used": capacity_tokens,
        "anchor_confirm": {
            "cold": cold,
            "warm_probes": warm_probes,
            "warm_mean_ttft": warm_mean,
            "speedup": speedup,
            "jump_midpoint_ttft": jump_midpoint,
        },
        "sweep": sweep,
        "jump_at_fraction": jump_at_fraction,
        "jump_at_tokens": jump_at_tokens,
    }


# --------------------------------------------------------------------------
# E1 -- timing characterization
# --------------------------------------------------------------------------

def linear_fit(xs: List[float], ys: List[float]) -> Dict[str, Optional[float]]:
    """Minimal least-squares slope/intercept, no numpy dependency."""
    n = len(xs)
    if n < 2:
        return {"slope_ms_per_token": None, "intercept_ms": None, "n_points": n}
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = sum((x - xbar) ** 2 for x in xs)
    if den == 0:
        return {"slope_ms_per_token": None, "intercept_ms": None, "n_points": n}
    slope = num / den
    intercept = ybar - slope * xbar
    return {"slope_ms_per_token": slope, "intercept_ms": intercept, "n_points": n}


def run_e1(client: httpx.Client, cfg: Config, rng: random.Random) -> Dict[str, Any]:
    """Hit: warm one fixed prompt, then sample it repeatedly. Miss: a
    fresh never-reused prompt per sample (reusing one would make the
    second probe a hit, per the design constraint).

    Hit/miss prompts target ~2000 tokens, not a short/arbitrary length --
    at short lengths (~195 tokens) prefill work is small relative to the
    fixed request/response floor, so the hit/miss gap is mostly overhead
    rather than cached work: a real number, but characterizing a regime
    the attack doesn't operate in, and not comparable to results at
    representative lengths."""
    hit_words = words_for_token_target(cfg, cfg.e1_hit_target_tokens)
    miss_words = words_for_token_target(cfg, cfg.e1_miss_target_tokens)

    hit_prompt = unique_prompt(rng, hit_words)
    _ = probe(client, cfg, hit_prompt)  # warm it (discarded)
    hit_probes = [probe(client, cfg, hit_prompt) for _ in range(cfg.e1_n_hit)]

    miss_probes = [probe(client, cfg, unique_prompt(rng, miss_words)) for _ in range(cfg.e1_n_miss)]

    hit_ttfts = [p["ttft"] for p in hit_probes if p["ok"]]
    miss_ttfts = [p["ttft"] for p in miss_probes if p["ok"]]

    hit_mean = statistics.mean(hit_ttfts) if hit_ttfts else None
    hit_std = statistics.stdev(hit_ttfts) if len(hit_ttfts) > 1 else 0.0
    miss_mean = statistics.mean(miss_ttfts) if miss_ttfts else None
    miss_std = statistics.stdev(miss_ttfts) if len(miss_ttfts) > 1 else 0.0

    D = None
    tau = None
    if hit_mean is not None and miss_mean is not None and (hit_std + miss_std) > 0:
        D = (miss_mean - hit_mean) / (hit_std + miss_std)
    if hit_mean is not None and miss_mean is not None:
        tau = (hit_mean + miss_mean) / 2.0

    # Length sweep -> ms/token fit. Fresh (miss-style) prompts at each
    # nominal length; x-values for the fit are the REAL measured
    # prompt_tokens, not the nominal target.
    length_points = []
    for target_tokens in cfg.e1_length_sweep_tokens:
        target_words = words_for_token_target(cfg, target_tokens)
        for _ in range(cfg.e1_length_sweep_repeats):
            p = probe(client, cfg, unique_prompt(rng, target_words))
            length_points.append({"target_tokens": target_tokens, **p})

    fit_xs = [p["prompt_tokens"] for p in length_points if p["ok"] and p.get("prompt_tokens") is not None]
    fit_ys = [p["ttft"] for p in length_points if p["ok"] and p.get("prompt_tokens") is not None]
    fit = linear_fit(fit_xs, fit_ys)

    return {
        "hit_probes": hit_probes,
        "miss_probes": miss_probes,
        "hit_mean_ttft": hit_mean,
        "hit_std_ttft": hit_std,
        "miss_mean_ttft": miss_mean,
        "miss_std_ttft": miss_std,
        "D": D,
        "tau": tau,
        "length_sweep": length_points,
        "ms_per_token_fit": fit,
    }


# --------------------------------------------------------------------------
# E2 -- cache presence detection
# --------------------------------------------------------------------------

def classify(ttft: Optional[float], tau: float) -> Optional[str]:
    if ttft is None:
        return None
    return "HIT" if ttft < tau else "MISS"


def run_e2(client: httpx.Client, cfg: Config, rng: random.Random, tau: float, inter_probe_delay_s: float = 0.0) -> Dict[str, Any]:
    """Balanced known-hit / known-miss set, one classification probe each.
    Returns accuracy + confusion matrix + the raw (ttft, truth) pairs so
    E4's jitter/threshold-shift sweeps can reuse them without firing new
    requests.

    inter_probe_delay_s: if >0, sleeps that long after EVERY probe fired
    in this battery (including the discardable warm-up fire for HIT
    trials). Used by E4's rate-limit-delay sweep, which needs a live
    rerun per delay value -- enforced spacing changes cache state (gives
    the cache time to evict) rather than just adding latency to an
    already-fixed reading, so it can't be simulated post-hoc the way
    jitter/threshold-shift can."""
    words = words_for_token_target(cfg, cfg.e2_target_tokens)
    trials = []

    for _ in range(cfg.e2_n_hit):
        p = unique_prompt(rng, words)
        _ = probe(client, cfg, p)  # warm it (discarded)
        if inter_probe_delay_s > 0:
            time.sleep(inter_probe_delay_s)
        result = probe(client, cfg, p)
        trials.append({"truth": "HIT", **result})
        if inter_probe_delay_s > 0:
            time.sleep(inter_probe_delay_s)

    for _ in range(cfg.e2_n_miss):
        p = unique_prompt(rng, words)
        result = probe(client, cfg, p)
        trials.append({"truth": "MISS", **result})
        if inter_probe_delay_s > 0:
            time.sleep(inter_probe_delay_s)

    rng.shuffle(trials)

    confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}  # positive class = HIT
    n_scored = 0
    n_correct = 0
    for t in trials:
        if not t["ok"]:
            continue
        pred = classify(t["ttft"], tau)
        n_scored += 1
        if pred == t["truth"]:
            n_correct += 1
        if t["truth"] == "HIT" and pred == "HIT":
            confusion["TP"] += 1
        elif t["truth"] == "HIT" and pred == "MISS":
            confusion["FN"] += 1
        elif t["truth"] == "MISS" and pred == "HIT":
            confusion["FP"] += 1
        elif t["truth"] == "MISS" and pred == "MISS":
            confusion["TN"] += 1

    accuracy = (n_correct / n_scored) if n_scored else None

    return {
        "tau_used": tau,
        "trials": trials,
        "n_scored": n_scored,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
    }


# --------------------------------------------------------------------------
# E3 -- prompt content inference
# --------------------------------------------------------------------------

def run_e3(client: httpx.Client, cfg: Config, rng: random.Random) -> Dict[str, Any]:
    """Closed candidate set of ~20 templates spanning a length range.
    Fingerprint = one cold (TTFT, token_count) pair per template. A
    'live observation' is a FRESH, never-before-seen instance drawn from
    one of the 20 length classes (ground truth known to us, not to a
    real attacker) -- classified by nearest fingerprint TTFT. This tests
    length-class inference via timing, not literal string recovery
    (the attacker never sees content either way)."""
    template_word_targets = [
        round(cfg.e3_template_min_words + i * (cfg.e3_template_max_words - cfg.e3_template_min_words) / (cfg.e3_n_templates - 1))
        for i in range(cfg.e3_n_templates)
    ]

    fingerprints = []
    for idx, words in enumerate(template_word_targets):
        p = probe(client, cfg, unique_prompt(rng, words))
        fingerprints.append({"template_id": idx, "target_words": words, **p})

    valid_fps = [f for f in fingerprints if f["ok"] and f.get("prompt_tokens") is not None]

    observations = []
    for _ in range(cfg.e3_n_live_observations):
        true_template = rng.choice(fingerprints)
        p = probe(client, cfg, unique_prompt(rng, true_template["target_words"]))
        pred_template = None
        if p["ok"] and valid_fps:
            pred_template = min(valid_fps, key=lambda f: abs(f["ttft"] - p["ttft"]))
        observations.append({
            "true_template_id": true_template["template_id"],
            "true_prompt_tokens": p.get("prompt_tokens"),
            "pred_template_id": pred_template["template_id"] if pred_template else None,
            "pred_prompt_tokens": pred_template["prompt_tokens"] if pred_template else None,
            **{f"obs_{k}": v for k, v in p.items()},
        })

    scored = [o for o in observations if o["pred_template_id"] is not None]
    top1_correct = sum(1 for o in scored if o["pred_template_id"] == o["true_template_id"])
    top1_accuracy = (top1_correct / len(scored)) if scored else None

    token_errors = [
        abs(o["pred_prompt_tokens"] - o["true_prompt_tokens"])
        for o in scored
        if o["pred_prompt_tokens"] is not None and o["true_prompt_tokens"] is not None
    ]
    mean_token_error = statistics.mean(token_errors) if token_errors else None

    return {
        "fingerprints": fingerprints,
        "observations": observations,
        "top1_accuracy": top1_accuracy,
        "mean_token_count_error": mean_token_error,
        "n_scored": len(scored),
    }


# --------------------------------------------------------------------------
# E4 -- defense resistance
# --------------------------------------------------------------------------

def run_e4(client: Optional[httpx.Client], cfg: Config, rng: random.Random, e2_result: Dict[str, Any], tau: float) -> Dict[str, Any]:
    """jitter and threshold-shift are re-classifications of E2's already-
    collected raw (ttft, truth) pairs -- valid post-hoc, since both are
    equivalent to having measured under the perturbation: additive noise
    on a fixed-cache-state reading, and a shifted decision boundary on
    that same reading, respectively. No new HTTP requests for these two.

    rate-limit delay is NOT simulated post-hoc. Constant added latency
    would shift hit and miss distributions equally (leaving D unchanged)
    and could only move accuracy through interaction with a fixed tau --
    a threshold-calibration artifact, not the defense's real effect. The
    actual mechanism is enforced spacing giving the cache time to evict,
    which changes cache STATE, not a fixed reading -- that can't be
    derived from data already collected under different spacing. So this
    reruns the full E2 battery live, once per delay value, with
    time.sleep(d) enforced between every consecutive probe (requires
    `client`; pass None to skip rate-limit evaluation entirely and record
    it as not evaluated rather than modeling it wrong).

    Parameters swept INDEPENDENTLY (each other at its neutral value), not
    a full 6x3x3 cross product -- ablation-style report per parameter.
    """
    valid = [t for t in e2_result["trials"] if t["ok"]]
    base_ttfts = [t["ttft"] for t in valid]
    truths = [t["truth"] for t in valid]

    def accuracy_with_perturbation(add_ms_fn, tau_eff: float) -> Optional[float]:
        if not valid:
            return None
        correct = 0
        for ttft, truth in zip(base_ttfts, truths):
            perturbed = ttft + add_ms_fn()
            pred = classify(perturbed, tau_eff)
            if pred == truth:
                correct += 1
        return correct / len(valid)

    jitter_results = []
    for J in cfg.e4_jitter_ms:
        accs = []
        for _ in range(cfg.e4_jitter_trials):
            acc = accuracy_with_perturbation(lambda: rng.gauss(0, J) if J > 0 else 0.0, tau)
            if acc is not None:
                accs.append(acc)
        jitter_results.append({
            "jitter_ms": J,
            "mean_accuracy": statistics.mean(accs) if accs else None,
            "std_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "resamples": len(accs),
        })

    threshold_shift_results = []
    for delta in cfg.e4_threshold_shift:
        tau_eff = tau * (1 + delta)
        acc = accuracy_with_perturbation(lambda: 0.0, tau_eff)
        threshold_shift_results.append({"delta": delta, "tau_eff": tau_eff, "accuracy": acc})

    rate_limit_results = []
    rate_limit_evaluated = client is not None
    if rate_limit_evaluated:
        for d in cfg.e4_rate_limit_delay_s:
            rerun = run_e2(client, cfg, rng, tau, inter_probe_delay_s=d)
            rate_limit_results.append({
                "delay_s": d,
                "accuracy": rerun["accuracy"],
                "n_scored": rerun["n_scored"],
                "confusion_matrix": rerun["confusion_matrix"],
            })

    return {
        "baseline_tau": tau,
        "n_reused_from_e2": len(valid),
        "jitter_sweep": jitter_results,
        "threshold_shift_sweep": threshold_shift_results,
        "rate_limit_delay_evaluated": rate_limit_evaluated,
        "rate_limit_delay_sweep": rate_limit_results,
    }


# --------------------------------------------------------------------------
# E5 -- agentic WORM workload
# --------------------------------------------------------------------------
#
# E0-E4 use synthetic prompts with clean binary hit/miss ground truth by
# construction. Real agentic serving produces a different structure: a
# fixed system prompt + tool schema, then conversation history that only
# ever grows by append (write-once-read-many) -- every turn re-reads the
# entire prior prefix. That's what makes the cached prefix both large and
# stable enough to be worth attacking in practice. E5 characterizes the
# channel under that realistic structure using banking_sim's actual
# ReAct agent as the victim, unmodified. It does NOT replace or feed into
# E0-E4 -- mixing synthetic and agentic data would confound prompt
# structure with transport in the three-arm comparison, which is the one
# thing that comparison exists to isolate.
#
# Agentic prefixes share structure by construction: a probe matching
# turns 1..k of a longer session is a genuine partial hit, not a
# misclassification, so results are recorded per depth k rather than
# collapsed to one binary per turn.
#
# The attacker never sends the victim's exact current full prompt --
# every probed depth k is strictly less than the current turn count, so
# it can never equal what the victim just sent (that would be a
# trivially self-warming probe, measuring nothing).

def _import_banking_sim():
    """banking_sim/ is a sibling directory to kv_attack_client_spark/, not
    a parent package -- add its parent to sys.path so the import works
    regardless of cwd. Raises ImportError if banking_sim/langgraph/
    langchain-openai aren't available; caller decides how to handle that
    (main() skips E5 and records it as not-evaluated rather than crashing
    the rest of the suite)."""
    import sys
    parent = str(Path(__file__).resolve().parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from banking_sim.agent import SYSTEM_PROMPT, build_graph
    from banking_sim.main import CONVERSATION_TEMPLATE
    from banking_sim.logger import JsonlLogger
    return SYSTEM_PROMPT, build_graph, CONVERSATION_TEMPLATE, JsonlLogger


def probe_chat(client: httpx.Client, cfg: Config, messages: List[Dict[str, Any]], max_tokens: int = 1) -> Dict[str, Any]:
    """Same streaming-TTFT methodology as probe(), but against
    /v1/chat/completions with a messages array instead of /v1/completions
    with a raw prompt string. E5 needs this specifically: the victim
    (banking_sim's ChatOpenAI client) sends chat-formatted requests, which
    the server renders through its chat template before tokenizing -- a
    raw /v1/completions prompt containing the same literal text would NOT
    produce the same token sequence, and so would never land on the
    victim's actual cached prefix. E0-E4 don't have this problem because
    the attacker generates both sides of every comparison there; E5's
    attacker must match a real chat-formatted victim."""
    payload = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    start = time.perf_counter()
    ttft_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None

    try:
        with client.stream(
            "POST", f"{cfg.base_url}/chat/completions", json=payload, headers=headers, timeout=cfg.request_timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
    except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
        return {"ttft": None, "prompt_tokens": None, "ok": False, "error": str(exc)}

    if ttft_ms is None:
        return {"ttft": None, "prompt_tokens": prompt_tokens, "ok": False, "error": "no data chunks received"}
    return {"ttft": round(ttft_ms, 3), "prompt_tokens": prompt_tokens, "ok": True}


def _build_candidate_messages(boundary_messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert a prefix slice of the victim's real LangChain message list
    into an OpenAI chat messages payload, using langchain_core's own
    convert_to_openai_messages rather than a hand-rolled converter --
    tool-call-bearing AIMessages/ToolMessages are easy to subtly
    misrender by hand (wrong tool_call_id, wrong tool_calls JSON shape),
    which would cause a spurious tokenization mismatch (and a spurious
    MISS) that's about the reconstruction being wrong, not the cache."""
    from langchain_core.messages import convert_to_openai_messages
    msgs = convert_to_openai_messages(boundary_messages)
    if len(boundary_messages) <= 1:
        # k=0: system prompt only, no turn has happened yet. Most chat
        # templates expect at least one non-system message; append the
        # same minimal generic turn used elsewhere in this project for
        # exactly this situation.
        msgs = msgs + [{"role": "user", "content": "Hi"}]
    return msgs


def run_e5(client: httpx.Client, cfg: Config, rng: random.Random, tau: float, out_dir: Path) -> Dict[str, Any]:
    import asyncio
    from langchain_core.messages import HumanMessage, SystemMessage

    SYSTEM_PROMPT, build_graph, CONVERSATION_TEMPLATE, JsonlLogger = _import_banking_sim()

    victim_logger = JsonlLogger(str(out_dir / "e5_victim_requests.jsonl"))
    graph = build_graph(
        base_url=cfg.base_url, model=cfg.model, api_key=cfg.api_key,
        logger=victim_logger, max_tokens=cfg.e5_victim_max_tokens,
    )

    state = {"messages": [SystemMessage(content=SYSTEM_PROMPT)], "session_id": "e5-victim", "turn_index": 0}
    boundaries = [list(state["messages"])]  # boundary[0] = system prompt only (k=0)

    turn_reports = []
    try:
        for t in range(1, cfg.e5_n_turns + 1):
            template = CONVERSATION_TEMPLATE[(t - 1) % len(CONVERSATION_TEMPLATE)]
            user_text = template.format(chk=cfg.e5_victim_chk_account, sav=cfg.e5_victim_sav_account)
            state["messages"] = state["messages"] + [HumanMessage(content=user_text)]
            state["turn_index"] = t - 1

            print(f"[E5] turn {t}/{cfg.e5_n_turns}: victim firing (banking_sim ReAct agent, unmodified)...")
            state = asyncio.run(graph.ainvoke(state))
            boundaries.append(list(state["messages"]))  # boundary[t] = state after turn t

            # Depth sweep k=0..t-1 -- every candidate here is strictly
            # shorter than the current full state (which now includes
            # turn t), so none of them can equal the victim's exact
            # current prompt.
            depth_probes = []
            for k in range(0, t):
                candidate = _build_candidate_messages(boundaries[k])
                result = probe_chat(client, cfg, candidate)
                pred = classify(result.get("ttft"), tau) if result["ok"] else None
                depth_probes.append({"k": k, **result, "classification": pred})

            deepest_tested_k = t - 1
            hit_ks = [dp["k"] for dp in depth_probes if dp["classification"] == "HIT"]
            deepest_hit_k = max(hit_ks) if hit_ks else None
            n_scored = sum(1 for dp in depth_probes if dp["classification"] is not None)
            detection_accuracy = (len(hit_ks) / n_scored) if n_scored else None

            turn_reports.append({
                "turn": t,
                "message_count_after_turn": len(state["messages"]),
                "depth_probes": depth_probes,
                "deepest_tested_k": deepest_tested_k,
                "deepest_hit_k": deepest_hit_k,
                "perfect_reconstruction": deepest_hit_k == deepest_tested_k,
                "detection_accuracy": detection_accuracy,
            })
            print(f"[E5] turn {t}/{cfg.e5_n_turns}: deepest_hit_k={deepest_hit_k} "
                  f"(tested up to {deepest_tested_k}), detection_accuracy={detection_accuracy}")
    finally:
        victim_logger.close()

    per_turn_accuracy = [r["detection_accuracy"] for r in turn_reports if r["detection_accuracy"] is not None]
    overall_accuracy = statistics.mean(per_turn_accuracy) if per_turn_accuracy else None
    perfect_turns = sum(1 for r in turn_reports if r["perfect_reconstruction"])

    return {
        "evaluated": True,
        "tau_used": tau,
        "n_turns": cfg.e5_n_turns,
        "turn_reports": turn_reports,
        "overall_detection_accuracy": overall_accuracy,
        "turns_with_perfect_reconstruction": perfect_turns,
        "turn_count_growth_pattern_reconstructable": perfect_turns == cfg.e5_n_turns,
    }


# --------------------------------------------------------------------------
# Metadata collection (best-effort; null/empty when not run on the actual
# serving host -- never fabricated).
# --------------------------------------------------------------------------

def try_import_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        try:
            out = subprocess.run(["pip", "show", module_name], capture_output=True, text=True, timeout=10)
            for line in out.stdout.splitlines():
                if line.lower().startswith("version:"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return None


def find_worker_pids() -> List[int]:
    """Best-effort scan of /proc/*/cmdline for vLLM/Dynamo worker
    processes. Only finds anything when run on the actual serving host(s)."""
    pids = []
    for cmdline_path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(cmdline_path, "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode(errors="ignore")
        except Exception:
            continue
        if "dynamo.vllm" in cmdline or "vllm.entrypoints" in cmdline or "dynamo.frontend" in cmdline:
            try:
                pids.append(int(cmdline_path.split("/")[2]))
            except (IndexError, ValueError):
                continue
    return pids


def read_ucx_env_for_pid(pid: int) -> Dict[str, str]:
    ucx_env = {}
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
        for entry in raw.split(b"\x00"):
            if not entry:
                continue
            try:
                k, _, v = entry.decode(errors="ignore").partition("=")
            except Exception:
                continue
            if k.startswith("UCX_"):
                ucx_env[k] = v
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        pass
    return ucx_env


def parse_worker_log(cfg: Config) -> Dict[str, Any]:
    result = {"log_path": None, "gpu_memory_utilization_arg": None, "gpu_kv_cache_size_tokens": None}
    matches = glob.glob(cfg.worker_log_glob)
    if not matches:
        return result
    log_path = matches[0]
    result["log_path"] = log_path
    try:
        text = Path(log_path).read_text(errors="ignore")
    except Exception:
        return result
    m = re.search(r"--gpu-memory-utilization[= ]([0-9.]+)", text)
    if m:
        result["gpu_memory_utilization_arg"] = float(m.group(1))
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    if m:
        result["gpu_kv_cache_size_tokens"] = int(m.group(1).replace(",", ""))
    return result


def nvidia_smi_snapshot() -> Optional[List[Dict[str, str]]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        if not lines:
            return None
        header = [h.strip() for h in lines[0].split(",")]
        rows = []
        for line in lines[1:]:
            values = [v.strip() for v in line.split(",")]
            rows.append(dict(zip(header, values)))
        return rows
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


def collect_metadata(cfg: Config, nvidia_smi_start: Optional[List[Dict[str, str]]]) -> Dict[str, Any]:
    worker_pids = find_worker_pids()
    ucx_env_by_pid = {pid: read_ucx_env_for_pid(pid) for pid in worker_pids}

    return {
        "arm": cfg.arm,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": cfg.base_url,
        "model": cfg.model,
        "vllm_version": try_import_version("vllm"),
        "nixl_version": try_import_version("nixl"),
        "worker_pids_found": worker_pids,
        "ucx_env_by_pid": ucx_env_by_pid,
        "worker_log": parse_worker_log(cfg),
        "nvidia_smi_start": nvidia_smi_start,
        "nvidia_smi_end": nvidia_smi_snapshot(),
    }


# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------

def print_summary(e0, e1, e2, e3, e4, e5) -> None:
    def fmt(x, nd=1):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    jump_frac = e0.get("jump_at_fraction")
    jump_desc = f"{jump_frac}x capacity ({e0.get('jump_at_tokens')} tokens)" if jump_frac is not None else "n/a (no jump observed in swept range)"
    print(f"E0  speedup(cold/warm)={fmt(e0['anchor_confirm']['speedup'], 2)}x  capacity={e0.get('capacity_tokens_used')} tokens  jump_at={jump_desc}")
    print(f"E1  D={fmt(e1['D'], 3)}  tau={fmt(e1['tau'])}ms  "
          f"ms/token={fmt(e1['ms_per_token_fit']['slope_ms_per_token'], 4)}")
    print(f"E2  accuracy={fmt((e2['accuracy'] or 0) * 100, 1)}%  confusion={e2['confusion_matrix']}")
    print(f"E3  top1_accuracy={fmt((e3['top1_accuracy'] or 0) * 100, 1)}%  "
          f"mean_token_error={fmt(e3['mean_token_count_error'], 1)}")
    print("E4  jitter sweep:")
    for row in e4["jitter_sweep"]:
        print(f"      J={row['jitter_ms']:>5}ms  acc={fmt((row['mean_accuracy'] or 0) * 100, 1)}%")
    print("E4  threshold-shift sweep:")
    for row in e4["threshold_shift_sweep"]:
        print(f"      delta={row['delta']:>+.2f}  acc={fmt((row['accuracy'] or 0) * 100, 1)}%")
    if e4.get("rate_limit_delay_evaluated"):
        print("E4  rate-limit-delay sweep (live rerun per d):")
        for row in e4["rate_limit_delay_sweep"]:
            print(f"      d={row['delay_s']:>4}s  acc={fmt((row['accuracy'] or 0) * 100, 1)}%")
    else:
        print("E4  rate-limit-delay sweep: NOT EVALUATED")
    if e5.get("evaluated"):
        print(f"E5  overall_detection_accuracy={fmt((e5['overall_detection_accuracy'] or 0) * 100, 1)}%  "
              f"turns_with_perfect_reconstruction={e5['turns_with_perfect_reconstruction']}/{e5['n_turns']}  "
              f"growth_pattern_reconstructable={e5['turn_count_growth_pattern_reconstructable']}")
    else:
        print(f"E5  NOT EVALUATED ({e5.get('reason', 'unknown')})")
    print("=" * 72)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=["flat", "disagg_tcp", "disagg_rdma"], default="flat")
    parser.add_argument("--base-url", default=None, help="Overrides Config.base_url (default http://localhost:8000/v1).")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    parser.add_argument("--kv-cache-tokens", type=int, default=None,
                         help="KV cache capacity in tokens, for E0's flood-fraction sweep. "
                              "Auto-detected from the worker log if available; required otherwise.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    cfg.arm = args.arm
    if args.base_url:
        cfg.base_url = args.base_url
    if args.model:
        cfg.model = args.model
    if args.api_key:
        cfg.api_key = args.api_key
    if args.output_root:
        cfg.output_root = args.output_root

    rng = random.Random(args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(cfg.output_root) / f"{cfg.arm}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    capacity_tokens = resolve_capacity_tokens(cfg, args.kv_cache_tokens)
    cfg.e0_capacity_tokens = capacity_tokens
    print(f"[suite] KV cache capacity: {capacity_tokens} tokens")

    nvidia_smi_start = nvidia_smi_snapshot()

    with httpx.Client() as client:
        print(f"[suite] warming up endpoint ({cfg.warmup_requests} discarded requests)...")
        for _ in range(cfg.warmup_requests):
            probe(client, cfg, unique_prompt(rng, cfg.e0_anchor_words))

        print("[suite] running E0 (cache capacity reconnaissance) -- this sweeps up to 1.5x measured capacity, expect it to be slow...")
        e0 = run_e0(client, cfg, rng, capacity_tokens)
        (out_dir / "e0.json").write_text(json.dumps(e0, indent=2))

        print("[suite] running E1 (timing characterization)...")
        e1 = run_e1(client, cfg, rng)
        (out_dir / "e1.json").write_text(json.dumps(e1, indent=2))
        tau = e1["tau"]
        if tau is None:
            raise RuntimeError("E1 failed to produce a tau (insufficient successful hit/miss probes) -- cannot run E2/E3 classification meaningfully.")

        print(f"[suite] running E2 (cache presence detection, tau={tau:.2f}ms)...")
        e2 = run_e2(client, cfg, rng, tau)
        (out_dir / "e2.json").write_text(json.dumps(e2, indent=2))

        print("[suite] running E3 (prompt content inference)...")
        e3 = run_e3(client, cfg, rng)
        (out_dir / "e3.json").write_text(json.dumps(e3, indent=2))

        print("[suite] running E4 (defense resistance: jitter/threshold-shift post-hoc, rate-limit live rerun)...")
        e4 = run_e4(client, cfg, rng, e2, tau)
        (out_dir / "e4.json").write_text(json.dumps(e4, indent=2))

        print("[suite] running E5 (agentic WORM workload, banking_sim victim)...")
        try:
            e5 = run_e5(client, cfg, rng, tau, out_dir)
        except ImportError as exc:
            print(f"[suite] E5 SKIPPED: banking_sim/langgraph/langchain-openai not importable here ({exc})")
            e5 = {"evaluated": False, "reason": f"ImportError: {exc}"}
        (out_dir / "e5.json").write_text(json.dumps(e5, indent=2))

    print("[suite] collecting run metadata...")
    metadata = collect_metadata(cfg, nvidia_smi_start)
    metadata["config"] = asdict(cfg)
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n[suite] done. Results in {out_dir}")
    print_summary(e0, e1, e2, e3, e4, e5)


if __name__ == "__main__":
    main()
