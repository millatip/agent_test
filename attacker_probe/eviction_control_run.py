"""Control run: eviction test with ZERO interim victim rechecks during the
flood, and explicit timed idle gaps between phases.

Rules out two things the earlier runs couldn't:
  1. Whether repeated victim rechecking itself was refreshing the victim's
     LRU recency and masking real eviction, regardless of flood volume.
  2. Gives the server's power logger clean, non-overlapping phase windows
     instead of back-to-back seconds.

Sequence: baseline fire -> idle gap -> flood (zero victim touches, no
exceptions) -> idle gap -> single final victim fire -> idle gap.

Exactly two victim fires in the entire run. No warmup fires either --
that would violate the same "no more than two victim touches" constraint
this run is specifically designed to satisfy.
"""

import argparse
import asyncio
import random
import time
from datetime import datetime, timezone

import httpx

from attacker_probe.probe import KNOWN_VICTIM_SYSTEM_PROMPT
from attacker_probe.eviction_threshold import build_filler_content, fire, created_to_utc
from banking_sim.config import load_config_file, resolve
from banking_sim.logger import JsonlLogger


async def run_control(args: argparse.Namespace) -> dict:
    logger = JsonlLogger(args.output)
    rng = random.Random()
    run_start = time.perf_counter()

    def log_row(kind, seq, ttft_ms, prompt_tokens, created):
        logger.log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "seq": seq,
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "prompt_tokens": prompt_tokens,
            "created_epoch": created,
            "created_utc": created_to_utc(created),
            "elapsed_run_seconds": round(time.perf_counter() - run_start, 2),
        })

    result = {}
    try:
        async with httpx.AsyncClient() as client:
            # Step 1: baseline fire (victim touch #1 of exactly 2).
            baseline_ttft, baseline_tokens, baseline_created = await fire(
                client, args, KNOWN_VICTIM_SYSTEM_PROMPT, args.request_timeout
            )
            log_row("victim_baseline", 0, baseline_ttft, baseline_tokens, baseline_created)
            print(f"[control] baseline: victim TTFT={baseline_ttft:.1f}ms at {created_to_utc(baseline_created)}")

            # Step 2: idle gap -- fire nothing.
            await asyncio.sleep(args.gap_seconds)
            print(f"[control] idle gap 1 ({args.gap_seconds}s) done, no requests fired")

            # Step 3: flood, zero victim touches, no exceptions.
            total_filler_count = 0
            total_filler_tokens = 0
            flood_start_created = None
            flood_end_created = None
            while total_filler_tokens < args.target_tokens and total_filler_count < args.max_filler_requests:
                content = build_filler_content(rng, args.filler_words)
                ttft, tokens, created = await fire(client, args, content, args.request_timeout)
                total_filler_count += 1
                total_filler_tokens += tokens or 0
                if created is not None:
                    flood_start_created = created if flood_start_created is None else min(flood_start_created, created)
                    flood_end_created = created if flood_end_created is None else max(flood_end_created, created)
                log_row("filler", total_filler_count, ttft, tokens, created)
                if total_filler_count % 100 == 0:
                    print(f"[control] flood progress: {total_filler_count} reqs / {total_filler_tokens} tokens")

            if total_filler_count >= args.max_filler_requests:
                print(f"[control] WARNING: hit --max-filler-requests={args.max_filler_requests} safety cap before reaching --target-tokens={args.target_tokens}")
            print(f"[control] flood done: {total_filler_count} filler requests / {total_filler_tokens} tokens, zero victim touches during flood")

            # Step 4: idle gap -- fire nothing.
            await asyncio.sleep(args.gap_seconds)
            print(f"[control] idle gap 2 ({args.gap_seconds}s) done, no requests fired")

            # Step 5: single final victim fire (victim touch #2 of exactly 2).
            final_ttft, final_tokens, final_created = await fire(
                client, args, KNOWN_VICTIM_SYSTEM_PROMPT, args.request_timeout
            )
            log_row("victim_final", 0, final_ttft, final_tokens, final_created)
            print(f"[control] final victim fire: TTFT={final_ttft:.1f}ms at {created_to_utc(final_created)}")

            # Step 6: idle gap -- fire nothing, closes out the window.
            await asyncio.sleep(args.gap_seconds)
            print(f"[control] idle gap 3 ({args.gap_seconds}s) done, no requests fired -- window closed")

            result = {
                "baseline_ttft_ms": baseline_ttft,
                "final_victim_ttft_ms": final_ttft,
                "total_filler_requests": total_filler_count,
                "total_filler_tokens": total_filler_tokens,
                "gap_seconds": args.gap_seconds,
                "timestamps_utc": {
                    "baseline_fire": created_to_utc(baseline_created),
                    "flood_start": created_to_utc(flood_start_created),
                    "flood_end": created_to_utc(flood_end_created),
                    "victim_final": created_to_utc(final_created),
                },
                "timestamps_epoch": {
                    "baseline_fire": baseline_created,
                    "flood_start": flood_start_created,
                    "flood_end": flood_end_created,
                    "victim_final": final_created,
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
    parser.add_argument("--filler-words", type=int, default=200)
    parser.add_argument("--target-tokens", type=int, default=300000)
    parser.add_argument("--max-filler-requests", type=int, default=2000, help="Safety cap independent of --target-tokens.")
    parser.add_argument("--gap-seconds", type=float, default=5.0)
    parser.add_argument("--output", default="runs/eviction_control_run.jsonl")
    args = parser.parse_args()

    config = load_config_file(args.config)
    args.base_url = resolve("base_url", args.base_url, config)
    args.model = resolve("model", args.model, config)
    args.api_key = resolve("api_key", args.api_key, config)
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_control(args))
    print("\n[control] result:", result)


if __name__ == "__main__":
    main()
