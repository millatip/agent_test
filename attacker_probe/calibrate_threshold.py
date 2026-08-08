"""Calibration pass: confirm hit vs. miss TTFT bands on THIS serving
config right now, rather than assuming last week's energy-channel
~213ms/~277ms baseline/post-flood values transfer directly.

known_hit: victim prompt fired after a short warm-up (a few discardable
fires establish a warm cache entry), then --n measured fires.
known_miss: --n fresh, never-before-seen prompts (same unique-content
generator used throughout this project), each necessarily a cold miss
by construction.

Reuses fire()/KNOWN_VICTIM_SYSTEM_PROMPT/build_filler_content from the
rest of attacker_probe/ -- no new HTTP logic.
"""

import argparse
import asyncio
import random
import statistics as st
from datetime import datetime, timezone

import httpx

from attacker_probe.probe import KNOWN_VICTIM_SYSTEM_PROMPT
from attacker_probe.eviction_threshold import build_filler_content, fire, created_to_utc
from banking_sim.config import load_config_file, resolve
from banking_sim.logger import JsonlLogger


async def run_calibration(args: argparse.Namespace) -> dict:
    logger = JsonlLogger(args.output)
    rng = random.Random()
    hit_ttfts, miss_ttfts = [], []

    try:
        async with httpx.AsyncClient() as client:
            for i in range(args.warmup_fires):
                await fire(client, args, KNOWN_VICTIM_SYSTEM_PROMPT, args.request_timeout)
            print(f"[calibrate] warmed up victim with {args.warmup_fires} discardable fires")

            for i in range(args.n):
                ttft, tokens, created = await fire(client, args, KNOWN_VICTIM_SYSTEM_PROMPT, args.request_timeout)
                hit_ttfts.append(ttft)
                logger.log({
                    "timestamp": datetime.now(timezone.utc).isoformat(), "kind": "known_hit", "trial": i,
                    "ttft_ms": round(ttft, 2), "prompt_tokens": tokens, "created_utc": created_to_utc(created),
                })
                print(f"[calibrate] known-hit {i+1}/{args.n}: {ttft:.1f}ms")
                await asyncio.sleep(args.gap_seconds)

            for i in range(args.n):
                content = build_filler_content(rng, args.filler_words)
                ttft, tokens, created = await fire(client, args, content, args.request_timeout)
                miss_ttfts.append(ttft)
                logger.log({
                    "timestamp": datetime.now(timezone.utc).isoformat(), "kind": "known_miss", "trial": i,
                    "ttft_ms": round(ttft, 2), "prompt_tokens": tokens, "created_utc": created_to_utc(created),
                })
                print(f"[calibrate] known-miss {i+1}/{args.n}: {ttft:.1f}ms")
                await asyncio.sleep(args.gap_seconds)
    finally:
        logger.close()

    hit_mean, hit_std = st.mean(hit_ttfts), (st.stdev(hit_ttfts) if len(hit_ttfts) > 1 else 0.0)
    miss_mean, miss_std = st.mean(miss_ttfts), (st.stdev(miss_ttfts) if len(miss_ttfts) > 1 else 0.0)
    result = {
        "hit_mean_ttft_ms": hit_mean, "hit_std_ttft_ms": hit_std, "hit_n": len(hit_ttfts),
        "miss_mean_ttft_ms": miss_mean, "miss_std_ttft_ms": miss_std, "miss_n": len(miss_ttfts),
        "midpoint_threshold_ms": (hit_mean + miss_mean) / 2.0,
    }
    print("\n[calibrate] result:", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--warmup-fires", type=int, default=5)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--filler-words", type=int, default=200)
    parser.add_argument("--gap-seconds", type=float, default=5.0)
    parser.add_argument("--output", default="runs/calibrate_threshold.jsonl")
    args = parser.parse_args()

    config = load_config_file(args.config)
    args.base_url = resolve("base_url", args.base_url, config)
    args.model = resolve("model", args.model, config)
    args.api_key = resolve("api_key", args.api_key, config)
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run_calibration(args))


if __name__ == "__main__":
    main()
