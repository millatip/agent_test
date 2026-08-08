"""Side-channel correlation test (client side): fire requests of clearly
different, known prompt sizes with precise gaps and timestamps, so the
server side can check afterward whether its passively-polled counter
deltas actually discriminate between them.

Passive data collection only -- no correlation analysis here, that
happens once both sides' logs are combined.

Gate: do not run this until the server agent has confirmed Phase 0/1
complete (counters confirmed unprivileged-readable, polling live). This
script does not and cannot check that itself -- it's an external
precondition enforced by whoever invokes it.

Each of the 3 size categories (small/medium/large) is fired 5 times, in
randomized order, with an explicit --gap-seconds (default 10) of silence
between every single request. Content is unique per request (nonce +
random words, same generator as attacker_probe.eviction_threshold) so no
request can be a cache hit against another -- size category is the only
systematic variable.

Captures, per request:
  - size category
  - request-fire timestamp (server's own `created` field, UTC)
  - response-complete timestamp (client wall-clock, right as the stream
    fully closes -- distinct from request-fire; with max_tokens=1 these
    are close together but not identical)
  - actual prompt token count from the response's usage field (not the
    word-count target -- the real measured number)
"""

import argparse
import asyncio
import json
import random
import time
from datetime import datetime, timezone

import httpx

from attacker_probe.eviction_threshold import build_filler_content, created_to_utc
from banking_sim.config import load_config_file, resolve
from banking_sim.logger import JsonlLogger

# Word-count targets calibrated against this project's own filler
# generator (empirically ~1.35 tokens/word for this nonce+word-salad
# content, from attacker_probe.eviction_threshold's prior runs). These
# are starting points, not guarantees -- the actual measured token count
# is what gets reported, per the task's explicit instruction not to
# assume the prompt landed exactly on target.
SIZE_CATEGORIES = {
    "small": 30,     # target ~50 tokens
    "medium": 360,   # target ~500 tokens
    "large": 1840,   # target ~2500 tokens
}


async def fire_and_time(client, base_url, model, api_key, content, timeout):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": content},
            {"role": "user", "content": "Hi"},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    created = None
    prompt_tokens = None

    async with client.stream(
        "POST", f"{base_url}/chat/completions", json=payload, headers=headers, timeout=timeout
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if created is None and "created" in chunk:
                created = chunk.get("created")
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens")

    completed_at = datetime.now(timezone.utc).isoformat()
    return created, prompt_tokens, completed_at


async def run(args: argparse.Namespace) -> None:
    logger = JsonlLogger(args.output)
    rng = random.Random()
    run_start = time.perf_counter()

    # 5x each of 3 categories, randomized order.
    plan = []
    for size, words in SIZE_CATEGORIES.items():
        plan.extend([(size, words)] * args.reps_per_size)
    rng.shuffle(plan)

    rows = []
    try:
        async with httpx.AsyncClient() as client:
            for i, (size, target_words) in enumerate(plan, 1):
                content = build_filler_content(rng, target_words)
                created, prompt_tokens, completed_at = await fire_and_time(
                    client, args.base_url, args.model, args.api_key, content, args.request_timeout
                )
                row = {
                    "seq": i,
                    "size_category": size,
                    "target_words": target_words,
                    "request_fire_utc": created_to_utc(created),
                    "request_fire_epoch": created,
                    "response_complete_utc": completed_at,
                    "prompt_tokens": prompt_tokens,
                    "elapsed_run_seconds": round(time.perf_counter() - run_start, 2),
                }
                rows.append(row)
                logger.log(row)
                print(f"[size_corr] {i}/{len(plan)} size={size:6s} prompt_tokens={prompt_tokens} fired_at={created_to_utc(created)}")
                await asyncio.sleep(args.gap_seconds)
    finally:
        logger.close()

    print(f"\n[size_corr] done: {len(rows)} requests logged to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--reps-per-size", type=int, default=5)
    parser.add_argument("--gap-seconds", type=float, default=10.0)
    parser.add_argument("--output", default="runs/size_correlation_probe.jsonl")
    args = parser.parse_args()

    config = load_config_file(args.config)
    args.base_url = resolve("base_url", args.base_url, config)
    args.model = resolve("model", args.model, config)
    args.api_key = resolve("api_key", args.api_key, config)
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
