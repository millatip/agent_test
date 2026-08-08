"""Time-matched no-flood control: isolate whether the prior control run's
+60.5ms rise (baseline 214.8ms -> final 275.3ms, after a 300K-token,
316s flood) was flood-caused or just ordinary wall-clock disuse over the
same elapsed time.

Same two-touch design, flood entirely removed: baseline fire -> wait
exactly --wait-seconds (matching the prior run's actual flood duration)
-> final fire. Zero filler traffic, zero other requests during the wait.
"""

import argparse
import asyncio
import time
from datetime import datetime, timezone

import httpx

from attacker_probe.probe import KNOWN_VICTIM_SYSTEM_PROMPT
from attacker_probe.eviction_threshold import fire, created_to_utc
from banking_sim.config import load_config_file, resolve
from banking_sim.logger import JsonlLogger


async def run_time_control(args: argparse.Namespace) -> dict:
    logger = JsonlLogger(args.output)
    run_start = time.perf_counter()

    def log_row(kind, ttft_ms, prompt_tokens, created):
        logger.log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "prompt_tokens": prompt_tokens,
            "created_epoch": created,
            "created_utc": created_to_utc(created),
            "elapsed_run_seconds": round(time.perf_counter() - run_start, 2),
        })

    try:
        async with httpx.AsyncClient() as client:
            baseline_ttft, baseline_tokens, baseline_created = await fire(
                client, args, KNOWN_VICTIM_SYSTEM_PROMPT, args.request_timeout
            )
            log_row("victim_baseline", baseline_ttft, baseline_tokens, baseline_created)
            print(f"[time_control] baseline: victim TTFT={baseline_ttft:.1f}ms at {created_to_utc(baseline_created)}")

            print(f"[time_control] waiting {args.wait_seconds}s, firing nothing...")
            await asyncio.sleep(args.wait_seconds)
            print("[time_control] wait done, no requests fired")

            final_ttft, final_tokens, final_created = await fire(
                client, args, KNOWN_VICTIM_SYSTEM_PROMPT, args.request_timeout
            )
            log_row("victim_final", final_ttft, final_tokens, final_created)
            print(f"[time_control] final: victim TTFT={final_ttft:.1f}ms at {created_to_utc(final_created)}")

            result = {
                "baseline_ttft_ms": baseline_ttft,
                "final_ttft_ms": final_ttft,
                "delta_ms": final_ttft - baseline_ttft,
                "wait_seconds": args.wait_seconds,
                "timestamps_utc": {
                    "baseline_fire": created_to_utc(baseline_created),
                    "victim_final": created_to_utc(final_created),
                },
                "timestamps_epoch": {
                    "baseline_fire": baseline_created,
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
    parser.add_argument("--wait-seconds", type=float, default=316.0)
    parser.add_argument("--output", default="runs/eviction_time_control.jsonl")
    args = parser.parse_args()

    config = load_config_file(args.config)
    args.base_url = resolve("base_url", args.base_url, config)
    args.model = resolve("model", args.model, config)
    args.api_key = resolve("api_key", args.api_key, config)
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_time_control(args))
    print("\n[time_control] result:", result)


if __name__ == "__main__":
    main()
