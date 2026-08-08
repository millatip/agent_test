"""Poll the serving stack's Prometheus /metrics endpoint at a fixed interval
and log a compact JSONL snapshot of queue-depth / worker-busy / cache
activity gauges, so scheduler burstiness can be time-aligned against
attacker_probe's TTFT time series during a run (e.g. banking_sim.main's
capped-concurrent victim traffic).

This is operator/diagnostic tooling, not attacker capability — it lives
outside attacker_probe/ and banking_sim/ on purpose. It reuses
banking_sim.config purely for endpoint resolution (same --base-url the
other tools point at); it derives the metrics URL by stripping the /v1
suffix, since Prometheus metrics are served off the root, not under /v1.

Usage:
    python scrape_metrics.py --duration-seconds 70 --interval-seconds 1 \
        --output runs/metrics_scrape.jsonl
"""

import argparse
import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import httpx

from banking_sim.config import load_config_file, resolve
from banking_sim.logger import JsonlLogger

_LINE_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(\{(?P<labels>[^}]*)\})?\s+(?P<value>[-\d.eE+]+)\s*$')
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')

# Cumulative-since-server-start counters we diff between polls to get a
# per-window rate/average, plus a handful of instantaneous gauges.
_GAUGES = [
    "dynamo_frontend_queued_requests",
    "dynamo_frontend_inflight_requests",
    "dynamo_frontend_active_requests",
]
_HISTOGRAM_SUM_COUNT = [
    "dynamo_frontend_cached_tokens",
    "dynamo_frontend_time_to_first_token_seconds",
]


def metrics_url(base_url: str) -> str:
    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    return root.rstrip("/") + "/metrics"


def parse_metrics_text(text: str) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
    out: Dict[str, List[Tuple[Dict[str, str], float]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
        value = float(m.group("value"))
        out.setdefault(name, []).append((labels, value))
    return out


def snapshot(parsed: Dict[str, List[Tuple[Dict[str, str], float]]]) -> Dict[str, float]:
    record: Dict[str, float] = {}

    for name in _GAUGES:
        entries = parsed.get(name, [])
        record[name] = sum(v for _, v in entries)

    for labels, value in parsed.get("dynamo_tokio_worker_busy_ratio", []):
        record[f"worker_busy_ratio_{labels.get('worker', '?')}"] = value

    for metric in _HISTOGRAM_SUM_COUNT:
        record[f"{metric}_sum"] = sum(v for _, v in parsed.get(f"{metric}_sum", []))
        record[f"{metric}_count"] = sum(v for _, v in parsed.get(f"{metric}_count", []))

    record["requests_total"] = sum(v for _, v in parsed.get("dynamo_frontend_requests_total", []))
    return record


def add_windowed_deltas(record: Dict[str, float], prev: Dict[str, float]) -> None:
    """Turn cumulative counters into a per-polling-window rate/average."""
    dt = record["elapsed_run_seconds"] - prev["elapsed_run_seconds"]

    d_count = record["dynamo_frontend_cached_tokens_count"] - prev["dynamo_frontend_cached_tokens_count"]
    d_sum = record["dynamo_frontend_cached_tokens_sum"] - prev["dynamo_frontend_cached_tokens_sum"]
    record["window_requests_completed"] = d_count
    record["window_avg_cached_tokens"] = (d_sum / d_count) if d_count > 0 else None

    d_ttft_count = record["dynamo_frontend_time_to_first_token_seconds_count"] - prev["dynamo_frontend_time_to_first_token_seconds_count"]
    d_ttft_sum = record["dynamo_frontend_time_to_first_token_seconds_sum"] - prev["dynamo_frontend_time_to_first_token_seconds_sum"]
    record["window_server_side_avg_ttft_ms"] = (d_ttft_sum / d_ttft_count * 1000.0) if d_ttft_count > 0 else None

    d_req = record["requests_total"] - prev["requests_total"]
    record["window_requests_per_sec"] = (d_req / dt) if dt > 0 else None


async def run(args: argparse.Namespace) -> None:
    logger = JsonlLogger(args.output)
    url = metrics_url(args.base_url)
    try:
        async with httpx.AsyncClient() as client:
            run_start = time.perf_counter()
            prev = None
            n = 0
            while time.perf_counter() - run_start < args.duration_seconds:
                try:
                    resp = await client.get(url, timeout=args.request_timeout)
                    resp.raise_for_status()
                    record = snapshot(parse_metrics_text(resp.text))
                except httpx.HTTPError as exc:
                    print(f"[scrape_metrics] poll failed: {exc}")
                    await asyncio.sleep(args.interval_seconds)
                    continue

                record["timestamp"] = datetime.now(timezone.utc).isoformat()
                record["elapsed_run_seconds"] = round(time.perf_counter() - run_start, 2)
                if prev is not None:
                    add_windowed_deltas(record, prev)

                logger.log(record)
                prev = record
                n += 1
                await asyncio.sleep(args.interval_seconds)
            print(f"[scrape_metrics] done: {n} snapshots logged to {args.output}")
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float, default=70.0)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--output", default="runs/metrics_scrape.jsonl")
    args = parser.parse_args()

    config = load_config_file(args.config)
    args.base_url = resolve("base_url", args.base_url, config)
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
