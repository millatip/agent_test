"""Multi-tenant synthetic banking-assistant load generator.

Spins up N concurrent simulated client sessions, each running a fixed
multi-turn scripted conversation against a LangGraph ReAct agent backed by
an OpenAI-compatible completions endpoint. Every session shares the exact
same system prompt (see banking_sim.agent.SYSTEM_PROMPT), so the whole run
is a WORM-style prefix-cache probe: the prefix is written once and re-read
by every other session/turn that follows.

Per-request timing/token records are appended as JSONL to --output, one
line per LLM call, ready to load with pandas for side-channel analysis.

Usage:
    python -m banking_sim.main --num-sessions 8 --output runs/log.jsonl
"""

import argparse
import asyncio
import os
import time

from langchain_core.messages import HumanMessage, SystemMessage

from .agent import SYSTEM_PROMPT, build_graph
from .config import load_config_file, resolve
from .logger import JsonlLogger

# Scripted multi-turn conversation. {chk}/{sav} are filled in per session so
# different tenants touch different (fake) accounts while sharing the same
# system prompt and conversational shape.
CONVERSATION_TEMPLATE = [
    "Hi, can you check the balance on account {chk}?",
    "Thanks. Now show me the last 5 transactions on {chk}.",
    "I'd like to transfer $250 from {chk} to {sav}.",
    "Great, can you confirm the new balance on {chk}?",
]


async def run_session(session_idx: int, graph, semaphore: asyncio.Semaphore) -> None:
    session_id = f"tenant-{session_idx:04d}"
    chk = f"CHK-{1000 + session_idx}"
    sav = f"SAV-{2000 + session_idx}"

    async with semaphore:
        state = {
            "messages": [SystemMessage(content=SYSTEM_PROMPT)],
            "session_id": session_id,
            "turn_index": 0,
        }
        for turn_index, template in enumerate(CONVERSATION_TEMPLATE):
            user_text = template.format(chk=chk, sav=sav)
            state["messages"] = state["messages"] + [HumanMessage(content=user_text)]
            state["turn_index"] = turn_index
            state = await graph.ainvoke(state)


async def run_all(args) -> None:
    logger = JsonlLogger(args.output)
    try:
        graph = build_graph(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            logger=logger,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        semaphore = asyncio.Semaphore(args.concurrency)
        start = time.perf_counter()
        await asyncio.gather(
            *(run_session(i, graph, semaphore) for i in range(args.num_sessions))
        )
        elapsed = time.perf_counter() - start
        print(
            f"Done: {args.num_sessions} sessions "
            f"({len(CONVERSATION_TEMPLATE)} turns each) in {elapsed:.1f}s. "
            f"Logs written to {args.output}"
        )
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=os.environ.get("BANKING_SIM_CONFIG"),
        help="Optional JSON file with base_url/model/api_key overrides.",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible endpoint base URL. Falls back to --config, then "
             "$BANKING_SIM_BASE_URL, then http://localhost:8000/v1. "
             "e.g. http://10.126.36.140:8000/v1 for direct access, or "
             "http://localhost:8000/v1 if tunneled over SSH.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name as registered on the serving endpoint. Falls back to "
             "--config, then $BANKING_SIM_MODEL, then a built-in default.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key. Falls back to --config, then $BANKING_SIM_API_KEY, then 'EMPTY'.",
    )
    parser.add_argument(
        "--num-sessions", type=int, default=4,
        help="Number of simulated tenant sessions to run.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Max sessions in flight at once (default: same as --num-sessions, i.e. all at once).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100,
        help="Max completion tokens per LLM call (caps how long a reasoning "
             "model can ramble in <think> traces per ReAct step).",
    )
    parser.add_argument(
        "--output", default="runs/requests.jsonl",
        help="Path to the JSONL log file (appended to).",
    )
    args = parser.parse_args()

    config = load_config_file(args.config)
    args.base_url = resolve("base_url", args.base_url, config)
    args.model = resolve("model", args.model, config)
    args.api_key = resolve("api_key", args.api_key, config)

    if args.concurrency is None:
        args.concurrency = args.num_sessions
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
