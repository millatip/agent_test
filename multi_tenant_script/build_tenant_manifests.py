#!/usr/bin/env python3
"""
build_tenant_manifest.py

Builds the 5-tenant manifest for the multi-tenant isolation calibration,
using REAL benchmark dataset content (not synthetic filler), per plan:

  Tenant 0: LMSYS-Chat-1M       (real user conversation opener)
  Tenant 1: system-prompts repo (real agent system prompt, Claude Code)
  Tenant 2: GSM8K                (math reasoning prompt)
  Tenant 3: Alpaca                (instruction-following prompt)
  Tenant 4: LMSYS-Chat-1M       (second, different real conversation opener)

Run this ON THE SERVER (millaone), where network access to HuggingFace
and GitHub is available -- this sandbox's network egress does not reach
HuggingFace, so this script could not be fully exercised before handoff.
Treat as a first draft: run it, inspect tenant_manifest.tsv, and adjust
indices/sampling if any selected prompt looks degenerate (too short,
mostly whitespace, etc.) before using it for calibration.

Requires: `datasets` (HuggingFace) and `requests`, both already used in
this project's earlier LMSYS-loading work per the project README.

Usage:
    python3 build_tenant_manifest.py [output_path]
"""
import sys
import time
import requests

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "tenant_manifest.tsv"

MARKER_PREFIX_LEN = 9  # matches the nanosecond-suffix marker style used in
                        # 04_concurrent_tenants.sh, for consistency


def make_marker(tenant_idx):
    return f"TENANT{tenant_idx}_{str(int(time.time() * 1e9))[-MARKER_PREFIX_LEN:]}"


def get_lmsys_openers(n=2):
    """Pull n distinct real first-user-turns from LMSYS-Chat-1M.
    Mirrors the loading approach already used in this project's
    victim/agent.py (per project_readme), just simplified to grab
    n samples instead of building a full victim session."""
    from datasets import load_dataset
    print("Loading LMSYS-Chat-1M (streaming)...")
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    openers = []
    for row in ds:
        conv = row.get("conversation", [])
        if not conv:
            continue
        first_user = next((m["content"] for m in conv if m.get("role") == "user"), None)
        if first_user and 20 <= len(first_user) <= 400:
            openers.append(first_user.strip())
        if len(openers) >= n:
            break
    if len(openers) < n:
        raise RuntimeError(f"Only found {len(openers)}/{n} usable LMSYS openers -- "
                            f"widen the length filter or scan further into the stream.")
    return openers


def get_system_prompt():
    """Pull one real agent system prompt from the system-prompts repo.
    Using Claude Code's prompt as a representative, real, high-value
    agentic system prompt (matches the agentic-framing motivation)."""
    url = ("https://raw.githubusercontent.com/x1xhlol/"
           "system-prompts-and-models-of-ai-tools/main/Anthropic/Claude%20Code/Prompt.txt")
    print(f"Fetching system prompt from {url} ...")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    text = r.text.strip()
    # Truncate to a reasonable single-request length -- the full Claude
    # Code prompt is long; for calibration purposes we want something
    # that fits comfortably as one prompt without hitting context/latency
    # extremes that could confound the timing-adjacent parts of the
    # toolkit. Adjust this cap if you want the full untruncated prompt.
    MAX_CHARS = 2000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text


def get_gsm8k_prompt():
    """Pull one real GSM8K question. Matches Shadow-in-the-Cache's own
    generalization dataset (their Appendix C3), for direct comparability
    if you want it later."""
    from datasets import load_dataset
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="test")
    return ds[0]["question"].strip()


def get_alpaca_prompt():
    """Pull one real Alpaca instruction. Also matches Shadow-in-the-Cache's
    generalization dataset."""
    from datasets import load_dataset
    print("Loading Alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    row = ds[0]
    instr = row["instruction"].strip()
    if row.get("input"):
        instr = f"{instr}\n{row['input'].strip()}"
    return instr


def main():
    print("Building 5-tenant manifest from real benchmark datasets...\n")

    lmsys_openers = get_lmsys_openers(n=2)
    system_prompt = get_system_prompt()
    gsm8k_prompt = get_gsm8k_prompt()
    alpaca_prompt = get_alpaca_prompt()

    tenant_sources = [
        ("tenant0", "lmsys_chat_1m", lmsys_openers[0]),
        ("tenant1", "system_prompts_claude_code", system_prompt),
        ("tenant2", "gsm8k", gsm8k_prompt),
        ("tenant3", "alpaca", alpaca_prompt),
        ("tenant4", "lmsys_chat_1m", lmsys_openers[1]),
    ]

    lines = []
    for idx, (tenant_id, source, content) in enumerate(tenant_sources):
        marker = make_marker(idx)
        # Wrap content with marker on both ends, matching the convention
        # in 04_concurrent_tenants.sh, so fingerprint matching has a
        # known, locatable anchor regardless of dataset content shape.
        full_prompt = f"{marker} {content} {marker}"
        full_prompt = full_prompt.replace("\t", " ").replace("\n", " ")
        lines.append(f"{tenant_id}\t{marker}\t{full_prompt}")
        print(f"  {tenant_id} ({source}): {len(content)} chars, marker={marker}")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {len(lines)} tenants to {OUT_PATH}")
    print("\nReview this file before running calibration -- check none of the "
          "prompts are degenerate (e.g. mostly whitespace, truncated mid-word "
          "in a way that changes meaning) and that dataset diversity looks "
          "right across tenants.")


if __name__ == "__main__":
    main()