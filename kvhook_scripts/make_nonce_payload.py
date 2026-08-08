#!/usr/bin/env python3
"""
Builds a never-before-sent variant of tenant1_payload.json for a clean
Phase A rerun -- same structure, but with a unique nonce prepended to the
prompt so tokenization can't overlap with anything sent earlier in this
session (avoiding the prefix-cache contamination that affected the
original Phase A captures). tenant1_payload.json itself is read-only,
never modified.
"""
import json
import os
import uuid

SRC = os.path.expanduser("~/LG2026/KVHOOK/tenant1_payload.json")
DST = os.path.expanduser("~/LG2026/KVHOOK/nonce_payload.json")

with open(SRC) as f:
    payload = json.load(f)

nonce = uuid.uuid4().hex
payload["prompt"] = f"[session nonce {nonce}, ignore this line]\n\n" + payload["prompt"]

with open(DST, "w") as f:
    json.dump(payload, f)

print(f"wrote {DST}")
print(f"nonce: {nonce}")
print(f"prompt length: {len(payload['prompt'])} chars (original {len(payload['prompt']) - len(nonce) - 27} chars)")
