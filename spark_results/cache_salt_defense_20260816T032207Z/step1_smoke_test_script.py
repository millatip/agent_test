import sys, time, uuid, json
sys.path.insert(0, "/home/s3lab-spark/LG2026/kv_attack_client_spark")
import httpx
from suite import Config, unique_prompt, words_for_token_target
import random

cfg = Config()
rng = random.Random(1234)

def probe_salt(client, cfg, prompt_text, cache_salt=None, max_tokens=1):
    payload = {
        "model": cfg.model, "prompt": prompt_text, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
    }
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    start = time.perf_counter()
    ttft_ms = None
    with client.stream("POST", f"{cfg.base_url}/completions", json=payload, headers=headers, timeout=cfg.request_timeout) as response:
        if response.status_code != 200:
            body = response.read()
            print("ERROR BODY:", body[:2000])
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
    return ttft_ms

with httpx.Client() as client:
    words = words_for_token_target(cfg, 2000)
    prompt = unique_prompt(rng, words)

    print("=== Test A: same prompt, different salts -> 2nd should be MISS (cold) ===")
    salt1 = f"testA-{uuid.uuid4().hex}"
    salt2 = f"testB-{uuid.uuid4().hex}"
    t1 = probe_salt(client, cfg, prompt, cache_salt=salt1)
    print(f"  salt1 first probe (cold, salt={salt1[:12]}...): {t1:.1f} ms")
    t2 = probe_salt(client, cfg, prompt, cache_salt=salt2)
    print(f"  salt2 first probe (different salt, salt={salt2[:12]}...): {t2:.1f} ms  <- expect ~cold/miss, similar to t1")

    print("\n=== Test B: same prompt, same salt twice -> 2nd should be HIT (warm) ===")
    prompt2 = unique_prompt(rng, words)
    salt3 = f"testC-{uuid.uuid4().hex}"
    t3 = probe_salt(client, cfg, prompt2, cache_salt=salt3)
    print(f"  first probe (cold, salt={salt3[:12]}...): {t3:.1f} ms")
    t4 = probe_salt(client, cfg, prompt2, cache_salt=salt3)
    print(f"  second probe (same salt again): {t4:.1f} ms  <- expect much faster (hit)")

    print("\n=== Test C: baseline no-salt hit/miss for reference ===")
    prompt3 = unique_prompt(rng, words)
    t5 = probe_salt(client, cfg, prompt3, cache_salt=None)
    print(f"  first probe (cold, no salt): {t5:.1f} ms")
    t6 = probe_salt(client, cfg, prompt3, cache_salt=None)
    print(f"  second probe (same prompt, no salt): {t6:.1f} ms  <- expect fast (hit)")
