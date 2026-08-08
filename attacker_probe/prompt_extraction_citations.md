# Prior art: prompt/system-prompt extraction

`attacker_probe/probe.py` assumes the attacker already possesses the victim's
exact system prompt string. This module does **not** implement how that
string was obtained — that's a separate, already well-studied attack class.
This file exists so that precondition isn't left unsourced.

## System-prompt / prompt leaking

- Perez, F. & Ribeiro, I. "Ignore Previous Prompt: Attack Techniques For
  Language Models." NeurIPS 2022 ML Safety Workshop. (Often cited as the
  original formalization of "prompt leaking" via adversarial instructions
  that induce a model to repeat its hidden system prompt.)
- Willison, S. "Prompt injection attacks against GPT-3" (2022) and
  subsequent posts on simonwillison.net — early, widely-cited practitioner
  documentation of prompt leaking as a subclass of prompt injection,
  including the "repeat everything above verbatim" family of attacks used
  against real deployed chatbots.
- Morris, J. X. et al. "Language Model Inversion." (2023) — recovers hidden
  prompts from a model's output distribution/logprobs rather than relying
  on the model complying with a leak instruction; relevant when the target
  API exposes logprobs and direct leak instructions are filtered.

## Why we don't re-implement it here

These attacks are behavioral (they get the *model* to disclose the prompt
through its outputs, or invert the prompt from output statistics) and are
orthogonal to the timing side channel this module studies. Treating
"attacker knows the exact system prompt string" as a given precondition —
rather than re-deriving it — keeps `attacker_probe/` focused on a single
claim: *given* a known candidate prefix, can its presence in a shared
KV-cache be inferred purely from request timing, with no other privileged
access to the victim.

## Note on the cache-timing side channel itself

KV-cache / prefix-cache reuse in multi-tenant LLM serving (vLLM's
`APC`/prefix caching, and similar mechanisms in other engines) is a
production performance optimization — see Gim, I. et al. "Prompt Cache:
Modular Attention Reuse for Low-Latency Inference," MLSys 2024, for the
caching mechanism this side channel piggybacks on. Timing side channels
against shared caches are a much older, general pattern in systems
security (e.g. Bernstein, D. J. "Cache-Timing Attacks on AES," 2005;
Percival, C. "Cache Missing for Fun and Profit," 2005, on Hyper-Threading).
Whether prefix-cache timing specifically has since been formalized as a
named LLM-serving attack in the security literature is an active and
fast-moving area — search current USENIX Security / IEEE S&P / arXiv cs.CR
proceedings rather than relying on a citation frozen at the time this file
was written, since deliberately guessing a specific paper title/URL here
would risk citing something that doesn't exist.
