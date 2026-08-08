"""Endpoint configuration, resolved with precedence:

    CLI flag  >  --config JSON file  >  environment variable  >  built-in default

This lets you flip between direct and SSH-tunneled access to the vLLM
server without touching code, e.g.:

    export BANKING_SIM_BASE_URL=http://localhost:8000/v1   # via SSH tunnel
    python -m banking_sim.main

or drop a config.json next to this package:

    {"base_url": "http://10.126.36.140:8000/v1"}
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

ENV_PREFIX = "BANKING_SIM_"

DEFAULTS = {
    "base_url": "http://localhost:8000/v1",
    "model": "/home/s3lab-spark/LG2026/models/Qwen3-8B-unsloth-bnb-4bit",
    "api_key": "EMPTY",
}


def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def resolve(key: str, cli_value: Optional[str], config: Dict[str, Any]) -> str:
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    env_value = os.environ.get(f"{ENV_PREFIX}{key.upper()}")
    if env_value is not None:
        return env_value
    return DEFAULTS[key]
