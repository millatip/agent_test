"""Thread/async-safe append-only JSONL logger for per-request timing records."""

import json
import threading
from pathlib import Path
from typing import Any, Dict


class JsonlLogger:
    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self._path, "a", buffering=1)

    def log(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.close()
