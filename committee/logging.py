"""Optional JSONL run logger.

Writes one JSONL line per LLM call so a run can be reproduced and audited
after the fact. Off by default — pass `LLMClient(logger=RunLogger("run.jsonl"))`
to turn on.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    """Append one JSON object per LLM call to a file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def write(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
