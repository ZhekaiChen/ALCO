"""JSON logging helpers for rollout artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target

