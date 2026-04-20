"""Stable JSON I/O helpers for data layer boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import RolloutState, TSPInstance


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_tsp_instance(path: str | Path) -> TSPInstance:
    return TSPInstance.from_dict(_read_json(path))


def save_tsp_instance(instance: TSPInstance, path: str | Path) -> None:
    _write_json(instance.to_dict(), path)


def load_rollout_state(path: str | Path) -> RolloutState:
    return RolloutState.from_dict(_read_json(path))


def save_rollout_state(state: RolloutState, path: str | Path) -> None:
    _write_json(state.to_dict(), path)

