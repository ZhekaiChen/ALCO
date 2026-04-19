"""RL environment configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from tsp_action_rl.rl import RLEnvironmentConfigError, TSPRLEnvSettings


def load_rl_env_settings(path: str | Path = "configs/rl.yaml") -> TSPRLEnvSettings:
    """Load and validate RL environment settings from YAML config."""
    config_path = Path(path)
    if not config_path.exists():
        raise RLEnvironmentConfigError(f"RL config file not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise RLEnvironmentConfigError(f"Expected YAML object at top-level in {config_path}.")

    return TSPRLEnvSettings.from_mapping(payload)


def dump_rl_env_settings(settings: TSPRLEnvSettings, path: str | Path) -> None:
    """Write RL environment settings snapshot to YAML."""
    payload: dict[str, Any] = settings.to_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
