"""LKH solver configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from tsp_action_rl.solvers.lkh_integration import LKHConfigError, LKHSettings


def load_lkh_settings(path: str | Path = "configs/lkh.yaml") -> LKHSettings:
    """Load and validate LKH settings from YAML config."""
    config_path = Path(path)
    if not config_path.exists():
        raise LKHConfigError(f"LKH config file not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise LKHConfigError(f"Expected YAML object at top-level in {config_path}.")

    return LKHSettings.from_mapping(payload)


def dump_lkh_settings(settings: LKHSettings, path: str | Path) -> None:
    """Write settings object to YAML for reproducible experiment snapshots."""
    payload: dict[str, Any] = {
        "solver_executable": settings.solver_executable,
        "source_archive": settings.source_archive,
        "runs": settings.runs,
        "max_trials": settings.max_trials,
        "seed": settings.seed,
        "trace_level": settings.trace_level,
        "time_limit": settings.time_limit,
        "extra_params": dict(settings.extra_params),
        "require_source_archive": settings.require_source_archive,
        "debug": {
            "enabled": settings.debug_enabled,
            "output_root": settings.debug_output_root,
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

