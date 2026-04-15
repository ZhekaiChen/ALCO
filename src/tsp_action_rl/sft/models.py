"""Typed records for SFT trace mining and export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _parse_int_list(raw: Any) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
        return tuple(int(chunk) for chunk in chunks)
    if isinstance(raw, list):
        return tuple(int(value) for value in raw)
    raise ValueError(f"Expected integer list or comma-separated string, got {type(raw).__name__}.")


def _parse_str_list(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
        return tuple(chunks)
    if isinstance(raw, list):
        return tuple(str(value).strip() for value in raw if str(value).strip())
    raise ValueError(f"Expected string list or comma-separated string, got {type(raw).__name__}.")


@dataclass(frozen=True)
class StepTraceRecord:
    """Indexed per-step trace record loaded from rollout episode logs."""

    run_name: str
    run_dir: str
    run_summary_path: str | None
    episode_path: str
    episode_id: str
    instance_id: str
    node_count: int
    step_index: int
    source_step_position: int
    model_id: str | None
    model_name: str | None
    prompt_text: str
    reasoning_text: str
    raw_model_output: str
    final_tag_status: str | None
    final_tag_count: int | None
    parsed_next_node: int | None
    action_is_valid: bool | None
    action_failure_reason: str | None
    episode_status: str
    solver_status: str | None
    final_gap_to_reference: float | None
    step_gap_to_reference: float | None
    reward_value: float | None

    def source_mapping(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "run_summary_path": self.run_summary_path,
            "episode_path": self.episode_path,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "source_step_position": self.source_step_position,
            "instance_id": self.instance_id,
            "node_count": self.node_count,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "episode_status": self.episode_status,
            "final_tag_status": self.final_tag_status,
            "action_is_valid": self.action_is_valid,
            "solver_status": self.solver_status,
            "final_gap_to_reference": self.final_gap_to_reference,
            "step_gap_to_reference": self.step_gap_to_reference,
        }


@dataclass(frozen=True)
class LoadedTraceCorpus:
    """Loaded run/episode/step corpus with scan counts."""

    run_dirs: tuple[str, ...]
    run_names: tuple[str, ...]
    episodes_scanned: int
    steps_scanned: int
    step_records: tuple[StepTraceRecord, ...]


@dataclass(frozen=True)
class TraceFilterConfig:
    """Configurable filters for selecting high-quality rollout steps."""

    require_parse_success: bool = True
    require_valid_action: bool = True
    require_episode_success: bool = False
    require_solver_completion: bool = False
    max_final_gap_to_reference: float | None = None
    max_step_gap_to_reference: float | None = None
    node_counts: tuple[int, ...] | None = None
    model_ids: tuple[str, ...] | None = None
    min_reasoning_chars: int | None = None
    max_reasoning_chars: int | None = None

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None) -> "TraceFilterConfig":
        source = raw or {}
        if not isinstance(source, Mapping):
            raise ValueError(f"filter config must be a mapping/object, got {type(source).__name__}.")
        return TraceFilterConfig(
            require_parse_success=bool(source.get("require_parse_success", True)),
            require_valid_action=bool(source.get("require_valid_action", True)),
            require_episode_success=bool(source.get("require_episode_success", False)),
            require_solver_completion=bool(source.get("require_solver_completion", False)),
            max_final_gap_to_reference=(
                None
                if source.get("max_final_gap_to_reference") is None
                else float(source.get("max_final_gap_to_reference"))
            ),
            max_step_gap_to_reference=(
                None
                if source.get("max_step_gap_to_reference") is None
                else float(source.get("max_step_gap_to_reference"))
            ),
            node_counts=_parse_int_list(source.get("node_counts")),
            model_ids=_parse_str_list(source.get("model_ids")),
            min_reasoning_chars=(
                None if source.get("min_reasoning_chars") is None else int(source.get("min_reasoning_chars"))
            ),
            max_reasoning_chars=(
                None if source.get("max_reasoning_chars") is None else int(source.get("max_reasoning_chars"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_parse_success": self.require_parse_success,
            "require_valid_action": self.require_valid_action,
            "require_episode_success": self.require_episode_success,
            "require_solver_completion": self.require_solver_completion,
            "max_final_gap_to_reference": self.max_final_gap_to_reference,
            "max_step_gap_to_reference": self.max_step_gap_to_reference,
            "node_counts": None if self.node_counts is None else list(self.node_counts),
            "model_ids": None if self.model_ids is None else list(self.model_ids),
            "min_reasoning_chars": self.min_reasoning_chars,
            "max_reasoning_chars": self.max_reasoning_chars,
        }


@dataclass(frozen=True)
class TraceFilterResult:
    """Filtered step records plus dropped-reason counters."""

    kept_records: tuple[StepTraceRecord, ...]
    dropped_reason_counts: dict[str, int]
