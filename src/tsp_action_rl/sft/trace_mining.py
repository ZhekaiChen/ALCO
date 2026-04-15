"""Loading, filtering, and export utilities for SFT trace mining."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .models import LoadedTraceCorpus, StepTraceRecord, TraceFilterConfig, TraceFilterResult


class TraceMiningError(RuntimeError):
    """Raised when trace inputs are malformed or cannot be loaded."""


def discover_run_directories(
    *,
    run_directories: Sequence[str | Path] | None = None,
    root_directories: Sequence[str | Path] | None = None,
) -> list[Path]:
    """Discover run directories from explicit run paths and/or root scans."""
    discovered: dict[str, Path] = {}

    def _add_run_dir(path: Path) -> None:
        resolved = path.resolve()
        if not resolved.exists():
            raise TraceMiningError(f"Run directory does not exist: {resolved}")
        if not resolved.is_dir():
            raise TraceMiningError(f"Run path must be a directory: {resolved}")
        discovered[str(resolved)] = resolved

    for raw in run_directories or ():
        _add_run_dir(Path(raw))

    for raw_root in root_directories or ():
        root = Path(raw_root).resolve()
        if not root.exists():
            raise TraceMiningError(f"Input root does not exist: {root}")
        if not root.is_dir():
            raise TraceMiningError(f"Input root must be a directory: {root}")

        if _looks_like_run_directory(root):
            _add_run_dir(root)
            continue

        root_has_summary_run = False
        for summary_path in sorted(root.rglob("run_summary.json")):
            _add_run_dir(summary_path.parent)
            root_has_summary_run = True

        if root_has_summary_run:
            continue
        for episodes_dir in sorted(root.rglob("episodes")):
            if episodes_dir.is_dir() and any(episodes_dir.glob("*.json")):
                _add_run_dir(episodes_dir.parent)

    if not discovered:
        raise TraceMiningError("No run directories were discovered from provided inputs.")
    return sorted(discovered.values())


def load_step_traces_from_runs(run_dirs: Sequence[str | Path]) -> LoadedTraceCorpus:
    """Load and index step-level records from one or more run directories."""
    all_records: list[StepTraceRecord] = []
    run_names: list[str] = []
    episodes_scanned = 0

    resolved_run_dirs: list[Path] = []
    for raw in run_dirs:
        path = Path(raw).resolve()
        if not path.exists() or not path.is_dir():
            raise TraceMiningError(f"Run directory does not exist: {path}")
        resolved_run_dirs.append(path)

    for run_dir in resolved_run_dirs:
        run_summary_path = run_dir / "run_summary.json"
        run_summary = _load_json_object(run_summary_path) if run_summary_path.exists() else None
        run_name = _as_nonempty_str(_get_mapping_value(run_summary, "run_name"), default=run_dir.name)
        run_names.append(run_name)

        episode_paths = _resolve_episode_paths(run_dir=run_dir, run_summary=run_summary)
        for episode_path in episode_paths:
            episode = _load_json_object(episode_path)
            episodes_scanned += 1
            all_records.extend(
                _index_episode_steps(
                    run_name=run_name,
                    run_dir=run_dir,
                    run_summary_path=run_summary_path if run_summary_path.exists() else None,
                    run_summary=run_summary,
                    episode_path=episode_path,
                    episode=episode,
                )
            )

    return LoadedTraceCorpus(
        run_dirs=tuple(str(path) for path in resolved_run_dirs),
        run_names=tuple(run_names),
        episodes_scanned=episodes_scanned,
        steps_scanned=len(all_records),
        step_records=tuple(all_records),
    )


def filter_step_traces(records: Sequence[StepTraceRecord], config: TraceFilterConfig) -> TraceFilterResult:
    """Apply configurable high-quality filters to indexed step records."""
    kept: list[StepTraceRecord] = []
    dropped_reasons: Counter[str] = Counter()

    node_count_filter = None if config.node_counts is None else set(config.node_counts)
    model_id_filter = None if config.model_ids is None else set(config.model_ids)

    for record in records:
        reasons: list[str] = []
        reasoning_len = len(record.reasoning_text)

        if config.require_parse_success and record.final_tag_status != "success":
            reasons.append("require_parse_success")
        if config.require_valid_action and record.action_is_valid is not True:
            reasons.append("require_valid_action")
        if config.require_episode_success and record.episode_status != "success":
            reasons.append("require_episode_success")
        if config.require_solver_completion and record.solver_status != "success":
            reasons.append("require_solver_completion")
        if config.max_final_gap_to_reference is not None:
            if record.final_gap_to_reference is None or record.final_gap_to_reference > config.max_final_gap_to_reference:
                reasons.append("max_final_gap_to_reference")
        if config.max_step_gap_to_reference is not None:
            if record.step_gap_to_reference is None or record.step_gap_to_reference > config.max_step_gap_to_reference:
                reasons.append("max_step_gap_to_reference")
        if node_count_filter is not None and record.node_count not in node_count_filter:
            reasons.append("node_counts")
        if model_id_filter is not None:
            if record.model_id is None or record.model_id not in model_id_filter:
                reasons.append("model_ids")
        if config.min_reasoning_chars is not None and reasoning_len < config.min_reasoning_chars:
            reasons.append("min_reasoning_chars")
        if config.max_reasoning_chars is not None and reasoning_len > config.max_reasoning_chars:
            reasons.append("max_reasoning_chars")

        if reasons:
            for reason in reasons:
                dropped_reasons[reason] += 1
            continue
        kept.append(record)

    return TraceFilterResult(kept_records=tuple(kept), dropped_reason_counts=dict(dropped_reasons))


def build_internal_sft_examples(records: Sequence[StepTraceRecord]) -> list[dict[str, Any]]:
    """Build project-internal JSONL SFT examples from selected step records."""
    examples: list[dict[str, Any]] = []
    for record in records:
        final_tag_output = (
            None
            if record.parsed_next_node is None
            else f"<FINAL_NEXT_NODE>{record.parsed_next_node}</FINAL_NEXT_NODE>"
        )
        assistant_text = _build_assistant_text(
            reasoning_text=record.reasoning_text,
            final_tag_output=final_tag_output,
        )

        examples.append(
            {
                "prompt_text": record.prompt_text,
                "reasoning_text": record.reasoning_text,
                "final_tag_output": final_tag_output,
                "next_node_label": record.parsed_next_node,
                "assistant_text": assistant_text,
                "source": record.source_mapping(),
            }
        )
    return examples


def build_chat_style_examples(records: Sequence[StepTraceRecord]) -> list[dict[str, Any]]:
    """Build a simple chat-style JSONL export format from selected step records."""
    examples: list[dict[str, Any]] = []
    for record in records:
        final_tag_output = (
            None
            if record.parsed_next_node is None
            else f"<FINAL_NEXT_NODE>{record.parsed_next_node}</FINAL_NEXT_NODE>"
        )
        assistant_text = _build_assistant_text(
            reasoning_text=record.reasoning_text,
            final_tag_output=final_tag_output,
        )
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": record.prompt_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                "next_node_label": record.parsed_next_node,
                "source": record.source_mapping(),
            }
        )
    return examples


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write JSONL rows with stable UTF-8 serialization."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return target


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write pretty JSON summary payload."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def build_export_summary(
    *,
    corpus: LoadedTraceCorpus,
    filter_config: TraceFilterConfig,
    filter_result: TraceFilterResult,
    output_jsonl: str | None,
    output_chat_jsonl: str | None,
) -> dict[str, Any]:
    """Build scan/selection/export summary for SFT mining runs."""
    kept = list(filter_result.kept_records)
    kept_count = len(kept)
    parse_success_count = sum(1 for record in kept if record.final_tag_status == "success")
    valid_action_count = sum(1 for record in kept if record.action_is_valid is True)
    reasoning_lengths = [len(record.reasoning_text) for record in kept]

    node_counts = Counter(record.node_count for record in kept)
    model_ids = Counter((record.model_id or "unknown") for record in kept)

    return {
        "runs_scanned": len(corpus.run_dirs),
        "run_names": list(corpus.run_names),
        "episodes_scanned": corpus.episodes_scanned,
        "steps_scanned": corpus.steps_scanned,
        "steps_kept": kept_count,
        "retention_rate": 0.0 if corpus.steps_scanned == 0 else kept_count / corpus.steps_scanned,
        "counts_by_node_count": {str(key): int(value) for key, value in sorted(node_counts.items())},
        "counts_by_model_id": {str(key): int(value) for key, value in sorted(model_ids.items())},
        "reasoning_length_chars": _length_stats(reasoning_lengths),
        "parse_success_rate_among_kept": 0.0 if kept_count == 0 else parse_success_count / kept_count,
        "valid_action_rate_among_kept": 0.0 if kept_count == 0 else valid_action_count / kept_count,
        "dropped_reason_counts": dict(filter_result.dropped_reason_counts),
        "filter_config": filter_config.to_dict(),
        "output_jsonl": output_jsonl,
        "output_chat_jsonl": output_chat_jsonl,
    }


def _looks_like_run_directory(path: Path) -> bool:
    return (path / "run_summary.json").exists() or (path / "episodes").is_dir()


def _resolve_episode_paths(*, run_dir: Path, run_summary: Mapping[str, Any] | None) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()

    episode_files_raw = _get_mapping_value(run_summary, "episode_files")
    if isinstance(episode_files_raw, list):
        for raw in episode_files_raw:
            if not isinstance(raw, str) or not raw.strip():
                continue
            path = _resolve_episode_path(raw=raw, run_dir=run_dir)
            if path is None:
                continue
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)

    if paths:
        return paths

    episodes_dir = run_dir / "episodes"
    if episodes_dir.is_dir():
        for path in sorted(episodes_dir.glob("*.json")):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    else:
        for path in sorted(run_dir.glob("*.json")):
            if path.name == "run_summary.json":
                continue
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def _resolve_episode_path(*, raw: str, run_dir: Path) -> Path | None:
    candidate = Path(raw)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                candidate,
                run_dir / candidate,
                run_dir / "episodes" / candidate.name,
            ]
        )
    for item in candidates:
        if item.exists() and item.is_file():
            return item
    return None


def _index_episode_steps(
    *,
    run_name: str,
    run_dir: Path,
    run_summary_path: Path | None,
    run_summary: Mapping[str, Any] | None,
    episode_path: Path,
    episode: Mapping[str, Any],
) -> list[StepTraceRecord]:
    episode_id = _as_nonempty_str(episode.get("episode_id"), default=episode_path.stem)
    instance_id = _as_nonempty_str(episode.get("instance_id"), default="unknown_instance")
    node_count = _as_int(episode.get("node_count"), default=0)
    episode_status = _as_nonempty_str(episode.get("status"), default="unknown")
    episode_metadata = _as_mapping(episode.get("metadata"))
    summary_metrics = _as_mapping(episode.get("summary_metrics"))
    final_gap = _as_float(summary_metrics.get("final_gap_to_reference"))

    run_model_id = _as_nonempty_str(
        _get_mapping_value(_as_mapping(_get_mapping_value(_as_mapping(_get_mapping_value(run_summary, "config")), "api")), "model_id"),
        default=None,
    )
    run_model_name = _as_nonempty_str(_get_mapping_value(run_summary, "model_name"), default=None)
    episode_model_name = _as_nonempty_str(episode_metadata.get("model_name"), default=None)
    episode_model_id = _as_nonempty_str(episode_metadata.get("model_id"), default=None)

    step_logs = episode.get("step_logs")
    if not isinstance(step_logs, list):
        raise TraceMiningError(f"Episode log missing step_logs list: {episode_path}")

    records: list[StepTraceRecord] = []
    for position, step in enumerate(step_logs):
        if not isinstance(step, Mapping):
            continue
        step_metadata = _as_mapping(step.get("metadata"))
        final_tag_parse = _as_mapping(step.get("final_tag_parse"))
        action_validation = _as_mapping(step.get("action_validation"))
        solver_completion = _as_mapping(step.get("solver_completion"))
        reward_signal = _as_mapping(step.get("reward_signal"))
        reward_components = _as_mapping(reward_signal.get("components"))

        model_id = _first_nonempty_str(
            [
                step_metadata.get("model_id"),
                episode_model_id,
                run_model_id,
            ]
        )
        model_name = _first_nonempty_str(
            [
                step_metadata.get("model_name"),
                episode_model_name,
                run_model_name,
            ]
        )

        step_instance_id = _as_nonempty_str(step.get("instance_id"), default=instance_id)
        step_node_count = _as_int(_get_mapping_value(_as_mapping(step.get("state_before")), "node_count"), default=node_count)

        records.append(
            StepTraceRecord(
                run_name=run_name,
                run_dir=str(run_dir),
                run_summary_path=None if run_summary_path is None else str(run_summary_path),
                episode_path=str(episode_path),
                episode_id=_as_nonempty_str(step.get("episode_id"), default=episode_id),
                instance_id=step_instance_id,
                node_count=step_node_count,
                step_index=_as_int(step.get("step_index"), default=position),
                source_step_position=position,
                model_id=model_id,
                model_name=model_name,
                prompt_text=_as_nonempty_str(step.get("prompt_text"), default=""),
                reasoning_text=_as_nonempty_str(step.get("reasoning_text"), default=""),
                raw_model_output=_as_nonempty_str(step.get("raw_model_output"), default=""),
                final_tag_status=_as_nonempty_str(final_tag_parse.get("status"), default=None),
                final_tag_count=_as_int(final_tag_parse.get("tag_count"), default=None),
                parsed_next_node=_as_int(final_tag_parse.get("parsed_next_node"), default=None),
                action_is_valid=_as_bool(action_validation.get("is_valid")),
                action_failure_reason=_as_nonempty_str(action_validation.get("failure_reason"), default=None),
                episode_status=episode_status,
                solver_status=_as_nonempty_str(solver_completion.get("status"), default=None),
                final_gap_to_reference=final_gap,
                step_gap_to_reference=_as_float(reward_components.get("gap_to_reference")),
                reward_value=_as_float(reward_signal.get("reward_value")),
            )
        )
    return records


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TraceMiningError(f"Failed to read JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TraceMiningError(f"Failed to parse JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise TraceMiningError(f"Expected top-level JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _get_mapping_value(mapping: Mapping[str, Any] | None, key: str) -> Any:
    if mapping is None:
        return None
    return mapping.get(key)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _as_nonempty_str(value: Any, *, default: str | None) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def _as_int(value: Any, *, default: int | None) -> int | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _first_nonempty_str(values: Sequence[Any]) -> str | None:
    for value in values:
        maybe = _as_nonempty_str(value, default=None)
        if maybe is not None:
            return maybe
    return None


def _build_assistant_text(*, reasoning_text: str, final_tag_output: str | None) -> str:
    if final_tag_output is None:
        return reasoning_text
    if reasoning_text.strip():
        return f"{reasoning_text}\n\n{final_tag_output}"
    return final_tag_output


def _length_stats(lengths: Sequence[int]) -> dict[str, float | int | None]:
    if not lengths:
        return {"min": None, "max": None, "mean": None, "std": None}
    length_values = [int(value) for value in lengths]
    mean = sum(length_values) / len(length_values)
    variance = sum((value - mean) ** 2 for value in length_values) / len(length_values)
    return {
        "min": int(min(length_values)),
        "max": int(max(length_values)),
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
    }
