#!/usr/bin/env python3
"""Mine and export SFT-ready per-step traces from zero-shot rollout logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

from tsp_action_rl.sft import (
    TraceFilterConfig,
    build_chat_style_examples,
    build_export_summary,
    build_internal_sft_examples,
    discover_run_directories,
    filter_step_traces,
    load_step_traces_from_runs,
    write_json,
    write_jsonl,
)


def _read_yaml_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at top-level in {path}.")
    return payload


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _parse_optional_csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not chunks:
        return None
    return tuple(int(chunk) for chunk in chunks)


def _parse_optional_csv_strings(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not chunks:
        return None
    return tuple(chunks)


def _csv_default(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return ",".join(str(item) for item in raw)
    return None


def _build_arg_parser(defaults: Mapping[str, Any]) -> argparse.ArgumentParser:
    filter_defaults = _as_mapping(defaults.get("filters"))
    output_defaults = _as_mapping(defaults.get("output"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    parser.add_argument("--input-run-dir", action="append", default=None)
    parser.add_argument("--input-root", action="append", default=None)

    parser.add_argument("--output-jsonl", type=Path, default=output_defaults.get("jsonl_path"))
    parser.add_argument("--output-chat-jsonl", type=Path, default=output_defaults.get("chat_jsonl_path"))
    parser.add_argument("--summary-path", type=Path, default=output_defaults.get("summary_path"))
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=bool(output_defaults.get("dry_run", False)),
    )

    parser.add_argument(
        "--require-parse-success",
        action=argparse.BooleanOptionalAction,
        default=bool(filter_defaults.get("require_parse_success", True)),
    )
    parser.add_argument(
        "--require-valid-action",
        action=argparse.BooleanOptionalAction,
        default=bool(filter_defaults.get("require_valid_action", True)),
    )
    parser.add_argument(
        "--require-episode-success",
        action=argparse.BooleanOptionalAction,
        default=bool(filter_defaults.get("require_episode_success", False)),
    )
    parser.add_argument(
        "--require-solver-completion",
        action=argparse.BooleanOptionalAction,
        default=bool(filter_defaults.get("require_solver_completion", False)),
    )
    parser.add_argument(
        "--max-final-gap-to-reference",
        type=float,
        default=filter_defaults.get("max_final_gap_to_reference"),
    )
    parser.add_argument(
        "--max-step-gap-to-reference",
        type=float,
        default=filter_defaults.get("max_step_gap_to_reference"),
    )
    parser.add_argument("--node-counts", type=str, default=_csv_default(filter_defaults.get("node_counts")))
    parser.add_argument("--model-ids", type=str, default=_csv_default(filter_defaults.get("model_ids")))
    parser.add_argument("--min-reasoning-chars", type=int, default=filter_defaults.get("min_reasoning_chars"))
    parser.add_argument("--max-reasoning-chars", type=int, default=filter_defaults.get("max_reasoning_chars"))
    return parser


def _build_filter_config(args: argparse.Namespace) -> TraceFilterConfig:
    return TraceFilterConfig(
        require_parse_success=args.require_parse_success,
        require_valid_action=args.require_valid_action,
        require_episode_success=args.require_episode_success,
        require_solver_completion=args.require_solver_completion,
        max_final_gap_to_reference=args.max_final_gap_to_reference,
        max_step_gap_to_reference=args.max_step_gap_to_reference,
        node_counts=_parse_optional_csv_ints(args.node_counts),
        model_ids=_parse_optional_csv_strings(args.model_ids),
        min_reasoning_chars=args.min_reasoning_chars,
        max_reasoning_chars=args.max_reasoning_chars,
    )


def _resolve_input_sources(
    *,
    defaults: Mapping[str, Any],
    input_run_dirs: list[str] | None,
    input_roots: list[str] | None,
) -> tuple[list[str], list[str]]:
    input_defaults = _as_mapping(defaults.get("input"))
    default_run_dirs = list(input_defaults.get("run_dirs", []))
    default_roots = list(input_defaults.get("root_dirs", []))

    has_cli_override = input_run_dirs is not None or input_roots is not None
    if has_cli_override:
        return (list(input_run_dirs or []), list(input_roots or []))
    return (default_run_dirs, default_roots)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    pre_args, _ = pre.parse_known_args()
    defaults = _read_yaml_defaults(pre_args.config)
    parser = _build_arg_parser(defaults)
    args = parser.parse_args()

    input_run_dirs, input_roots = _resolve_input_sources(
        defaults=defaults,
        input_run_dirs=args.input_run_dir,
        input_roots=args.input_root,
    )
    run_dirs = discover_run_directories(
        run_directories=input_run_dirs,
        root_directories=input_roots,
    )
    filter_config = _build_filter_config(args)
    corpus = load_step_traces_from_runs(run_dirs)
    filter_result = filter_step_traces(corpus.step_records, filter_config)

    output_jsonl_path: str | None = None
    output_chat_jsonl_path: str | None = None
    if not args.dry_run:
        if args.output_jsonl is None:
            raise ValueError("output_jsonl is required unless --dry-run is enabled.")
        sft_rows = build_internal_sft_examples(filter_result.kept_records)
        output_jsonl_path = str(write_jsonl(sft_rows, args.output_jsonl))

        if args.output_chat_jsonl is not None:
            chat_rows = build_chat_style_examples(filter_result.kept_records)
            output_chat_jsonl_path = str(write_jsonl(chat_rows, args.output_chat_jsonl))

    summary = build_export_summary(
        corpus=corpus,
        filter_config=filter_config,
        filter_result=filter_result,
        output_jsonl=output_jsonl_path,
        output_chat_jsonl=output_chat_jsonl_path,
    )
    if args.summary_path is not None:
        summary_path = write_json(summary, args.summary_path)
        print(f"Summary written to: {summary_path}")

    print(
        f"Scanned runs={summary['runs_scanned']} episodes={summary['episodes_scanned']} "
        f"steps={summary['steps_scanned']} kept={summary['steps_kept']}"
    )
    if args.dry_run:
        print("Dry-run mode enabled: no JSONL examples written.")
    else:
        print(f"SFT JSONL: {output_jsonl_path}")
        if output_chat_jsonl_path is not None:
            print(f"Chat JSONL: {output_chat_jsonl_path}")


if __name__ == "__main__":
    main()
