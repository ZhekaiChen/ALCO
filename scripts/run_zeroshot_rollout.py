#!/usr/bin/env python3
"""Run zero-shot action-level TSP rollout episodes and save structured logs."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from tsp_action_rl.config import load_lkh_settings
from tsp_action_rl.data import TSPInstance, generate_random_euclidean_instance, load_tsp_instance, save_tsp_instance
from tsp_action_rl.inference import (
    DmxOpenAICompatibleConfig,
    available_dmx_model_profiles,
    build_model_backend,
    supported_backends,
)
from tsp_action_rl.rollout import RolloutProgressUpdate, ZeroShotRolloutConfig, ZeroShotRolloutRunner, save_json
from tsp_action_rl.solvers import LKHIntegration


def _read_yaml_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at top-level in {path}.")
    return payload


def _parse_coordinate_range(raw: Any) -> tuple[float, float]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"coordinate_range must be a 2-item list, got: {raw!r}")
    low = float(raw[0])
    high = float(raw[1])
    if high <= low:
        raise ValueError(f"coordinate_range must satisfy min < max, got ({low}, {high}).")
    return low, high


def _parse_node_counts(raw: Any) -> list[int]:
    if isinstance(raw, str):
        values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError(f"node_counts must be a list or comma-separated string, got {type(raw).__name__}.")

    parsed: list[int] = []
    for value in values:
        count = int(value)
        if count < 2:
            raise ValueError(f"Each node count must be >= 2, got {count}.")
        parsed.append(count)
    if not parsed:
        raise ValueError("node_counts must not be empty.")
    return parsed


def _parse_csv_fields(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _resolve_api_defaults(defaults: Mapping[str, Any]) -> dict[str, Any]:
    api_raw = defaults.get("api", {})
    if api_raw is None:
        api_raw = {}
    if not isinstance(api_raw, Mapping):
        raise ValueError("api config block must be a mapping/object.")
    return dict(api_raw)


def _build_arg_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    coord_min_default, coord_max_default = _parse_coordinate_range(defaults.get("coordinate_range", [0, 10000]))
    node_counts_default = _parse_node_counts(defaults.get("node_counts", [10, 25, 50]))
    node_counts_default_text = ",".join(str(v) for v in node_counts_default)
    api_defaults = _resolve_api_defaults(defaults)
    api_debug_defaults = api_defaults.get("debug", {})
    if api_debug_defaults is None:
        api_debug_defaults = {}
    if not isinstance(api_debug_defaults, Mapping):
        raise ValueError("api.debug must be a mapping/object.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/zeroshot_eval.yaml"))

    parser.add_argument("--instance-path", type=Path, default=defaults.get("instance_path"))
    parser.add_argument(
        "--generate-random",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("generate_random", True)),
    )
    parser.add_argument(
        "--save-generated-instances",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("save_generated_instances", True)),
    )
    parser.add_argument("--instance-id-prefix", type=str, default=str(defaults.get("instance_id_prefix", "tsp")))
    parser.add_argument("--node-counts", type=str, default=node_counts_default_text)
    parser.add_argument(
        "--instances-per-node-count",
        type=int,
        default=int(defaults.get("instances_per_node_count", 1)),
    )
    parser.add_argument("--random-seed", type=int, default=int(defaults.get("random_seed", 12345)))
    parser.add_argument("--coord-min", type=float, default=coord_min_default)
    parser.add_argument("--coord-max", type=float, default=coord_max_default)
    parser.add_argument(
        "--integer-coordinates",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("integer_coordinates", True)),
    )
    parser.add_argument("--episodes-per-instance", type=int, default=int(defaults.get("episodes_per_instance", 1)))

    parser.add_argument("--backend", choices=supported_backends(), default=defaults.get("backend", "local_deterministic"))
    parser.add_argument("--model-name", type=str, default=defaults.get("model_name"))
    parser.add_argument("--static-response-path", type=str, default=defaults.get("static_response_path"))

    parser.add_argument("--api-base-url", type=str, default=api_defaults.get("base_url"))
    parser.add_argument("--api-base-url-env", type=str, default=api_defaults.get("base_url_env", "DMXAPI_BASE_URL"))
    parser.add_argument("--api-key-env", type=str, default=api_defaults.get("api_key_env", "DMXAPI_API_KEY"))
    parser.add_argument("--api-endpoint-path", type=str, default=api_defaults.get("endpoint_path", "/chat/completions"))
    parser.add_argument(
        "--api-model-profile",
        type=str,
        default=None,
        help=(
            "Named DMX request-shaping profile "
            f"(supported: {', '.join(available_dmx_model_profiles())})."
        ),
    )
    parser.add_argument(
        "--api-model-id",
        type=str,
        default=None,
    )
    parser.add_argument("--api-thinking-effort", type=str, default=None)
    parser.add_argument(
        "--api-thinking-effort-field",
        type=str,
        default=None,
    )
    parser.add_argument("--api-max-tokens", type=int, default=int(api_defaults.get("max_tokens", 4096)))
    parser.add_argument("--api-temperature", type=float, default=float(api_defaults.get("temperature", 0.2)))
    parser.add_argument("--api-top-p", type=float, default=float(api_defaults.get("top_p", 1.0)))
    parser.add_argument("--api-timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--api-omit-request-fields",
        type=str,
        default=None,
        help="Comma-separated request body fields to omit (applied after profile + extra_body merge).",
    )
    parser.add_argument(
        "--api-debug-enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(api_debug_defaults.get("enabled", False)),
    )
    parser.add_argument(
        "--api-debug-output-root",
        type=str,
        default=str(api_debug_defaults.get("output_root", "outputs/debug/dmxapi")),
    )
    parser.add_argument(
        "--api-dry-run",
        action=argparse.BooleanOptionalAction,
        default=bool(api_debug_defaults.get("dry_run", False)),
    )

    parser.add_argument(
        "--include-current-position",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("include_current_position", True)),
    )
    parser.add_argument(
        "--include-visited-nodes",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("include_visited_nodes", False)),
    )
    parser.add_argument(
        "--include-unvisited-nodes",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("include_unvisited_nodes", True)),
    )

    parser.add_argument(
        "--start-node-policy",
        choices=["fixed", "random"],
        default=str(defaults.get("start_node_policy", "random")),
    )
    parser.add_argument("--fixed-start-node", type=int, default=int(defaults.get("fixed_start_node", 1)))
    parser.add_argument(
        "--rollout-step-policy",
        choices=["until_terminal", "node_count_minus_2", "fixed"],
        default=str(defaults.get("rollout_step_policy", "node_count_minus_2")),
    )
    parser.add_argument("--fixed-prediction-steps", type=int, default=defaults.get("fixed_prediction_steps"))
    parser.add_argument(
        "--auto-complete-last-node",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("auto_complete_last_node", True)),
    )
    parser.add_argument(
        "--close-tour-to-start",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("close_tour_to_start", True)),
    )
    parser.add_argument("--max-steps", type=int, default=defaults.get("max_steps"))
    parser.add_argument("--max-step-retries", type=int, default=int(defaults.get("max_step_retries", 0)))
    parser.add_argument(
        "--retry-on-parse-failure",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("retry_on_parse_failure", True)),
    )
    parser.add_argument(
        "--retry-on-provider-error",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("retry_on_provider_error", False)),
    )
    parser.add_argument("--retry-backoff-seconds", type=float, default=float(defaults.get("retry_backoff_seconds", 0.0)))

    parser.add_argument(
        "--enable-solver-completion",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("enable_solver_completion", True)),
    )
    parser.add_argument("--lkh-config", type=Path, default=Path(defaults.get("lkh_config", "configs/lkh.yaml")))

    parser.add_argument("--output-root", type=Path, default=Path(defaults.get("output_root", "outputs/zeroshot")))
    parser.add_argument("--run-name", type=str, default=defaults.get("run_name"))
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("show_progress", True)),
        help="Print operator-facing progress line for each rollout step.",
    )
    return parser


def _build_api_config(args: argparse.Namespace, defaults: Mapping[str, Any]) -> dict[str, Any]:
    api_defaults = _resolve_api_defaults(defaults)
    api_defaults.update(
        {
            "base_url": args.api_base_url,
            "base_url_env": args.api_base_url_env,
            "api_key_env": args.api_key_env,
            "endpoint_path": args.api_endpoint_path,
            "max_tokens": args.api_max_tokens,
            "temperature": args.api_temperature,
            "top_p": args.api_top_p,
            "debug": {
                "enabled": args.api_debug_enabled,
                "output_root": args.api_debug_output_root,
                "dry_run": args.api_dry_run,
            },
        }
    )
    if args.api_model_profile is not None:
        api_defaults["model_profile"] = args.api_model_profile
        if args.api_model_id is None:
            api_defaults.pop("model_id", None)
        if args.api_thinking_effort is None:
            api_defaults.pop("thinking_effort", None)
        if args.api_thinking_effort_field is None:
            api_defaults.pop("thinking_effort_field", None)
    if args.api_model_id is not None:
        api_defaults["model_id"] = args.api_model_id
    if args.api_thinking_effort is not None:
        api_defaults["thinking_effort"] = args.api_thinking_effort
    if args.api_thinking_effort_field is not None:
        api_defaults["thinking_effort_field"] = args.api_thinking_effort_field
    if args.api_timeout_seconds is not None:
        api_defaults["timeout_seconds"] = args.api_timeout_seconds
    if args.api_omit_request_fields is not None:
        api_defaults["omit_request_fields"] = _parse_csv_fields(args.api_omit_request_fields)
    return dict(api_defaults)


def _resolve_model_name(backend: str, explicit_model_name: str | None, api_config: Mapping[str, Any]) -> str:
    if explicit_model_name is not None and explicit_model_name.strip():
        return explicit_model_name.strip()
    if backend == "dmx_openai_compatible":
        return DmxOpenAICompatibleConfig.from_mapping(api_config).model_id
    if backend == "local_vllm_openai_compatible":
        return DmxOpenAICompatibleConfig.from_mapping(api_config).model_id
    if backend == "local_deterministic":
        return "local-deterministic-nearest-neighbor"
    if backend == "local_static":
        return "local-static-response"
    if backend == "api_todo":
        return "api-todo"
    return ""


def _generate_random_instances(args: argparse.Namespace) -> tuple[list[TSPInstance], list[dict[str, Any]]]:
    if args.instances_per_node_count < 1:
        raise ValueError(f"instances_per_node_count must be >= 1, got {args.instances_per_node_count}.")
    if args.episodes_per_instance < 1:
        raise ValueError(f"episodes_per_instance must be >= 1, got {args.episodes_per_instance}.")
    if args.coord_max <= args.coord_min:
        raise ValueError(f"coord range must satisfy coord_min < coord_max, got ({args.coord_min}, {args.coord_max}).")

    node_counts = _parse_node_counts(args.node_counts)
    rng = random.Random(args.random_seed)
    instances: list[TSPInstance] = []
    specs: list[dict[str, Any]] = []

    for node_count in node_counts:
        for index in range(args.instances_per_node_count):
            instance_seed = rng.randint(0, 2_147_483_647)
            instance_id = f"{args.instance_id_prefix}_n{node_count:04d}_i{index:03d}_s{instance_seed}"
            instance = generate_random_euclidean_instance(
                node_count=node_count,
                seed=instance_seed,
                coordinate_range=(args.coord_min, args.coord_max),
                integer_coordinates=args.integer_coordinates,
                instance_id=instance_id,
            )
            instances.append(instance)
            specs.append({"instance_id": instance_id, "node_count": node_count, "seed": instance_seed})

    return instances, specs


def _load_instances(args: argparse.Namespace) -> tuple[list[TSPInstance], list[dict[str, Any]], bool]:
    if args.generate_random:
        instances, specs = _generate_random_instances(args)
        return instances, specs, True

    if args.instance_path is None:
        raise ValueError("Specify --instance-path or set --generate-random.")
    instance = load_tsp_instance(args.instance_path)
    return [instance], [{"instance_id": instance.instance_id, "node_count": instance.node_count}], False


def _summarize_episodes(episode_logs: list[dict[str, Any]]) -> dict[str, Any]:
    if not episode_logs:
        return {
            "total_episodes": 0,
            "status_counts": {},
            "avg_parse_success_rate": 0.0,
            "avg_valid_action_rate": 0.0,
            "avg_final_gap_to_reference": None,
            "total_step_retries": 0,
            "total_step_attempts": 0,
            "episodes_with_retries": 0,
        }

    status_counts = Counter(log["status"] for log in episode_logs)
    parse_avg = sum(float(log["summary_metrics"]["parse_success_rate"]) for log in episode_logs) / len(episode_logs)
    valid_avg = sum(float(log["summary_metrics"]["valid_action_rate"]) for log in episode_logs) / len(episode_logs)
    final_gaps = [
        log["summary_metrics"]["final_gap_to_reference"]
        for log in episode_logs
        if log["summary_metrics"]["final_gap_to_reference"] is not None
    ]
    gap_avg = None if not final_gaps else sum(float(v) for v in final_gaps) / len(final_gaps)
    total_step_retries = sum(int(log.get("metadata", {}).get("total_step_retries", 0)) for log in episode_logs)
    total_step_attempts = sum(int(log.get("metadata", {}).get("total_step_attempts", 0)) for log in episode_logs)
    episodes_with_retries = sum(
        1 for log in episode_logs if int(log.get("metadata", {}).get("steps_with_retries", 0)) > 0
    )

    return {
        "total_episodes": len(episode_logs),
        "status_counts": dict(status_counts),
        "avg_parse_success_rate": parse_avg,
        "avg_valid_action_rate": valid_avg,
        "avg_final_gap_to_reference": gap_avg,
        "total_step_retries": total_step_retries,
        "total_step_attempts": total_step_attempts,
        "episodes_with_retries": episodes_with_retries,
    }


def _format_duration(total_seconds: float) -> str:
    seconds = max(int(total_seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=Path("configs/zeroshot_eval.yaml"))
    pre_args, _ = pre.parse_known_args()
    defaults = _read_yaml_defaults(pre_args.config)
    parser = _build_arg_parser(defaults)
    args = parser.parse_args()

    instances, instance_specs, generated = _load_instances(args)
    api_config = _build_api_config(args, defaults)
    resolved_api_config = DmxOpenAICompatibleConfig.from_mapping(api_config)
    resolved_model_name = _resolve_model_name(args.backend, args.model_name, api_config)

    backend = build_model_backend(
        backend=args.backend,
        model_name=resolved_model_name,
        static_response_path=args.static_response_path,
        api_config=api_config,
    )

    solver = None
    if args.enable_solver_completion:
        solver = LKHIntegration(load_lkh_settings(args.lkh_config))

    rollout_config = ZeroShotRolloutConfig(
        start_node_policy=args.start_node_policy,
        fixed_start_node=args.fixed_start_node,
        rollout_step_policy=args.rollout_step_policy,
        fixed_prediction_steps=args.fixed_prediction_steps,
        auto_complete_last_node=args.auto_complete_last_node,
        close_tour_to_start=args.close_tour_to_start,
        random_seed=args.random_seed,
        max_steps=args.max_steps,
        max_step_retries=args.max_step_retries,
        retry_on_parse_failure=args.retry_on_parse_failure,
        retry_on_provider_error=args.retry_on_provider_error,
        retry_backoff_seconds=args.retry_backoff_seconds,
        enable_solver_completion=args.enable_solver_completion,
        include_current_position=args.include_current_position,
        include_visited_nodes=args.include_visited_nodes,
        include_unvisited_nodes=args.include_unvisited_nodes,
    )
    runner = ZeroShotRolloutRunner(model_backend=backend, config=rollout_config, solver=solver)

    run_name = args.run_name or datetime.utcnow().strftime("zeroshot_%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    episodes_dir = run_dir / "episodes"
    instances_dir = run_dir / "instances"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    if generated and args.save_generated_instances:
        instances_dir.mkdir(parents=True, exist_ok=True)

    all_episode_logs: list[dict[str, Any]] = []
    episode_files: list[str] = []
    saved_instance_files: list[str] = []
    total_episodes_planned = len(instances) * args.episodes_per_instance
    episode_index_offset = 0

    progress_callback = None
    if args.show_progress:

        def _on_progress(update: RolloutProgressUpdate) -> None:
            elapsed_text = _format_duration(update.elapsed_seconds)
            eta_text = _format_duration(update.eta_seconds)
            attempt_text = f"attempt {update.step_attempt}/{update.max_step_retries + 1}"
            retry_text = (
                f" retry {update.step_retry_count}/{update.max_step_retries}"
                if update.step_retry_count > 0 and update.max_step_retries > 0
                else ""
            )
            print(
                (
                    f"[progress] episode {update.episode_index}/{update.total_episodes} "
                    f"step {update.step_index}/{update.expected_steps} "
                    f"{attempt_text}{retry_text} "
                    f"node_count={update.node_count} "
                    f"elapsed={elapsed_text} eta={eta_text}"
                ),
                flush=True,
            )

        progress_callback = _on_progress

    for instance in instances:
        if generated and args.save_generated_instances:
            instance_path = instances_dir / f"{instance.instance_id}.json"
            save_tsp_instance(instance, instance_path)
            saved_instance_files.append(str(instance_path))

        episode_logs = runner.run_episodes(
            instance=instance,
            num_episodes=args.episodes_per_instance,
            episode_id_prefix=f"{run_name}_{instance.instance_id}",
            episode_index_offset=episode_index_offset,
            total_episodes=total_episodes_planned,
            progress_callback=progress_callback,
        )
        episode_index_offset += args.episodes_per_instance
        for episode_log in episode_logs:
            episode_path = episodes_dir / f"{episode_log['episode_id']}.json"
            save_json(episode_log, episode_path)
            episode_files.append(str(episode_path))
        all_episode_logs.extend(episode_logs)

    run_summary = {
        "run_name": run_name,
        "generated_instances": generated,
        "backend": args.backend,
        "model_name": backend.model_name,
        "output_dir": str(run_dir),
        "episode_files": episode_files,
        "generated_instance_files": saved_instance_files,
        "instance_specs": instance_specs,
        "metrics": _summarize_episodes(all_episode_logs),
        "config": {
            "instance_path": None if args.instance_path is None else str(args.instance_path),
            "generate_random": args.generate_random,
            "save_generated_instances": args.save_generated_instances,
            "instance_id_prefix": args.instance_id_prefix,
            "node_counts": _parse_node_counts(args.node_counts),
            "instances_per_node_count": args.instances_per_node_count,
            "random_seed": args.random_seed,
            "coordinate_range": [args.coord_min, args.coord_max],
            "integer_coordinates": args.integer_coordinates,
            "episodes_per_instance": args.episodes_per_instance,
            "start_node_policy": args.start_node_policy,
            "fixed_start_node": args.fixed_start_node,
            "rollout_step_policy": args.rollout_step_policy,
            "fixed_prediction_steps": args.fixed_prediction_steps,
            "auto_complete_last_node": args.auto_complete_last_node,
            "close_tour_to_start": args.close_tour_to_start,
            "max_steps": args.max_steps,
            "max_step_retries": args.max_step_retries,
            "retry_on_parse_failure": args.retry_on_parse_failure,
            "retry_on_provider_error": args.retry_on_provider_error,
            "retry_backoff_seconds": args.retry_backoff_seconds,
            "enable_solver_completion": args.enable_solver_completion,
            "lkh_config": str(args.lkh_config),
            "api": {
                "base_url": args.api_base_url,
                "base_url_env": args.api_base_url_env,
                "api_key_env": args.api_key_env,
                "endpoint_path": args.api_endpoint_path,
                "model_profile": resolved_api_config.model_profile,
                "model_id": resolved_api_config.model_id,
                "require_api_key": bool(api_config.get("require_api_key", True)),
                "thinking_effort": resolved_api_config.thinking_effort,
                "thinking_effort_field": resolved_api_config.thinking_effort_field,
                "max_tokens": args.api_max_tokens,
                "temperature": args.api_temperature,
                "top_p": args.api_top_p,
                "timeout_seconds": resolved_api_config.timeout_seconds,
                "omit_request_fields": list(resolved_api_config.omit_request_fields),
                "provider_name": api_config.get("provider_name"),
                "backend_name": api_config.get("backend_name"),
                "local_model_path": api_config.get("local_model_path"),
                "served_model_name": api_config.get("served_model_name"),
                "server_host": api_config.get("server_host"),
                "server_port": api_config.get("server_port"),
                "server_tensor_parallel_size": api_config.get("server_tensor_parallel_size"),
                "debug": {
                    "enabled": args.api_debug_enabled,
                    "output_root": args.api_debug_output_root,
                    "dry_run": args.api_dry_run,
                },
            },
            "include_current_position": args.include_current_position,
            "include_visited_nodes": args.include_visited_nodes,
            "include_unvisited_nodes": args.include_unvisited_nodes,
            "show_progress": args.show_progress,
        },
    }
    summary_path = save_json(run_summary, run_dir / "run_summary.json")
    print(f"Saved run artifacts to {run_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
