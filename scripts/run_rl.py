#!/usr/bin/env python3
"""Run RL training with either SLIME integration or adapter-level validation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from tsp_action_rl.config import load_lkh_settings, load_rl_env_settings
from tsp_action_rl.data import generate_random_euclidean_instance, load_tsp_instance, save_tsp_instance
from tsp_action_rl.rl import (
    TRAINING_ALGORITHM_ACTOR_CRITIC,
    TRAINING_ALGORITHM_GRPO,
    SLIMEAdapterSettings,
    SUPPORTED_TRAINING_ALGORITHMS,
    SUPPORTED_SLIME_POLICIES,
    TSPRLSlimeAdapter,
    TSPRLStepEnvironment,
    apply_slime_training_overrides,
    load_slime_training_settings,
    run_slime_training_job,
    run_slime_train,
)
from tsp_action_rl.rollout import save_json
from tsp_action_rl.solvers import LKHIntegration


def _read_yaml_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected top-level mapping in config: {path}")
    return dict(payload)


def _quick_instance_defaults(defaults: Mapping[str, Any]) -> dict[str, Any]:
    raw = defaults.get("quick_instance")
    if raw is None:
        raw = defaults.get("smoke_instance", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("quick_instance must be a mapping/object.")
    return dict(raw)


def _slime_adapter_defaults(defaults: Mapping[str, Any]) -> dict[str, Any]:
    raw = defaults.get("slime_adapter", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("slime_adapter must be a mapping/object.")
    return dict(raw)


def _parse_pipeline_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"phase6_slime"}:
        return "slime_training"
    if normalized in {"phase5_smoke"}:
        return "adapter_validation"
    if normalized in {"slime_training", "adapter_validation"}:
        return normalized
    raise ValueError(f"Unsupported pipeline '{value}'.")


def _build_arg_parser(defaults: Mapping[str, Any]) -> argparse.ArgumentParser:
    quick_defaults = _quick_instance_defaults(defaults)
    slime_defaults = _slime_adapter_defaults(defaults)
    train_defaults = slime_defaults.get("train", {}) if isinstance(slime_defaults.get("train"), Mapping) else {}

    coord_range = quick_defaults.get("coordinate_range", [0, 1000])
    if not isinstance(coord_range, list) or len(coord_range) != 2:
        raise ValueError("quick_instance.coordinate_range must be a 2-item list.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rl.yaml"))
    parser.add_argument("--lkh-config", type=Path, default=Path("configs/lkh.yaml"))

    parser.add_argument("--instance-path", type=Path, default=quick_defaults.get("instance_path"))
    parser.add_argument(
        "--generate-random",
        action=argparse.BooleanOptionalAction,
        default=bool(quick_defaults.get("generate_random", True)),
    )
    parser.add_argument("--node-count", type=int, default=int(quick_defaults.get("node_count", 10)))
    parser.add_argument("--instance-seed", type=int, default=int(quick_defaults.get("seed", 12345)))
    parser.add_argument("--coord-min", type=float, default=float(coord_range[0]))
    parser.add_argument("--coord-max", type=float, default=float(coord_range[1]))
    parser.add_argument(
        "--integer-coordinates",
        action=argparse.BooleanOptionalAction,
        default=bool(quick_defaults.get("integer_coordinates", True)),
    )
    parser.add_argument("--start-node", type=int, default=None)
    parser.add_argument(
        "--save-generated-instance",
        action=argparse.BooleanOptionalAction,
        default=bool(quick_defaults.get("save_generated_instance", False)),
    )

    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--train-max-steps-per-episode", type=int, default=None)
    parser.add_argument(
        "--train-policy",
        choices=SUPPORTED_SLIME_POLICIES,
        default=None,
    )
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--train-entrypoint", type=str, default=None)
    parser.add_argument(
        "--use-real-slime-rollout-contract",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--slime-repo-path", type=str, default=None)
    parser.add_argument("--slime-rollout-function-path", type=str, default=None)
    parser.add_argument("--slime-eval-function-path", type=str, default=None)
    parser.add_argument("--slime-rollout-batch-size", type=int, default=None)
    parser.add_argument("--slime-n-samples-per-prompt", type=int, default=None)
    parser.add_argument("--slime-eval-dataset-name", type=str, default=None)

    parser.add_argument("--output-root", type=Path, default=Path("outputs/rl"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument(
        "--pipeline",
        type=_parse_pipeline_mode,
        choices=["slime_training", "adapter_validation"],
        default="slime_training",
        help="Choose SLIME training pipeline or adapter-level validation pipeline.",
    )
    parser.add_argument(
        "--slime-algorithm",
        "--phase6-algorithm",
        dest="slime_algorithm",
        choices=SUPPORTED_TRAINING_ALGORITHMS,
        default=None,
        help="Training algorithm: actor_critic (maps to PPO) or grpo.",
    )
    parser.add_argument("--slime-checkpoint-root", "--phase6-checkpoint-root", dest="slime_checkpoint_root", type=str, default=None)
    parser.add_argument("--slime-num-rollout", "--phase6-num-rollout", dest="slime_num_rollout", type=int, default=None)
    parser.add_argument(
        "--training-rollout-batch-size",
        "--phase6-rollout-batch-size",
        dest="training_rollout_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--training-n-samples-per-prompt",
        "--phase6-n-samples-per-prompt",
        dest="training_n_samples_per_prompt",
        type=int,
        default=None,
    )
    parser.add_argument("--slime-eval-interval", "--phase6-eval-interval", dest="slime_eval_interval", type=int, default=None)
    parser.add_argument("--slime-save-interval", "--phase6-save-interval", dest="slime_save_interval", type=int, default=None)
    parser.add_argument("--slime-node-min", "--phase6-node-min", dest="slime_node_min", type=int, default=None)
    parser.add_argument("--slime-node-max", "--phase6-node-max", dest="slime_node_max", type=int, default=None)
    parser.add_argument("--slime-coord-min", "--phase6-coord-min", dest="slime_coord_min", type=float, default=None)
    parser.add_argument("--slime-coord-max", "--phase6-coord-max", dest="slime_coord_max", type=float, default=None)
    parser.add_argument("--slime-seed", "--phase6-seed", dest="slime_seed", type=int, default=None)
    parser.add_argument(
        "--slime-tracking",
        "--phase6-tracking",
        dest="slime_tracking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable swanlab(wandb-style) tracking for training pipeline.",
    )
    parser.add_argument("--slime-wandb-project", "--phase6-wandb-project", dest="slime_wandb_project", type=str, default=None)
    parser.add_argument("--slime-wandb-group", "--phase6-wandb-group", dest="slime_wandb_group", type=str, default=None)
    parser.add_argument(
        "--slime-wandb-mode",
        "--phase6-wandb-mode",
        dest="slime_wandb_mode",
        choices=["online", "offline", "disabled"],
        default=None,
    )
    parser.add_argument("--slime-wandb-team", "--phase6-wandb-team", dest="slime_wandb_team", type=str, default=None)
    parser.add_argument("--slime-wandb-host", "--phase6-wandb-host", dest="slime_wandb_host", type=str, default=None)
    parser.add_argument("--slime-hf-checkpoint", "--phase6-hf-checkpoint", dest="slime_hf_checkpoint", type=str, default=None)
    parser.add_argument("--slime-model-name", "--phase6-model-name", dest="slime_model_name", type=str, default=None)
    parser.add_argument(
        "--slime-load-checkpoint",
        "--phase6-load-checkpoint",
        dest="slime_load_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument("--slime-ref-load", "--phase6-ref-load", dest="slime_ref_load", type=str, default=None)
    parser.add_argument(
        "--plan-only",
        "--phase6-dry-run",
        dest="plan_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build and persist training run plan without launching the runtime.",
    )
    return parser


def _build_adapter_settings(args: argparse.Namespace, defaults: Mapping[str, Any]) -> SLIMEAdapterSettings:
    raw = _slime_adapter_defaults(defaults)
    train_raw = raw.get("train", {}) if isinstance(raw.get("train"), Mapping) else {}
    train_raw = dict(train_raw)

    if args.train_episodes is not None:
        train_raw["episodes"] = args.train_episodes
    if args.train_max_steps_per_episode is not None:
        train_raw["max_steps_per_episode"] = args.train_max_steps_per_episode
    if args.train_policy is not None:
        train_raw["policy"] = args.train_policy
    if args.train_seed is not None:
        train_raw["seed"] = args.train_seed
    if args.slime_rollout_function_path is not None:
        train_raw["slime_rollout_function_path"] = args.slime_rollout_function_path
    if args.slime_eval_function_path is not None:
        train_raw["slime_eval_function_path"] = args.slime_eval_function_path
    if args.slime_rollout_batch_size is not None:
        train_raw["slime_rollout_batch_size"] = args.slime_rollout_batch_size
    if args.slime_n_samples_per_prompt is not None:
        train_raw["slime_n_samples_per_prompt"] = args.slime_n_samples_per_prompt
    if args.slime_eval_dataset_name is not None:
        train_raw["slime_eval_dataset_name"] = args.slime_eval_dataset_name

    raw = dict(raw)
    raw["train"] = train_raw

    if args.train_entrypoint is not None:
        raw["train_entrypoint"] = args.train_entrypoint
    if args.use_real_slime_rollout_contract is not None:
        raw["use_real_slime_rollout_contract"] = args.use_real_slime_rollout_contract
    if args.slime_repo_path is not None:
        raw["slime_repo_path"] = args.slime_repo_path

    return SLIMEAdapterSettings.from_mapping(raw)


def _build_instance(args: argparse.Namespace) -> tuple[Any, bool]:
    if args.generate_random:
        instance = generate_random_euclidean_instance(
            node_count=args.node_count,
            seed=args.instance_seed,
            coordinate_range=(args.coord_min, args.coord_max),
            integer_coordinates=args.integer_coordinates,
        )
        return instance, True

    if args.instance_path is None:
        raise ValueError("Specify --instance-path when --generate-random is false.")
    return load_tsp_instance(args.instance_path), False


def _detect_slime_repo_state(path: Path = Path("third_party/slime")) -> dict[str, Any]:
    exists = path.exists()
    is_dir = path.is_dir()
    non_placeholder_files = 0

    if exists and is_dir:
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            if item.name == ".gitkeep":
                continue
            non_placeholder_files += 1

    return {
        "path": str(path),
        "exists": exists,
        "is_dir": is_dir,
        "non_placeholder_files": non_placeholder_files,
        "is_populated": non_placeholder_files > 0,
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    defaults = _read_yaml_defaults(args.config)
    env_settings = load_rl_env_settings(args.config)
    adapter_settings = _build_adapter_settings(args, defaults)

    instance, generated = _build_instance(args)

    solver = LKHIntegration(load_lkh_settings(args.lkh_config))
    env = TSPRLStepEnvironment(solver=solver, settings=env_settings)
    adapter = TSPRLSlimeAdapter(env=env, settings=adapter_settings)

    adapter.reset(instance=instance, start_node=args.start_node)
    train_summary = run_slime_train(adapter=adapter, settings=adapter_settings)

    run_name = args.run_name or datetime.utcnow().strftime("rl_train_%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_instance_path: str | None = None
    if generated and args.save_generated_instance:
        instance_path = run_dir / f"{instance.instance_id}.json"
        save_tsp_instance(instance, instance_path)
        saved_instance_path = str(instance_path)

    payload = {
        "run_name": run_name,
        "mode": "train",
        "generated_instance": generated,
        "saved_instance_path": saved_instance_path,
        "instance_id": instance.instance_id,
        "node_count": instance.node_count,
        "slime_repo": _detect_slime_repo_state(),
        "config": {
            "rl_config": str(args.config),
            "lkh_config": str(args.lkh_config),
            "env": env_settings.to_dict(),
            "adapter": adapter_settings.to_dict(),
            "quick_instance": {
                "instance_path": None if args.instance_path is None else str(args.instance_path),
                "generate_random": args.generate_random,
                "node_count": args.node_count,
                "instance_seed": args.instance_seed,
                "coordinate_range": [args.coord_min, args.coord_max],
                "integer_coordinates": args.integer_coordinates,
                "start_node": args.start_node,
            },
        },
        "train_summary": train_summary,
    }

    summary_path = save_json(payload, run_dir / "train_summary.json")
    payload["summary_path"] = str(summary_path)
    return payload


def _slime_training_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    training_overrides: dict[str, Any] = {}

    if args.slime_algorithm is not None:
        training_overrides["algorithm"] = args.slime_algorithm
    if args.slime_checkpoint_root is not None:
        training_overrides["checkpoint_output_root"] = args.slime_checkpoint_root
    if args.slime_num_rollout is not None:
        training_overrides["num_rollout"] = args.slime_num_rollout
    if args.training_rollout_batch_size is not None:
        training_overrides["rollout_batch_size"] = args.training_rollout_batch_size
    if args.training_n_samples_per_prompt is not None:
        training_overrides["n_samples_per_prompt"] = args.training_n_samples_per_prompt
    if args.slime_eval_interval is not None:
        training_overrides["eval_interval"] = args.slime_eval_interval
    if args.slime_save_interval is not None:
        training_overrides["save_interval"] = args.slime_save_interval
    if args.slime_seed is not None:
        training_overrides["seed"] = args.slime_seed
    if args.slime_hf_checkpoint is not None:
        training_overrides["hf_checkpoint"] = args.slime_hf_checkpoint
    if args.slime_model_name is not None:
        training_overrides["model_name"] = args.slime_model_name
    if args.slime_load_checkpoint is not None:
        training_overrides["load_checkpoint"] = args.slime_load_checkpoint
    if args.slime_ref_load is not None:
        training_overrides["ref_load"] = args.slime_ref_load

    task_sampling_override: dict[str, Any] = {}
    node_min = args.slime_node_min
    node_max = args.slime_node_max
    if node_min is not None or node_max is not None:
        if node_min is None or node_max is None:
            raise ValueError("Use both --slime-node-min and --slime-node-max together.")
        task_sampling_override["node_count_range"] = [node_min, node_max]

    coord_min = args.slime_coord_min
    coord_max = args.slime_coord_max
    if coord_min is not None or coord_max is not None:
        if coord_min is None or coord_max is None:
            raise ValueError("Use both --slime-coord-min and --slime-coord-max together.")
        task_sampling_override["coordinate_range"] = [coord_min, coord_max]

    if task_sampling_override:
        training_overrides["task_sampling"] = task_sampling_override

    logging_override: dict[str, Any] = {}
    if args.slime_tracking is not None:
        logging_override["enabled"] = args.slime_tracking
    if args.slime_wandb_project is not None:
        logging_override["wandb_project"] = args.slime_wandb_project
    if args.slime_wandb_group is not None:
        logging_override["wandb_group"] = args.slime_wandb_group
    if args.slime_wandb_mode is not None:
        logging_override["wandb_mode"] = args.slime_wandb_mode
    if args.slime_wandb_team is not None:
        logging_override["wandb_team"] = args.slime_wandb_team
    if args.slime_wandb_host is not None:
        logging_override["wandb_host"] = args.slime_wandb_host
    if logging_override:
        training_overrides["logging"] = logging_override

    return training_overrides


def run_slime_training_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_slime_training_settings(config_path=args.config, overrides=_slime_training_overrides_from_args(args))

    if args.run_name is not None:
        settings = apply_slime_training_overrides(settings, run_name=args.run_name)
    if args.slime_algorithm is not None:
        settings = apply_slime_training_overrides(settings, algorithm=args.slime_algorithm)
    if args.slime_checkpoint_root is not None:
        settings = apply_slime_training_overrides(settings, checkpoint_output_root=args.slime_checkpoint_root)
    if args.slime_num_rollout is not None:
        settings = apply_slime_training_overrides(settings, num_rollout=args.slime_num_rollout)
    if args.training_rollout_batch_size is not None:
        settings = apply_slime_training_overrides(settings, rollout_batch_size=args.training_rollout_batch_size)
    if args.training_n_samples_per_prompt is not None:
        settings = apply_slime_training_overrides(settings, n_samples_per_prompt=args.training_n_samples_per_prompt)
    if args.slime_eval_interval is not None:
        settings = apply_slime_training_overrides(settings, eval_interval=args.slime_eval_interval)
    if args.slime_save_interval is not None:
        settings = apply_slime_training_overrides(settings, save_interval=args.slime_save_interval)

    if settings.algorithm == TRAINING_ALGORITHM_ACTOR_CRITIC and settings.n_samples_per_prompt < 1:
        raise ValueError("actor_critic requires n_samples_per_prompt >= 1.")
    if settings.algorithm == TRAINING_ALGORITHM_GRPO and settings.n_samples_per_prompt < 2:
        raise ValueError("grpo usually requires n_samples_per_prompt >= 2.")

    return run_slime_training_job(
        settings=settings,
        lkh_config_path=args.lkh_config,
        mode="train",
        run_name_override=args.run_name,
        plan_only=bool(args.plan_only),
    )


# Legacy aliases for compatibility with previous internal names.
_phase6_overrides_from_args = _slime_training_overrides_from_args
run_phase6_training = run_slime_training_pipeline


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=Path("configs/rl.yaml"))
    pre_args, _ = pre.parse_known_args()
    defaults = _read_yaml_defaults(pre_args.config)

    parser = _build_arg_parser(defaults)
    args = parser.parse_args()

    if args.pipeline == "adapter_validation":
        result = run_training(args)
        print(f"Saved RL train summary: {result['summary_path']}")
        return

    result = run_slime_training_pipeline(args)
    print(f"Saved RL training summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
