"""SLIME training integration for step-level TSP RL."""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from tsp_action_rl.config import load_lkh_settings
from tsp_action_rl.data import (
    INDEXING_TSPLIB_1_BASED,
    Position2D,
    RolloutState,
    TSPInstance,
    generate_random_euclidean_instance,
)
from tsp_action_rl.parsing import parse_final_next_node
from tsp_action_rl.prompts import PromptRenderConfig, render_tsp_next_node_prompt
from tsp_action_rl.rollout import save_json
from tsp_action_rl.solvers import LKHIntegration

from .reward import (
    REWARD_MODE_GAP_ACTION_INVERSE,
    RewardContext,
    RewardSettings,
    build_reward_function,
)

TRAINING_ALGORITHM_ACTOR_CRITIC = "actor_critic"
TRAINING_ALGORITHM_GRPO = "grpo"
SUPPORTED_TRAINING_ALGORITHMS = (
    TRAINING_ALGORITHM_ACTOR_CRITIC,
    TRAINING_ALGORITHM_GRPO,
)

DEFAULT_TRAINING_ROLLOUT_FUNCTION_PATH = "tsp_action_rl.rl.slime_training.tsp_step_rollout"
DEFAULT_TRAINING_DATA_SOURCE_PATH = "tsp_action_rl.rl.slime_training.TSPStepRolloutDataSource"


class SLIMETrainingConfigError(ValueError):
    """Raised when SLIME training settings are invalid."""


class SLIMETrainingRuntimeError(RuntimeError):
    """Raised when SLIME training runtime setup or rollout fails."""


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise SLIMETrainingConfigError(f"Expected top-level mapping in config: {path}")
    return dict(payload)


def _load_slime_sample_type() -> Any:
    from slime.utils.types import Sample

    return Sample


@dataclass(frozen=True)
class TrainingTaskSamplingSettings:
    """Sampling config for one-step TSP training tasks."""

    node_count_min: int = 10
    node_count_max: int = 100
    coordinate_min: float = 0.0
    coordinate_max: float = 10000.0
    integer_coordinates: bool = True
    random_seed: int = 12345
    prefix_min_length: int = 1
    prefix_max_length: int | None = None

    def __post_init__(self) -> None:
        if self.node_count_min < 3:
            raise SLIMETrainingConfigError("node_count_min must be >= 3 for step-level action sampling.")
        if self.node_count_max < self.node_count_min:
            raise SLIMETrainingConfigError("node_count_max must be >= node_count_min.")
        if self.coordinate_max <= self.coordinate_min:
            raise SLIMETrainingConfigError("coordinate_max must be > coordinate_min.")
        if self.prefix_min_length < 1:
            raise SLIMETrainingConfigError("prefix_min_length must be >= 1.")
        if self.prefix_max_length is not None and self.prefix_max_length < self.prefix_min_length:
            raise SLIMETrainingConfigError("prefix_max_length must be >= prefix_min_length when set.")

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None) -> "TrainingTaskSamplingSettings":
        source = raw or {}
        if not isinstance(source, Mapping):
            raise SLIMETrainingConfigError(
                "slime_training.task_sampling must be a mapping/object."
            )

        node_range = source.get("node_count_range", [10, 100])
        if not isinstance(node_range, list) or len(node_range) != 2:
            raise SLIMETrainingConfigError(
                "slime_training.task_sampling.node_count_range must be a length-2 list."
            )

        coord_range = source.get("coordinate_range", [0, 10000])
        if not isinstance(coord_range, list) or len(coord_range) != 2:
            raise SLIMETrainingConfigError(
                "slime_training.task_sampling.coordinate_range must be a length-2 list."
            )

        return TrainingTaskSamplingSettings(
            node_count_min=int(node_range[0]),
            node_count_max=int(node_range[1]),
            coordinate_min=float(coord_range[0]),
            coordinate_max=float(coord_range[1]),
            integer_coordinates=bool(source.get("integer_coordinates", True)),
            random_seed=int(source.get("random_seed", 12345)),
            prefix_min_length=int(source.get("prefix_min_length", 1)),
            prefix_max_length=(
                None if source.get("prefix_max_length") is None else int(source.get("prefix_max_length"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count_range": [self.node_count_min, self.node_count_max],
            "coordinate_range": [self.coordinate_min, self.coordinate_max],
            "integer_coordinates": self.integer_coordinates,
            "random_seed": self.random_seed,
            "prefix_min_length": self.prefix_min_length,
            "prefix_max_length": self.prefix_max_length,
        }


@dataclass(frozen=True)
class TrainingTrackingSettings:
    """SwanLab/W&B-compatible tracking options."""

    enabled: bool = True
    wandb_project: str = "tsp-action-rl"
    wandb_group: str = "slime-training-step-rl"
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    wandb_team: str | None = None
    wandb_host: str | None = None

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None) -> "TrainingTrackingSettings":
        source = raw or {}
        if not isinstance(source, Mapping):
            raise SLIMETrainingConfigError(
                "slime_training.logging must be a mapping/object."
            )
        mode = str(source.get("wandb_mode", "offline"))
        if mode not in {"online", "offline", "disabled"}:
            raise SLIMETrainingConfigError(
                "slime_training.logging.wandb_mode must be one of: online, offline, disabled."
            )
        return TrainingTrackingSettings(
            enabled=bool(source.get("enabled", True)),
            wandb_project=str(source.get("wandb_project", "tsp-action-rl")),
            wandb_group=str(source.get("wandb_group", "slime-training-step-rl")),
            wandb_mode=mode,  # type: ignore[arg-type]
            wandb_team=None if source.get("wandb_team") is None else str(source.get("wandb_team")),
            wandb_host=None if source.get("wandb_host") is None else str(source.get("wandb_host")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "wandb_project": self.wandb_project,
            "wandb_group": self.wandb_group,
            "wandb_mode": self.wandb_mode,
            "wandb_team": self.wandb_team,
            "wandb_host": self.wandb_host,
        }


@dataclass(frozen=True)
class SLIMETrainingPipelineSettings:
    """SLIME launch settings for train/eval."""

    algorithm: Literal["actor_critic", "grpo"] = TRAINING_ALGORITHM_GRPO
    checkpoint_output_root: str = "/opt/aeon/container/"
    run_name: str | None = None
    slime_repo_path: str = "third_party/slime"
    run_train_async: bool = False

    # Core SLIME runtime knobs.
    hf_checkpoint: str | None = None
    model_name: str | None = None
    load_checkpoint: str | None = None
    ref_load: str | None = None
    seed: int = 12345
    num_rollout: int = 4
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    save_interval: int | None = 1
    eval_interval: int | None = None

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 1
    rollout_num_gpus: int = 1
    rollout_num_gpus_per_engine: int = 1
    num_gpus_per_node: int = 1

    rollout_temperature: float = 0.7
    rollout_top_p: float = 0.95
    rollout_top_k: int = -1
    rollout_max_prompt_len: int = 2048
    rollout_max_response_len: int = 384
    rollout_max_context_len: int = 4096
    rollout_stop: tuple[str, ...] = ()
    rollout_stop_token_ids: tuple[int, ...] = ()

    eval_dataset_name: str = "tsp_step_eval"
    save_rollout_traces: bool = True
    save_debug_rollout_data: bool = False
    save_debug_train_data: bool = False

    task_sampling: TrainingTaskSamplingSettings = field(default_factory=TrainingTaskSamplingSettings)
    tracking: TrainingTrackingSettings = field(default_factory=TrainingTrackingSettings)
    reward: RewardSettings = field(
        default_factory=lambda: RewardSettings(mode=REWARD_MODE_GAP_ACTION_INVERSE, parse_failure_penalty_enabled=True)
    )

    def __post_init__(self) -> None:
        if self.algorithm not in SUPPORTED_TRAINING_ALGORITHMS:
            raise SLIMETrainingConfigError(
                f"Unsupported training algorithm '{self.algorithm}'. Supported: {', '.join(SUPPORTED_TRAINING_ALGORITHMS)}"
            )
        if self.num_rollout < 1:
            raise SLIMETrainingConfigError("slime_training.num_rollout must be >= 1.")
        if self.rollout_batch_size < 1:
            raise SLIMETrainingConfigError("slime_training.rollout_batch_size must be >= 1.")
        if self.n_samples_per_prompt < 1:
            raise SLIMETrainingConfigError("slime_training.n_samples_per_prompt must be >= 1.")
        if self.save_interval is not None and self.save_interval < 1:
            raise SLIMETrainingConfigError("slime_training.save_interval must be >= 1 when set.")
        if self.eval_interval is not None and self.eval_interval < 1:
            raise SLIMETrainingConfigError("slime_training.eval_interval must be >= 1 when set.")
        if self.rollout_num_gpus < 1:
            raise SLIMETrainingConfigError("slime_training.rollout_num_gpus must be >= 1.")
        if self.rollout_num_gpus_per_engine < 1:
            raise SLIMETrainingConfigError("slime_training.rollout_num_gpus_per_engine must be >= 1.")
        if self.actor_num_nodes < 1 or self.actor_num_gpus_per_node < 1:
            raise SLIMETrainingConfigError("slime_training actor node/gpu settings must be >= 1.")
        if self.rollout_max_prompt_len < 1 or self.rollout_max_response_len < 1 or self.rollout_max_context_len < 2:
            raise SLIMETrainingConfigError("slime_training rollout length settings must be positive.")
        if self.hf_checkpoint is None or not self.hf_checkpoint.strip():
            raise SLIMETrainingConfigError(
                "slime_training.hf_checkpoint is required for real SLIME training/eval (model path or HF checkpoint id)."
            )
        if self.reward.mode != REWARD_MODE_GAP_ACTION_INVERSE:
            raise SLIMETrainingConfigError(
                f"slime_training.reward.mode must be '{REWARD_MODE_GAP_ACTION_INVERSE}' for this training task."
            )

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None, *, fallback_reward: Mapping[str, Any] | None = None) -> "SLIMETrainingPipelineSettings":
        source = raw or {}
        if not isinstance(source, Mapping):
            raise SLIMETrainingConfigError("slime_training config must be a mapping/object.")

        reward_raw = source.get("reward")
        if reward_raw is None:
            reward_raw = dict(fallback_reward or {})
        if not isinstance(reward_raw, Mapping):
            raise SLIMETrainingConfigError("slime_training.reward must be a mapping/object.")
        if "mode" not in reward_raw:
            reward_raw = dict(reward_raw)
            reward_raw["mode"] = REWARD_MODE_GAP_ACTION_INVERSE

        stop_raw = source.get("rollout_stop", [])
        if stop_raw is None:
            stop_raw = []
        if not isinstance(stop_raw, list):
            raise SLIMETrainingConfigError("slime_training.rollout_stop must be a list of strings.")

        stop_token_raw = source.get("rollout_stop_token_ids", [])
        if stop_token_raw is None:
            stop_token_raw = []
        if not isinstance(stop_token_raw, list):
            raise SLIMETrainingConfigError("slime_training.rollout_stop_token_ids must be a list of integers.")

        return SLIMETrainingPipelineSettings(
            algorithm=str(source.get("algorithm", TRAINING_ALGORITHM_GRPO)),  # type: ignore[arg-type]
            checkpoint_output_root=str(source.get("checkpoint_output_root", "/opt/aeon/container/")),
            run_name=None if source.get("run_name") is None else str(source.get("run_name")),
            slime_repo_path=str(source.get("slime_repo_path", "third_party/slime")),
            run_train_async=bool(source.get("run_train_async", False)),
            hf_checkpoint=None if source.get("hf_checkpoint") is None else str(source.get("hf_checkpoint")),
            model_name=None if source.get("model_name") is None else str(source.get("model_name")),
            load_checkpoint=None if source.get("load_checkpoint") is None else str(source.get("load_checkpoint")),
            ref_load=None if source.get("ref_load") is None else str(source.get("ref_load")),
            seed=int(source.get("seed", 12345)),
            num_rollout=int(source.get("num_rollout", 4)),
            rollout_batch_size=int(source.get("rollout_batch_size", 2)),
            n_samples_per_prompt=int(source.get("n_samples_per_prompt", 4)),
            save_interval=None if source.get("save_interval") is None else int(source.get("save_interval")),
            eval_interval=None if source.get("eval_interval") is None else int(source.get("eval_interval")),
            actor_num_nodes=int(source.get("actor_num_nodes", 1)),
            actor_num_gpus_per_node=int(source.get("actor_num_gpus_per_node", 1)),
            rollout_num_gpus=int(source.get("rollout_num_gpus", 1)),
            rollout_num_gpus_per_engine=int(source.get("rollout_num_gpus_per_engine", 1)),
            num_gpus_per_node=int(source.get("num_gpus_per_node", 1)),
            rollout_temperature=float(source.get("rollout_temperature", 0.7)),
            rollout_top_p=float(source.get("rollout_top_p", 0.95)),
            rollout_top_k=int(source.get("rollout_top_k", -1)),
            rollout_max_prompt_len=int(source.get("rollout_max_prompt_len", 2048)),
            rollout_max_response_len=int(source.get("rollout_max_response_len", 384)),
            rollout_max_context_len=int(source.get("rollout_max_context_len", 4096)),
            rollout_stop=tuple(str(item) for item in stop_raw),
            rollout_stop_token_ids=tuple(int(item) for item in stop_token_raw),
            eval_dataset_name=str(source.get("eval_dataset_name", "tsp_step_eval")),
            save_rollout_traces=bool(source.get("save_rollout_traces", True)),
            save_debug_rollout_data=bool(source.get("save_debug_rollout_data", False)),
            save_debug_train_data=bool(source.get("save_debug_train_data", False)),
            task_sampling=TrainingTaskSamplingSettings.from_mapping(source.get("task_sampling")),
            tracking=TrainingTrackingSettings.from_mapping(source.get("logging")),
            reward=RewardSettings.from_mapping(reward_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "checkpoint_output_root": self.checkpoint_output_root,
            "run_name": self.run_name,
            "slime_repo_path": self.slime_repo_path,
            "run_train_async": self.run_train_async,
            "hf_checkpoint": self.hf_checkpoint,
            "model_name": self.model_name,
            "load_checkpoint": self.load_checkpoint,
            "ref_load": self.ref_load,
            "seed": self.seed,
            "num_rollout": self.num_rollout,
            "rollout_batch_size": self.rollout_batch_size,
            "n_samples_per_prompt": self.n_samples_per_prompt,
            "save_interval": self.save_interval,
            "eval_interval": self.eval_interval,
            "actor_num_nodes": self.actor_num_nodes,
            "actor_num_gpus_per_node": self.actor_num_gpus_per_node,
            "rollout_num_gpus": self.rollout_num_gpus,
            "rollout_num_gpus_per_engine": self.rollout_num_gpus_per_engine,
            "num_gpus_per_node": self.num_gpus_per_node,
            "rollout_temperature": self.rollout_temperature,
            "rollout_top_p": self.rollout_top_p,
            "rollout_top_k": self.rollout_top_k,
            "rollout_max_prompt_len": self.rollout_max_prompt_len,
            "rollout_max_response_len": self.rollout_max_response_len,
            "rollout_max_context_len": self.rollout_max_context_len,
            "rollout_stop": list(self.rollout_stop),
            "rollout_stop_token_ids": list(self.rollout_stop_token_ids),
            "eval_dataset_name": self.eval_dataset_name,
            "save_rollout_traces": self.save_rollout_traces,
            "save_debug_rollout_data": self.save_debug_rollout_data,
            "save_debug_train_data": self.save_debug_train_data,
            "task_sampling": self.task_sampling.to_dict(),
            "logging": self.tracking.to_dict(),
            "reward": self.reward.to_dict(),
        }


def load_slime_training_settings(
    *,
    config_path: str | Path = "configs/rl.yaml",
    overrides: Mapping[str, Any] | None = None,
) -> SLIMETrainingPipelineSettings:
    config_mapping = _read_yaml_mapping(Path(config_path))
    legacy_raw = config_mapping.get("phase6", {})
    if legacy_raw is None:
        legacy_raw = {}
    if not isinstance(legacy_raw, Mapping):
        raise SLIMETrainingConfigError("Legacy phase6 section must be a mapping/object when present.")

    training_raw = config_mapping.get("slime_training")
    if training_raw is None:
        training_raw = legacy_raw
    if not isinstance(training_raw, Mapping):
        raise SLIMETrainingConfigError("slime_training section must be a mapping/object.")

    # Keep backward compatibility by allowing legacy `phase6` defaults.
    merged = _deep_merge(dict(legacy_raw), dict(training_raw))
    if overrides:
        merged = _deep_merge(merged, dict(overrides))
    fallback_reward = config_mapping.get("reward")
    if fallback_reward is not None and not isinstance(fallback_reward, Mapping):
        raise SLIMETrainingConfigError("Top-level reward section must be a mapping/object when present.")
    return SLIMETrainingPipelineSettings.from_mapping(merged, fallback_reward=fallback_reward)


def algorithm_to_advantage_estimator(algorithm: str) -> str:
    if algorithm == TRAINING_ALGORITHM_ACTOR_CRITIC:
        return "ppo"
    if algorithm == TRAINING_ALGORITHM_GRPO:
        return "grpo"
    raise SLIMETrainingConfigError(
        f"Unsupported training algorithm '{algorithm}'. Supported: {', '.join(SUPPORTED_TRAINING_ALGORITHMS)}"
    )


@dataclass(frozen=True)
class SLIMETrainingRunContext:
    mode: Literal["train", "eval"]
    run_name: str
    run_dir: Path
    checkpoint_dir: Path
    trace_dir: Path
    summary_path: Path
    custom_config_path: Path
    dummy_eval_prompt_path: Path


def prepare_slime_training_run_context(
    *,
    settings: SLIMETrainingPipelineSettings,
    mode: Literal["train", "eval"],
    run_name_override: str | None = None,
) -> SLIMETrainingRunContext:
    run_name = run_name_override or settings.run_name or datetime.utcnow().strftime(f"slime_training_{mode}_%Y%m%d_%H%M%S")
    run_dir = Path(settings.checkpoint_output_root) / "tsp_action_rl" / "training" / mode / run_name
    checkpoint_dir = run_dir / "checkpoints"
    trace_dir = run_dir / "rollout_traces"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if settings.save_rollout_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)

    dummy_eval_prompt_path = run_dir / "dummy_eval_prompts.jsonl"
    dummy_eval_prompt_path.write_text('{"input": "tsp_step_eval_placeholder"}\n', encoding="utf-8")

    return SLIMETrainingRunContext(
        mode=mode,
        run_name=run_name,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        trace_dir=trace_dir,
        summary_path=run_dir / ("train_summary.json" if mode == "train" else "eval_summary.json"),
        custom_config_path=run_dir / "slime_custom_config.yaml",
        dummy_eval_prompt_path=dummy_eval_prompt_path,
    )


def _write_slime_training_custom_config(
    *,
    context: SLIMETrainingRunContext,
    settings: SLIMETrainingPipelineSettings,
    lkh_config_path: Path,
) -> None:
    payload = {
        "tsp_lkh_config_path": str(lkh_config_path),
        "tsp_task_sampling": settings.task_sampling.to_dict(),
        "tsp_reward_settings": settings.reward.to_dict(),
        "tsp_eval_dataset_name": settings.eval_dataset_name,
        "tsp_trace_output_dir": str(context.trace_dir) if settings.save_rollout_traces else None,
    }
    context.custom_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_slime_cli_args(
    *,
    settings: SLIMETrainingPipelineSettings,
    context: SLIMETrainingRunContext,
) -> list[str]:
    args: list[str] = [
        "--seed",
        str(settings.seed),
        "--rollout-seed",
        str(settings.seed),
        "--advantage-estimator",
        algorithm_to_advantage_estimator(settings.algorithm),
        "--rollout-function-path",
        DEFAULT_TRAINING_ROLLOUT_FUNCTION_PATH,
        "--eval-function-path",
        DEFAULT_TRAINING_ROLLOUT_FUNCTION_PATH,
        "--data-source-path",
        DEFAULT_TRAINING_DATA_SOURCE_PATH,
        "--rollout-batch-size",
        str(settings.rollout_batch_size),
        "--n-samples-per-prompt",
        str(settings.n_samples_per_prompt),
        "--num-rollout",
        str(settings.num_rollout if context.mode == "train" else 0),
        "--save",
        str(context.checkpoint_dir),
        "--custom-config-path",
        str(context.custom_config_path),
        "--actor-num-nodes",
        str(settings.actor_num_nodes),
        "--actor-num-gpus-per-node",
        str(settings.actor_num_gpus_per_node),
        "--rollout-num-gpus",
        str(settings.rollout_num_gpus),
        "--rollout-num-gpus-per-engine",
        str(settings.rollout_num_gpus_per_engine),
        "--num-gpus-per-node",
        str(settings.num_gpus_per_node),
        "--rollout-temperature",
        str(settings.rollout_temperature),
        "--rollout-top-p",
        str(settings.rollout_top_p),
        "--rollout-top-k",
        str(settings.rollout_top_k),
        "--rollout-max-prompt-len",
        str(settings.rollout_max_prompt_len),
        "--rollout-max-response-len",
        str(settings.rollout_max_response_len),
        "--rollout-max-context-len",
        str(settings.rollout_max_context_len),
        "--hf-checkpoint",
        settings.hf_checkpoint or "",
    ]

    if settings.model_name:
        args.extend(["--model-name", settings.model_name])
    if settings.load_checkpoint:
        args.extend(["--load", settings.load_checkpoint])
    if settings.ref_load:
        args.extend(["--ref-load", settings.ref_load])
    if settings.save_interval is not None:
        args.extend(["--save-interval", str(settings.save_interval)])
    if settings.eval_interval is not None and context.mode == "train":
        args.extend(["--eval-interval", str(settings.eval_interval)])
        args.extend(["--eval-prompt-data", settings.eval_dataset_name, str(context.dummy_eval_prompt_path)])
    if context.mode == "eval":
        args.extend(["--eval-interval", "1"])
        args.extend(["--eval-prompt-data", settings.eval_dataset_name, str(context.dummy_eval_prompt_path)])

    if settings.rollout_stop:
        args.extend(["--rollout-stop", *settings.rollout_stop])
    if settings.rollout_stop_token_ids:
        args.extend(["--rollout-stop-token-ids", *(str(item) for item in settings.rollout_stop_token_ids)])

    if settings.save_debug_rollout_data:
        args.extend(
            [
                "--save-debug-rollout-data",
                str(context.run_dir / "debug_rollout" / "{rollout_id}.pt"),
            ]
        )
    if settings.save_debug_train_data:
        args.extend(
            [
                "--save-debug-train-data",
                str(context.run_dir / "debug_train" / "{rollout_id}_{rank}.pt"),
            ]
        )

    if settings.tracking.enabled:
        args.extend(
            [
                "--use-wandb",
                "--wandb-project",
                settings.tracking.wandb_project,
                "--wandb-group",
                settings.tracking.wandb_group,
                "--wandb-mode",
                settings.tracking.wandb_mode,
                "--wandb-dir",
                str(context.run_dir / "wandb"),
            ]
        )
        if settings.tracking.wandb_team:
            args.extend(["--wandb-team", settings.tracking.wandb_team])
        if settings.tracking.wandb_host:
            args.extend(["--wandb-host", settings.tracking.wandb_host])

    return args


def _ensure_wandb_compat_import() -> str:
    # Required because SLIME imports `wandb` at module import time.
    try:
        import swanlab as wandb  # type: ignore[import-not-found]

        sys.modules["wandb"] = wandb
        return "swanlab"
    except Exception:  # noqa: BLE001
        try:
            import wandb  # type: ignore[import-not-found]  # noqa: F401

            return "wandb"
        except Exception as exc:  # noqa: BLE001
            raise SLIMETrainingRuntimeError(
                "Neither `swanlab` nor `wandb` is importable. Install one of them before running the SLIME training pipeline."
            ) from exc


def _ensure_slime_repo_on_sys_path(repo_path: str) -> Path:
    path = Path(repo_path).resolve()
    if not path.exists() or not path.is_dir():
        raise SLIMETrainingRuntimeError(f"SLIME repo path is missing or invalid: {path}")
    if not (path / "slime").exists():
        raise SLIMETrainingRuntimeError(f"SLIME repo path does not contain `slime/`: {path}")
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return path


def _parse_slime_args(cli_args: list[str]) -> Any:
    from slime.utils.arguments import parse_args

    argv_before = list(sys.argv)
    try:
        sys.argv = ["train.py", *cli_args]
        return parse_args()
    finally:
        sys.argv = argv_before


def _load_slime_train_callable(*, run_train_async: bool) -> Any:
    module_name = "train_async" if run_train_async else "train"
    module = importlib.import_module(module_name)
    train_fn = getattr(module, "train", None)
    if train_fn is None or not callable(train_fn):
        raise SLIMETrainingRuntimeError(f"Could not resolve callable `train` from module `{module_name}`.")
    return train_fn


def run_slime_training_job(
    *,
    settings: SLIMETrainingPipelineSettings,
    lkh_config_path: str | Path,
    mode: Literal["train", "eval"],
    run_name_override: str | None = None,
    plan_only: bool = False,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    lkh_path = Path(lkh_config_path)
    context = prepare_slime_training_run_context(settings=settings, mode=mode, run_name_override=run_name_override)
    _write_slime_training_custom_config(context=context, settings=settings, lkh_config_path=lkh_path)
    cli_args = build_slime_cli_args(settings=settings, context=context)

    summary: dict[str, Any] = {
        "mode": mode,
        "run_name": context.run_name,
        "run_dir": str(context.run_dir),
        "checkpoint_output_root": settings.checkpoint_output_root,
        "checkpoint_dir": str(context.checkpoint_dir),
        "trace_dir": str(context.trace_dir),
        "slime_repo_path": str(Path(settings.slime_repo_path).resolve()),
        "config": settings.to_dict(),
        "lkh_config_path": str(lkh_path),
        "custom_config_path": str(context.custom_config_path),
        "slime_cli_args": cli_args,
    }

    if dry_run is not None:
        plan_only = bool(dry_run)

    if plan_only:
        summary["status"] = "plan_only"
        save_json(summary, context.summary_path)
        summary["summary_path"] = str(context.summary_path)
        return summary

    _ensure_slime_repo_on_sys_path(settings.slime_repo_path)
    summary["wandb_backend"] = _ensure_wandb_compat_import()

    start = time.time()
    args = _parse_slime_args(cli_args)
    train_fn = _load_slime_train_callable(run_train_async=settings.run_train_async)
    train_fn(args)
    elapsed = time.time() - start

    summary["status"] = "completed"
    summary["elapsed_seconds"] = elapsed
    save_json(summary, context.summary_path)
    summary["summary_path"] = str(context.summary_path)
    return summary


@dataclass
class _SLIMETrainingRuntime:
    args: Any
    solver: LKHIntegration
    reward_fn: Any
    reward_settings: RewardSettings
    task_sampling: TrainingTaskSamplingSettings
    prompt_config: PromptRenderConfig


def _get_slime_training_runtime(args: Any) -> _SLIMETrainingRuntime:
    cached = getattr(args, "_tsp_training_runtime", None)
    if cached is None:
        cached = getattr(args, "_tsp_phase6_runtime", None)
    if cached is not None:
        return cached

    lkh_config_path = Path(str(getattr(args, "tsp_lkh_config_path", "configs/lkh.yaml")))
    reward_raw = getattr(args, "tsp_reward_settings", {})
    if not isinstance(reward_raw, Mapping):
        raise SLIMETrainingRuntimeError("Expected mapping in args.tsp_reward_settings.")
    task_sampling_raw = getattr(args, "tsp_task_sampling", {})
    if not isinstance(task_sampling_raw, Mapping):
        raise SLIMETrainingRuntimeError("Expected mapping in args.tsp_task_sampling.")

    reward_settings = RewardSettings.from_mapping(reward_raw)
    if reward_settings.mode != REWARD_MODE_GAP_ACTION_INVERSE:
        raise SLIMETrainingRuntimeError(
            f"Expected training reward mode '{REWARD_MODE_GAP_ACTION_INVERSE}', got '{reward_settings.mode}'."
        )

    runtime = _SLIMETrainingRuntime(
        args=args,
        solver=LKHIntegration(load_lkh_settings(lkh_config_path)),
        reward_fn=build_reward_function(reward_settings),
        reward_settings=reward_settings,
        task_sampling=TrainingTaskSamplingSettings.from_mapping(task_sampling_raw),
        prompt_config=PromptRenderConfig(
            include_current_position=True,
            include_visited_nodes=False,
            include_unvisited_nodes=True,
        ),
    )
    setattr(args, "_tsp_training_runtime", runtime)
    # Backward-compat for old integrations.
    setattr(args, "_tsp_phase6_runtime", runtime)
    return runtime


def _compute_open_path_length(instance: TSPInstance, route_prefix: tuple[int, ...]) -> float:
    if len(route_prefix) < 2:
        return 0.0
    coords = {node.node_id: (node.x, node.y) for node in instance.nodes}
    total = 0.0
    for left, right in zip(route_prefix[:-1], route_prefix[1:], strict=True):
        x1, y1 = coords[left]
        x2, y2 = coords[right]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _build_rollout_state_from_prefix(instance: TSPInstance, prefix: tuple[int, ...]) -> RolloutState:
    if len(prefix) == 0:
        raise SLIMETrainingRuntimeError("Prefix route must be non-empty.")
    nodes_by_id = {node.node_id: node for node in instance.nodes}
    current_node = prefix[-1]
    if current_node not in nodes_by_id:
        raise SLIMETrainingRuntimeError(f"Prefix node {current_node} not found in instance node set.")
    node = nodes_by_id[current_node]
    unvisited = tuple(node_id for node_id in range(1, instance.node_count + 1) if node_id not in set(prefix))
    return RolloutState(
        instance_id=instance.instance_id,
        step_index=len(prefix),
        node_count=instance.node_count,
        partial_route=prefix,
        visited_nodes=prefix,
        unvisited_nodes=unvisited,
        current_node=current_node,
        current_position=Position2D(x=node.x, y=node.y),
        is_terminal=(len(unvisited) == 0),
        indexing=INDEXING_TSPLIB_1_BASED,
        notes={"training_source": "sampled_reference_prefix"},
    )


class TSPStepRolloutDataSource:
    """SLIME DataSource for one-step TSP action-level RL tasks."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.runtime = _get_slime_training_runtime(args)
        self._rng = random.Random(self.runtime.task_sampling.random_seed + int(getattr(args, "rollout_seed", 0)))
        self._n_samples_per_prompt = int(args.n_samples_per_prompt)
        self._sample_group_index = 0
        self._sample_index = 0
        self._buffer: list[list[Any]] = []

    def _choose_prefix_length(self, node_count: int) -> int:
        lower = max(1, self.runtime.task_sampling.prefix_min_length)
        upper = node_count - 1
        if self.runtime.task_sampling.prefix_max_length is not None:
            upper = min(upper, self.runtime.task_sampling.prefix_max_length)
        if lower > upper:
            raise SLIMETrainingRuntimeError(
                f"Invalid prefix length bounds for node_count={node_count}: lower={lower}, upper={upper}"
            )
        return self._rng.randint(lower, upper)

    def _build_one_task(self) -> tuple[str, dict[str, Any]]:
        node_count = self._rng.randint(self.runtime.task_sampling.node_count_min, self.runtime.task_sampling.node_count_max)
        seed = self._rng.randint(0, 2**31 - 1)
        instance = generate_random_euclidean_instance(
            node_count=node_count,
            seed=seed,
            coordinate_range=(
                self.runtime.task_sampling.coordinate_min,
                self.runtime.task_sampling.coordinate_max,
            ),
            integer_coordinates=self.runtime.task_sampling.integer_coordinates,
        )
        reference = self.runtime.solver.solve_reference(instance)
        prefix_length = self._choose_prefix_length(node_count)
        prefix = tuple(reference.tour[:prefix_length])
        state = _build_rollout_state_from_prefix(instance, prefix)
        prompt_text = render_tsp_next_node_prompt(instance=instance, state=state, config=self.runtime.prompt_config)

        reference_next_node = reference.tour[prefix_length] if prefix_length < len(reference.tour) else None
        task_metadata = {
            "instance": instance.to_dict(),
            "instance_id": instance.instance_id,
            "node_count": instance.node_count,
            "partial_route": list(prefix),
            "unvisited_nodes": list(state.unvisited_nodes),
            "reference_tour": list(reference.tour),
            "reference_tour_length": reference.tour_length,
            "prefix_partial_tour_length": _compute_open_path_length(instance, prefix),
            "reference_next_node": reference_next_node,
            "prefix_cut_length": prefix_length,
            "task_seed": seed,
            "indexing": INDEXING_TSPLIB_1_BASED,
        }
        return prompt_text, task_metadata

    def get_samples(self, num_samples: int) -> list[list[Any]]:
        Sample = _load_slime_sample_type()
        groups: list[list[Any]] = []

        while self._buffer and len(groups) < num_samples:
            groups.append(self._buffer.pop(0))

        while len(groups) < num_samples:
            prompt_text, task_metadata = self._build_one_task()
            group: list[Any] = []
            for _ in range(self._n_samples_per_prompt):
                sample = Sample(
                    group_index=self._sample_group_index,
                    index=self._sample_index,
                    prompt=prompt_text,
                    response="",
                    tokens=[],
                    response_length=0,
                    reward=0.0,
                    status=Sample.Status.PENDING,
                    metadata={
                        "tsp_step_task": dict(task_metadata),
                        "tsp_prompt_text": prompt_text,
                    },
                )
                self._sample_index += 1
                group.append(sample)
            self._sample_group_index += 1
            groups.append(group)

        return groups

    def add_samples(self, samples: list[list[Any]]) -> None:
        if samples:
            self._buffer.extend(samples)

    def save(self, rollout_id: int) -> None:  # noqa: ARG002
        return

    def load(self, rollout_id: int | None = None) -> None:  # noqa: ARG002
        return

    def __len__(self) -> int:
        # Keep positive/stable length for framework internals; training uses explicit --num-rollout.
        return max(int(getattr(self.args, "num_rollout", 1)) * int(getattr(self.args, "rollout_batch_size", 1)), 1)


def _validate_action(node_count: int, visited_nodes: set[int], parsed_action: int | None, parse_status: str) -> tuple[bool, str | None]:
    if parse_status != "success" or parsed_action is None:
        return False, "parse_failure"
    if parsed_action < 1 or parsed_action > node_count:
        return False, "out_of_range"
    if parsed_action in visited_nodes:
        return False, "already_visited"
    return True, None


def _score_tsp_step_sample(runtime: _SLIMETrainingRuntime, sample: Any) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, Mapping) else {}
    task = metadata.get("tsp_step_task")
    if not isinstance(task, Mapping):
        raise SLIMETrainingRuntimeError(
            "Sample is missing `metadata.tsp_step_task` required for training reward."
        )

    instance_raw = task.get("instance")
    if not isinstance(instance_raw, Mapping):
        raise SLIMETrainingRuntimeError("Sample task metadata is missing instance payload.")
    instance = TSPInstance.from_dict(instance_raw)

    partial_route = tuple(int(node_id) for node_id in task.get("partial_route", []))
    if len(partial_route) == 0:
        raise SLIMETrainingRuntimeError("Sample task metadata has empty partial_route.")
    visited_nodes = set(partial_route)
    node_count = int(task.get("node_count", instance.node_count))

    reference_tour_length = float(task.get("reference_tour_length"))
    prefix_partial_tour_length = float(task.get("prefix_partial_tour_length"))

    parse_result = parse_final_next_node(str(sample.response))
    action_is_valid, action_failure_reason = _validate_action(
        node_count=node_count,
        visited_nodes=visited_nodes,
        parsed_action=parse_result.parsed_next_node,
        parse_status=parse_result.status,
    )

    constrained_tour_length: float | None = None
    constrained_tour: list[int] | None = None
    solver_debug_paths: dict[str, str] = {}

    if action_is_valid and parse_result.parsed_next_node is not None:
        chosen_action = int(parse_result.parsed_next_node)
        constrained = runtime.solver.solve_with_fixed_prefix(instance, partial_route + (chosen_action,))
        constrained_tour_length = float(constrained.tour_length)
        constrained_tour = [int(node_id) for node_id in constrained.tour]
        solver_debug_paths = dict(constrained.debug_paths)
    done_reason = "single_step_completed" if action_is_valid else "invalid_action"

    reward_context = RewardContext(
        action_is_valid=action_is_valid,
        action_failure_reason=action_failure_reason,
        parse_status=parse_result.status,
        reference_tour_length=reference_tour_length,
        constrained_tour_length=constrained_tour_length,
        previous_constrained_tour_length=None,
        prefix_partial_tour_length=prefix_partial_tour_length,
        is_terminal_step=True,
    )
    reward_signal = runtime.reward_fn.compute(reward_context)
    sample.reward = float(reward_signal.reward_value)

    components = reward_signal.components
    gap_action = components.get("gap_action")
    if not isinstance(gap_action, (int, float)):
        gap_action = None

    reference_next_node = task.get("reference_next_node")
    chosen_node = parse_result.parsed_next_node if parse_result.status == "success" else None
    is_reference_match = (
        isinstance(reference_next_node, int) and isinstance(chosen_node, int) and reference_next_node == chosen_node
    )

    sample.metadata = dict(sample.metadata)
    sample.metadata["tsp_step_result"] = {
        "parse_status": parse_result.status,
        "tag_count": parse_result.tag_count,
        "parsed_next_node": parse_result.parsed_next_node,
        "reasoning_text": parse_result.reasoning_text,
        "action_is_valid": action_is_valid,
        "action_failure_reason": action_failure_reason,
        "done_reason": done_reason,
        "reference_tour_length": reference_tour_length,
        "prefix_partial_tour_length": prefix_partial_tour_length,
        "constrained_tour_length": constrained_tour_length,
        "reward_signal": reward_signal.to_dict(),
        "reference_next_node": reference_next_node,
        "is_reference_next_node_match": is_reference_match,
        "solver_debug_paths": solver_debug_paths,
    }
    sample.metadata["raw_reward"] = float(reward_signal.reward_value)

    return {
        "sample_index": sample.index,
        "group_index": sample.group_index,
        "instance_id": task.get("instance_id"),
        "node_count": node_count,
        "prompt_text": metadata.get("tsp_prompt_text"),
        "raw_model_output": sample.response,
        "reasoning_text": parse_result.reasoning_text,
        "parse_status": parse_result.status,
        "tag_count": parse_result.tag_count,
        "parsed_next_node": parse_result.parsed_next_node,
        "action_validation": {
            "is_valid": action_is_valid,
            "failure_reason": action_failure_reason,
        },
        "done_reason": done_reason,
        "reference_next_node": reference_next_node,
        "is_reference_next_node_match": is_reference_match,
        "reference_tour_length": reference_tour_length,
        "prefix_partial_tour_length": prefix_partial_tour_length,
        "constrained_tour_length": constrained_tour_length,
        "constrained_tour": constrained_tour,
        "gap_action": gap_action,
        "reward_signal": reward_signal.to_dict(),
        "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
    }


def _write_rollout_traces(args: Any, rollout_id: int, traces: list[dict[str, Any]]) -> str | None:
    trace_dir_raw = getattr(args, "tsp_trace_output_dir", None)
    if trace_dir_raw is None:
        return None
    trace_dir = Path(str(trace_dir_raw))
    trace_dir.mkdir(parents=True, exist_ok=True)
    target = trace_dir / f"rollout_{rollout_id:06d}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for item in traces:
            handle.write(json.dumps(item) + "\n")
    return str(target)


async def _generate_groups_with_sglang(args: Any, groups: list[list[Any]], evaluation: bool) -> list[list[Any]]:
    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group

    state = GenerateState(args)
    tasks: list[Any] = []
    for group in groups:
        sampling_params = state.sampling_params.copy()
        tasks.append(
            asyncio.create_task(
                generate_and_rm_group(args, group, sampling_params=sampling_params, evaluation=evaluation)
            )
        )
    generated = await asyncio.gather(*tasks)
    state.reset()
    return [list(group) for group in generated]


def tsp_step_rollout(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """SLIME rollout entrypoint for step-level TSP training."""
    from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
    from slime.utils.async_utils import run
    from slime.utils.types import Sample

    runtime = _get_slime_training_runtime(args)
    requested = int(getattr(args, "rollout_batch_size", 1))
    groups = data_source.get_samples(requested)
    if len(groups) != requested:
        raise SLIMETrainingRuntimeError(f"Expected {requested} sample groups, got {len(groups)}.")

    generated_groups = run(_generate_groups_with_sglang(args, groups, evaluation=evaluation))
    flat_samples: list[Any] = [sample for group in generated_groups for sample in group]

    traces: list[dict[str, Any]] = []
    parse_success_count = 0
    valid_action_count = 0
    reference_match_count = 0
    gap_action_values: list[float] = []

    for sample in flat_samples:
        trace = _score_tsp_step_sample(runtime, sample)
        traces.append(trace)
        if trace["parse_status"] == "success":
            parse_success_count += 1
        if bool(trace["action_validation"]["is_valid"]):
            valid_action_count += 1
        if bool(trace["is_reference_next_node_match"]):
            reference_match_count += 1
        gap_value = trace.get("gap_action")
        if isinstance(gap_value, (int, float)):
            gap_action_values.append(float(gap_value))

    trace_path = _write_rollout_traces(args, rollout_id, traces)

    sample_count = max(1, len(flat_samples))
    metrics: dict[str, Any] = {
        "source": "tsp_step_training",
        "rollout/tsp_sample_count": len(flat_samples),
        "rollout/tsp_parse_success_rate": parse_success_count / sample_count,
        "rollout/tsp_valid_action_rate": valid_action_count / sample_count,
        "rollout/tsp_reference_next_node_match_rate": reference_match_count / sample_count,
        "rollout/tsp_raw_reward_mean": sum(float(sample.reward) for sample in flat_samples) / sample_count,
    }
    if gap_action_values:
        metrics["rollout/tsp_gap_action_mean"] = sum(gap_action_values) / len(gap_action_values)
    if trace_path is not None:
        metrics["rollout/tsp_trace_file"] = trace_path

    if evaluation:
        dataset_name = str(getattr(args, "tsp_eval_dataset_name", "tsp_step_eval"))
        eval_payload = {
            dataset_name: {
                "rewards": [float(sample.reward) for sample in flat_samples],
                "truncated": [sample.status == Sample.Status.TRUNCATED for sample in flat_samples],
                "samples": flat_samples,
            }
        }
        return RolloutFnEvalOutput(data=eval_payload, metrics=metrics)

    return RolloutFnTrainOutput(samples=generated_groups, metrics=metrics)


def apply_slime_training_overrides(
    settings: SLIMETrainingPipelineSettings,
    *,
    algorithm: str | None = None,
    checkpoint_output_root: str | None = None,
    run_name: str | None = None,
    num_rollout: int | None = None,
    rollout_batch_size: int | None = None,
    n_samples_per_prompt: int | None = None,
    eval_interval: int | None = None,
    save_interval: int | None = None,
) -> SLIMETrainingPipelineSettings:
    updated = settings
    if algorithm is not None:
        updated = replace(updated, algorithm=algorithm)  # type: ignore[arg-type]
    if checkpoint_output_root is not None:
        updated = replace(updated, checkpoint_output_root=checkpoint_output_root)
    if run_name is not None:
        updated = replace(updated, run_name=run_name)
    if num_rollout is not None:
        updated = replace(updated, num_rollout=num_rollout)
    if rollout_batch_size is not None:
        updated = replace(updated, rollout_batch_size=rollout_batch_size)
    if n_samples_per_prompt is not None:
        updated = replace(updated, n_samples_per_prompt=n_samples_per_prompt)
    if eval_interval is not None:
        updated = replace(updated, eval_interval=eval_interval)
    if save_interval is not None:
        updated = replace(updated, save_interval=save_interval)
    return updated


# Legacy aliases kept for low-friction migration from earlier internal names.
PHASE6_ALGORITHM_ACTOR_CRITIC = TRAINING_ALGORITHM_ACTOR_CRITIC
PHASE6_ALGORITHM_GRPO = TRAINING_ALGORITHM_GRPO
SUPPORTED_PHASE6_ALGORITHMS = SUPPORTED_TRAINING_ALGORITHMS
DEFAULT_PHASE6_ROLLOUT_FUNCTION_PATH = DEFAULT_TRAINING_ROLLOUT_FUNCTION_PATH
DEFAULT_PHASE6_DATA_SOURCE_PATH = DEFAULT_TRAINING_DATA_SOURCE_PATH
SLIMEPhase6ConfigError = SLIMETrainingConfigError
SLIMEPhase6RuntimeError = SLIMETrainingRuntimeError
Phase6TaskSamplingSettings = TrainingTaskSamplingSettings
Phase6TrackingSettings = TrainingTrackingSettings
Phase6PipelineSettings = SLIMETrainingPipelineSettings
Phase6RunContext = SLIMETrainingRunContext
load_phase6_settings = load_slime_training_settings
prepare_phase6_run_context = prepare_slime_training_run_context
_write_phase6_custom_config = _write_slime_training_custom_config
run_phase6_slime_job = run_slime_training_job
_get_phase6_runtime = _get_slime_training_runtime
apply_phase6_overrides = apply_slime_training_overrides
