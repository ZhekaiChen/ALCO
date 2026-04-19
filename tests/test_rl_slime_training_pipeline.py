"""SLIME training pipeline plan-only wiring tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tsp_action_rl.rl import (
    apply_slime_training_overrides,
    load_slime_training_settings,
    run_slime_training_job,
)


def _arg_value(args: list[str], flag: str) -> str:
    idx = args.index(flag)
    return args[idx + 1]


def test_load_slime_training_settings_requires_hf_checkpoint() -> None:
    with pytest.raises(Exception):
        load_slime_training_settings(config_path="configs/rl.yaml")


def test_training_plan_actor_critic_builds_ppo_args(tmp_path: Path) -> None:
    settings = load_slime_training_settings(
        config_path="configs/rl.yaml",
        overrides={
            "algorithm": "actor_critic",
            "hf_checkpoint": "Qwen/Qwen3-4B",
            "checkpoint_output_root": str(tmp_path),
            "num_rollout": 2,
            "rollout_batch_size": 2,
            "n_samples_per_prompt": 1,
        },
    )
    settings = apply_slime_training_overrides(settings, run_name="training_actor_critic_unit")

    summary = run_slime_training_job(
        settings=settings,
        lkh_config_path="configs/lkh.yaml",
        mode="train",
        plan_only=True,
    )

    args = summary["slime_cli_args"]
    assert _arg_value(args, "--advantage-estimator") == "ppo"
    assert _arg_value(args, "--num-rollout") == "2"
    assert _arg_value(args, "--rollout-function-path") == "tsp_action_rl.rl.slime_training.tsp_step_rollout"
    assert _arg_value(args, "--data-source-path") == "tsp_action_rl.rl.slime_training.TSPStepRolloutDataSource"
    assert Path(summary["summary_path"]).exists()
    assert summary["status"] == "plan_only"


def test_evaluation_plan_grpo_builds_eval_args(tmp_path: Path) -> None:
    settings = load_slime_training_settings(
        config_path="configs/rl.yaml",
        overrides={
            "algorithm": "grpo",
            "hf_checkpoint": "Qwen/Qwen3-4B",
            "checkpoint_output_root": str(tmp_path),
            "num_rollout": 3,
            "rollout_batch_size": 2,
            "n_samples_per_prompt": 4,
        },
    )
    settings = apply_slime_training_overrides(settings, run_name="training_eval_unit")

    summary = run_slime_training_job(
        settings=settings,
        lkh_config_path="configs/lkh.yaml",
        mode="eval",
        plan_only=True,
    )

    args = summary["slime_cli_args"]
    assert _arg_value(args, "--advantage-estimator") == "grpo"
    assert _arg_value(args, "--num-rollout") == "0"
    assert _arg_value(args, "--eval-interval") == "1"
    assert "--eval-prompt-data" in args
    assert Path(summary["summary_path"]).exists()
    assert summary["status"] == "plan_only"
