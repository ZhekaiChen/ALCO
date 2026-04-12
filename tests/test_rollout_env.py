"""Zero-shot rollout smoke tests."""

from __future__ import annotations

from pathlib import Path

from tsp_action_rl.data import load_rollout_state, load_tsp_instance
from tsp_action_rl.inference import LocalDeterministicModelBackend, LocalStaticResponseBackend, ModelOutput
from tsp_action_rl.prompts import PromptRenderConfig, render_tsp_next_node_prompt
from tsp_action_rl.rollout import ZeroShotRolloutConfig, ZeroShotRolloutRunner

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class _FailingBackend:
    backend_type = "api"
    model_name = "failing-backend"

    def generate(self, prompt_text: str, *, instance, state) -> ModelOutput:  # type: ignore[no-untyped-def]
        del prompt_text, instance, state
        raise RuntimeError("simulated backend failure")


def test_prompt_renderer_contains_required_fields() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    state = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json")
    prompt = render_tsp_next_node_prompt(
        instance=instance,
        state=state,
        config=PromptRenderConfig(include_current_position=True, include_visited_nodes=False, include_unvisited_nodes=True),
    )
    assert "Number of nodes: 5" in prompt
    assert "1: (0.0, 0.0)" in prompt
    assert "[1, 2, 3]" in prompt
    assert "Current node:" in prompt
    assert "<FINAL_NEXT_NODE>INTEGER</FINAL_NEXT_NODE>" in prompt


def test_zero_shot_rollout_success_smoke() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    backend = LocalDeterministicModelBackend()
    runner = ZeroShotRolloutRunner(
        model_backend=backend,
        config=ZeroShotRolloutConfig(enable_solver_completion=False),
        solver=None,
    )
    episode = runner.run_episode(instance=instance, episode_id="episode_smoke_success")

    assert episode["status"] == "success"
    assert episode["final_route"] is not None
    assert len(episode["final_route"]) == instance.node_count
    assert episode["summary_metrics"]["parse_success_rate"] == 1.0
    assert episode["summary_metrics"]["valid_action_rate"] == 1.0
    assert episode["step_logs"][0]["final_tag_parse"]["status"] == "success"


def test_zero_shot_rollout_invalid_repeated_node_failure() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    initial_state = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json")
    static_response = (FIXTURES_DIR / "sample_model_output_invalid.txt").read_text(encoding="utf-8")
    backend = LocalStaticResponseBackend([static_response])
    runner = ZeroShotRolloutRunner(
        model_backend=backend,
        config=ZeroShotRolloutConfig(enable_solver_completion=False),
        solver=None,
    )
    episode = runner.run_episode(
        instance=instance,
        episode_id="episode_smoke_invalid_repeated",
        initial_state=initial_state,
    )

    assert episode["status"] == "failed_invalid_action"
    assert episode["final_route"] is None
    assert len(episode["step_logs"]) == 1
    step = episode["step_logs"][0]
    assert step["final_tag_parse"]["status"] == "success"
    assert step["final_tag_parse"]["parsed_next_node"] == 2
    assert step["action_validation"]["is_valid"] is False
    assert step["action_validation"]["failure_reason"] == "already_visited"
    assert step["state_after"] is None


def test_rollout_policy_node_count_minus_2_with_auto_complete_last_node() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    backend = LocalDeterministicModelBackend()
    runner = ZeroShotRolloutRunner(
        model_backend=backend,
        config=ZeroShotRolloutConfig(
            start_node_policy="fixed",
            fixed_start_node=1,
            rollout_step_policy="node_count_minus_2",
            auto_complete_last_node=True,
            enable_solver_completion=False,
        ),
        solver=None,
    )
    episode = runner.run_episode(instance=instance, episode_id="episode_policy_node_count_minus_2")

    assert episode["status"] == "success"
    assert episode["summary_metrics"]["num_steps"] == instance.node_count - 2
    assert episode["final_route"] is not None
    assert len(episode["final_route"]) == instance.node_count
    assert episode["metadata"]["prediction_budget"] == instance.node_count - 2
    assert episode["metadata"]["prediction_steps_used"] == instance.node_count - 2
    assert len(episode["metadata"]["auto_completed_nodes"]) == 1


def test_random_start_policy_is_reproducible_with_seed() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    backend_a = LocalDeterministicModelBackend()
    backend_b = LocalDeterministicModelBackend()
    config = ZeroShotRolloutConfig(
        start_node_policy="random",
        random_seed=2026,
        rollout_step_policy="until_terminal",
        enable_solver_completion=False,
    )

    episode_a = ZeroShotRolloutRunner(model_backend=backend_a, config=config, solver=None).run_episode(
        instance=instance,
        episode_id="episode_random_start_a",
    )
    episode_b = ZeroShotRolloutRunner(model_backend=backend_b, config=config, solver=None).run_episode(
        instance=instance,
        episode_id="episode_random_start_b",
    )

    assert episode_a["initial_state"]["partial_route"] == episode_b["initial_state"]["partial_route"]


def test_rollout_records_step_when_model_backend_raises() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    runner = ZeroShotRolloutRunner(
        model_backend=_FailingBackend(),  # type: ignore[arg-type]
        config=ZeroShotRolloutConfig(enable_solver_completion=False),
        solver=None,
    )

    episode = runner.run_episode(instance=instance, episode_id="episode_backend_error")
    assert episode["status"] == "failed_other"
    assert episode["final_route"] is None
    assert len(episode["step_logs"]) == 1
    step = episode["step_logs"][0]
    assert step["prompt_text"]
    assert step["raw_model_output"] == ""
    assert step["final_tag_parse"]["status"] == "missing_tag"
    assert step["action_validation"]["failure_reason"] == "parse_failure"
    assert "simulated backend failure" in step["metadata"]["model_error"]
