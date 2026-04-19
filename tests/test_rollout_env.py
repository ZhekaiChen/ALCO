"""Zero-shot rollout smoke tests."""

from __future__ import annotations

from pathlib import Path

from tsp_action_rl.data import load_rollout_state, load_tsp_instance
from tsp_action_rl.inference import (
    DMXAPIBackendError,
    LocalDeterministicModelBackend,
    LocalStaticResponseBackend,
    ModelOutput,
)
from tsp_action_rl.prompts import PromptRenderConfig, render_tsp_next_node_prompt
from tsp_action_rl.rollout import RolloutProgressUpdate, ZeroShotRolloutConfig, ZeroShotRolloutRunner

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class _FailingBackend:
    backend_type = "api"
    model_name = "failing-backend"

    def generate(self, prompt_text: str, *, instance, state) -> ModelOutput:  # type: ignore[no-untyped-def]
        del prompt_text, instance, state
        raise RuntimeError("simulated backend failure")


class _SequenceBackend:
    backend_type = "api"
    model_name = "sequence-backend"

    def __init__(self, responses: list[str | Exception]) -> None:
        if not responses:
            raise ValueError("responses must not be empty")
        self._responses = responses
        self._index = 0
        self.calls = 0

    def generate(self, prompt_text: str, *, instance, state) -> ModelOutput:  # type: ignore[no-untyped-def]
        del prompt_text, instance, state
        self.calls += 1
        current_index = min(self._index, len(self._responses) - 1)
        self._index += 1
        response = self._responses[current_index]
        if isinstance(response, Exception):
            raise response
        return ModelOutput(raw_text=response, metadata={"call_index": self.calls})


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


def test_rollout_progress_callback_reports_episode_step_elapsed_and_eta() -> None:
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

    updates: list[RolloutProgressUpdate] = []
    runner.run_episodes(
        instance=instance,
        num_episodes=2,
        episode_id_prefix="episode_progress",
        episode_index_offset=3,
        total_episodes=7,
        progress_callback=updates.append,
    )

    # 5-node instance with node_count_minus_2 policy produces 3 model-predicted steps per episode.
    assert len(updates) == 6
    assert updates[0].episode_index == 4
    assert updates[0].total_episodes == 7
    assert updates[0].step_index == 1
    assert updates[0].expected_steps == 3
    assert updates[0].node_count == 5
    assert updates[0].step_attempt == 1
    assert updates[0].step_retry_count == 0
    assert updates[0].max_step_retries == 0
    assert updates[-1].episode_index == 5
    assert updates[-1].step_index == 3


def test_step_retry_parse_failure_then_success() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    initial_state = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json")
    backend = _SequenceBackend(
        [
            "Reasoning without final tag.",
            "Valid reasoning.\n\n<FINAL_NEXT_NODE>4</FINAL_NEXT_NODE>",
        ]
    )
    runner = ZeroShotRolloutRunner(
        model_backend=backend,  # type: ignore[arg-type]
        config=ZeroShotRolloutConfig(
            rollout_step_policy="fixed",
            fixed_prediction_steps=1,
            auto_complete_last_node=True,
            enable_solver_completion=False,
            max_step_retries=1,
            retry_on_parse_failure=True,
            retry_on_provider_error=False,
            retry_backoff_seconds=0.0,
        ),
        solver=None,
    )
    episode = runner.run_episode(instance=instance, episode_id="episode_retry_parse_success", initial_state=initial_state)

    assert episode["status"] == "success"
    assert backend.calls == 2
    step = episode["step_logs"][0]
    retry = step["metadata"]["retry"]
    assert retry["attempts_made"] == 2
    assert retry["succeeded_on_attempt"] == 2
    assert retry["failed_attempts"][0]["kind"] == "parse_failure"
    assert retry["failed_attempts"][0]["parse_status"] == "missing_tag"
    assert episode["metadata"]["parse_retries_used"] == 1


def test_step_retry_parse_failure_exhausted() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    initial_state = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json")
    backend = _SequenceBackend(
        [
            "Missing final tag in attempt one.",
            "Still missing final tag in attempt two.",
        ]
    )
    runner = ZeroShotRolloutRunner(
        model_backend=backend,  # type: ignore[arg-type]
        config=ZeroShotRolloutConfig(
            rollout_step_policy="fixed",
            fixed_prediction_steps=1,
            auto_complete_last_node=True,
            enable_solver_completion=False,
            max_step_retries=1,
            retry_on_parse_failure=True,
            retry_on_provider_error=False,
            retry_backoff_seconds=0.0,
        ),
        solver=None,
    )
    episode = runner.run_episode(instance=instance, episode_id="episode_retry_parse_exhausted", initial_state=initial_state)

    assert episode["status"] == "failed_parse"
    assert backend.calls == 2
    step = episode["step_logs"][0]
    assert step["final_tag_parse"]["status"] == "missing_tag"
    retry = step["metadata"]["retry"]
    assert retry["attempts_made"] == 2
    assert retry["succeeded_on_attempt"] is None
    assert len(retry["failed_attempts"]) == 2
    assert retry["failed_attempts"][0]["retryable"] is True
    assert retry["failed_attempts"][1]["retryable"] is False


def test_step_retry_provider_error_then_success() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    initial_state = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json")
    backend = _SequenceBackend(
        [
            DMXAPIBackendError("gateway timeout", metadata={"http_status": 504}),
            "Valid reasoning.\n\n<FINAL_NEXT_NODE>4</FINAL_NEXT_NODE>",
        ]
    )
    runner = ZeroShotRolloutRunner(
        model_backend=backend,  # type: ignore[arg-type]
        config=ZeroShotRolloutConfig(
            rollout_step_policy="fixed",
            fixed_prediction_steps=1,
            auto_complete_last_node=True,
            enable_solver_completion=False,
            max_step_retries=1,
            retry_on_parse_failure=True,
            retry_on_provider_error=True,
            retry_backoff_seconds=0.0,
        ),
        solver=None,
    )
    episode = runner.run_episode(instance=instance, episode_id="episode_retry_provider_success", initial_state=initial_state)

    assert episode["status"] == "success"
    assert backend.calls == 2
    step = episode["step_logs"][0]
    retry = step["metadata"]["retry"]
    assert retry["attempts_made"] == 2
    assert retry["succeeded_on_attempt"] == 2
    assert retry["failed_attempts"][0]["kind"] == "provider_error"
    assert retry["failed_attempts"][0]["provider_retry_reason"] == "http_504"
    assert episode["metadata"]["provider_retries_used"] == 1


def test_provider_error_logging_without_model_output_has_structured_context() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    initial_state = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json")
    backend = _SequenceBackend(
        [
            DMXAPIBackendError(
                "DMXAPI request failed: <urlopen error [Errno 111] Connection refused>",
                metadata={
                    "response_metadata": {
                        "endpoint": "https://dmx.example/v1/chat/completions",
                        "model_id": "glm-5.1",
                        "model_profile": "glm-5.1",
                    },
                    "provider_error": {
                        "type": "url_error",
                        "message": "<urlopen error [Errno 111] Connection refused>",
                        "reason_errno": 111,
                        "has_error_body": False,
                    },
                },
            ),
        ]
    )
    runner = ZeroShotRolloutRunner(
        model_backend=backend,  # type: ignore[arg-type]
        config=ZeroShotRolloutConfig(
            rollout_step_policy="fixed",
            fixed_prediction_steps=1,
            auto_complete_last_node=True,
            enable_solver_completion=False,
            max_step_retries=0,
            retry_on_parse_failure=True,
            retry_on_provider_error=True,
            retry_backoff_seconds=0.0,
        ),
        solver=None,
    )
    episode = runner.run_episode(instance=instance, episode_id="episode_provider_error_context", initial_state=initial_state)

    assert episode["status"] == "failed_other"
    step = episode["step_logs"][0]
    assert step["raw_model_output"] == ""
    provider_error = step["metadata"]["provider_error"]
    assert provider_error["type"] == "url_error"
    assert "Connection refused" in provider_error["message"]
    assert provider_error["endpoint"] == "https://dmx.example/v1/chat/completions"
    assert provider_error["model_id"] == "glm-5.1"
    assert provider_error["model_profile"] == "glm-5.1"
    assert provider_error["has_response_body"] is False
    assert provider_error["raw_error_payload"] is None


def test_step_retry_connection_refused_provider_error_then_success() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    initial_state = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json")
    backend = _SequenceBackend(
        [
            DMXAPIBackendError(
                "DMXAPI request failed: <urlopen error [Errno 111] Connection refused>",
                metadata={
                    "response_metadata": {
                        "endpoint": "https://dmx.example/v1/chat/completions",
                        "model_id": "glm-5.1",
                        "model_profile": "glm-5.1",
                    },
                    "provider_error": {
                        "type": "url_error",
                        "message": "<urlopen error [Errno 111] Connection refused>",
                        "reason_message": "[Errno 111] Connection refused",
                        "reason_errno": 111,
                        "has_error_body": False,
                    },
                },
            ),
            "Valid reasoning.\n\n<FINAL_NEXT_NODE>4</FINAL_NEXT_NODE>",
        ]
    )
    runner = ZeroShotRolloutRunner(
        model_backend=backend,  # type: ignore[arg-type]
        config=ZeroShotRolloutConfig(
            rollout_step_policy="fixed",
            fixed_prediction_steps=1,
            auto_complete_last_node=True,
            enable_solver_completion=False,
            max_step_retries=1,
            retry_on_parse_failure=True,
            retry_on_provider_error=True,
            retry_backoff_seconds=0.0,
        ),
        solver=None,
    )
    episode = runner.run_episode(instance=instance, episode_id="episode_retry_connection_refused", initial_state=initial_state)

    assert episode["status"] == "success"
    retry = episode["step_logs"][0]["metadata"]["retry"]
    assert retry["attempts_made"] == 2
    assert retry["failed_attempts"][0]["provider_retry_reason"] == "connection_refused"
    assert retry["succeeded_on_attempt"] == 2


def test_step_retry_timeout_error_then_success() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    initial_state = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json")
    backend = _SequenceBackend(
        [
            TimeoutError("The read operation timed out"),
            "Valid reasoning.\n\n<FINAL_NEXT_NODE>4</FINAL_NEXT_NODE>",
        ]
    )
    runner = ZeroShotRolloutRunner(
        model_backend=backend,  # type: ignore[arg-type]
        config=ZeroShotRolloutConfig(
            rollout_step_policy="fixed",
            fixed_prediction_steps=1,
            auto_complete_last_node=True,
            enable_solver_completion=False,
            max_step_retries=1,
            retry_on_parse_failure=True,
            retry_on_provider_error=True,
            retry_backoff_seconds=0.0,
        ),
        solver=None,
    )
    episode = runner.run_episode(instance=instance, episode_id="episode_retry_timeout_success", initial_state=initial_state)

    assert episode["status"] == "success"
    assert backend.calls == 2
    retry = episode["step_logs"][0]["metadata"]["retry"]
    assert retry["attempts_made"] == 2
    assert retry["failed_attempts"][0]["provider_retry_reason"] == "timeout"
    assert retry["succeeded_on_attempt"] == 2


def test_retry_progress_update_includes_attempt_indicator() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    initial_state = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json")
    backend = _SequenceBackend(
        [
            "Missing final tag on first attempt.",
            "Valid reasoning.\n\n<FINAL_NEXT_NODE>4</FINAL_NEXT_NODE>",
        ]
    )
    runner = ZeroShotRolloutRunner(
        model_backend=backend,  # type: ignore[arg-type]
        config=ZeroShotRolloutConfig(
            rollout_step_policy="fixed",
            fixed_prediction_steps=1,
            auto_complete_last_node=True,
            enable_solver_completion=False,
            max_step_retries=2,
            retry_on_parse_failure=True,
            retry_on_provider_error=False,
            retry_backoff_seconds=0.0,
        ),
        solver=None,
    )

    updates: list[RolloutProgressUpdate] = []
    runner.run_episode(
        instance=instance,
        episode_id="episode_retry_progress",
        initial_state=initial_state,
        progress_callback=updates.append,
    )

    assert len(updates) == 1
    assert updates[0].step_index == 1
    assert updates[0].step_attempt == 2
    assert updates[0].step_retry_count == 1
    assert updates[0].max_step_retries == 2
