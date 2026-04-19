"""Zero-shot rollout episode runner for action-level TSP next-node prediction."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from tsp_action_rl.data import Position2D, RolloutState, TSPInstance, build_initial_rollout_state
from tsp_action_rl.inference import ModelBackend
from tsp_action_rl.parsing import FinalTagParseResult, parse_final_next_node
from tsp_action_rl.prompts import PromptRenderConfig, render_tsp_next_node_prompt
from tsp_action_rl.solvers import LKHIntegration


@dataclass(frozen=True)
class ActionValidationResult:
    """Validation result for parsed next-node action."""

    is_valid: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"is_valid": self.is_valid, "failure_reason": self.failure_reason}


@dataclass(frozen=True)
class ZeroShotRolloutConfig:
    """Execution options for zero-shot episode rollout."""

    start_node_policy: Literal["fixed", "random"] = "fixed"
    fixed_start_node: int = 1
    rollout_step_policy: Literal["until_terminal", "node_count_minus_2", "fixed"] = "until_terminal"
    fixed_prediction_steps: int | None = None
    auto_complete_last_node: bool = False
    close_tour_to_start: bool = True
    random_seed: int = 0
    max_steps: int | None = None
    enable_solver_completion: bool = False
    include_current_position: bool = True
    include_visited_nodes: bool = False
    include_unvisited_nodes: bool = True
    max_step_retries: int = 0
    retry_on_parse_failure: bool = True
    retry_on_provider_error: bool = False
    retry_backoff_seconds: float = 0.0


@dataclass(frozen=True)
class RolloutProgressUpdate:
    """Operator-facing per-step progress snapshot."""

    episode_id: str
    episode_index: int
    total_episodes: int
    step_index: int
    expected_steps: int
    node_count: int
    elapsed_seconds: float
    eta_seconds: float
    step_attempt: int
    step_retry_count: int
    max_step_retries: int


ProgressCallback = Callable[[RolloutProgressUpdate], None]


class ZeroShotRolloutRunner:
    """Runs zero-shot episodes by prompt -> model -> parse -> validate -> transition."""

    def __init__(
        self,
        *,
        model_backend: ModelBackend,
        config: ZeroShotRolloutConfig,
        solver: LKHIntegration | None = None,
    ) -> None:
        if config.enable_solver_completion and solver is None:
            raise ValueError("solver must be provided when enable_solver_completion=True.")
        if config.start_node_policy not in {"fixed", "random"}:
            raise ValueError("start_node_policy must be one of: fixed, random.")
        if config.rollout_step_policy not in {"until_terminal", "node_count_minus_2", "fixed"}:
            raise ValueError("rollout_step_policy must be one of: until_terminal, node_count_minus_2, fixed.")
        if config.rollout_step_policy == "fixed":
            if config.fixed_prediction_steps is None or config.fixed_prediction_steps < 0:
                raise ValueError("fixed_prediction_steps must be >= 0 when rollout_step_policy='fixed'.")
        if config.max_step_retries < 0:
            raise ValueError("max_step_retries must be >= 0.")
        if config.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0.")

        self.model_backend = model_backend
        self.config = config
        self.solver = solver
        self._rng = random.Random(config.random_seed)
        self.prompt_config = PromptRenderConfig(
            include_current_position=config.include_current_position,
            include_visited_nodes=config.include_visited_nodes,
            include_unvisited_nodes=config.include_unvisited_nodes,
        )

    def run_episode(
        self,
        *,
        instance: TSPInstance,
        episode_id: str,
        initial_state: RolloutState | None = None,
        episode_index: int = 1,
        total_episodes: int = 1,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        state = initial_state or build_initial_rollout_state(instance, start_node=self._choose_start_node(instance))
        initial_state_payload = state.to_dict()
        prediction_budget = self._resolve_prediction_budget(node_count=instance.node_count)
        expected_steps_for_progress = self._resolve_expected_steps_for_progress(
            initial_state=state,
            prediction_budget=prediction_budget,
        )
        episode_start_monotonic = time.monotonic()
        prediction_steps_used = 0
        auto_completed_nodes: list[int] = []

        parse_successes = 0
        valid_actions = 0
        total_step_attempts = 0
        total_step_retries = 0
        steps_with_retries = 0
        parse_retry_count = 0
        provider_retry_count = 0
        step_logs: list[dict[str, Any]] = []
        final_route: list[int] | None = None
        latest_constrained_tour_length: float | None = None

        reference_tour_length, reference_metadata = self._resolve_reference_tour_length(instance)
        if reference_metadata.get("reference_status") == "failed_solver":
            return self._finalize_episode_log(
                instance=instance,
                episode_id=episode_id,
                status="failed_solver",
                initial_state_payload=initial_state_payload,
                final_route=None,
                step_logs=[],
                parse_successes=0,
                valid_actions=0,
                reference_tour_length=reference_tour_length,
                latest_constrained_tour_length=None,
                extra_metadata=reference_metadata,
            )

        episode_status = "failed_other"
        while True:
            if state.is_terminal:
                episode_status = "success"
                final_route = list(state.partial_route)
                break

            if prediction_budget is not None and prediction_steps_used >= prediction_budget:
                if self.config.auto_complete_last_node and len(state.unvisited_nodes) == 1:
                    auto_node = state.unvisited_nodes[0]
                    state = self._advance_state(instance=instance, state=state, next_node=auto_node)
                    auto_completed_nodes.append(auto_node)
                    continue
                episode_status = "failed_other"
                break

            if self.config.max_steps is not None and len(step_logs) >= self.config.max_steps:
                episode_status = "failed_other"
                break

            prompt_text = render_tsp_next_node_prompt(instance=instance, state=state, config=self.prompt_config)
            attempts_made = 0
            failed_attempts: list[dict[str, Any]] = []
            step_failed_due_to_provider_error = False
            model_output = None
            parse = None
            action_validation = None

            while True:
                attempts_made += 1
                has_retries_remaining = attempts_made <= self.config.max_step_retries
                try:
                    model_output = self.model_backend.generate(prompt_text, instance=instance, state=state)
                except Exception as exc:  # noqa: BLE001
                    provider_retry_reason = self._provider_retry_reason(exc)
                    can_retry_provider = (
                        self.config.retry_on_provider_error
                        and provider_retry_reason is not None
                        and has_retries_remaining
                    )
                    failed_attempts.append(
                        {
                            "attempt_index": attempts_made,
                            "kind": "provider_error",
                            "retryable": can_retry_provider,
                            "provider_retry_reason": provider_retry_reason,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "error_metadata": self._extract_exception_metadata(exc),
                        }
                    )
                    if can_retry_provider:
                        provider_retry_count += 1
                        if self.config.retry_backoff_seconds > 0:
                            time.sleep(self.config.retry_backoff_seconds)
                        continue

                    step_metadata = self._build_model_exception_metadata(exc)
                    step_metadata["retry"] = self._build_retry_metadata(
                        attempts_made=attempts_made,
                        failed_attempts=failed_attempts,
                        succeeded_on_attempt=None,
                    )
                    step_logs.append(
                        {
                            "instance_id": instance.instance_id,
                            "episode_id": episode_id,
                            "step_index": state.step_index,
                            "prompt_text": prompt_text,
                            "raw_model_output": "",
                            "reasoning_text": "",
                            "final_tag_parse": {
                                "status": "missing_tag",
                                "tag_count": 0,
                                "parsed_next_node": None,
                            },
                            "action_validation": {
                                "is_valid": False,
                                "failure_reason": "parse_failure",
                            },
                            "reward_signal": {
                                "reward_mode": "phase3_zero_shot_placeholder",
                                "reward_value": 0.0,
                                "components": {"reference_tour_length": reference_tour_length},
                            },
                            "solver_completion": {
                                "status": "skipped",
                                "constrained_tour": None,
                                "constrained_tour_length": None,
                                "reference_tour_length": reference_tour_length,
                                "debug_paths": {},
                            },
                            "state_before": state.to_dict(),
                            "state_after": None,
                            "metadata": step_metadata,
                        }
                    )
                    total_step_attempts += attempts_made
                    total_step_retries += max(attempts_made - 1, 0)
                    if attempts_made > 1:
                        steps_with_retries += 1
                    episode_status = "failed_other"
                    step_failed_due_to_provider_error = True
                    break

                assert model_output is not None
                parse = parse_final_next_node(model_output.raw_text)
                action_validation = self._validate_action(state=state, parse=parse)

                if parse.status != "success":
                    can_retry_parse_failure = (
                        self.config.retry_on_parse_failure
                        and parse.status in {"missing_tag", "multiple_tags", "malformed_tag"}
                        and has_retries_remaining
                    )
                    failed_attempts.append(
                        {
                            "attempt_index": attempts_made,
                            "kind": "parse_failure",
                            "retryable": can_retry_parse_failure,
                            "parse_status": parse.status,
                            "tag_count": parse.tag_count,
                        }
                    )
                    if can_retry_parse_failure:
                        parse_retry_count += 1
                        if self.config.retry_backoff_seconds > 0:
                            time.sleep(self.config.retry_backoff_seconds)
                        continue
                break

            if step_failed_due_to_provider_error:
                break

            assert model_output is not None
            assert parse is not None
            assert action_validation is not None

            prediction_steps_used += 1
            total_step_attempts += attempts_made
            total_step_retries += max(attempts_made - 1, 0)
            if attempts_made > 1:
                steps_with_retries += 1

            if parse.status == "success":
                parse_successes += 1
            if action_validation.is_valid:
                valid_actions += 1
            elif parse.status == "success":
                failed_attempts.append(
                    {
                        "attempt_index": attempts_made,
                        "kind": "action_validation_failure",
                        "retryable": False,
                        "failure_reason": action_validation.failure_reason,
                    }
                )

            reward_signal: dict[str, Any] = {
                "reward_mode": "phase3_zero_shot_placeholder",
                "reward_value": 0.0,
                "components": {"reference_tour_length": reference_tour_length},
            }
            solver_completion: dict[str, Any] = {
                "status": "skipped",
                "constrained_tour": None,
                "constrained_tour_length": None,
                "reference_tour_length": reference_tour_length,
                "debug_paths": {},
            }
            next_state: RolloutState | None = None
            step_metadata: dict[str, Any] = {
                "model_name": self.model_backend.model_name,
                "model_backend": self.model_backend.backend_type,
            }
            step_metadata.update(model_output.metadata)
            step_metadata["retry"] = self._build_retry_metadata(
                attempts_made=attempts_made,
                failed_attempts=failed_attempts,
                succeeded_on_attempt=(attempts_made if action_validation.is_valid else None),
            )

            if action_validation.is_valid:
                assert parse.parsed_next_node is not None
                next_state = self._advance_state(instance=instance, state=state, next_node=parse.parsed_next_node)

                if self.config.enable_solver_completion:
                    try:
                        assert self.solver is not None
                        completion = self.solver.solve_with_fixed_prefix(instance, next_state.partial_route)
                        solver_completion = {
                            "status": "success",
                            "constrained_tour": list(completion.tour),
                            "constrained_tour_length": completion.tour_length,
                            "reference_tour_length": reference_tour_length,
                            "debug_paths": completion.debug_paths,
                        }
                        latest_constrained_tour_length = completion.tour_length
                        reward_signal["components"]["constrained_tour_length"] = completion.tour_length
                        if reference_tour_length > 0:
                            reward_signal["components"]["gap_to_reference"] = (
                                completion.tour_length - reference_tour_length
                            ) / reference_tour_length
                    except Exception as exc:  # noqa: BLE001
                        solver_completion = {
                            "status": "failed",
                            "constrained_tour": None,
                            "constrained_tour_length": None,
                            "reference_tour_length": reference_tour_length,
                            "debug_paths": {"error": str(exc)},
                        }
                        episode_status = "failed_solver"
                        step_metadata["solver_error"] = str(exc)

            step_logs.append(
                {
                    "instance_id": instance.instance_id,
                    "episode_id": episode_id,
                    "step_index": state.step_index,
                    "prompt_text": prompt_text,
                    "raw_model_output": model_output.raw_text,
                    "reasoning_text": parse.reasoning_text,
                    "final_tag_parse": parse.to_dict(),
                    "action_validation": action_validation.to_dict(),
                    "reward_signal": reward_signal,
                    "solver_completion": solver_completion,
                    "state_before": state.to_dict(),
                    "state_after": None if next_state is None else next_state.to_dict(),
                    "metadata": step_metadata,
                }
            )

            if progress_callback is not None and expected_steps_for_progress > 0:
                elapsed_seconds = time.monotonic() - episode_start_monotonic
                remaining_steps = max(expected_steps_for_progress - prediction_steps_used, 0)
                eta_seconds = (
                    0.0
                    if prediction_steps_used == 0
                    else (elapsed_seconds / prediction_steps_used) * remaining_steps
                )
                progress_callback(
                    RolloutProgressUpdate(
                        episode_id=episode_id,
                        episode_index=episode_index,
                        total_episodes=total_episodes,
                        step_index=prediction_steps_used,
                        expected_steps=expected_steps_for_progress,
                        node_count=instance.node_count,
                        elapsed_seconds=elapsed_seconds,
                        eta_seconds=eta_seconds,
                        step_attempt=attempts_made,
                        step_retry_count=max(attempts_made - 1, 0),
                        max_step_retries=self.config.max_step_retries,
                    )
                )

            if not action_validation.is_valid:
                episode_status = "failed_parse" if parse.status != "success" else "failed_invalid_action"
                break

            if solver_completion["status"] == "failed":
                break

            assert next_state is not None
            state = next_state

        if episode_status == "success":
            final_route = list(state.partial_route)

        closed_tour: list[int] | None = None
        if final_route is not None and self.config.close_tour_to_start and final_route:
            closed_tour = list(final_route) + [final_route[0]]

        return self._finalize_episode_log(
            instance=instance,
            episode_id=episode_id,
            status=episode_status,
            initial_state_payload=initial_state_payload,
            final_route=final_route,
            step_logs=step_logs,
            parse_successes=parse_successes,
            valid_actions=valid_actions,
            reference_tour_length=reference_tour_length,
            latest_constrained_tour_length=latest_constrained_tour_length,
            extra_metadata={
                **reference_metadata,
                "start_node_policy": self.config.start_node_policy,
                "fixed_start_node": self.config.fixed_start_node,
                "rollout_step_policy": self.config.rollout_step_policy,
                "fixed_prediction_steps": self.config.fixed_prediction_steps,
                "prediction_budget": prediction_budget,
                "prediction_steps_used": prediction_steps_used,
                "max_step_retries": self.config.max_step_retries,
                "retry_on_parse_failure": self.config.retry_on_parse_failure,
                "retry_on_provider_error": self.config.retry_on_provider_error,
                "retry_backoff_seconds": self.config.retry_backoff_seconds,
                "total_step_attempts": total_step_attempts,
                "total_step_retries": total_step_retries,
                "steps_with_retries": steps_with_retries,
                "parse_retries_used": parse_retry_count,
                "provider_retries_used": provider_retry_count,
                "auto_complete_last_node": self.config.auto_complete_last_node,
                "auto_completed_nodes": auto_completed_nodes,
                "close_tour_to_start": self.config.close_tour_to_start,
                "closed_tour": closed_tour,
            },
        )

    def run_episodes(
        self,
        *,
        instance: TSPInstance,
        num_episodes: int,
        episode_id_prefix: str,
        episode_index_offset: int = 0,
        total_episodes: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        if num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {num_episodes}.")
        resolved_total_episodes = total_episodes if total_episodes is not None else num_episodes
        if resolved_total_episodes < 1:
            raise ValueError("total_episodes must be >= 1 when provided.")
        if episode_index_offset < 0:
            raise ValueError("episode_index_offset must be >= 0.")

        logs: list[dict[str, Any]] = []
        for idx in range(1, num_episodes + 1):
            episode_id = f"{episode_id_prefix}_{idx:06d}"
            logs.append(
                self.run_episode(
                    instance=instance,
                    episode_id=episode_id,
                    episode_index=episode_index_offset + idx,
                    total_episodes=resolved_total_episodes,
                    progress_callback=progress_callback,
                )
            )
        return logs

    def _choose_start_node(self, instance: TSPInstance) -> int:
        if self.config.start_node_policy == "fixed":
            return self.config.fixed_start_node
        return self._rng.choice(list(range(1, instance.node_count + 1)))

    def _resolve_prediction_budget(self, *, node_count: int) -> int | None:
        if self.config.rollout_step_policy == "until_terminal":
            return None
        if self.config.rollout_step_policy == "node_count_minus_2":
            return max(node_count - 2, 0)
        assert self.config.fixed_prediction_steps is not None
        return self.config.fixed_prediction_steps

    def _resolve_expected_steps_for_progress(
        self,
        *,
        initial_state: RolloutState,
        prediction_budget: int | None,
    ) -> int:
        if prediction_budget is not None:
            expected = prediction_budget
        else:
            expected = len(initial_state.unvisited_nodes)
        if self.config.max_steps is not None:
            expected = min(expected, self.config.max_steps)
        return max(expected, 0)

    def _resolve_reference_tour_length(self, instance: TSPInstance) -> tuple[float, dict[str, Any]]:
        if self.config.enable_solver_completion:
            try:
                assert self.solver is not None
                reference = self.solver.solve_reference(instance)
                return (
                    reference.tour_length,
                    {
                        "reference_status": "success",
                        "reference_tour": list(reference.tour),
                        "reference_debug_paths": reference.debug_paths,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                return (0.0, {"reference_status": "failed_solver", "reference_error": str(exc)})

        if instance.reference_solution is not None:
            return (instance.reference_solution.tour_length, {"reference_status": "fixture_reference"})
        return (0.0, {"reference_status": "unavailable"})

    @staticmethod
    def _validate_action(*, state: RolloutState, parse: FinalTagParseResult) -> ActionValidationResult:
        if parse.status != "success":
            return ActionValidationResult(is_valid=False, failure_reason="parse_failure")
        if state.is_terminal:
            return ActionValidationResult(is_valid=False, failure_reason="terminal_state")
        assert parse.parsed_next_node is not None
        next_node = parse.parsed_next_node
        if next_node < 1 or next_node > state.node_count:
            return ActionValidationResult(is_valid=False, failure_reason="out_of_range")
        if next_node in state.visited_nodes:
            return ActionValidationResult(is_valid=False, failure_reason="already_visited")
        return ActionValidationResult(is_valid=True, failure_reason=None)

    @staticmethod
    def _advance_state(*, instance: TSPInstance, state: RolloutState, next_node: int) -> RolloutState:
        nodes_by_id = {node.node_id: node for node in instance.nodes}
        node = nodes_by_id[next_node]
        new_partial_route = tuple(list(state.partial_route) + [next_node])
        new_unvisited_nodes = tuple(node_id for node_id in state.unvisited_nodes if node_id != next_node)
        return RolloutState(
            instance_id=state.instance_id,
            step_index=len(new_partial_route),
            node_count=state.node_count,
            partial_route=new_partial_route,
            visited_nodes=new_partial_route,
            unvisited_nodes=new_unvisited_nodes,
            current_node=next_node,
            current_position=Position2D(x=node.x, y=node.y),
            is_terminal=(len(new_unvisited_nodes) == 0),
            indexing=state.indexing,
            notes=dict(state.notes),
        )

    @staticmethod
    def _compute_tour_length(*, instance: TSPInstance, tour: list[int]) -> float:
        coords = {node.node_id: (node.x, node.y) for node in instance.nodes}
        total = 0.0
        for idx, node_id in enumerate(tour):
            next_node = tour[(idx + 1) % len(tour)]
            x1, y1 = coords[node_id]
            x2, y2 = coords[next_node]
            total += math.hypot(x2 - x1, y2 - y1)
        return total

    def _finalize_episode_log(
        self,
        *,
        instance: TSPInstance,
        episode_id: str,
        status: str,
        initial_state_payload: dict[str, Any],
        final_route: list[int] | None,
        step_logs: list[dict[str, Any]],
        parse_successes: int,
        valid_actions: int,
        reference_tour_length: float,
        latest_constrained_tour_length: float | None,
        extra_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        num_steps = len(step_logs)
        parse_success_rate = 0.0 if num_steps == 0 else parse_successes / num_steps
        valid_action_rate = 0.0 if num_steps == 0 else valid_actions / num_steps
        total_reward = sum(float(step["reward_signal"]["reward_value"]) for step in step_logs)

        final_gap_to_reference: float | None = None
        if status == "success" and reference_tour_length > 0:
            if latest_constrained_tour_length is not None:
                final_gap_to_reference = (latest_constrained_tour_length - reference_tour_length) / reference_tour_length
            elif final_route is not None:
                final_tour_length = self._compute_tour_length(instance=instance, tour=final_route)
                final_gap_to_reference = (final_tour_length - reference_tour_length) / reference_tour_length

        return {
            "episode_id": episode_id,
            "instance_id": instance.instance_id,
            "status": status,
            "node_count": instance.node_count,
            "initial_state": initial_state_payload,
            "final_route": final_route,
            "step_logs": step_logs,
            "summary_metrics": {
                "num_steps": num_steps,
                "parse_success_rate": parse_success_rate,
                "valid_action_rate": valid_action_rate,
                "total_reward": total_reward,
                "final_gap_to_reference": final_gap_to_reference,
            },
            "metadata": {
                "model_name": self.model_backend.model_name,
                "model_backend": self.model_backend.backend_type,
                **extra_metadata,
            },
        }

    def _build_retry_metadata(
        self,
        *,
        attempts_made: int,
        failed_attempts: list[dict[str, Any]],
        succeeded_on_attempt: int | None,
    ) -> dict[str, Any]:
        return {
            "attempts_made": attempts_made,
            "max_step_retries": self.config.max_step_retries,
            "retry_on_parse_failure": self.config.retry_on_parse_failure,
            "retry_on_provider_error": self.config.retry_on_provider_error,
            "retry_backoff_seconds": self.config.retry_backoff_seconds,
            "failed_attempts": failed_attempts,
            "succeeded_on_attempt": succeeded_on_attempt,
        }

    @staticmethod
    def _extract_exception_metadata(exc: Exception) -> dict[str, Any] | None:
        metadata = getattr(exc, "metadata", None)
        if isinstance(metadata, dict):
            return dict(metadata)
        return None

    @staticmethod
    def _provider_retry_reason(exc: Exception) -> str | None:
        metadata = getattr(exc, "metadata", None)
        if isinstance(metadata, dict):
            http_status = metadata.get("http_status")
            if isinstance(http_status, int) and 500 <= http_status < 600:
                return f"http_{http_status}"

            provider_error = metadata.get("provider_error")
            if isinstance(provider_error, dict):
                provider_type = provider_error.get("type")
                if isinstance(provider_type, str) and "timeout" in provider_type.lower():
                    return provider_type

                reason_errno = provider_error.get("reason_errno")
                if isinstance(reason_errno, int) and reason_errno == 111:
                    return "connection_refused"

                reason_text = provider_error.get("reason_message")
                if isinstance(reason_text, str) and "connection refused" in reason_text.lower():
                    return "connection_refused"

            nested_failure = metadata.get("failure")
            if isinstance(nested_failure, dict):
                failure_type = nested_failure.get("type")
                if isinstance(failure_type, str) and "timeout" in failure_type.lower():
                    return failure_type

        if isinstance(exc, TimeoutError):
            return "timeout"

        message = str(exc).lower()
        if "connection refused" in message:
            return "connection_refused"
        if "timed out" in message or "timeout" in message:
            return "timeout"
        return None

    def _build_model_exception_metadata(self, exc: Exception) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "model_name": self.model_backend.model_name,
            "model_backend": self.model_backend.backend_type,
            "model_error": str(exc),
            "model_error_type": type(exc).__name__,
        }
        extra = getattr(exc, "metadata", None)
        if isinstance(extra, dict):
            metadata["model_error_metadata"] = extra
        metadata["provider_error"] = self._build_provider_error_context(exc)
        return metadata

    @staticmethod
    def _build_provider_error_context(exc: Exception) -> dict[str, Any]:
        extra = getattr(exc, "metadata", None)
        extra_dict = extra if isinstance(extra, dict) else {}
        response_meta = extra_dict.get("response_metadata")
        response_meta_dict = response_meta if isinstance(response_meta, dict) else {}
        provider_error = extra_dict.get("provider_error")
        provider_error_dict = provider_error if isinstance(provider_error, dict) else {}

        endpoint = response_meta_dict.get("endpoint") or extra_dict.get("endpoint")
        model_id = response_meta_dict.get("model_id") or extra_dict.get("model_id")
        model_profile = response_meta_dict.get("model_profile") or extra_dict.get("model_profile")

        http_status = extra_dict.get("http_status")
        if not isinstance(http_status, int):
            provider_http_status = provider_error_dict.get("http_status")
            http_status = provider_http_status if isinstance(provider_http_status, int) else None

        raw_error_payload = extra_dict.get("http_error_body")
        if not isinstance(raw_error_payload, str):
            payload_from_provider_error = provider_error_dict.get("error_body")
            raw_error_payload = payload_from_provider_error if isinstance(payload_from_provider_error, str) else None

        has_response_body = provider_error_dict.get("has_error_body")
        if not isinstance(has_response_body, bool):
            has_response_body = bool(raw_error_payload)

        provider_type = provider_error_dict.get("type")
        if not isinstance(provider_type, str):
            provider_type = type(exc).__name__

        provider_message = provider_error_dict.get("message")
        if not isinstance(provider_message, str):
            provider_message = str(exc)

        return {
            "type": provider_type,
            "message": provider_message,
            "http_status": http_status,
            "endpoint": endpoint if isinstance(endpoint, str) else None,
            "model_id": model_id if isinstance(model_id, str) else None,
            "model_profile": model_profile if isinstance(model_profile, str) else None,
            "has_response_body": has_response_body,
            "raw_error_payload": raw_error_payload,
        }
