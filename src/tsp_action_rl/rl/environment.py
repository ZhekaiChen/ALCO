"""Step-level RL environment for TSP next-node action learning."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from tsp_action_rl.data import (
    INDEXING_TSPLIB_1_BASED,
    Position2D,
    RolloutState,
    TSPInstance,
    build_initial_rollout_state,
)
from tsp_action_rl.solvers import LKHIntegration, LKHSolveResult

from .reward import (
    RewardContext,
    RewardFunction,
    RewardResult,
    RewardSettings,
    build_reward_function,
)


class RLEnvironmentError(RuntimeError):
    """Raised when the environment is misused or in an invalid lifecycle state."""


class RLEnvironmentConfigError(ValueError):
    """Raised when RL environment config is invalid."""


@dataclass(frozen=True)
class RLActionValidationResult:
    """Validation result for one step-level next-node action."""

    is_valid: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class TSPRLEnvSettings:
    """Configurable settings for the TSP step-level RL environment."""

    start_node_policy: Literal["fixed", "random"] = "fixed"
    fixed_start_node: int = 1
    random_seed: int = 0

    max_steps: int | None = None
    invalid_action_handling: Literal["terminate_episode", "continue_episode"] = "terminate_episode"
    max_invalid_actions: int | None = None

    solver_completion_diagnostics: bool = True

    reward: RewardSettings = field(default_factory=RewardSettings)

    def __post_init__(self) -> None:
        if self.start_node_policy not in {"fixed", "random"}:
            raise RLEnvironmentConfigError("start_node_policy must be one of: fixed, random.")
        if self.fixed_start_node < 1:
            raise RLEnvironmentConfigError(f"fixed_start_node must be >= 1, got {self.fixed_start_node}.")
        if self.max_steps is not None and self.max_steps < 1:
            raise RLEnvironmentConfigError(f"max_steps must be >= 1 when set, got {self.max_steps}.")
        if self.invalid_action_handling not in {"terminate_episode", "continue_episode"}:
            raise RLEnvironmentConfigError(
                "invalid_action_handling must be one of: terminate_episode, continue_episode."
            )
        if self.max_invalid_actions is not None and self.max_invalid_actions < 1:
            raise RLEnvironmentConfigError(
                f"max_invalid_actions must be >= 1 when set, got {self.max_invalid_actions}."
            )

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None) -> "TSPRLEnvSettings":
        source = raw or {}
        if not isinstance(source, Mapping):
            raise RLEnvironmentConfigError(
                f"rl env config must be a mapping/object, got {type(source).__name__}."
            )

        invalid_raw = source.get("invalid_action", {})
        if invalid_raw is None:
            invalid_raw = {}
        if not isinstance(invalid_raw, Mapping):
            raise RLEnvironmentConfigError("invalid_action config must be a mapping/object.")

        solver_completion_raw = source.get("solver_completion", {})
        if solver_completion_raw is None:
            solver_completion_raw = {}
        if not isinstance(solver_completion_raw, Mapping):
            raise RLEnvironmentConfigError("solver_completion config must be a mapping/object.")

        reward_raw = source.get("reward", {})
        if reward_raw is None:
            reward_raw = {}

        return TSPRLEnvSettings(
            start_node_policy=str(source.get("start_node_policy", "fixed")),
            fixed_start_node=int(source.get("fixed_start_node", 1)),
            random_seed=int(source.get("random_seed", 0)),
            max_steps=None if source.get("max_steps") is None else int(source.get("max_steps")),
            invalid_action_handling=str(
                invalid_raw.get(
                    "handling",
                    source.get("invalid_action_handling", "terminate_episode"),
                )
            ),
            max_invalid_actions=(
                int(invalid_raw.get("max_invalid_actions"))
                if invalid_raw.get("max_invalid_actions") is not None
                else (
                    None if source.get("max_invalid_actions") is None else int(source.get("max_invalid_actions"))
                )
            ),
            solver_completion_diagnostics=bool(
                solver_completion_raw.get(
                    "diagnostics",
                    source.get("solver_completion_diagnostics", True),
                )
            ),
            reward=RewardSettings.from_mapping(reward_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_node_policy": self.start_node_policy,
            "fixed_start_node": self.fixed_start_node,
            "random_seed": self.random_seed,
            "max_steps": self.max_steps,
            "invalid_action": {
                "handling": self.invalid_action_handling,
                "max_invalid_actions": self.max_invalid_actions,
            },
            "solver_completion": {
                "diagnostics": self.solver_completion_diagnostics,
            },
            "reward": self.reward.to_dict(),
        }


@dataclass(frozen=True)
class TSPRLObservation:
    """Environment observation for one step-level TSP RL decision."""

    instance: TSPInstance
    rollout_state: RolloutState
    reference_tour_length: float
    reference_tour: tuple[int, ...]
    latest_constrained_tour_length: float | None
    latest_gap_to_reference: float | None
    step_count: int
    invalid_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance.to_dict(),
            "rollout_state": self.rollout_state.to_dict(),
            "reference_tour_length": self.reference_tour_length,
            "reference_tour": list(self.reference_tour),
            "latest_constrained_tour_length": self.latest_constrained_tour_length,
            "latest_gap_to_reference": self.latest_gap_to_reference,
            "step_count": self.step_count,
            "invalid_action_count": self.invalid_action_count,
        }


@dataclass(frozen=True)
class TSPRLStepResult:
    """Single-step environment transition output."""

    observation: TSPRLObservation
    reward: float
    done: bool
    done_reason: str | None
    reward_signal: RewardResult
    diagnostics: dict[str, Any]


class TSPRLStepEnvironment:
    """TSP step-level RL environment with online constrained LKH completion."""

    def __init__(
        self,
        *,
        solver: LKHIntegration,
        settings: TSPRLEnvSettings,
        reward_function: RewardFunction | None = None,
    ) -> None:
        self.solver = solver
        self.settings = settings
        self.reward_function: RewardFunction = reward_function or build_reward_function(settings.reward)
        self._rng = random.Random(settings.random_seed)

        self._instance: TSPInstance | None = None
        self._state: RolloutState | None = None
        self._reference_result: LKHSolveResult | None = None
        self._latest_constrained_tour_length: float | None = None

        self._step_count = 0
        self._invalid_action_count = 0
        self._done = False
        self._done_reason: str | None = None
        self._step_history: list[dict[str, Any]] = []

    def reset(self, *, instance: TSPInstance, start_node: int | None = None) -> TSPRLObservation:
        """Start a fresh RL episode for one instance and return initial observation."""
        resolved_start = self._resolve_start_node(instance=instance, explicit_start_node=start_node)

        # Full-solve reference tour is required for online step-level reward comparisons.
        reference = self.solver.solve_reference(instance)

        self._instance = instance
        self._state = build_initial_rollout_state(instance, start_node=resolved_start)
        self._reference_result = reference
        self._latest_constrained_tour_length = None

        self._step_count = 0
        self._invalid_action_count = 0
        self._done = False
        self._done_reason = None
        self._step_history = []

        return self._build_observation()

    def current_observation(self) -> TSPRLObservation:
        """Return the current observation without stepping."""
        return self._build_observation()

    def step(self, action: int, *, action_metadata: Mapping[str, Any] | None = None) -> TSPRLStepResult:
        """Apply one next-node action and return reward, done flag, and diagnostics."""
        state_before = self._require_state()
        instance = self._require_instance()
        reference = self._require_reference_result()

        if self._done:
            raise RLEnvironmentError(
                f"Episode is already done (done_reason={self._done_reason}). Call reset(...) before step(...)."
            )

        self._step_count += 1
        parse_status = self._parse_status_from_metadata(action_metadata)
        validation = self._validate_action(state=state_before, action=action)
        prefix_partial_tour_length = self._compute_partial_route_length(instance=instance, partial_route=state_before.partial_route)

        solver_completion: dict[str, Any] = {
            "status": "skipped",
            "constrained_tour": None,
            "constrained_tour_length": None,
            "reference_tour_length": reference.tour_length,
            "debug_paths": {},
        }

        state_after: RolloutState | None = None
        done = False
        done_reason: str | None = None
        reward_context: RewardContext

        if not validation.is_valid:
            self._invalid_action_count += 1
            done, done_reason = self._resolve_invalid_done()
            reward_context = RewardContext(
                action_is_valid=False,
                action_failure_reason=validation.failure_reason,
                parse_status=parse_status,
                reference_tour_length=reference.tour_length,
                constrained_tour_length=self._latest_constrained_tour_length,
                previous_constrained_tour_length=self._latest_constrained_tour_length,
                is_terminal_step=False,
                prefix_partial_tour_length=prefix_partial_tour_length,
            )
        else:
            state_after = self._advance_state(instance=instance, state=state_before, next_node=action)
            previous_constrained = self._latest_constrained_tour_length

            completion = self.solver.solve_with_fixed_prefix(instance, state_after.partial_route)
            self._latest_constrained_tour_length = completion.tour_length

            solver_completion = {
                "status": "success",
                "constrained_tour": list(completion.tour),
                "constrained_tour_length": completion.tour_length,
                "reference_tour_length": reference.tour_length,
                "debug_paths": completion.debug_paths if self.settings.solver_completion_diagnostics else {},
            }

            self._state = state_after

            if state_after.is_terminal:
                done = True
                done_reason = "route_completed"
            elif self.settings.max_steps is not None and self._step_count >= self.settings.max_steps:
                done = True
                done_reason = "max_steps"

            reward_context = RewardContext(
                action_is_valid=True,
                action_failure_reason=None,
                parse_status=parse_status,
                reference_tour_length=reference.tour_length,
                constrained_tour_length=self._latest_constrained_tour_length,
                previous_constrained_tour_length=previous_constrained,
                is_terminal_step=(done_reason == "route_completed"),
                prefix_partial_tour_length=prefix_partial_tour_length,
            )

        reward_signal = self.reward_function.compute(reward_context)

        if done:
            self._done = True
            self._done_reason = done_reason

        observation = self._build_observation()
        diagnostics = {
            "action": action,
            "action_validation": validation.to_dict(),
            "parse_status": parse_status,
            "step_count": self._step_count,
            "invalid_action_count": self._invalid_action_count,
            "reward_signal": reward_signal.to_dict(),
            "solver_completion": solver_completion,
            "reference_tour_length": reference.tour_length,
            "prefix_partial_tour_length": prefix_partial_tour_length,
            "state_before": state_before.to_dict(),
            "state_after": None if state_after is None else state_after.to_dict(),
            "done": done,
            "done_reason": done_reason,
        }
        self._step_history.append(diagnostics)

        return TSPRLStepResult(
            observation=observation,
            reward=reward_signal.reward_value,
            done=done,
            done_reason=done_reason,
            reward_signal=reward_signal,
            diagnostics=diagnostics,
        )

    def episode_done(self) -> bool:
        """Whether the current episode is terminal."""
        return self._done

    def done_reason(self) -> str | None:
        """Terminal reason for the current episode, if done."""
        return self._done_reason

    def step_history(self) -> tuple[dict[str, Any], ...]:
        """Return immutable snapshot of per-step diagnostics for the current episode."""
        return tuple(self._step_history)

    def _build_observation(self) -> TSPRLObservation:
        instance = self._require_instance()
        state = self._require_state()
        reference = self._require_reference_result()

        latest_gap: float | None = None
        if self._latest_constrained_tour_length is not None and reference.tour_length > 0:
            latest_gap = (self._latest_constrained_tour_length - reference.tour_length) / reference.tour_length

        return TSPRLObservation(
            instance=instance,
            rollout_state=state,
            reference_tour_length=reference.tour_length,
            reference_tour=reference.tour,
            latest_constrained_tour_length=self._latest_constrained_tour_length,
            latest_gap_to_reference=latest_gap,
            step_count=self._step_count,
            invalid_action_count=self._invalid_action_count,
        )

    def _resolve_invalid_done(self) -> tuple[bool, str | None]:
        if self.settings.invalid_action_handling == "terminate_episode":
            return True, "invalid_action"

        if self.settings.max_invalid_actions is not None and self._invalid_action_count >= self.settings.max_invalid_actions:
            return True, "max_invalid_actions"

        if self.settings.max_steps is not None and self._step_count >= self.settings.max_steps:
            return True, "max_steps"

        return False, None

    def _resolve_start_node(self, *, instance: TSPInstance, explicit_start_node: int | None) -> int:
        if explicit_start_node is not None:
            start_node = explicit_start_node
        elif self.settings.start_node_policy == "fixed":
            start_node = self.settings.fixed_start_node
        else:
            start_node = self._rng.choice(list(range(1, instance.node_count + 1)))

        if start_node < 1 or start_node > instance.node_count:
            raise RLEnvironmentConfigError(
                f"start_node must be in 1..{instance.node_count}, got {start_node}."
            )
        return start_node

    @staticmethod
    def _validate_action(*, state: RolloutState, action: int) -> RLActionValidationResult:
        if state.is_terminal:
            return RLActionValidationResult(is_valid=False, failure_reason="terminal_state")

        if not isinstance(action, int) or isinstance(action, bool):
            return RLActionValidationResult(is_valid=False, failure_reason="non_integer_action")

        if action < 1 or action > state.node_count:
            return RLActionValidationResult(is_valid=False, failure_reason="out_of_range")

        if action in state.visited_nodes:
            return RLActionValidationResult(is_valid=False, failure_reason="already_visited")

        return RLActionValidationResult(is_valid=True, failure_reason=None)

    @staticmethod
    def _advance_state(*, instance: TSPInstance, state: RolloutState, next_node: int) -> RolloutState:
        nodes_by_id = {node.node_id: node for node in instance.nodes}
        if next_node not in nodes_by_id:
            raise RLEnvironmentError(
                f"next_node {next_node} missing from instance node map; expected ids 1..{instance.node_count}."
            )

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
            indexing=INDEXING_TSPLIB_1_BASED,
            notes=dict(state.notes),
        )

    @staticmethod
    def _parse_status_from_metadata(action_metadata: Mapping[str, Any] | None) -> str | None:
        if action_metadata is None:
            return None
        if not isinstance(action_metadata, Mapping):
            return None
        raw_status = action_metadata.get("parse_status")
        if isinstance(raw_status, str) and raw_status.strip():
            return raw_status.strip()
        return None

    @staticmethod
    def _compute_partial_route_length(*, instance: TSPInstance, partial_route: tuple[int, ...]) -> float:
        if len(partial_route) < 2:
            return 0.0
        coords = {node.node_id: (node.x, node.y) for node in instance.nodes}
        total = 0.0
        for left, right in zip(partial_route[:-1], partial_route[1:], strict=True):
            x1, y1 = coords[left]
            x2, y2 = coords[right]
            total += math.hypot(x2 - x1, y2 - y1)
        return total

    def _require_instance(self) -> TSPInstance:
        if self._instance is None:
            raise RLEnvironmentError("No active instance. Call reset(instance=...) first.")
        return self._instance

    def _require_state(self) -> RolloutState:
        if self._state is None:
            raise RLEnvironmentError("No active rollout state. Call reset(instance=...) first.")
        return self._state

    def _require_reference_result(self) -> LKHSolveResult:
        if self._reference_result is None:
            raise RLEnvironmentError("Reference solution is unavailable. Call reset(instance=...) first.")
        return self._reference_result
