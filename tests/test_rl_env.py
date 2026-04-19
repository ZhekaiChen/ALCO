"""Step-level RL environment and reward interface tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tsp_action_rl.config import load_rl_env_settings
from tsp_action_rl.data import generate_random_euclidean_instance, load_tsp_instance
from tsp_action_rl.rl import (
    REWARD_MODE_GAP_ACTION_INVERSE,
    REWARD_MODE_GAP_TO_REFERENCE_ABSOLUTE,
    REWARD_MODE_GAP_TO_REFERENCE_DELTA,
    RewardContext,
    RewardSettings,
    TSPRLEnvSettings,
    TSPRLStepEnvironment,
    build_reward_function,
)
from tsp_action_rl.solvers import LKHSolveResult

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class FakeSolver:
    """Simple deterministic solver stub for RL environment tests."""

    def __init__(self) -> None:
        self.reference_calls = 0
        self.completion_calls: list[tuple[int, ...]] = []

    def solve_reference(self, instance):  # type: ignore[no-untyped-def]
        self.reference_calls += 1
        tour = tuple(range(1, instance.node_count + 1))
        return LKHSolveResult(
            mode="full_reference",
            tour=tour,
            tour_length=100.0,
            fixed_edges=(),
            solver_params={},
            debug_paths={"reference": "fake"},
        )

    def solve_with_fixed_prefix(self, instance, partial_route):  # type: ignore[no-untyped-def]
        prefix = tuple(int(node_id) for node_id in partial_route)
        self.completion_calls.append(prefix)
        remaining = tuple(node_id for node_id in range(1, instance.node_count + 1) if node_id not in prefix)
        fixed_edges = tuple((prefix[idx], prefix[idx + 1]) for idx in range(len(prefix) - 1))
        return LKHSolveResult(
            mode="constrained_completion",
            tour=prefix + remaining,
            tour_length=100.0 + 5.0 * len(prefix),
            fixed_edges=fixed_edges,
            solver_params={},
            debug_paths={"constrained": "fake"},
        )


def test_rl_env_reset_initializes_reference_and_state() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    solver = FakeSolver()
    env = TSPRLStepEnvironment(
        solver=solver,  # type: ignore[arg-type]
        settings=TSPRLEnvSettings(start_node_policy="fixed", fixed_start_node=1),
    )

    observation = env.reset(instance=instance)

    assert solver.reference_calls == 1
    assert list(observation.rollout_state.partial_route) == [1]
    assert observation.reference_tour_length == 100.0
    assert observation.latest_constrained_tour_length is None
    assert observation.step_count == 0
    assert observation.invalid_action_count == 0


def test_rl_env_valid_step_updates_prefix_and_computes_gap_reward() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    solver = FakeSolver()
    env = TSPRLStepEnvironment(
        solver=solver,  # type: ignore[arg-type]
        settings=TSPRLEnvSettings(
            start_node_policy="fixed",
            fixed_start_node=1,
            reward=RewardSettings(
                mode=REWARD_MODE_GAP_TO_REFERENCE_ABSOLUTE,
                dense_gap_scale=-1.0,
                invalid_action_penalty=-2.0,
            ),
        ),
    )

    env.reset(instance=instance)
    step = env.step(2)

    assert step.done is False
    assert step.done_reason is None
    assert solver.completion_calls == [(1, 2)]
    assert list(step.observation.rollout_state.partial_route) == [1, 2]
    assert step.observation.latest_constrained_tour_length == 110.0
    assert step.observation.latest_gap_to_reference == pytest.approx(0.1)
    assert step.reward == pytest.approx(-0.1)
    assert step.reward_signal.components["gap_to_reference"] == pytest.approx(0.1)
    assert step.diagnostics["solver_completion"]["status"] == "success"


def test_rl_env_invalid_action_can_terminate_episode() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    solver = FakeSolver()
    env = TSPRLStepEnvironment(
        solver=solver,  # type: ignore[arg-type]
        settings=TSPRLEnvSettings(
            start_node_policy="fixed",
            fixed_start_node=1,
            invalid_action_handling="terminate_episode",
            reward=RewardSettings(invalid_action_penalty=-3.5),
        ),
    )

    env.reset(instance=instance)
    step = env.step(1)

    assert step.done is True
    assert step.done_reason == "invalid_action"
    assert step.reward == pytest.approx(-3.5)
    assert list(step.observation.rollout_state.partial_route) == [1]
    assert step.diagnostics["action_validation"]["is_valid"] is False
    assert step.diagnostics["action_validation"]["failure_reason"] == "already_visited"


def test_rl_env_continue_mode_respects_max_invalid_actions() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    solver = FakeSolver()
    env = TSPRLStepEnvironment(
        solver=solver,  # type: ignore[arg-type]
        settings=TSPRLEnvSettings(
            start_node_policy="fixed",
            fixed_start_node=1,
            invalid_action_handling="continue_episode",
            max_invalid_actions=2,
        ),
    )

    env.reset(instance=instance)
    step1 = env.step(1)
    step2 = env.step(1)

    assert step1.done is False
    assert step1.done_reason is None
    assert step2.done is True
    assert step2.done_reason == "max_invalid_actions"


def test_rl_env_done_on_route_completion() -> None:
    instance = generate_random_euclidean_instance(node_count=3, seed=7)
    solver = FakeSolver()
    env = TSPRLStepEnvironment(
        solver=solver,  # type: ignore[arg-type]
        settings=TSPRLEnvSettings(
            start_node_policy="fixed",
            fixed_start_node=1,
            invalid_action_handling="continue_episode",
        ),
    )

    env.reset(instance=instance)
    step1 = env.step(2)
    step2 = env.step(3)

    assert step1.done is False
    assert step2.done is True
    assert step2.done_reason == "route_completed"
    assert step2.observation.rollout_state.is_terminal is True
    assert list(step2.observation.rollout_state.partial_route) == [1, 2, 3]


def test_reward_interface_supports_delta_gap_mode() -> None:
    reward_fn = build_reward_function(
        RewardSettings(
            mode=REWARD_MODE_GAP_TO_REFERENCE_DELTA,
            dense_gap_scale=0.0,
            gap_delta_scale=2.0,
        )
    )

    result = reward_fn.compute(
        RewardContext(
            action_is_valid=True,
            action_failure_reason=None,
            parse_status=None,
            reference_tour_length=100.0,
            constrained_tour_length=110.0,
            previous_constrained_tour_length=120.0,
            is_terminal_step=False,
        )
    )

    assert result.reward_mode == REWARD_MODE_GAP_TO_REFERENCE_DELTA
    assert result.components["gap_delta"] == pytest.approx(0.1)
    assert result.reward_value == pytest.approx(0.2)


def test_reward_interface_applies_parse_failure_penalty_hook() -> None:
    reward_fn = build_reward_function(
        RewardSettings(
            mode=REWARD_MODE_GAP_TO_REFERENCE_ABSOLUTE,
            invalid_action_penalty=-2.0,
            parse_failure_penalty_enabled=True,
            parse_failure_penalty=-0.5,
            step_penalty=-0.1,
        )
    )

    result = reward_fn.compute(
        RewardContext(
            action_is_valid=False,
            action_failure_reason="parse_failure",
            parse_status="missing_tag",
            reference_tour_length=100.0,
            constrained_tour_length=None,
            previous_constrained_tour_length=None,
            is_terminal_step=False,
        )
    )

    assert result.reward_value == pytest.approx(-2.6)
    assert result.components["invalid_action_penalty"] == pytest.approx(-2.0)
    assert result.components["parse_failure_penalty"] == pytest.approx(-0.5)
    assert result.components["step_penalty"] == pytest.approx(-0.1)


def test_reward_interface_supports_gap_action_inverse_mode() -> None:
    reward_fn = build_reward_function(
        RewardSettings(
            mode=REWARD_MODE_GAP_ACTION_INVERSE,
            gap_action_scale=1.0,
            invalid_action_penalty=-2.0,
            parse_failure_penalty_enabled=True,
            parse_failure_penalty=-0.25,
        )
    )

    result = reward_fn.compute(
        RewardContext(
            action_is_valid=True,
            action_failure_reason=None,
            parse_status="success",
            reference_tour_length=100.0,
            constrained_tour_length=112.0,
            previous_constrained_tour_length=None,
            is_terminal_step=True,
            prefix_partial_tour_length=40.0,
        )
    )

    # gap_action = (112 - 100) / (100 - 40) = 0.2 -> reward = 1 / 1.2 = 0.833333...
    assert result.reward_mode == REWARD_MODE_GAP_ACTION_INVERSE
    assert result.components["gap_action"] == pytest.approx(0.2)
    assert result.reward_value == pytest.approx(0.8333333333)


def test_load_rl_env_settings_validation() -> None:
    settings = load_rl_env_settings("configs/rl.yaml")
    assert settings.invalid_action_handling in {"terminate_episode", "continue_episode"}
    assert settings.reward.mode in {
        REWARD_MODE_GAP_ACTION_INVERSE,
        REWARD_MODE_GAP_TO_REFERENCE_ABSOLUTE,
        REWARD_MODE_GAP_TO_REFERENCE_DELTA,
        "sparse_terminal",
    }
