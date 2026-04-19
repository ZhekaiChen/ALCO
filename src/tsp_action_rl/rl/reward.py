"""Step-level reward interface and implementations for TSP RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol


REWARD_MODE_GAP_TO_REFERENCE_ABSOLUTE = "gap_to_reference_absolute"
REWARD_MODE_GAP_TO_REFERENCE_DELTA = "gap_to_reference_delta"
REWARD_MODE_SPARSE_TERMINAL = "sparse_terminal"
REWARD_MODE_GAP_ACTION_INVERSE = "gap_action_inverse"

SUPPORTED_REWARD_MODES = (
    REWARD_MODE_GAP_TO_REFERENCE_ABSOLUTE,
    REWARD_MODE_GAP_TO_REFERENCE_DELTA,
    REWARD_MODE_SPARSE_TERMINAL,
    REWARD_MODE_GAP_ACTION_INVERSE,
)

_PARSE_FAILURE_STATUSES = {"missing_tag", "multiple_tags", "malformed_tag"}


class RewardConfigError(ValueError):
    """Raised when reward config or reward inputs are invalid."""


@dataclass(frozen=True)
class RewardSettings:
    """Configurable reward settings for step-level RL transitions."""

    mode: Literal[
        "gap_to_reference_absolute",
        "gap_to_reference_delta",
        "sparse_terminal",
        "gap_action_inverse",
    ] = REWARD_MODE_GAP_TO_REFERENCE_ABSOLUTE

    # Gap-based shaping controls.
    dense_gap_scale: float = -1.0
    gap_delta_scale: float = 1.0
    gap_action_scale: float = 1.0
    min_gap_action_denominator: float = 1e-9

    # Invalid/format penalties.
    invalid_action_penalty: float = -1.0
    parse_failure_penalty_enabled: bool = False
    parse_failure_penalty: float = -0.25

    # Optional shaping controls.
    step_penalty: float = 0.0
    success_bonus: float = 0.0
    sparse_success_reward: float = 1.0
    sparse_nonterminal_reward: float = 0.0

    # Optional reward clipping.
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_REWARD_MODES:
            raise RewardConfigError(
                f"Unsupported reward mode '{self.mode}'. Supported modes: {', '.join(SUPPORTED_REWARD_MODES)}"
            )
        if self.min_gap_action_denominator <= 0:
            raise RewardConfigError(
                f"min_gap_action_denominator must be > 0, got {self.min_gap_action_denominator}."
            )
        if self.clip_min is not None and self.clip_max is not None and self.clip_min > self.clip_max:
            raise RewardConfigError(f"clip_min ({self.clip_min}) must be <= clip_max ({self.clip_max}).")

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None) -> "RewardSettings":
        source = raw or {}
        if not isinstance(source, Mapping):
            raise RewardConfigError(f"reward config must be a mapping/object, got {type(source).__name__}.")

        mode_raw = source.get("mode", REWARD_MODE_GAP_TO_REFERENCE_ABSOLUTE)
        mode = str(mode_raw)

        return RewardSettings(
            mode=mode,  # type: ignore[arg-type]
            dense_gap_scale=float(source.get("dense_gap_scale", -1.0)),
            gap_delta_scale=float(source.get("gap_delta_scale", 1.0)),
            gap_action_scale=float(source.get("gap_action_scale", 1.0)),
            min_gap_action_denominator=float(source.get("min_gap_action_denominator", 1e-9)),
            invalid_action_penalty=float(source.get("invalid_action_penalty", -1.0)),
            parse_failure_penalty_enabled=bool(source.get("parse_failure_penalty_enabled", False)),
            parse_failure_penalty=float(source.get("parse_failure_penalty", -0.25)),
            step_penalty=float(source.get("step_penalty", 0.0)),
            success_bonus=float(source.get("success_bonus", 0.0)),
            sparse_success_reward=float(source.get("sparse_success_reward", 1.0)),
            sparse_nonterminal_reward=float(source.get("sparse_nonterminal_reward", 0.0)),
            clip_min=None if source.get("clip_min") is None else float(source.get("clip_min")),
            clip_max=None if source.get("clip_max") is None else float(source.get("clip_max")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "dense_gap_scale": self.dense_gap_scale,
            "gap_delta_scale": self.gap_delta_scale,
            "gap_action_scale": self.gap_action_scale,
            "min_gap_action_denominator": self.min_gap_action_denominator,
            "invalid_action_penalty": self.invalid_action_penalty,
            "parse_failure_penalty_enabled": self.parse_failure_penalty_enabled,
            "parse_failure_penalty": self.parse_failure_penalty,
            "step_penalty": self.step_penalty,
            "success_bonus": self.success_bonus,
            "sparse_success_reward": self.sparse_success_reward,
            "sparse_nonterminal_reward": self.sparse_nonterminal_reward,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }


@dataclass(frozen=True)
class RewardContext:
    """Inputs required to compute step-level reward."""

    action_is_valid: bool
    action_failure_reason: str | None
    parse_status: str | None
    reference_tour_length: float
    constrained_tour_length: float | None
    previous_constrained_tour_length: float | None
    is_terminal_step: bool
    prefix_partial_tour_length: float | None = None


@dataclass(frozen=True)
class RewardResult:
    """Computed reward value plus explicit components for diagnostics."""

    reward_mode: str
    reward_value: float
    components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward_mode": self.reward_mode,
            "reward_value": self.reward_value,
            "components": dict(self.components),
        }


class RewardFunction(Protocol):
    """Simple interface for reward implementations."""

    def compute(self, context: RewardContext) -> RewardResult:
        """Compute reward for one environment step."""


class _BaseRewardFunction:
    def __init__(self, settings: RewardSettings) -> None:
        self.settings = settings

    def _is_parse_failure(self, parse_status: str | None) -> bool:
        return parse_status in _PARSE_FAILURE_STATUSES

    def _gap_to_reference(self, *, constrained_tour_length: float, reference_tour_length: float) -> float:
        if reference_tour_length <= 0:
            raise RewardConfigError(
                f"reference_tour_length must be > 0 for gap-based rewards, got {reference_tour_length}."
            )
        return (constrained_tour_length - reference_tour_length) / reference_tour_length

    def _common_components(self, context: RewardContext) -> tuple[float, dict[str, Any]]:
        components: dict[str, Any] = {
            "action_is_valid": context.action_is_valid,
            "action_failure_reason": context.action_failure_reason,
            "parse_status": context.parse_status,
            "reference_tour_length": context.reference_tour_length,
        }
        if context.prefix_partial_tour_length is not None:
            components["prefix_partial_tour_length"] = context.prefix_partial_tour_length

        reward_total = 0.0

        if self.settings.step_penalty != 0.0:
            reward_total += self.settings.step_penalty
            components["step_penalty"] = self.settings.step_penalty

        if not context.action_is_valid:
            reward_total += self.settings.invalid_action_penalty
            components["invalid_action_penalty"] = self.settings.invalid_action_penalty

        parse_penalty_applied = (
            self.settings.parse_failure_penalty_enabled and self._is_parse_failure(context.parse_status)
        )
        if parse_penalty_applied:
            reward_total += self.settings.parse_failure_penalty
            components["parse_failure_penalty"] = self.settings.parse_failure_penalty

        return reward_total, components

    def _finalize(self, *, reward_mode: str, reward_value: float, components: dict[str, Any]) -> RewardResult:
        clipped = reward_value
        if self.settings.clip_min is not None:
            clipped = max(clipped, self.settings.clip_min)
        if self.settings.clip_max is not None:
            clipped = min(clipped, self.settings.clip_max)
        components["reward_unclipped"] = reward_value
        if clipped != reward_value:
            components["reward_clipped"] = clipped
        return RewardResult(reward_mode=reward_mode, reward_value=clipped, components=components)


class GapToReferenceAbsoluteReward(_BaseRewardFunction):
    """Dense reward based on current constrained-tour gap to reference."""

    def compute(self, context: RewardContext) -> RewardResult:
        reward_total, components = self._common_components(context)

        if context.action_is_valid:
            if context.constrained_tour_length is None:
                raise RewardConfigError(
                    "constrained_tour_length is required for valid actions in gap_to_reference_absolute mode."
                )
            gap = self._gap_to_reference(
                constrained_tour_length=context.constrained_tour_length,
                reference_tour_length=context.reference_tour_length,
            )
            gap_component = self.settings.dense_gap_scale * gap
            reward_total += gap_component
            components["gap_to_reference"] = gap
            components["dense_gap_component"] = gap_component

            if context.is_terminal_step and self.settings.success_bonus != 0.0:
                reward_total += self.settings.success_bonus
                components["success_bonus"] = self.settings.success_bonus

        return self._finalize(
            reward_mode=REWARD_MODE_GAP_TO_REFERENCE_ABSOLUTE,
            reward_value=reward_total,
            components=components,
        )


class GapToReferenceDeltaReward(_BaseRewardFunction):
    """Dense reward based on improvement in constrained-tour gap across steps."""

    def compute(self, context: RewardContext) -> RewardResult:
        reward_total, components = self._common_components(context)

        if context.action_is_valid:
            if context.constrained_tour_length is None:
                raise RewardConfigError(
                    "constrained_tour_length is required for valid actions in gap_to_reference_delta mode."
                )

            current_gap = self._gap_to_reference(
                constrained_tour_length=context.constrained_tour_length,
                reference_tour_length=context.reference_tour_length,
            )
            previous_gap = (
                current_gap
                if context.previous_constrained_tour_length is None
                else self._gap_to_reference(
                    constrained_tour_length=context.previous_constrained_tour_length,
                    reference_tour_length=context.reference_tour_length,
                )
            )

            gap_delta = previous_gap - current_gap
            gap_delta_component = self.settings.gap_delta_scale * gap_delta
            dense_gap_component = self.settings.dense_gap_scale * current_gap

            reward_total += gap_delta_component
            reward_total += dense_gap_component

            components["gap_to_reference"] = current_gap
            components["previous_gap_to_reference"] = previous_gap
            components["gap_delta"] = gap_delta
            components["gap_delta_component"] = gap_delta_component
            components["dense_gap_component"] = dense_gap_component

            if context.is_terminal_step and self.settings.success_bonus != 0.0:
                reward_total += self.settings.success_bonus
                components["success_bonus"] = self.settings.success_bonus

        return self._finalize(
            reward_mode=REWARD_MODE_GAP_TO_REFERENCE_DELTA,
            reward_value=reward_total,
            components=components,
        )


class SparseTerminalReward(_BaseRewardFunction):
    """Sparse reward: terminal success signal plus optional penalties."""

    def compute(self, context: RewardContext) -> RewardResult:
        reward_total, components = self._common_components(context)

        if context.action_is_valid:
            sparse_component = (
                self.settings.sparse_success_reward
                if context.is_terminal_step
                else self.settings.sparse_nonterminal_reward
            )
            reward_total += sparse_component
            components["sparse_component"] = sparse_component

            if context.is_terminal_step and self.settings.success_bonus != 0.0:
                reward_total += self.settings.success_bonus
                components["success_bonus"] = self.settings.success_bonus

        return self._finalize(
            reward_mode=REWARD_MODE_SPARSE_TERMINAL,
            reward_value=reward_total,
            components=components,
        )


class GapActionInverseReward(_BaseRewardFunction):
    """Action-level reward: 1 / (1 + gap_action)."""

    def compute(self, context: RewardContext) -> RewardResult:
        reward_total, components = self._common_components(context)

        if context.action_is_valid:
            if context.constrained_tour_length is None:
                raise RewardConfigError(
                    "constrained_tour_length is required for valid actions in gap_action_inverse mode."
                )
            if context.prefix_partial_tour_length is None:
                raise RewardConfigError(
                    "prefix_partial_tour_length is required for valid actions in gap_action_inverse mode."
                )

            denominator = context.reference_tour_length - context.prefix_partial_tour_length
            if denominator <= self.settings.min_gap_action_denominator:
                raise RewardConfigError(
                    "Invalid denominator for gap_action computation. "
                    f"Expected reference_tour_length - prefix_partial_tour_length > {self.settings.min_gap_action_denominator}, "
                    f"got reference_tour_length={context.reference_tour_length}, "
                    f"prefix_partial_tour_length={context.prefix_partial_tour_length}."
                )

            gap_action = (context.constrained_tour_length - context.reference_tour_length) / denominator
            inverse_denominator = 1.0 + gap_action
            if inverse_denominator <= 0:
                raise RewardConfigError(
                    "Invalid 1 + gap_action denominator for inverse reward. "
                    f"Computed gap_action={gap_action}, so 1 + gap_action={inverse_denominator}."
                )

            action_component = self.settings.gap_action_scale * (1.0 / inverse_denominator)
            reward_total += action_component

            components["gap_action"] = gap_action
            components["inverse_gap_action_component"] = action_component

            if context.is_terminal_step and self.settings.success_bonus != 0.0:
                reward_total += self.settings.success_bonus
                components["success_bonus"] = self.settings.success_bonus

        return self._finalize(
            reward_mode=REWARD_MODE_GAP_ACTION_INVERSE,
            reward_value=reward_total,
            components=components,
        )


def build_reward_function(settings: RewardSettings) -> RewardFunction:
    """Build the configured reward implementation."""
    if settings.mode == REWARD_MODE_GAP_TO_REFERENCE_ABSOLUTE:
        return GapToReferenceAbsoluteReward(settings)
    if settings.mode == REWARD_MODE_GAP_TO_REFERENCE_DELTA:
        return GapToReferenceDeltaReward(settings)
    if settings.mode == REWARD_MODE_SPARSE_TERMINAL:
        return SparseTerminalReward(settings)
    if settings.mode == REWARD_MODE_GAP_ACTION_INVERSE:
        return GapActionInverseReward(settings)
    raise RewardConfigError(
        f"Unsupported reward mode '{settings.mode}'. Supported modes: {', '.join(SUPPORTED_REWARD_MODES)}"
    )
