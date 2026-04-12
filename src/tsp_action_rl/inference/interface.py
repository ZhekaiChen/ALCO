"""Common model inference interfaces for zero-shot rollout."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tsp_action_rl.data import RolloutState, TSPInstance


@dataclass(frozen=True)
class ModelOutput:
    """Raw model output plus optional backend metadata."""

    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelBackend(ABC):
    """Simple explicit interface for API-based and local model backends."""

    backend_type: str
    model_name: str

    @abstractmethod
    def generate(
        self,
        prompt_text: str,
        *,
        instance: TSPInstance,
        state: RolloutState,
    ) -> ModelOutput:
        """Generate one model response for the current rollout step."""

