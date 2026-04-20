"""Data layer: models, generators, and JSON I/O."""

from .generation import build_initial_rollout_state, generate_random_euclidean_instance
from .io import load_rollout_state, load_tsp_instance, save_rollout_state, save_tsp_instance
from .models import (
    DISTANCE_METRIC_EUCLIDEAN_2D,
    INDEXING_TSPLIB_1_BASED,
    PROBLEM_TYPE_TSP,
    DataValidationError,
    GenerationSpec,
    Position2D,
    ReferenceSolution,
    RolloutState,
    TSPInstance,
    TSPNode,
)

__all__ = [
    "DISTANCE_METRIC_EUCLIDEAN_2D",
    "INDEXING_TSPLIB_1_BASED",
    "PROBLEM_TYPE_TSP",
    "DataValidationError",
    "GenerationSpec",
    "Position2D",
    "ReferenceSolution",
    "RolloutState",
    "TSPInstance",
    "TSPNode",
    "build_initial_rollout_state",
    "generate_random_euclidean_instance",
    "load_rollout_state",
    "load_tsp_instance",
    "save_rollout_state",
    "save_tsp_instance",
]

