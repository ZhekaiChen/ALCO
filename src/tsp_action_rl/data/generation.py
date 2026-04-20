"""Random TSP instance generation with explicit 1-based node ids."""

from __future__ import annotations

import random

from .models import (
    DISTANCE_METRIC_EUCLIDEAN_2D,
    INDEXING_TSPLIB_1_BASED,
    PROBLEM_TYPE_TSP,
    DataValidationError,
    GenerationSpec,
    Position2D,
    RolloutState,
    TSPInstance,
    TSPNode,
)


def generate_random_euclidean_instance(
    *,
    node_count: int,
    seed: int,
    coordinate_range: tuple[float, float] = (0.0, 1.0),
    integer_coordinates: bool = False,
    instance_id: str | None = None,
) -> TSPInstance:
    """Generate a reproducible Euclidean 2D TSP instance."""
    if node_count < 2:
        raise DataValidationError(f"node_count must be >= 2, got {node_count}.")

    low, high = coordinate_range
    if not high > low:
        raise DataValidationError(f"coordinate_range must satisfy min < max, got {coordinate_range}.")

    rng = random.Random(seed)
    if integer_coordinates:
        low_int = int(low)
        high_int = int(high)
        if float(low_int) != float(low) or float(high_int) != float(high):
            raise DataValidationError(
                "integer_coordinates=True requires integer-valued coordinate_range bounds."
            )
        nodes = tuple(
            TSPNode(
                node_id=node_id,
                x=float(rng.randint(low_int, high_int)),
                y=float(rng.randint(low_int, high_int)),
            )
            for node_id in range(1, node_count + 1)
        )
    else:
        nodes = tuple(
            TSPNode(node_id=node_id, x=rng.uniform(low, high), y=rng.uniform(low, high))
            for node_id in range(1, node_count + 1)
        )
    resolved_instance_id = instance_id or f"tsp_n{node_count:04d}_seed{seed}"

    return TSPInstance(
        instance_id=resolved_instance_id,
        problem_type=PROBLEM_TYPE_TSP,
        node_count=node_count,
        nodes=nodes,
        distance_metric=DISTANCE_METRIC_EUCLIDEAN_2D,
        indexing=INDEXING_TSPLIB_1_BASED,
        generation=GenerationSpec(
            generator="random_euclidean",
            seed=seed,
            coordinate_range=coordinate_range,
        ),
        reference_solution=None,
        metadata={},
    )


def build_initial_rollout_state(instance: TSPInstance, *, start_node: int = 1) -> RolloutState:
    """Create the initial rollout state with a fixed single-node prefix."""
    if start_node < 1 or start_node > instance.node_count:
        raise DataValidationError(f"start_node must be in 1..{instance.node_count}, got {start_node}.")

    node_by_id = {node.node_id: node for node in instance.nodes}
    start = node_by_id[start_node]
    unvisited = [node_id for node_id in range(1, instance.node_count + 1) if node_id != start_node]

    return RolloutState(
        instance_id=instance.instance_id,
        step_index=1,
        node_count=instance.node_count,
        partial_route=(start_node,),
        visited_nodes=(start_node,),
        unvisited_nodes=tuple(unvisited),
        current_node=start_node,
        current_position=Position2D(x=start.x, y=start.y),
        is_terminal=False,
        indexing=INDEXING_TSPLIB_1_BASED,
        notes={"prefix_semantics": "ordered_fixed_prefix"},
    )
