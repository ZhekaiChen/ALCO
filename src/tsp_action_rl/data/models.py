"""Typed data models and strict boundary validation for TSP research artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

PROBLEM_TYPE_TSP = "tsp"
DISTANCE_METRIC_EUCLIDEAN_2D = "euclidean_2d"
INDEXING_TSPLIB_1_BASED = "tsplib_1_based"


class DataValidationError(ValueError):
    """Raised when external data violates project schemas/invariants."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: Any, *, field_name: str, minimum: int | None = None) -> int:
    _check(_is_int(value), f"{field_name} must be an integer, got {type(value).__name__}.")
    integer_value = int(value)
    if minimum is not None:
        _check(integer_value >= minimum, f"{field_name} must be >= {minimum}, got {integer_value}.")
    return integer_value


def _require_number(value: Any, *, field_name: str) -> float:
    _check(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field_name} must be numeric.")
    return float(value)


def _require_string(value: Any, *, field_name: str) -> str:
    _check(isinstance(value, str), f"{field_name} must be a string.")
    _check(len(value) > 0, f"{field_name} must be non-empty.")
    return value


def _disallow_unknown_keys(raw: Mapping[str, Any], *, allowed: set[str], field_name: str) -> None:
    unknown = set(raw.keys()) - allowed
    _check(not unknown, f"{field_name} contains unknown fields: {sorted(unknown)}")


def _require_keys(raw: Mapping[str, Any], *, required: set[str], field_name: str) -> None:
    missing = required - set(raw.keys())
    _check(not missing, f"{field_name} is missing required fields: {sorted(missing)}")


@dataclass(frozen=True)
class TSPNode:
    """Single 2D node with explicit 1-based TSPLIB-style id."""

    node_id: int
    x: float
    y: float

    def __post_init__(self) -> None:
        _check(self.node_id >= 1, f"node_id must be >= 1, got {self.node_id}.")

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "TSPNode":
        _disallow_unknown_keys(raw, allowed={"node_id", "x", "y"}, field_name="TSPNode")
        _require_keys(raw, required={"node_id", "x", "y"}, field_name="TSPNode")
        return TSPNode(
            node_id=_require_int(raw.get("node_id"), field_name="node_id", minimum=1),
            x=_require_number(raw.get("x"), field_name="x"),
            y=_require_number(raw.get("y"), field_name="y"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "x": self.x, "y": self.y}


@dataclass(frozen=True)
class GenerationSpec:
    """Generation metadata for synthetic instances."""

    generator: str
    seed: int
    coordinate_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.coordinate_range is not None:
            low, high = self.coordinate_range
            _check(high > low, f"coordinate_range must satisfy min < max, got {self.coordinate_range}.")

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "GenerationSpec":
        _disallow_unknown_keys(raw, allowed={"generator", "seed", "coordinate_range"}, field_name="generation")
        _require_keys(raw, required={"generator", "seed"}, field_name="generation")
        coordinate_range_raw = raw.get("coordinate_range")
        coordinate_range: tuple[float, float] | None
        if coordinate_range_raw is None:
            coordinate_range = None
        else:
            _check(isinstance(coordinate_range_raw, list), "coordinate_range must be a length-2 array.")
            _check(len(coordinate_range_raw) == 2, "coordinate_range must contain exactly two numbers.")
            coordinate_range = (
                _require_number(coordinate_range_raw[0], field_name="coordinate_range[0]"),
                _require_number(coordinate_range_raw[1], field_name="coordinate_range[1]"),
            )

        return GenerationSpec(
            generator=_require_string(raw.get("generator"), field_name="generation.generator"),
            seed=_require_int(raw.get("seed"), field_name="generation.seed"),
            coordinate_range=coordinate_range,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"generator": self.generator, "seed": self.seed}
        if self.coordinate_range is not None:
            payload["coordinate_range"] = [self.coordinate_range[0], self.coordinate_range[1]]
        return payload


@dataclass(frozen=True)
class ReferenceSolution:
    """Reference full-tour solution (typically from full LKH3 solve)."""

    solver: str
    tour: tuple[int, ...]
    tour_length: float

    def __post_init__(self) -> None:
        _check(len(self.tour) >= 2, "reference_solution.tour must contain at least 2 nodes.")
        for idx, node_id in enumerate(self.tour):
            _check(node_id >= 1, f"reference_solution.tour[{idx}] must be >= 1.")

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "ReferenceSolution":
        _disallow_unknown_keys(
            raw,
            allowed={"solver", "tour", "tour_length"},
            field_name="reference_solution",
        )
        _require_keys(raw, required={"solver", "tour", "tour_length"}, field_name="reference_solution")
        tour_raw = raw.get("tour")
        _check(isinstance(tour_raw, list), "reference_solution.tour must be an array.")
        tour = tuple(_require_int(node, field_name="reference_solution.tour[]", minimum=1) for node in tour_raw)
        return ReferenceSolution(
            solver=_require_string(raw.get("solver"), field_name="reference_solution.solver"),
            tour=tour,
            tour_length=_require_number(raw.get("tour_length"), field_name="reference_solution.tour_length"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "tour": list(self.tour),
            "tour_length": self.tour_length,
        }


@dataclass(frozen=True)
class TSPInstance:
    """Full TSP instance record aligned with `specs/schemas/tsp_instance.schema.json`."""

    instance_id: str
    problem_type: str
    node_count: int
    nodes: tuple[TSPNode, ...]
    distance_metric: str
    indexing: str
    generation: GenerationSpec
    reference_solution: ReferenceSolution | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check(self.problem_type == PROBLEM_TYPE_TSP, f"problem_type must be '{PROBLEM_TYPE_TSP}'.")
        _check(self.distance_metric == DISTANCE_METRIC_EUCLIDEAN_2D, f"distance_metric must be '{DISTANCE_METRIC_EUCLIDEAN_2D}'.")
        _check(self.indexing == INDEXING_TSPLIB_1_BASED, f"indexing must be '{INDEXING_TSPLIB_1_BASED}'.")
        _check(self.node_count >= 2, f"node_count must be >= 2, got {self.node_count}.")
        _check(len(self.nodes) == self.node_count, "node_count must match len(nodes).")

        node_ids = [node.node_id for node in self.nodes]
        expected_node_ids = list(range(1, self.node_count + 1))
        _check(node_ids == expected_node_ids, f"nodes must use contiguous 1-based ids {expected_node_ids}, got {node_ids}.")

        if self.reference_solution is not None:
            ref_tour = self.reference_solution.tour
            _check(len(ref_tour) == self.node_count, "reference_solution.tour must contain exactly node_count nodes.")
            _check(set(ref_tour) == set(expected_node_ids), "reference_solution.tour must be a permutation of 1..node_count.")

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "TSPInstance":
        allowed = {
            "instance_id",
            "problem_type",
            "node_count",
            "nodes",
            "distance_metric",
            "indexing",
            "generation",
            "reference_solution",
            "metadata",
        }
        _disallow_unknown_keys(raw, allowed=allowed, field_name="TSPInstance")
        _require_keys(
            raw,
            required={
                "instance_id",
                "problem_type",
                "node_count",
                "nodes",
                "distance_metric",
                "indexing",
                "generation",
            },
            field_name="TSPInstance",
        )

        nodes_raw = raw.get("nodes")
        _check(isinstance(nodes_raw, list), "nodes must be an array.")
        nodes = tuple(TSPNode.from_dict(item) for item in nodes_raw)

        generation_raw = raw.get("generation")
        _check(isinstance(generation_raw, Mapping), "generation must be an object.")

        reference_raw = raw.get("reference_solution")
        if reference_raw is None:
            reference: ReferenceSolution | None = None
        else:
            _check(isinstance(reference_raw, Mapping), "reference_solution must be an object if present.")
            reference = ReferenceSolution.from_dict(reference_raw)

        metadata_raw = raw.get("metadata", {})
        _check(isinstance(metadata_raw, Mapping), "metadata must be an object if present.")

        return TSPInstance(
            instance_id=_require_string(raw.get("instance_id"), field_name="instance_id"),
            problem_type=_require_string(raw.get("problem_type"), field_name="problem_type"),
            node_count=_require_int(raw.get("node_count"), field_name="node_count", minimum=2),
            nodes=nodes,
            distance_metric=_require_string(raw.get("distance_metric"), field_name="distance_metric"),
            indexing=_require_string(raw.get("indexing"), field_name="indexing"),
            generation=GenerationSpec.from_dict(generation_raw),
            reference_solution=reference,
            metadata=dict(metadata_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instance_id": self.instance_id,
            "problem_type": self.problem_type,
            "node_count": self.node_count,
            "distance_metric": self.distance_metric,
            "indexing": self.indexing,
            "nodes": [node.to_dict() for node in self.nodes],
            "generation": self.generation.to_dict(),
        }
        if self.reference_solution is not None:
            payload["reference_solution"] = self.reference_solution.to_dict()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class Position2D:
    """Optional position payload for rollout state."""

    x: float
    y: float

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "Position2D":
        _disallow_unknown_keys(raw, allowed={"x", "y"}, field_name="current_position")
        _require_keys(raw, required={"x", "y"}, field_name="current_position")
        return Position2D(
            x=_require_number(raw.get("x"), field_name="current_position.x"),
            y=_require_number(raw.get("y"), field_name="current_position.y"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class RolloutState:
    """Per-step rollout state aligned with `rollout_state.schema.json`."""

    instance_id: str
    step_index: int
    node_count: int
    partial_route: tuple[int, ...]
    visited_nodes: tuple[int, ...]
    unvisited_nodes: tuple[int, ...]
    current_node: int | None
    current_position: Position2D | None
    is_terminal: bool
    indexing: str
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check(isinstance(self.is_terminal, bool), "is_terminal must be a boolean.")
        _check(self.node_count >= 2, "node_count must be >= 2.")
        _check(self.step_index >= 0, "step_index must be >= 0.")
        _check(self.indexing == INDEXING_TSPLIB_1_BASED, f"indexing must be '{INDEXING_TSPLIB_1_BASED}'.")

        all_nodes = set(range(1, self.node_count + 1))
        partial = list(self.partial_route)
        visited = list(self.visited_nodes)
        unvisited = list(self.unvisited_nodes)

        for node_id in partial:
            _check(node_id in all_nodes, f"partial_route contains out-of-range node_id: {node_id}")
        for node_id in visited:
            _check(node_id in all_nodes, f"visited_nodes contains out-of-range node_id: {node_id}")
        for node_id in unvisited:
            _check(node_id in all_nodes, f"unvisited_nodes contains out-of-range node_id: {node_id}")

        _check(len(set(partial)) == len(partial), "partial_route must not contain repeated node ids.")
        _check(len(set(visited)) == len(visited), "visited_nodes must not contain duplicates.")
        _check(len(set(unvisited)) == len(unvisited), "unvisited_nodes must not contain duplicates.")
        _check(set(visited) == set(partial), "visited_nodes must match the node set of partial_route.")
        _check(set(visited).isdisjoint(set(unvisited)), "visited_nodes and unvisited_nodes must be disjoint.")
        _check(set(visited) | set(unvisited) == all_nodes, "visited_nodes + unvisited_nodes must cover 1..node_count exactly.")

        if partial:
            _check(self.current_node == partial[-1], "current_node must equal the last node of partial_route.")
        else:
            _check(self.current_node is None, "current_node must be null when partial_route is empty.")

        expected_terminal = len(unvisited) == 0
        _check(self.is_terminal == expected_terminal, "is_terminal must match whether unvisited_nodes is empty.")

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "RolloutState":
        allowed = {
            "instance_id",
            "step_index",
            "node_count",
            "partial_route",
            "visited_nodes",
            "unvisited_nodes",
            "current_node",
            "current_position",
            "is_terminal",
            "indexing",
            "notes",
        }
        _disallow_unknown_keys(raw, allowed=allowed, field_name="RolloutState")
        _require_keys(
            raw,
            required={
                "instance_id",
                "step_index",
                "node_count",
                "partial_route",
                "visited_nodes",
                "unvisited_nodes",
                "current_node",
                "is_terminal",
                "indexing",
            },
            field_name="RolloutState",
        )

        partial_raw = raw.get("partial_route")
        visited_raw = raw.get("visited_nodes")
        unvisited_raw = raw.get("unvisited_nodes")

        _check(isinstance(partial_raw, list), "partial_route must be an array.")
        _check(isinstance(visited_raw, list), "visited_nodes must be an array.")
        _check(isinstance(unvisited_raw, list), "unvisited_nodes must be an array.")

        current_node_raw = raw.get("current_node")
        if current_node_raw is None:
            current_node: int | None = None
        else:
            current_node = _require_int(current_node_raw, field_name="current_node", minimum=1)

        current_position_raw = raw.get("current_position")
        if current_position_raw is None:
            current_position: Position2D | None = None
        else:
            _check(isinstance(current_position_raw, Mapping), "current_position must be null or an object.")
            current_position = Position2D.from_dict(current_position_raw)

        notes_raw = raw.get("notes", {})
        _check(isinstance(notes_raw, Mapping), "notes must be an object if present.")

        return RolloutState(
            instance_id=_require_string(raw.get("instance_id"), field_name="instance_id"),
            step_index=_require_int(raw.get("step_index"), field_name="step_index", minimum=0),
            node_count=_require_int(raw.get("node_count"), field_name="node_count", minimum=2),
            partial_route=tuple(_require_int(n, field_name="partial_route[]", minimum=1) for n in partial_raw),
            visited_nodes=tuple(_require_int(n, field_name="visited_nodes[]", minimum=1) for n in visited_raw),
            unvisited_nodes=tuple(_require_int(n, field_name="unvisited_nodes[]", minimum=1) for n in unvisited_raw),
            current_node=current_node,
            current_position=current_position,
            is_terminal=raw.get("is_terminal"),
            indexing=_require_string(raw.get("indexing"), field_name="indexing"),
            notes=dict(notes_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instance_id": self.instance_id,
            "step_index": self.step_index,
            "node_count": self.node_count,
            "partial_route": list(self.partial_route),
            "visited_nodes": list(self.visited_nodes),
            "unvisited_nodes": list(self.unvisited_nodes),
            "current_node": self.current_node,
            "current_position": None if self.current_position is None else self.current_position.to_dict(),
            "is_terminal": self.is_terminal,
            "indexing": self.indexing,
        }
        if self.notes:
            payload["notes"] = dict(self.notes)
        return payload
