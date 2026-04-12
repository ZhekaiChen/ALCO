"""Phase-1 smoke tests for typed data models and serializers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tsp_action_rl.data import (
    DataValidationError,
    INDEXING_TSPLIB_1_BASED,
    build_initial_rollout_state,
    generate_random_euclidean_instance,
    load_rollout_state,
    load_tsp_instance,
    save_rollout_state,
    save_tsp_instance,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_load_tsp_fixture() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    assert instance.instance_id == "tsp_000001"
    assert instance.node_count == 5
    assert [node.node_id for node in instance.nodes] == [1, 2, 3, 4, 5]
    assert instance.indexing == INDEXING_TSPLIB_1_BASED
    assert instance.reference_solution is not None


def test_load_rollout_state_fixture() -> None:
    state = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json")
    assert state.step_index == 3
    assert list(state.partial_route) == [1, 2, 3]
    assert state.current_node == 3
    assert list(state.unvisited_nodes) == [4, 5]
    assert state.indexing == INDEXING_TSPLIB_1_BASED


def test_random_generation_is_deterministic() -> None:
    a = generate_random_euclidean_instance(node_count=6, seed=7, coordinate_range=(0.0, 2.0))
    b = generate_random_euclidean_instance(node_count=6, seed=7, coordinate_range=(0.0, 2.0))
    assert a.to_dict() == b.to_dict()


def test_generated_ids_are_strictly_one_based() -> None:
    instance = generate_random_euclidean_instance(node_count=8, seed=19)
    assert [node.node_id for node in instance.nodes] == list(range(1, 9))


def test_random_generation_supports_integer_coordinates() -> None:
    instance = generate_random_euclidean_instance(
        node_count=6,
        seed=123,
        coordinate_range=(0.0, 10000.0),
        integer_coordinates=True,
    )
    for node in instance.nodes:
        assert float(node.x).is_integer()
        assert float(node.y).is_integer()
        assert 0.0 <= node.x <= 10000.0
        assert 0.0 <= node.y <= 10000.0


def test_save_reload_roundtrip_instance_and_state(tmp_path: Path) -> None:
    instance = generate_random_euclidean_instance(node_count=9, seed=21)
    state = build_initial_rollout_state(instance, start_node=1)

    instance_path = tmp_path / "instance.json"
    state_path = tmp_path / "state.json"
    save_tsp_instance(instance, instance_path)
    save_rollout_state(state, state_path)

    loaded_instance = load_tsp_instance(instance_path)
    loaded_state = load_rollout_state(state_path)

    assert loaded_instance.to_dict() == instance.to_dict()
    assert loaded_state.to_dict() == state.to_dict()


def test_rollout_state_rejects_inconsistent_current_node() -> None:
    state_payload = load_rollout_state(FIXTURES_DIR / "rollout_state_prefix3.json").to_dict()
    state_payload["current_node"] = 2
    with pytest.raises(DataValidationError):
        load_rollout_state_from_payload(state_payload)


def load_rollout_state_from_payload(payload: dict) -> object:
    """Test helper to reuse production parsing path without direct model import."""
    from tsp_action_rl.data.models import RolloutState

    return RolloutState.from_dict(payload)
