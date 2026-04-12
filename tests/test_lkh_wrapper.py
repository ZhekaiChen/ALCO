"""Mocked unit tests for Phase-2 LKH integration behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from tsp_action_rl.config import load_lkh_settings
from tsp_action_rl.data import load_tsp_instance
from tsp_action_rl.solvers import (
    CANONICAL_PINNED_LKH_ARCHIVE,
    LKHConstraintError,
    LKHIntegration,
    LKHSettings,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class FakeLKHModule:
    """Simple fake module exposing lkh.solve(...) for wrapper tests."""

    def __init__(self, routes: list[list[int]]) -> None:
        self.routes = routes
        self.calls: list[dict] = []

    def solve(self, solver: str = "LKH", problem_file: str | None = None, **params: object) -> list[list[int]]:
        self.calls.append(
            {
                "solver": solver,
                "problem_file": problem_file,
                "params": dict(params),
            }
        )
        return self.routes


def _test_settings(tmp_path: Path) -> LKHSettings:
    return LKHSettings(
        solver_executable="python",
        source_archive=CANONICAL_PINNED_LKH_ARCHIVE,
        runs=2,
        trace_level=0,
        debug_enabled=True,
        debug_output_root=str(tmp_path / "debug"),
    )


def test_reference_solve_uses_lkh_and_writes_debug_artifacts(tmp_path: Path) -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    fake_lkh = FakeLKHModule(routes=[[1, 2, 3, 5, 4]])
    solver = LKHIntegration(_test_settings(tmp_path), lkh_module=fake_lkh)

    result = solver.solve_reference(instance)

    assert result.mode == "full_reference"
    assert list(result.tour) == [1, 2, 3, 5, 4]
    assert result.tour_length > 0.0
    assert result.fixed_edges == ()
    assert fake_lkh.calls[0]["solver"] == "python"
    assert Path(result.debug_paths["problem_file"]).exists()
    assert Path(result.debug_paths["param_preview_file"]).exists()
    assert Path(result.debug_paths["result_preview_file"]).exists()


def test_constrained_completion_uses_fixed_edges_and_preserves_prefix(tmp_path: Path) -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    fake_lkh = FakeLKHModule(routes=[[3, 2, 1, 4, 5]])
    solver = LKHIntegration(_test_settings(tmp_path), lkh_module=fake_lkh)

    result = solver.solve_with_fixed_prefix(instance, partial_route=[1, 2, 3])

    assert list(result.tour[:3]) == [1, 2, 3]
    assert result.fixed_edges == ((1, 2), (2, 3))
    problem_text = Path(result.debug_paths["problem_file"]).read_text(encoding="utf-8")
    assert "FIXED_EDGES_SECTION" in problem_text
    assert "1 2" in problem_text
    assert "2 3" in problem_text


def test_constrained_completion_raises_on_prefix_violation(tmp_path: Path) -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    fake_lkh = FakeLKHModule(routes=[[1, 4, 2, 3, 5]])
    solver = LKHIntegration(_test_settings(tmp_path), lkh_module=fake_lkh)

    with pytest.raises(LKHConstraintError):
        solver.solve_with_fixed_prefix(instance, partial_route=[1, 2, 3])


def test_load_lkh_settings_from_config() -> None:
    settings = load_lkh_settings("configs/lkh.yaml")
    assert settings.source_archive == CANONICAL_PINNED_LKH_ARCHIVE
    assert settings.solver_executable
