"""Real-solver integration tests for local LKH build + Python `lkh` interface."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

from tsp_action_rl.config import load_lkh_settings
from tsp_action_rl.data import build_initial_rollout_state, generate_random_euclidean_instance
from tsp_action_rl.solvers import LKHIntegration

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _real_settings(tmp_path: Path):
    base = load_lkh_settings(REPO_ROOT / "configs/lkh.yaml")
    solver_path = _resolve_repo_path(base.solver_executable)
    archive_path = _resolve_repo_path(base.source_archive)
    return replace(
        base,
        solver_executable=str(solver_path),
        source_archive=str(archive_path),
        runs=1,
        trace_level=0,
        debug_enabled=True,
        debug_output_root=str(tmp_path / "lkh_debug"),
    )


def test_real_lkh_environment_ready() -> None:
    settings = load_lkh_settings(REPO_ROOT / "configs/lkh.yaml")
    solver_path = _resolve_repo_path(settings.solver_executable)
    archive_path = _resolve_repo_path(settings.source_archive)

    assert importlib.util.find_spec("lkh") is not None, "Python package 'lkh' is not installed."
    assert archive_path.exists(), f"Pinned archive missing: {archive_path}"
    assert solver_path.exists(), f"LKH executable missing: {solver_path}"
    assert solver_path.is_file(), f"LKH executable path is not a file: {solver_path}"
    assert solver_path.stat().st_mode & 0o111, f"LKH executable is not marked executable: {solver_path}"


def test_real_lkh_end_to_end_reference_and_constrained(tmp_path: Path) -> None:
    settings = _real_settings(tmp_path)
    solver = LKHIntegration(settings)

    instance = generate_random_euclidean_instance(
        node_count=8,
        seed=20260411,
        coordinate_range=(0.0, 10.0),
        instance_id="phase25_test_instance",
    )
    reference = solver.solve_reference(instance)

    state = build_initial_rollout_state(instance, start_node=1)
    fixed_prefix = list(state.partial_route) + [2, 3]
    constrained = solver.solve_with_fixed_prefix(instance, partial_route=fixed_prefix)

    assert len(reference.tour) == instance.node_count
    assert len(constrained.tour) == instance.node_count
    assert list(constrained.tour[: len(fixed_prefix)]) == fixed_prefix
    assert constrained.fixed_edges == ((1, 2), (2, 3))

    ref_debug = {key: Path(value) for key, value in reference.debug_paths.items()}
    con_debug = {key: Path(value) for key, value in constrained.debug_paths.items()}
    for debug_paths in (ref_debug, con_debug):
        assert debug_paths["problem_file"].exists()
        assert debug_paths["param_preview_file"].exists()
        assert debug_paths["tour_file"].exists()
        assert debug_paths["result_preview_file"].exists()

    constrained_problem = con_debug["problem_file"].read_text(encoding="utf-8")
    assert "FIXED_EDGES_SECTION" in constrained_problem
    assert "1 2" in constrained_problem
    assert "2 3" in constrained_problem

