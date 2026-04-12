"""Phase-2 LKH integration using the `lkh` Python package as primary interface."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tsp_action_rl.data.models import (
    DISTANCE_METRIC_EUCLIDEAN_2D,
    INDEXING_TSPLIB_1_BASED,
    DataValidationError,
    TSPInstance,
)

CANONICAL_PINNED_LKH_ARCHIVE = "third_party/LKH3/LKH-3.0.14.tar"


class LKHConfigError(ValueError):
    """Raised when solver configuration is invalid."""


class LKHDependencyError(RuntimeError):
    """Raised when required runtime dependencies are unavailable."""


class LKHExecutionError(RuntimeError):
    """Raised when `lkh.solve(...)` fails."""


class LKHConstraintError(RuntimeError):
    """Raised when constrained completion does not preserve fixed-prefix semantics."""


@dataclass
class LKHSettings:
    """User-configurable settings for LKH integration."""

    solver_executable: str
    source_archive: str = CANONICAL_PINNED_LKH_ARCHIVE
    runs: int = 1
    max_trials: int | None = None
    seed: int | None = None
    trace_level: int = 1
    time_limit: int | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    debug_enabled: bool = False
    debug_output_root: str = "outputs/debug/lkh"
    require_source_archive: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.solver_executable, str) or not self.solver_executable.strip():
            raise LKHConfigError("solver_executable must be a non-empty string.")
        if self.runs < 1:
            raise LKHConfigError(f"runs must be >= 1, got {self.runs}.")
        if self.max_trials is not None and self.max_trials < 1:
            raise LKHConfigError(f"max_trials must be >= 1 when set, got {self.max_trials}.")
        if self.trace_level < 0:
            raise LKHConfigError(f"trace_level must be >= 0, got {self.trace_level}.")
        if self.time_limit is not None and self.time_limit < 1:
            raise LKHConfigError(f"time_limit must be >= 1 when set, got {self.time_limit}.")

    @staticmethod
    def from_mapping(raw: Mapping[str, Any]) -> "LKHSettings":
        extra_params_raw = raw.get("extra_params", {})
        if not isinstance(extra_params_raw, Mapping):
            raise LKHConfigError("extra_params must be an object/map.")

        debug_raw = raw.get("debug", {})
        if not isinstance(debug_raw, Mapping):
            raise LKHConfigError("debug must be an object/map.")

        return LKHSettings(
            solver_executable=str(raw.get("solver_executable", "LKH")),
            source_archive=str(raw.get("source_archive", CANONICAL_PINNED_LKH_ARCHIVE)),
            runs=int(raw.get("runs", 1)),
            max_trials=None if raw.get("max_trials") is None else int(raw.get("max_trials")),
            seed=None if raw.get("seed") is None else int(raw.get("seed")),
            trace_level=int(raw.get("trace_level", 1)),
            time_limit=None if raw.get("time_limit") is None else int(raw.get("time_limit")),
            extra_params=dict(extra_params_raw),
            debug_enabled=bool(debug_raw.get("enabled", False)),
            debug_output_root=str(debug_raw.get("output_root", "outputs/debug/lkh")),
            require_source_archive=bool(raw.get("require_source_archive", True)),
        )

    def to_lkh_params(self, *, tour_file: str) -> dict[str, Any]:
        """Build keyword params forwarded directly to `lkh.solve(...)`."""
        params: dict[str, Any] = {
            "tour_file": tour_file,
            "runs": self.runs,
            "trace_level": self.trace_level,
        }
        if self.max_trials is not None:
            params["max_trials"] = self.max_trials
        if self.seed is not None:
            params["seed"] = self.seed
        if self.time_limit is not None:
            params["time_limit"] = self.time_limit
        params.update(self.extra_params)
        return params


@dataclass(frozen=True)
class LKHSolveResult:
    """Structured solve result for full and constrained modes."""

    mode: str
    tour: tuple[int, ...]
    tour_length: float
    fixed_edges: tuple[tuple[int, int], ...]
    solver_params: dict[str, Any]
    debug_paths: dict[str, str]


class LKHIntegration:
    """Primary LKH integration entrypoint using `lkh.solve(...)`."""

    def __init__(self, settings: LKHSettings, *, lkh_module: Any | None = None) -> None:
        self.settings = settings
        self._lkh_module = lkh_module

    def solve_reference(self, instance: TSPInstance) -> LKHSolveResult:
        """Solve a full reference tour from scratch with no fixed prefix."""
        return self._solve(mode="full_reference", instance=instance, partial_route=None)

    def solve_with_fixed_prefix(self, instance: TSPInstance, partial_route: Sequence[int]) -> LKHSolveResult:
        """Solve constrained completion while preserving ordered fixed prefix."""
        validated_prefix = self._validate_partial_route(instance=instance, partial_route=partial_route)
        return self._solve(mode="constrained_completion", instance=instance, partial_route=validated_prefix)

    def _solve(
        self,
        *,
        mode: str,
        instance: TSPInstance,
        partial_route: tuple[int, ...] | None,
    ) -> LKHSolveResult:
        self._validate_instance_for_solver(instance)
        self._validate_source_archive()
        self._validate_solver_executable()

        run_dir, should_cleanup = self._make_run_dir(mode=mode, instance_id=instance.instance_id)
        try:
            fixed_edges = () if partial_route is None else self._prefix_to_fixed_edges(partial_route)
            problem_path = run_dir / "problem.tsp"
            tour_path = run_dir / "solution.tour"
            params_path = run_dir / "params.par"

            self._write_problem_file(instance=instance, path=problem_path, fixed_edges=fixed_edges)
            solver_params = self.settings.to_lkh_params(tour_file=str(tour_path))
            self._write_param_preview(path=params_path, problem_path=problem_path, solver_params=solver_params)

            routes = self._run_lkh(problem_path=problem_path, solver_params=solver_params)
            if not routes:
                raise LKHExecutionError("lkh.solve(...) returned no routes.")
            if len(routes) != 1:
                raise LKHExecutionError(f"Expected a single TSP route, got {len(routes)} routes.")

            raw_route = tuple(int(node_id) for node_id in routes[0])
            self._validate_route_permutation(route=raw_route, node_count=instance.node_count)

            if partial_route is None:
                final_route = self._canonicalize_to_start_node(raw_route, start_node=1)
            else:
                final_route = self._canonicalize_to_prefix(raw_route, partial_route)

            tour_length = self._compute_tour_length(instance=instance, tour=final_route)
            debug_paths = self._build_debug_paths(
                enabled=self.settings.debug_enabled,
                run_dir=run_dir,
                problem_path=problem_path,
                params_path=params_path,
                tour_path=tour_path,
            )
            if self.settings.debug_enabled:
                self._write_result_preview(
                    path=run_dir / "result.json",
                    mode=mode,
                    route=final_route,
                    tour_length=tour_length,
                    fixed_edges=fixed_edges,
                    solver_params=solver_params,
                )

            return LKHSolveResult(
                mode=mode,
                tour=final_route,
                tour_length=tour_length,
                fixed_edges=fixed_edges,
                solver_params=solver_params,
                debug_paths=debug_paths,
            )
        finally:
            if should_cleanup:
                shutil.rmtree(run_dir, ignore_errors=True)

    def _run_lkh(self, *, problem_path: Path, solver_params: Mapping[str, Any]) -> list[list[int]]:
        lkh_module = self._get_lkh_module()
        try:
            routes = lkh_module.solve(
                solver=self.settings.solver_executable,
                problem_file=str(problem_path),
                **dict(solver_params),
            )
        except AssertionError as exc:
            raise LKHExecutionError(f"LKH assertion failure: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LKHExecutionError(f"LKH solve failed: {exc}") from exc

        if not isinstance(routes, list):
            raise LKHExecutionError(f"Expected routes list from lkh.solve(...), got {type(routes).__name__}.")
        return routes

    def _get_lkh_module(self) -> Any:
        if self._lkh_module is not None:
            return self._lkh_module
        try:
            import lkh  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LKHDependencyError(
                "Python package 'lkh' is required for Phase-2 solver integration. "
                "Install it in your environment before running LKH solves."
            ) from exc
        self._lkh_module = lkh
        return lkh

    def _validate_source_archive(self) -> None:
        if not self.settings.require_source_archive:
            return
        archive = Path(self.settings.source_archive)
        if not archive.exists():
            raise LKHConfigError(
                f"Configured source archive not found at '{archive}'. "
                f"Expected canonical path '{CANONICAL_PINNED_LKH_ARCHIVE}'."
            )

    @staticmethod
    def _validate_instance_for_solver(instance: TSPInstance) -> None:
        if instance.indexing != INDEXING_TSPLIB_1_BASED:
            raise DataValidationError(
                f"Only TSPLIB 1-based indexing is supported, got indexing='{instance.indexing}'."
            )
        if instance.distance_metric != DISTANCE_METRIC_EUCLIDEAN_2D:
            raise DataValidationError(
                f"Only '{DISTANCE_METRIC_EUCLIDEAN_2D}' is supported, got '{instance.distance_metric}'."
            )

    @staticmethod
    def _validate_partial_route(instance: TSPInstance, partial_route: Sequence[int]) -> tuple[int, ...]:
        if len(partial_route) == 0:
            raise DataValidationError("partial_route must be non-empty for constrained completion.")

        route = tuple(int(node_id) for node_id in partial_route)
        all_nodes = set(range(1, instance.node_count + 1))

        if len(set(route)) != len(route):
            raise DataValidationError("partial_route contains repeated node ids.")
        for node_id in route:
            if node_id not in all_nodes:
                raise DataValidationError(
                    f"partial_route contains out-of-range node id {node_id}; expected in 1..{instance.node_count}."
                )
        return route

    @staticmethod
    def _prefix_to_fixed_edges(partial_route: Sequence[int]) -> tuple[tuple[int, int], ...]:
        if len(partial_route) < 2:
            return ()
        return tuple((partial_route[idx], partial_route[idx + 1]) for idx in range(len(partial_route) - 1))

    @staticmethod
    def _make_run_dir_internal(
        *,
        mode: str,
        instance_id: str,
        debug_enabled: bool = False,
        debug_root: str = "",
    ) -> tuple[Path, bool]:
        if debug_enabled:
            root = Path(debug_root)
            root.mkdir(parents=True, exist_ok=True)
            run_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{mode}_{instance_id}_",
                    dir=str(root),
                )
            )
            return run_dir, False
        run_dir = Path(tempfile.mkdtemp(prefix=f"{mode}_{instance_id}_"))
        return run_dir, True

    def _make_run_dir(self, *, mode: str, instance_id: str) -> tuple[Path, bool]:
        return self._make_run_dir_internal(
            mode=mode,
            instance_id=instance_id,
            debug_enabled=self.settings.debug_enabled,
            debug_root=self.settings.debug_output_root,
        )

    def _validate_solver_executable(self) -> None:
        solver = self.settings.solver_executable
        if "/" in solver:
            path = Path(solver)
            if not path.exists():
                raise LKHConfigError(f"Configured solver_executable does not exist: {solver}")
            if not path.is_file():
                raise LKHConfigError(f"Configured solver_executable is not a file: {solver}")
            if not (path.stat().st_mode & 0o111):
                raise LKHConfigError(f"Configured solver_executable is not executable: {solver}")
            return
        if shutil.which(solver) is None:
            raise LKHConfigError(
                f"Configured solver_executable '{solver}' was not found on PATH and is not a direct file path."
            )

    @staticmethod
    def _write_problem_file(
        *,
        instance: TSPInstance,
        path: Path,
        fixed_edges: Sequence[tuple[int, int]],
    ) -> None:
        lines: list[str] = [
            f"NAME: {instance.instance_id}",
            "TYPE: TSP",
            f"DIMENSION: {instance.node_count}",
            "EDGE_WEIGHT_TYPE: EUC_2D",
            "NODE_COORD_SECTION",
        ]
        for node in instance.nodes:
            lines.append(f"{node.node_id} {node.x:.12f} {node.y:.12f}")
        if fixed_edges:
            lines.append("FIXED_EDGES_SECTION")
            for left, right in fixed_edges:
                lines.append(f"{left} {right}")
            lines.append("-1")
        lines.append("EOF")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_param_preview(*, path: Path, problem_path: Path, solver_params: Mapping[str, Any]) -> None:
        lines = ["SPECIAL", f"PROBLEM_FILE = {problem_path}"]
        for key in sorted(solver_params.keys()):
            lines.append(f"{key.upper()} = {solver_params[key]}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_result_preview(
        *,
        path: Path,
        mode: str,
        route: Sequence[int],
        tour_length: float,
        fixed_edges: Sequence[tuple[int, int]],
        solver_params: Mapping[str, Any],
    ) -> None:
        payload = {
            "mode": mode,
            "tour": list(route),
            "tour_length": tour_length,
            "fixed_edges": [list(edge) for edge in fixed_edges],
            "solver_params": dict(solver_params),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _validate_route_permutation(*, route: Sequence[int], node_count: int) -> None:
        expected = set(range(1, node_count + 1))
        got = set(route)
        if len(route) != node_count:
            raise LKHExecutionError(f"Expected route length {node_count}, got {len(route)}.")
        if got != expected:
            raise LKHExecutionError(f"Expected tour permutation of 1..{node_count}, got {list(route)}.")

    @staticmethod
    def _rotate_to_start(route: Sequence[int], start_node: int) -> tuple[int, ...]:
        if start_node not in route:
            raise LKHConstraintError(f"start_node {start_node} is missing from route {list(route)}")
        start_idx = route.index(start_node)
        rotated = tuple(route[start_idx:]) + tuple(route[:start_idx])
        return rotated

    @classmethod
    def _canonicalize_to_start_node(cls, route: Sequence[int], *, start_node: int) -> tuple[int, ...]:
        forward = cls._rotate_to_start(list(route), start_node)
        backward = cls._rotate_to_start(list(reversed(route)), start_node)
        return min(forward, backward)

    @classmethod
    def _canonicalize_to_prefix(cls, route: Sequence[int], partial_route: Sequence[int]) -> tuple[int, ...]:
        start_node = partial_route[0]
        forward = cls._rotate_to_start(list(route), start_node)
        backward = cls._rotate_to_start(list(reversed(route)), start_node)

        if tuple(forward[: len(partial_route)]) == tuple(partial_route):
            return forward
        if tuple(backward[: len(partial_route)]) == tuple(partial_route):
            return backward
        raise LKHConstraintError(
            "Constrained completion violated ordered fixed-prefix semantics. "
            f"Expected prefix {list(partial_route)}, got forward {list(forward)} and backward {list(backward)}."
        )

    @staticmethod
    def _compute_tour_length(*, instance: TSPInstance, tour: Sequence[int]) -> float:
        coords = {node.node_id: (node.x, node.y) for node in instance.nodes}
        total = 0.0
        for idx, node_id in enumerate(tour):
            next_node_id = tour[(idx + 1) % len(tour)]
            x1, y1 = coords[node_id]
            x2, y2 = coords[next_node_id]
            total += math.hypot(x2 - x1, y2 - y1)
        return total

    @staticmethod
    def _build_debug_paths(
        *,
        enabled: bool,
        run_dir: Path,
        problem_path: Path,
        params_path: Path,
        tour_path: Path,
    ) -> dict[str, str]:
        if not enabled:
            return {}
        return {
            "run_dir": str(run_dir),
            "problem_file": str(problem_path),
            "param_preview_file": str(params_path),
            "tour_file": str(tour_path),
            "result_preview_file": str(run_dir / "result.json"),
        }
