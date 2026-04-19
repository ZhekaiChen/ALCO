"""SLIME adapter and RL entrypoint validation tests."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from tsp_action_rl.data import generate_random_euclidean_instance, load_tsp_instance
from tsp_action_rl.rl import (
    SLIMEAdapterSettings,
    SLIMERunSettings,
    TSPRLSlimeAdapter,
    TSPRLStepEnvironment,
    TSPRLEnvSettings,
    run_slime_eval,
    run_slime_train,
)
from tsp_action_rl.solvers import LKHSolveResult

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class FakeSolver:
    """Deterministic fake solver for adapter/train validation tests."""

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


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_adapter(*, include_diag: bool = True) -> tuple[TSPRLSlimeAdapter, FakeSolver]:
    solver = FakeSolver()
    env = TSPRLStepEnvironment(
        solver=solver,  # type: ignore[arg-type]
        settings=TSPRLEnvSettings(start_node_policy="fixed", fixed_start_node=1),
    )
    adapter = TSPRLSlimeAdapter(
        env=env,
        settings=SLIMEAdapterSettings(
            include_instance_payload=True,
            include_action_mask=True,
            include_step_diagnostics_in_info=include_diag,
            train=SLIMERunSettings(episodes=1, policy="first_unvisited", seed=7),
            eval=SLIMERunSettings(episodes=1, policy="first_unvisited", seed=7),
        ),
    )
    return adapter, solver


def test_slime_adapter_reset_and_step_validation() -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    adapter, solver = _build_adapter()

    obs = adapter.reset(instance=instance)
    assert obs["node_count"] == instance.node_count
    assert obs["partial_route"] == [1]
    assert len(obs["valid_action_mask"]) == instance.node_count + 1
    assert obs["valid_action_mask"][1] is False
    assert obs["valid_action_mask"][2] is True

    next_obs, reward, done, info = adapter.step(2)
    assert done is False
    assert reward == pytest.approx(-0.1)
    assert next_obs["partial_route"] == [1, 2]
    assert info["action_validation"]["is_valid"] is True
    assert "diagnostics" in info
    assert solver.reference_calls == 1
    assert solver.completion_calls == [(1, 2)]


def test_run_slime_train_validation_collects_step_traces() -> None:
    instance = generate_random_euclidean_instance(node_count=5, seed=11)
    adapter, _ = _build_adapter()
    adapter.reset(instance=instance)

    summary = run_slime_train(adapter=adapter, settings=adapter.settings)
    assert summary["mode"] == "train_validation"
    assert summary["episodes"] == 1
    assert summary["status_counts"]
    assert len(summary["episode_summaries"]) == 1
    assert len(summary["episode_summaries"][0]["step_traces"]) >= 1
    assert "action_validation" in summary["episode_summaries"][0]["step_traces"][0]


def test_run_slime_eval_external_entrypoint_hook() -> None:
    instance = generate_random_euclidean_instance(node_count=5, seed=13)
    adapter, _ = _build_adapter()
    adapter.reset(instance=instance)

    module = types.ModuleType("fake_slime_hooks")

    def fake_eval_runner(*, adapter, run_settings, mode):  # type: ignore[no-untyped-def]
        return {
            "mode": mode,
            "episodes": run_settings.episodes,
            "external": True,
            "instance_id": adapter.current_instance.instance_id,
        }

    module.fake_eval_runner = fake_eval_runner
    sys.modules["fake_slime_hooks"] = module
    try:
        settings = SLIMEAdapterSettings(
            eval_entrypoint="fake_slime_hooks:fake_eval_runner",
            eval=SLIMERunSettings(episodes=2, policy="random_unvisited", seed=3),
        )
        summary = run_slime_eval(adapter=adapter, settings=settings)
        assert summary["external"] is True
        assert summary["mode"] == "eval"
        assert summary["episodes"] == 2
    finally:
        del sys.modules["fake_slime_hooks"]


def test_run_rl_script_wiring_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module("run_rl.py")

    defaults = module._read_yaml_defaults(Path("configs/rl.yaml"))
    parser = module._build_arg_parser(defaults)
    args = parser.parse_args(
        [
            "--config",
            "configs/rl.yaml",
            "--lkh-config",
            "configs/lkh.yaml",
            "--output-root",
            str(tmp_path),
            "--run-name",
            "rl_script_validation",
            "--train-episodes",
            "1",
            "--node-count",
            "5",
            "--instance-seed",
            "19",
        ]
    )

    monkeypatch.setattr(module, "LKHIntegration", lambda settings: FakeSolver())

    result = module.run_training(args)
    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
    assert result["train_summary"]["episodes"] == 1
    assert result["train_summary"]["episode_summaries"]


def test_real_slime_rollout_contract_train_validation() -> None:
    slime_repo = Path(__file__).resolve().parents[1] / "third_party" / "slime"
    if not slime_repo.exists():
        pytest.skip("third_party/slime is not present")

    instance = generate_random_euclidean_instance(node_count=5, seed=17)
    adapter, _ = _build_adapter(include_diag=False)
    adapter.reset(instance=instance)

    settings = SLIMEAdapterSettings(
        use_real_slime_rollout_contract=True,
        slime_repo_path=str(slime_repo),
        train=SLIMERunSettings(
            episodes=1,
            policy="first_unvisited",
            seed=11,
            slime_rollout_batch_size=1,
            slime_n_samples_per_prompt=1,
        ),
    )
    summary = run_slime_train(adapter=adapter, settings=settings)

    assert summary["mode"] == "train_slime_rollout_contract_validation"
    assert summary["episodes"] == 1
    assert summary["episode_summaries"]
    assert summary["episode_summaries"][0]["num_steps"] == 1
    assert summary["integration"]["slime_repo_path"] == str(slime_repo.resolve())


def test_real_slime_rollout_contract_eval_validation() -> None:
    slime_repo = Path(__file__).resolve().parents[1] / "third_party" / "slime"
    if not slime_repo.exists():
        pytest.skip("third_party/slime is not present")

    instance = generate_random_euclidean_instance(node_count=5, seed=23)
    adapter, _ = _build_adapter(include_diag=False)
    adapter.reset(instance=instance)

    settings = SLIMEAdapterSettings(
        use_real_slime_rollout_contract=True,
        slime_repo_path=str(slime_repo),
        eval=SLIMERunSettings(
            episodes=1,
            policy="nearest_unvisited",
            seed=13,
            slime_rollout_batch_size=1,
            slime_n_samples_per_prompt=1,
        ),
    )
    summary = run_slime_eval(adapter=adapter, settings=settings)

    assert summary["mode"] == "eval_slime_rollout_contract_validation"
    assert summary["episodes"] == 1
    assert summary["episode_summaries"]
    assert summary["episode_summaries"][0]["num_steps"] == 1
