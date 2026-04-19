"""Project-owned SLIME adapter for the step-level TSP RL environment.

This module keeps SLIME integration project-owned and non-invasive by exposing:
- a lightweight reset/step adapter API around ``TSPRLStepEnvironment``;
- configurable external callable hooks (``module:function``) for future real SLIME
  trainer/evaluator entrypoints;
- built-in quick validation train/eval loops for immediate end-to-end validation.
"""

from __future__ import annotations

import importlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Mapping, Protocol

from tsp_action_rl.data import TSPInstance

from .environment import TSPRLObservation, TSPRLStepEnvironment


SLIME_POLICY_RANDOM_UNVISITED = "random_unvisited"
SLIME_POLICY_FIRST_UNVISITED = "first_unvisited"
SLIME_POLICY_NEAREST_UNVISITED = "nearest_unvisited"

SUPPORTED_SLIME_POLICIES = (
    SLIME_POLICY_RANDOM_UNVISITED,
    SLIME_POLICY_FIRST_UNVISITED,
    SLIME_POLICY_NEAREST_UNVISITED,
)

DEFAULT_SLIME_REPO_PATH = "third_party/slime"
DEFAULT_SLIME_ROLLOUT_FUNCTION_PATH = "tsp_action_rl.rl.slime_adapter.tsp_slime_rollout"
DEFAULT_SLIME_EVAL_DATASET_NAME = "tsp_step_eval"


class SLIMEAdapterConfigError(ValueError):
    """Raised when SLIME adapter config is invalid."""


class SLIMEAdapterRuntimeError(RuntimeError):
    """Raised when the SLIME adapter is used in an invalid runtime state."""


class ExternalSLIMERunner(Protocol):
    """Protocol for externally supplied train/eval hook callables."""

    def __call__(
        self,
        *,
        adapter: "TSPRLSlimeAdapter",
        run_settings: "SLIMERunSettings",
        mode: str,
    ) -> Mapping[str, Any]:
        """Execute a train/eval run and return summary mapping."""


@dataclass(frozen=True)
class SLIMERunSettings:
    """Minimal run settings for train/eval validation loops."""

    episodes: int = 1
    max_steps_per_episode: int | None = None
    policy: Literal[
        "random_unvisited",
        "first_unvisited",
        "nearest_unvisited",
    ] = SLIME_POLICY_RANDOM_UNVISITED
    seed: int = 0

    # Real SLIME rollout contract knobs (used only when enabled in adapter settings).
    slime_rollout_batch_size: int = 1
    slime_n_samples_per_prompt: int = 1
    slime_rollout_function_path: str = DEFAULT_SLIME_ROLLOUT_FUNCTION_PATH
    slime_eval_function_path: str | None = None
    slime_eval_dataset_name: str = DEFAULT_SLIME_EVAL_DATASET_NAME

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise SLIMEAdapterConfigError(f"episodes must be >= 1, got {self.episodes}.")
        if self.max_steps_per_episode is not None and self.max_steps_per_episode < 1:
            raise SLIMEAdapterConfigError(
                f"max_steps_per_episode must be >= 1 when set, got {self.max_steps_per_episode}."
            )
        if self.policy not in SUPPORTED_SLIME_POLICIES:
            raise SLIMEAdapterConfigError(
                f"Unsupported policy '{self.policy}'. Supported: {', '.join(SUPPORTED_SLIME_POLICIES)}"
            )
        if self.slime_rollout_batch_size < 1:
            raise SLIMEAdapterConfigError(
                f"slime_rollout_batch_size must be >= 1, got {self.slime_rollout_batch_size}."
            )
        if self.slime_n_samples_per_prompt < 1:
            raise SLIMEAdapterConfigError(
                f"slime_n_samples_per_prompt must be >= 1, got {self.slime_n_samples_per_prompt}."
            )
        if not self.slime_rollout_function_path.strip():
            raise SLIMEAdapterConfigError("slime_rollout_function_path must be a non-empty dotted path.")
        if self.slime_eval_function_path is not None and not self.slime_eval_function_path.strip():
            raise SLIMEAdapterConfigError("slime_eval_function_path must be null or a non-empty dotted path.")
        if not self.slime_eval_dataset_name.strip():
            raise SLIMEAdapterConfigError("slime_eval_dataset_name must be non-empty.")

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None, *, default_seed: int = 0) -> "SLIMERunSettings":
        source = raw or {}
        if not isinstance(source, Mapping):
            raise SLIMEAdapterConfigError(
                f"run settings must be a mapping/object, got {type(source).__name__}."
            )
        return SLIMERunSettings(
            episodes=int(source.get("episodes", 1)),
            max_steps_per_episode=(
                None
                if source.get("max_steps_per_episode") is None
                else int(source.get("max_steps_per_episode"))
            ),
            policy=str(source.get("policy", SLIME_POLICY_RANDOM_UNVISITED)),
            seed=int(source.get("seed", default_seed)),
            slime_rollout_batch_size=int(source.get("slime_rollout_batch_size", 1)),
            slime_n_samples_per_prompt=int(source.get("slime_n_samples_per_prompt", 1)),
            slime_rollout_function_path=str(
                source.get("slime_rollout_function_path", DEFAULT_SLIME_ROLLOUT_FUNCTION_PATH)
            ),
            slime_eval_function_path=(
                None
                if source.get("slime_eval_function_path") is None
                else str(source.get("slime_eval_function_path"))
            ),
            slime_eval_dataset_name=str(source.get("slime_eval_dataset_name", DEFAULT_SLIME_EVAL_DATASET_NAME)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "policy": self.policy,
            "seed": self.seed,
            "slime_rollout_batch_size": self.slime_rollout_batch_size,
            "slime_n_samples_per_prompt": self.slime_n_samples_per_prompt,
            "slime_rollout_function_path": self.slime_rollout_function_path,
            "slime_eval_function_path": self.slime_eval_function_path,
            "slime_eval_dataset_name": self.slime_eval_dataset_name,
        }


@dataclass(frozen=True)
class SLIMEAdapterSettings:
    """Configurable knobs for SLIME adapter behavior and hook resolution."""

    include_instance_payload: bool = False
    include_reference_metadata: bool = True
    include_action_mask: bool = True
    include_step_counters: bool = True
    include_step_diagnostics_in_info: bool = True

    use_real_slime_rollout_contract: bool = False
    slime_repo_path: str = DEFAULT_SLIME_REPO_PATH

    train_entrypoint: str | None = None
    eval_entrypoint: str | None = None

    train: SLIMERunSettings = SLIMERunSettings()
    eval: SLIMERunSettings = SLIMERunSettings(policy=SLIME_POLICY_NEAREST_UNVISITED)

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None) -> "SLIMEAdapterSettings":
        source = raw or {}
        if not isinstance(source, Mapping):
            raise SLIMEAdapterConfigError(
                f"slime_adapter config must be a mapping/object, got {type(source).__name__}."
            )

        default_seed = int(source.get("seed", 0))

        return SLIMEAdapterSettings(
            include_instance_payload=bool(source.get("include_instance_payload", False)),
            include_reference_metadata=bool(source.get("include_reference_metadata", True)),
            include_action_mask=bool(source.get("include_action_mask", True)),
            include_step_counters=bool(source.get("include_step_counters", True)),
            include_step_diagnostics_in_info=bool(source.get("include_step_diagnostics_in_info", True)),
            use_real_slime_rollout_contract=bool(source.get("use_real_slime_rollout_contract", False)),
            slime_repo_path=str(source.get("slime_repo_path", DEFAULT_SLIME_REPO_PATH)),
            train_entrypoint=(
                None
                if source.get("train_entrypoint") is None
                else str(source.get("train_entrypoint"))
            ),
            eval_entrypoint=(
                None
                if source.get("eval_entrypoint") is None
                else str(source.get("eval_entrypoint"))
            ),
            train=SLIMERunSettings.from_mapping(
                source.get("train"),
                default_seed=default_seed,
            ),
            eval=SLIMERunSettings.from_mapping(
                source.get("eval"),
                default_seed=default_seed,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_instance_payload": self.include_instance_payload,
            "include_reference_metadata": self.include_reference_metadata,
            "include_action_mask": self.include_action_mask,
            "include_step_counters": self.include_step_counters,
            "include_step_diagnostics_in_info": self.include_step_diagnostics_in_info,
            "use_real_slime_rollout_contract": self.use_real_slime_rollout_contract,
            "slime_repo_path": self.slime_repo_path,
            "train_entrypoint": self.train_entrypoint,
            "eval_entrypoint": self.eval_entrypoint,
            "train": self.train.to_dict(),
            "eval": self.eval.to_dict(),
        }


class TSPRLSlimeAdapter:
    """Thin adapter exposing a stable reset/step interface for SLIME integration."""

    def __init__(self, *, env: TSPRLStepEnvironment, settings: SLIMEAdapterSettings) -> None:
        self.env = env
        self.settings = settings
        self._instance: TSPInstance | None = None
        self._observation: TSPRLObservation | None = None

    @property
    def action_space_n(self) -> int:
        """Action-space size (node count) for current instance."""
        if self._instance is None:
            raise SLIMEAdapterRuntimeError("No active instance. Call reset(instance=...) before querying action_space_n.")
        return self._instance.node_count

    @property
    def current_instance(self) -> TSPInstance:
        """Current instance bound to the adapter."""
        if self._instance is None:
            raise SLIMEAdapterRuntimeError("No active instance. Call reset(instance=...) first.")
        return self._instance

    @property
    def current_observation(self) -> TSPRLObservation:
        """Current raw environment observation payload."""
        if self._observation is None:
            raise SLIMEAdapterRuntimeError("No active observation. Call reset(instance=...) first.")
        return self._observation

    def reset(self, *, instance: TSPInstance | None = None, start_node: int | None = None) -> dict[str, Any]:
        """Reset adapter/environment and return adapter-formatted observation."""
        if instance is not None:
            self._instance = instance
        if self._instance is None:
            raise SLIMEAdapterRuntimeError("reset(instance=...) must provide an instance on first call.")

        self._observation = self.env.reset(instance=self._instance, start_node=start_node)
        return self._format_observation(self._observation)

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Apply one next-node action and return (obs, reward, done, info)."""
        result = self.env.step(action)
        self._observation = result.observation

        info: dict[str, Any] = {
            "done_reason": result.done_reason,
            "reward_signal": result.reward_signal.to_dict(),
            "action_validation": dict(result.diagnostics.get("action_validation", {})),
        }
        if self.settings.include_step_diagnostics_in_info:
            info["diagnostics"] = dict(result.diagnostics)

        return self._format_observation(result.observation), result.reward, result.done, info

    def _format_observation(self, observation: TSPRLObservation) -> dict[str, Any]:
        state = observation.rollout_state
        payload: dict[str, Any] = {
            "instance_id": observation.instance.instance_id,
            "node_count": observation.instance.node_count,
            "partial_route": list(state.partial_route),
            "visited_nodes": list(state.visited_nodes),
            "unvisited_nodes": list(state.unvisited_nodes),
            "current_node": state.current_node,
            "current_position": (
                None if state.current_position is None else state.current_position.to_dict()
            ),
            "is_terminal": state.is_terminal,
            "indexing": state.indexing,
        }

        if self.settings.include_reference_metadata:
            payload.update(
                {
                    "reference_tour_length": observation.reference_tour_length,
                    "latest_constrained_tour_length": observation.latest_constrained_tour_length,
                    "latest_gap_to_reference": observation.latest_gap_to_reference,
                }
            )

        if self.settings.include_action_mask:
            mask = [False] * (observation.instance.node_count + 1)
            for node_id in state.unvisited_nodes:
                mask[node_id] = True
            payload["valid_action_mask"] = mask

        if self.settings.include_step_counters:
            payload["step_count"] = observation.step_count
            payload["invalid_action_count"] = observation.invalid_action_count

        if self.settings.include_instance_payload:
            payload["nodes"] = [
                {
                    "node_id": node.node_id,
                    "x": node.x,
                    "y": node.y,
                }
                for node in observation.instance.nodes
            ]

        return payload


def resolve_entrypoint_callable(spec: str) -> Callable[..., Mapping[str, Any]]:
    """Resolve an external callable from ``module:function`` path."""
    if ":" not in spec:
        raise SLIMEAdapterConfigError(
            f"Entrypoint '{spec}' must use format 'module:function'."
        )
    module_name, attr_name = spec.split(":", maxsplit=1)
    module_name = module_name.strip()
    attr_name = attr_name.strip()
    if not module_name or not attr_name:
        raise SLIMEAdapterConfigError(
            f"Entrypoint '{spec}' must use non-empty 'module:function' parts."
        )

    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name, None)
    if attr is None or not callable(attr):
        raise SLIMEAdapterConfigError(
            f"Entrypoint '{spec}' did not resolve to a callable."
        )
    return attr


def _choose_action(observation: TSPRLObservation, *, policy: str, rng: random.Random) -> int:
    unvisited = list(observation.rollout_state.unvisited_nodes)
    if not unvisited:
        raise SLIMEAdapterRuntimeError("No unvisited nodes remain; action selection is undefined.")

    if policy == SLIME_POLICY_RANDOM_UNVISITED:
        return int(rng.choice(unvisited))

    if policy == SLIME_POLICY_FIRST_UNVISITED:
        return int(min(unvisited))

    if policy == SLIME_POLICY_NEAREST_UNVISITED:
        current_node = observation.rollout_state.current_node
        if current_node is None:
            return int(min(unvisited))
        coords = {node.node_id: (node.x, node.y) for node in observation.instance.nodes}
        cx, cy = coords[current_node]
        best = min(
            unvisited,
            key=lambda node_id: (
                (coords[node_id][0] - cx) ** 2 + (coords[node_id][1] - cy) ** 2,
                node_id,
            ),
        )
        return int(best)

    raise SLIMEAdapterConfigError(
        f"Unsupported policy '{policy}'. Supported: {', '.join(SUPPORTED_SLIME_POLICIES)}"
    )


def _run_validation_loop(
    *,
    adapter: TSPRLSlimeAdapter,
    run_settings: SLIMERunSettings,
    mode: str,
) -> dict[str, Any]:
    rng = random.Random(run_settings.seed)
    episode_summaries: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    for episode_idx in range(1, run_settings.episodes + 1):
        adapter.reset()
        total_reward = 0.0
        steps = 0
        done = False
        done_reason: str | None = None
        step_traces: list[dict[str, Any]] = []

        while True:
            obs = adapter.current_observation
            if obs.rollout_state.is_terminal:
                done = True
                done_reason = "route_completed"
                break

            action = _choose_action(obs, policy=run_settings.policy, rng=rng)
            _, reward, done, info = adapter.step(action)
            total_reward += float(reward)
            steps += 1
            done_reason = info.get("done_reason")

            trace: dict[str, Any] = {
                "step_in_episode": steps,
                "action": action,
                "reward": float(reward),
                "done": bool(done),
                "done_reason": done_reason,
                "action_validation": dict(info.get("action_validation", {})),
                "reward_signal": dict(info.get("reward_signal", {})),
            }
            diagnostics = info.get("diagnostics")
            if isinstance(diagnostics, Mapping):
                trace["solver_completion"] = diagnostics.get("solver_completion")
                trace["state_before"] = diagnostics.get("state_before")
                trace["state_after"] = diagnostics.get("state_after")
            step_traces.append(trace)

            if done:
                break

            if run_settings.max_steps_per_episode is not None and steps >= run_settings.max_steps_per_episode:
                done = True
                done_reason = "max_steps_per_episode"
                break

        final_status = done_reason or "unknown"
        status_counts[final_status] = status_counts.get(final_status, 0) + 1

        episode_summaries.append(
            {
                "episode_index": episode_idx,
                "return": total_reward,
                "num_steps": steps,
                "done": done,
                "done_reason": done_reason,
                "final_partial_route": list(adapter.current_observation.rollout_state.partial_route),
                "step_traces": step_traces,
            }
        )

    avg_return = sum(item["return"] for item in episode_summaries) / len(episode_summaries)
    avg_steps = sum(item["num_steps"] for item in episode_summaries) / len(episode_summaries)

    return {
        "mode": mode,
        "episodes": run_settings.episodes,
        "policy": run_settings.policy,
        "avg_return": avg_return,
        "avg_episode_steps": avg_steps,
        "status_counts": status_counts,
        "episode_summaries": episode_summaries,
    }


def _resolve_slime_repo_path(settings: SLIMEAdapterSettings) -> Path:
    path = Path(settings.slime_repo_path).resolve()
    if not path.exists() or not path.is_dir():
        raise SLIMEAdapterRuntimeError(f"SLIME repository path does not exist or is not a directory: {path}")
    if not (path / "slime").exists():
        raise SLIMEAdapterRuntimeError(
            f"SLIME repository path is missing expected package directory 'slime/': {path}"
        )
    return path


def _ensure_slime_repo_on_sys_path(repo_path: Path) -> None:
    text = str(repo_path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _load_dotted_callable(path: str) -> Callable[..., Any]:
    if "." not in path:
        raise SLIMEAdapterConfigError(f"Expected dotted path 'module.attr', got '{path}'.")
    module_name, attr_name = path.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name, None)
    if attr is None or not callable(attr):
        raise SLIMEAdapterConfigError(f"Dotted path '{path}' did not resolve to a callable.")
    return attr


def _load_slime_contract_symbols() -> tuple[Any, Any, Any]:
    from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput, call_rollout_fn

    return call_rollout_fn, RolloutFnTrainOutput, RolloutFnEvalOutput


def _load_slime_sample_type() -> Any:
    from slime.utils.types import Sample

    return Sample


def _build_step_prompt_from_observation(observation: dict[str, Any]) -> str:
    payload = {
        "instance_id": observation.get("instance_id"),
        "node_count": observation.get("node_count"),
        "partial_route": observation.get("partial_route"),
        "unvisited_nodes": observation.get("unvisited_nodes"),
        "current_node": observation.get("current_node"),
        "indexing": observation.get("indexing"),
    }
    return json.dumps(payload, sort_keys=True)


def _sample_status_for_validation(sample_type: Any, *, is_valid: bool) -> Any:
    if is_valid:
        return sample_type.Status.COMPLETED
    if hasattr(sample_type.Status, "FAILED"):
        return sample_type.Status.FAILED
    return sample_type.Status.ABORTED


def tsp_slime_rollout(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """SLIME rollout-function compatible wrapper for one-step TSP RL sampling.

    Signature intentionally matches SLIME's rollout function contract:
    ``(args, rollout_id, data_source, evaluation=False)``.
    """
    del data_source

    adapter = getattr(args, "tsp_adapter", None)
    if adapter is None:
        raise SLIMEAdapterRuntimeError("SLIME rollout args must provide `tsp_adapter`.")

    policy = str(getattr(args, "tsp_policy", SLIME_POLICY_RANDOM_UNVISITED))
    seed = int(getattr(args, "tsp_seed", 0))
    rollout_batch_size = int(getattr(args, "rollout_batch_size", 1))
    n_samples_per_prompt = int(getattr(args, "n_samples_per_prompt", 1))
    eval_dataset_name = str(getattr(args, "tsp_eval_dataset_name", DEFAULT_SLIME_EVAL_DATASET_NAME))
    start_node = getattr(args, "tsp_start_node", None)

    if rollout_batch_size < 1:
        raise SLIMEAdapterConfigError(f"rollout_batch_size must be >= 1, got {rollout_batch_size}.")
    if n_samples_per_prompt < 1:
        raise SLIMEAdapterConfigError(f"n_samples_per_prompt must be >= 1, got {n_samples_per_prompt}.")
    if policy not in SUPPORTED_SLIME_POLICIES:
        raise SLIMEAdapterConfigError(
            f"Unsupported policy '{policy}'. Supported: {', '.join(SUPPORTED_SLIME_POLICIES)}"
        )

    Sample = _load_slime_sample_type()
    _, RolloutFnTrainOutput, RolloutFnEvalOutput = _load_slime_contract_symbols()

    rng = random.Random(seed + rollout_id)
    groups: list[list[Any]] = []
    flat_samples: list[Any] = []
    total_reward = 0.0
    valid_count = 0
    sample_index = rollout_id * rollout_batch_size * n_samples_per_prompt

    for group_index in range(rollout_batch_size):
        group: list[Any] = []
        for _ in range(n_samples_per_prompt):
            observation_payload = adapter.reset(start_node=start_node)
            action = _choose_action(adapter.current_observation, policy=policy, rng=rng)
            _, reward, done, info = adapter.step(action)

            action_validation = info.get("action_validation", {})
            is_valid = bool(action_validation.get("is_valid", False))
            if is_valid:
                valid_count += 1
            total_reward += float(reward)

            trace = {
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                "done_reason": info.get("done_reason"),
                "action_validation": dict(action_validation),
                "reward_signal": dict(info.get("reward_signal", {})),
                "diagnostics": dict(info.get("diagnostics", {})),
            }

            sample = Sample(
                group_index=group_index,
                index=sample_index,
                prompt=_build_step_prompt_from_observation(observation_payload),
                response=f"<FINAL_NEXT_NODE>{action}</FINAL_NEXT_NODE>",
                tokens=[int(action)],
                response_length=1,
                reward=float(reward),
                status=_sample_status_for_validation(Sample, is_valid=is_valid),
                metadata={
                    "tsp_rollout_id": rollout_id,
                    "tsp_step_trace": trace,
                },
            )
            sample_index += 1
            group.append(sample)
            flat_samples.append(sample)
        groups.append(group)

    metrics = {
        "source": "tsp_step_adapter",
        "rollout_id": rollout_id,
        "sample_count": len(flat_samples),
        "valid_action_rate": (valid_count / len(flat_samples)) if flat_samples else 0.0,
        "avg_reward": (total_reward / len(flat_samples)) if flat_samples else 0.0,
    }

    if evaluation:
        eval_data = {
            eval_dataset_name: {
                "rewards": [float(sample.reward) for sample in flat_samples],
                "truncated": [sample.status == Sample.Status.TRUNCATED for sample in flat_samples],
                "samples": flat_samples,
            }
        }
        return RolloutFnEvalOutput(data=eval_data, metrics=metrics)

    return RolloutFnTrainOutput(samples=groups, metrics=metrics)


def _run_validation_loop_via_real_slime_rollout_contract(
    *,
    adapter: TSPRLSlimeAdapter,
    run_settings: SLIMERunSettings,
    settings: SLIMEAdapterSettings,
    evaluation: bool,
) -> dict[str, Any]:
    repo_path = _resolve_slime_repo_path(settings)
    _ensure_slime_repo_on_sys_path(repo_path)
    call_rollout_fn, _, _ = _load_slime_contract_symbols()

    rollout_fn = _load_dotted_callable(run_settings.slime_rollout_function_path)
    eval_fn_path = run_settings.slime_eval_function_path or run_settings.slime_rollout_function_path
    eval_fn = _load_dotted_callable(eval_fn_path)

    args = SimpleNamespace(
        rollout_batch_size=run_settings.slime_rollout_batch_size,
        n_samples_per_prompt=run_settings.slime_n_samples_per_prompt,
        reward_key=None,
        tsp_adapter=adapter,
        tsp_policy=run_settings.policy,
        tsp_seed=run_settings.seed,
        tsp_eval_dataset_name=run_settings.slime_eval_dataset_name,
        tsp_start_node=None,
    )

    episode_summaries: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    aggregate_return = 0.0
    aggregate_steps = 0

    for rollout_id in range(run_settings.episodes):
        fn = eval_fn if evaluation else rollout_fn
        output = call_rollout_fn(fn, args, rollout_id, None, evaluation=evaluation)

        if evaluation:
            datasets = output.data
            dataset_name = run_settings.slime_eval_dataset_name
            dataset = datasets.get(dataset_name)
            if dataset is None:
                dataset = next(iter(datasets.values()))

            samples = list(dataset.get("samples", []))
            traces = [dict(getattr(sample, "metadata", {}).get("tsp_step_trace", {})) for sample in samples]
            episode_return = float(sum(float(sample.reward) for sample in samples))
            num_steps = len(traces)
            done_reason = "slime_eval_rollout_contract_batch"
        else:
            groups = output.samples
            samples = [sample for group in groups for sample in group]
            traces = [dict(getattr(sample, "metadata", {}).get("tsp_step_trace", {})) for sample in samples]
            episode_return = float(sum(float(sample.reward) for sample in samples))
            num_steps = len(traces)
            done_reason = "slime_train_rollout_contract_batch"

        aggregate_return += episode_return
        aggregate_steps += num_steps
        status_counts[done_reason] = status_counts.get(done_reason, 0) + 1

        final_partial_route = []
        if traces:
            diagnostics = traces[-1].get("diagnostics")
            if isinstance(diagnostics, Mapping):
                state_after = diagnostics.get("state_after")
                if isinstance(state_after, Mapping):
                    partial_route = state_after.get("partial_route")
                    if isinstance(partial_route, (list, tuple)):
                        final_partial_route = [int(node_id) for node_id in partial_route]

        episode_summaries.append(
            {
                "episode_index": rollout_id + 1,
                "return": episode_return,
                "num_steps": num_steps,
                "done": True,
                "done_reason": done_reason,
                "final_partial_route": final_partial_route,
                "step_traces": traces,
            }
        )

    episodes = max(1, run_settings.episodes)
    mode = "eval_slime_rollout_contract_validation" if evaluation else "train_slime_rollout_contract_validation"
    return {
        "mode": mode,
        "episodes": run_settings.episodes,
        "policy": run_settings.policy,
        "avg_return": aggregate_return / episodes,
        "avg_episode_steps": aggregate_steps / episodes,
        "status_counts": status_counts,
        "episode_summaries": episode_summaries,
        "integration": {
            "slime_repo_path": str(repo_path),
            "rollout_function_path": run_settings.slime_rollout_function_path,
            "eval_function_path": eval_fn_path,
            "rollout_batch_size": run_settings.slime_rollout_batch_size,
            "n_samples_per_prompt": run_settings.slime_n_samples_per_prompt,
        },
    }


def run_slime_train(
    *,
    adapter: TSPRLSlimeAdapter,
    settings: SLIMEAdapterSettings,
) -> dict[str, Any]:
    """Run adapter training path using external hook or local validation fallback."""
    if settings.train_entrypoint:
        callable_obj = resolve_entrypoint_callable(settings.train_entrypoint)
        payload = callable_obj(adapter=adapter, run_settings=settings.train, mode="train")
        return dict(payload)

    if settings.use_real_slime_rollout_contract:
        return _run_validation_loop_via_real_slime_rollout_contract(
            adapter=adapter,
            run_settings=settings.train,
            settings=settings,
            evaluation=False,
        )

    return _run_validation_loop(adapter=adapter, run_settings=settings.train, mode="train_validation")


def run_slime_eval(
    *,
    adapter: TSPRLSlimeAdapter,
    settings: SLIMEAdapterSettings,
) -> dict[str, Any]:
    """Run adapter evaluation path using external hook or local validation fallback."""
    if settings.eval_entrypoint:
        callable_obj = resolve_entrypoint_callable(settings.eval_entrypoint)
        payload = callable_obj(adapter=adapter, run_settings=settings.eval, mode="eval")
        return dict(payload)

    if settings.use_real_slime_rollout_contract:
        return _run_validation_loop_via_real_slime_rollout_contract(
            adapter=adapter,
            run_settings=settings.eval,
            settings=settings,
            evaluation=True,
        )

    return _run_validation_loop(adapter=adapter, run_settings=settings.eval, mode="eval_validation")


# Legacy aliases for compatibility with existing integrations.
SLIMESmokeRunSettings = SLIMERunSettings
