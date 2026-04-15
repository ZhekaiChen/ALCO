"""Config merge tests for zero-shot CLI profile/retry behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tsp_action_rl.inference import DmxOpenAICompatibleConfig


def _load_run_zeroshot_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_zeroshot_rollout.py"
    spec = importlib.util.spec_from_file_location("run_zeroshot_rollout_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_profile_override_keeps_configured_timeout_default() -> None:
    module = _load_run_zeroshot_module()
    defaults = {
        "api": {
            "model_id": "claude-opus-4-6-thinking",
            "thinking_effort": "high",
            "thinking_effort_field": "reasoning_effort",
            "timeout_seconds": 420,
            "omit_request_fields": ["temperature"],
        }
    }
    parser = module._build_arg_parser(defaults)
    args = parser.parse_args(["--api-model-profile", "glm-5.1"])
    api_config = module._build_api_config(args, defaults)
    resolved = DmxOpenAICompatibleConfig.from_mapping(api_config)

    assert resolved.model_profile == "glm-5.1"
    assert resolved.model_id == "glm-5.1"
    assert resolved.thinking_effort is None
    assert resolved.timeout_seconds == 420
    assert resolved.omit_request_fields == ("temperature",)


def test_cli_profile_override_honors_explicit_timeout_flag() -> None:
    module = _load_run_zeroshot_module()
    defaults = {
        "api": {
            "model_id": "claude-opus-4-6-thinking",
            "thinking_effort": "high",
            "thinking_effort_field": "reasoning_effort",
            "timeout_seconds": 420,
        }
    }
    parser = module._build_arg_parser(defaults)
    args = parser.parse_args(
        [
            "--api-model-profile",
            "gpt-5.4",
            "--api-timeout-seconds",
            "75",
        ]
    )
    api_config = module._build_api_config(args, defaults)
    resolved = DmxOpenAICompatibleConfig.from_mapping(api_config)

    assert resolved.model_profile == "gpt-5.4"
    assert resolved.model_id == "gpt-5.4"
    assert resolved.thinking_effort == "high"
    assert resolved.timeout_seconds == 75
