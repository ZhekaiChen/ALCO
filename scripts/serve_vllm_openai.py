#!/usr/bin/env python3
"""Launch a local vLLM OpenAI-compatible server for zero-shot rollout runs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("configs/local_vllm.yaml")
DEFAULT_MODEL_PATH = "/mnt/zc/models/Qwen/Qwen3-30B-A3B-Thinking-2507"
DEFAULT_SERVED_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"


def _read_yaml_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected YAML object at top-level in {path}.")
    return dict(payload)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _int_or(value: Any, default: int) -> int:
    if value in {None, "", "null"}:
        return default
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value in {None, "", "null"}:
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value in {None, "", "null"}:
        return None
    return int(value)


def _build_arg_parser(defaults: Mapping[str, Any]) -> argparse.ArgumentParser:
    model_defaults = _as_mapping(defaults.get("model"))
    serving_defaults = _as_mapping(defaults.get("serving"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--launcher",
        choices=["uvx", "vllm", "python-module"],
        default="uvx",
        help=(
            "How to start vLLM. 'uvx' avoids installing vLLM into this project env "
            "and falls back to 'uv tool run' when uvx is unavailable."
        ),
    )
    parser.add_argument("--model-path", type=Path, default=model_defaults.get("model_path", DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=str(serving_defaults.get("served_model_name", DEFAULT_SERVED_MODEL_NAME)),
    )
    parser.add_argument("--host", type=str, default=str(serving_defaults.get("host", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=_int_or(serving_defaults.get("port"), 8000))
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=_int_or(serving_defaults.get("tensor_parallel_size"), 1),
        help="vLLM tensor parallel world size (single-node multi-GPU path uses 8 on this host).",
    )
    parser.add_argument("--dtype", type=str, default=str(serving_defaults.get("dtype", "auto")))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=_float_or_none(serving_defaults.get("gpu_memory_utilization")),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=_int_or_none(serving_defaults.get("max_model_len")),
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=str(serving_defaults.get("api_key_env", "LOCAL_VLLM_API_KEY")),
        help="If set, pass this env var value through vLLM --api-key.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the launch command without starting the server.",
    )
    return parser


def _build_uvx_fallback_prefix() -> list[str]:
    if shutil.which("uvx") is not None:
        return ["uvx", "--from", "vllm"]
    if shutil.which("uv") is not None:
        return ["uv", "tool", "run", "--from", "vllm"]
    base_executable = getattr(sys, "_base_executable", None)
    if base_executable:
        return [base_executable, "-m", "uv", "tool", "run", "--from", "vllm"]
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv", "tool", "run", "--from", "vllm"]
    return ["python3", "-m", "uv", "tool", "run", "--from", "vllm"]


def _build_launch_command(args: argparse.Namespace, api_key: str | None) -> list[str]:
    model_path = str(Path(args.model_path).expanduser().resolve())

    if args.launcher == "uvx":
        cmd = _build_uvx_fallback_prefix() + ["vllm", "serve", model_path]
    elif args.launcher == "vllm":
        cmd = ["vllm", "serve", model_path]
    else:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
        ]

    cmd.extend(
        [
            "--host",
            str(args.host),
            "--port",
            str(args.port),
            "--served-model-name",
            str(args.served_model_name),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--dtype",
            str(args.dtype),
        ]
    )

    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if api_key:
        cmd.extend(["--api-key", api_key])
    return cmd


def _redact_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    for idx, token in enumerate(redacted):
        if token == "--api-key" and idx + 1 < len(redacted):
            redacted[idx + 1] = "***REDACTED***"
    return redacted


def _build_subprocess_env() -> dict[str, str]:
    """Prepare a stable subprocess environment for vLLM launch.

    Some environments export LIBRARY_PATH with a trailing ':'.
    That injects the current directory into gcc search paths and can make
    gcc try to read a local './specs' directory as a spec file.
    """
    env = dict(os.environ)
    library_path = env.get("LIBRARY_PATH")
    if library_path:
        cleaned = ":".join(part for part in library_path.split(":") if part)
        if cleaned:
            env["LIBRARY_PATH"] = cleaned
        else:
            env.pop("LIBRARY_PATH", None)
    return env


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre.parse_known_args()

    defaults = _read_yaml_defaults(pre_args.config)
    parser = _build_arg_parser(defaults)
    args = parser.parse_args()

    if args.launcher == "vllm" and shutil.which("vllm") is None:
        raise RuntimeError(
            "'vllm' was not found in PATH. "
            "Install vLLM CLI or choose --launcher uvx / --launcher python-module."
        )

    api_key = os.getenv(args.api_key_env, "").strip() or None
    cmd = _build_launch_command(args, api_key)
    cmd_preview = " ".join(_redact_command(cmd))

    print(f"vLLM launch command: {cmd_preview}")
    if args.dry_run:
        return

    subprocess.run(cmd, check=True, env=_build_subprocess_env())


if __name__ == "__main__":
    main()
