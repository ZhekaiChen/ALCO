#!/usr/bin/env python3
"""Download a Hugging Face model snapshot to an external local model store."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
DEFAULT_MODELS_ROOT = "/mnt/zc/models"
DEFAULT_CONFIG_PATH = Path("configs/local_vllm.yaml")


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


def _default_model_path(*, models_root: str, model_id: str) -> Path:
    return Path(models_root).expanduser().resolve() / Path(*model_id.split("/"))


def _build_arg_parser(defaults: Mapping[str, Any]) -> argparse.ArgumentParser:
    model_defaults = _as_mapping(defaults.get("model"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-id", type=str, default=model_defaults.get("hf_model_id", DEFAULT_MODEL_ID))
    parser.add_argument(
        "--models-root",
        type=str,
        default=model_defaults.get("model_store_root", DEFAULT_MODELS_ROOT),
        help="External root directory for model snapshots.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=model_defaults.get("model_path"),
        help="Explicit full local directory for the snapshot. Overrides --models-root.",
    )
    parser.add_argument("--revision", type=str, default=model_defaults.get("revision"))
    parser.add_argument("--hf-token-env", type=str, default=model_defaults.get("hf_token_env", "HF_TOKEN"))
    parser.add_argument("--allow-pattern", action="append", default=None)
    parser.add_argument("--ignore-pattern", action="append", default=None)
    return parser


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre.parse_known_args()
    defaults = _read_yaml_defaults(pre_args.config)
    parser = _build_arg_parser(defaults)
    args = parser.parse_args()

    model_id = str(args.model_id).strip()
    if not model_id:
        raise ValueError("--model-id must be non-empty.")

    target_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path is not None
        else _default_model_path(models_root=str(args.models_root), model_id=model_id)
    )
    target_path.mkdir(parents=True, exist_ok=True)

    token = os.getenv(args.hf_token_env, "").strip() or None
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - runtime environment specific.
        raise RuntimeError(
            "huggingface_hub is required for model download. "
            "Install local-infer dependencies with: uv sync --group core --group local-infer"
        ) from exc

    downloaded_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(target_path),
        revision=None if args.revision in {None, "", "null"} else str(args.revision),
        token=token,
        allow_patterns=None if not args.allow_pattern else list(args.allow_pattern),
        ignore_patterns=None if not args.ignore_pattern else list(args.ignore_pattern),
    )
    print(f"Downloaded model '{model_id}' to: {downloaded_path}")


if __name__ == "__main__":
    main()
