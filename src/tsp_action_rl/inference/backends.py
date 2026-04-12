"""Concrete zero-shot model backends."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from dataclasses import dataclass, field

from tsp_action_rl.data import RolloutState, TSPInstance

from .interface import ModelBackend, ModelOutput


class DMXAPIBackendError(RuntimeError):
    """Raised when DMXAPI request/response handling fails."""

    def __init__(self, message: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata: dict[str, Any] = dict(metadata or {})


class LocalDeterministicModelBackend(ModelBackend):
    """Simple local backend that chooses nearest unvisited node from current node."""

    backend_type = "local"

    def __init__(self, model_name: str = "local-deterministic-nearest-neighbor") -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt_text: str,
        *,
        instance: TSPInstance,
        state: RolloutState,
    ) -> ModelOutput:
        if state.is_terminal or not state.unvisited_nodes:
            raw = (
                "The route is already complete, so no valid next action exists in this state.\n\n"
                "<FINAL_NEXT_NODE>1</FINAL_NEXT_NODE>"
            )
            return ModelOutput(raw_text=raw, metadata={"backend": self.model_name})

        current_node = state.current_node
        if current_node is None:
            raise ValueError("current_node is required for LocalDeterministicModelBackend.")

        coords = {node.node_id: (node.x, node.y) for node in instance.nodes}
        cx, cy = coords[current_node]
        candidates = []
        for node_id in state.unvisited_nodes:
            nx, ny = coords[node_id]
            distance = math.hypot(nx - cx, ny - cy)
            candidates.append((distance, node_id))
        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen = candidates[0][1]

        reasoning = (
            "I evaluate the unvisited candidates from the current position while considering tour continuity. "
            f"The nearest unvisited node from node {current_node} is node {chosen}, which is a reasonable next move "
            "for this step while preserving feasibility for the remaining nodes."
        )
        raw = f"{reasoning}\n\n<FINAL_NEXT_NODE>{chosen}</FINAL_NEXT_NODE>"
        return ModelOutput(raw_text=raw, metadata={"backend": self.model_name, "chosen_node": chosen})


class LocalStaticResponseBackend(ModelBackend):
    """Local backend that replays static response text(s), useful for tests and fixtures."""

    backend_type = "local"

    def __init__(self, responses: Sequence[str], model_name: str = "local-static-response") -> None:
        if not responses:
            raise ValueError("responses must contain at least one response string.")
        self.model_name = model_name
        self._responses = list(responses)
        self._index = 0

    def generate(
        self,
        prompt_text: str,
        *,
        instance: TSPInstance,
        state: RolloutState,
    ) -> ModelOutput:
        idx = self._index
        if self._index < len(self._responses) - 1:
            self._index += 1
        return ModelOutput(raw_text=self._responses[idx], metadata={"backend": self.model_name, "response_index": idx})

    @classmethod
    def from_text_file(cls, path: str | Path, model_name: str = "local-static-response") -> "LocalStaticResponseBackend":
        content = Path(path).read_text(encoding="utf-8")
        return cls(responses=[content], model_name=model_name)


class ApiTodoModelBackend(ModelBackend):
    """Explicit placeholder for API-based backends."""

    backend_type = "api"

    def __init__(self, model_name: str = "api-todo") -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt_text: str,
        *,
        instance: TSPInstance,
        state: RolloutState,
    ) -> ModelOutput:
        raise NotImplementedError(
            "API backend is not implemented yet. "
            "TODO: add provider-specific API call while keeping the ModelBackend interface unchanged."
        )


@dataclass(frozen=True)
class DmxOpenAICompatibleConfig:
    """Config for DMXAPI OpenAI-compatible backend."""

    base_url: str | None = None
    base_url_env: str = "DMXAPI_BASE_URL"
    api_key_env: str = "DMXAPI_API_KEY"
    endpoint_path: str = "/chat/completions"
    model_id: str = "claude-opus-4-6-thinking"
    thinking_effort: str = "high"
    thinking_effort_field: str = "reasoning_effort"
    max_tokens: int = 4096
    temperature: float | None = 0.2
    top_p: float | None = 1.0
    timeout_seconds: int = 180
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    debug_enabled: bool = False
    debug_output_root: str = "outputs/debug/dmxapi"
    dry_run: bool = False

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None) -> "DmxOpenAICompatibleConfig":
        source = raw or {}
        if not isinstance(source, Mapping):
            raise ValueError("api_config must be a mapping/object.")

        extra_headers_raw = source.get("extra_headers", {})
        extra_body_raw = source.get("extra_body", {})
        debug_raw = source.get("debug", {})
        if not isinstance(extra_headers_raw, Mapping):
            raise ValueError("api_config.extra_headers must be a mapping/object.")
        if not isinstance(extra_body_raw, Mapping):
            raise ValueError("api_config.extra_body must be a mapping/object.")
        if not isinstance(debug_raw, Mapping):
            raise ValueError("api_config.debug must be a mapping/object.")

        return DmxOpenAICompatibleConfig(
            base_url=source.get("base_url"),
            base_url_env=str(source.get("base_url_env", "DMXAPI_BASE_URL")),
            api_key_env=str(source.get("api_key_env", "DMXAPI_API_KEY")),
            endpoint_path=str(source.get("endpoint_path", "/chat/completions")),
            model_id=str(source.get("model_id", "claude-opus-4-6-thinking")),
            thinking_effort=str(source.get("thinking_effort", "high")),
            thinking_effort_field=str(source.get("thinking_effort_field", "reasoning_effort")),
            max_tokens=int(source.get("max_tokens", 4096)),
            temperature=None if source.get("temperature") is None else float(source.get("temperature")),
            top_p=None if source.get("top_p") is None else float(source.get("top_p")),
            timeout_seconds=int(source.get("timeout_seconds", 180)),
            extra_headers={str(k): str(v) for k, v in dict(extra_headers_raw).items()},
            extra_body=dict(extra_body_raw),
            debug_enabled=bool(debug_raw.get("enabled", False)),
            debug_output_root=str(debug_raw.get("output_root", "outputs/debug/dmxapi")),
            dry_run=bool(debug_raw.get("dry_run", False)),
        )


class DMXOpenAICompatibleBackend(ModelBackend):
    """Real DMXAPI backend using OpenAI-compatible `/chat/completions` requests."""

    backend_type = "api"
    _SENSITIVE_KEY_MARKERS = ("key", "token", "secret", "password", "authorization", "auth")

    def __init__(self, *, config: DmxOpenAICompatibleConfig, model_name: str | None = None) -> None:
        self.config = config
        self.model_name = model_name or config.model_id
        self._debug_first_request_written = False

    def generate(
        self,
        prompt_text: str,
        *,
        instance: TSPInstance,
        state: RolloutState,
    ) -> ModelOutput:
        base_url = self._resolve_base_url()
        api_key = self._resolve_api_key()
        endpoint = base_url.rstrip("/") + "/" + self.config.endpoint_path.lstrip("/")

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": self.config.max_tokens,
        }
        if self.config.thinking_effort:
            payload[self.config.thinking_effort_field] = self.config.thinking_effort
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        payload.update(self.config.extra_body)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.extra_headers)

        request_payload = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(endpoint, data=request_payload, headers=headers, method="POST")
        response_metadata: dict[str, Any] = {"endpoint": endpoint, "model_id": self.model_name}

        debug_context = self._make_debug_context(
            prompt_text=prompt_text,
            endpoint=endpoint,
            payload=payload,
            headers=headers,
        )

        if self.config.dry_run:
            failure = {
                "type": "dry_run",
                "message": "DMXAPI request skipped because api.debug.dry_run=true.",
            }
            debug_path = self._write_first_debug_record(
                debug_context=debug_context,
                response_metadata=response_metadata,
                failure=failure,
            )
            raise DMXAPIBackendError(
                "DMXAPI dry-run enabled; request not sent.",
                metadata={
                    "debug_record_path": debug_path,
                    "failure": failure,
                },
            )

        try:
            with urllib_request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw_bytes = response.read()
        except urllib_error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            failure = {
                "type": "http_error",
                "http_status": exc.code,
                "http_error_body": error_body,
                "message": str(exc),
            }
            debug_path = self._write_first_debug_record(
                debug_context=debug_context,
                response_metadata=response_metadata,
                failure=failure,
            )
            raise DMXAPIBackendError(
                f"DMXAPI HTTP error {exc.code}: {error_body}",
                metadata={
                    "http_status": exc.code,
                    "http_error_body": error_body,
                    "debug_record_path": debug_path,
                    "response_metadata": response_metadata,
                },
            ) from exc
        except urllib_error.URLError as exc:
            failure = {
                "type": "url_error",
                "message": str(exc),
            }
            debug_path = self._write_first_debug_record(
                debug_context=debug_context,
                response_metadata=response_metadata,
                failure=failure,
            )
            raise DMXAPIBackendError(
                f"DMXAPI request failed: {exc}",
                metadata={
                    "debug_record_path": debug_path,
                    "response_metadata": response_metadata,
                },
            ) from exc

        try:
            payload_json = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            failure = {
                "type": "json_decode_error",
                "message": "DMXAPI returned non-JSON response.",
                "raw_response_preview": raw_bytes.decode("utf-8", errors="replace")[:2000],
            }
            debug_path = self._write_first_debug_record(
                debug_context=debug_context,
                response_metadata=response_metadata,
                failure=failure,
            )
            raise DMXAPIBackendError(
                "DMXAPI returned non-JSON response.",
                metadata={
                    "debug_record_path": debug_path,
                    "response_metadata": response_metadata,
                },
            ) from exc

        response_metadata.update(self._extract_response_metadata(payload_json))

        try:
            text = self._extract_text_from_response(payload_json)
        except RuntimeError as exc:
            failure = {
                "type": "response_format_error",
                "message": str(exc),
            }
            debug_path = self._write_first_debug_record(
                debug_context=debug_context,
                response_metadata=response_metadata,
                failure=failure,
            )
            raise DMXAPIBackendError(
                f"DMXAPI response format error: {exc}",
                metadata={
                    "debug_record_path": debug_path,
                    "response_metadata": response_metadata,
                },
            ) from exc

        debug_path = self._write_first_debug_record(
            debug_context=debug_context,
            response_metadata=response_metadata,
            failure=None,
        )
        metadata = {
            "backend": "dmx_openai_compatible",
            "provider": "dmxapi",
            "model_id": self.model_name,
            "endpoint": endpoint,
            "response_id": payload_json.get("id"),
            "usage": payload_json.get("usage"),
            "finish_reason": self._extract_finish_reason(payload_json),
        }
        if debug_path is not None:
            metadata["debug_record_path"] = debug_path
        return ModelOutput(raw_text=text, metadata=metadata)

    def _resolve_base_url(self) -> str:
        if self.config.base_url:
            return self.config.base_url
        value = os.getenv(self.config.base_url_env, "").strip()
        if value:
            return value
        raise RuntimeError(
            f"DMXAPI base URL is missing. Set env var {self.config.base_url_env} or api_config.base_url."
        )

    def _resolve_api_key(self) -> str:
        value = os.getenv(self.config.api_key_env, "").strip()
        if value:
            return value
        raise RuntimeError(f"DMXAPI API key is missing. Set env var {self.config.api_key_env}.")

    @staticmethod
    def _extract_finish_reason(payload_json: Mapping[str, Any]) -> str | None:
        choices = payload_json.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                finish = first.get("finish_reason")
                if isinstance(finish, str):
                    return finish
        return None

    @staticmethod
    def _extract_text_from_response(payload_json: Mapping[str, Any]) -> str:
        choices = payload_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("DMXAPI response missing choices[0].")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise RuntimeError("DMXAPI response choices[0] is malformed.")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("DMXAPI response choices[0].message is missing.")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, Mapping):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
            if parts:
                return "\n".join(parts)
        raise RuntimeError("DMXAPI response message.content is not a supported text format.")

    def _make_debug_context(
        self,
        *,
        prompt_text: str,
        endpoint: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> dict[str, Any] | None:
        if not self.config.debug_enabled or self._debug_first_request_written:
            return None
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend": "dmx_openai_compatible",
            "model_id": self.model_name,
            "endpoint": endpoint,
            "prompt_text": prompt_text,
            "request_headers_redacted": self._redact_headers_for_debug(headers),
            "request_body_redacted": self._redact_for_debug(payload),
        }

    def _write_first_debug_record(
        self,
        *,
        debug_context: Mapping[str, Any] | None,
        response_metadata: Mapping[str, Any],
        failure: Mapping[str, Any] | None,
    ) -> str | None:
        if debug_context is None or self._debug_first_request_written:
            return None

        payload = dict(debug_context)
        payload["response_metadata"] = dict(response_metadata)
        payload["status"] = "failed" if failure is not None else "success"
        if failure is not None:
            payload["failure"] = dict(failure)

        root = Path(self.config.debug_output_root)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = root / f"dmx_first_request_{stamp}.json"
        try:
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError:
            return None

        self._debug_first_request_written = True
        return str(path)

    @classmethod
    def _redact_for_debug(cls, value: Any, *, key_name: str | None = None) -> Any:
        if key_name is not None and cls._is_sensitive_key(key_name):
            return "***REDACTED***"
        if isinstance(value, Mapping):
            return {str(k): cls._redact_for_debug(v, key_name=str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._redact_for_debug(item) for item in value]
        return value

    @classmethod
    def _redact_headers_for_debug(cls, headers: Mapping[str, str]) -> dict[str, str]:
        redacted: dict[str, str] = {}
        for key, value in headers.items():
            if cls._is_sensitive_key(key):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = value
        return redacted

    @classmethod
    def _is_sensitive_key(cls, key: str) -> bool:
        lowered = key.lower()
        return any(marker in lowered for marker in cls._SENSITIVE_KEY_MARKERS)

    def _extract_response_metadata(self, payload_json: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "response_id": payload_json.get("id"),
            "usage": payload_json.get("usage"),
            "finish_reason": self._extract_finish_reason(payload_json),
        }


def build_model_backend(
    *,
    backend: str,
    model_name: str,
    static_response_path: str | None = None,
    api_config: Mapping[str, Any] | None = None,
) -> ModelBackend:
    """Factory for supported backends."""
    if backend == "local_deterministic":
        return LocalDeterministicModelBackend(model_name=model_name)
    if backend == "local_static":
        if static_response_path is None:
            raise ValueError("static_response_path is required for backend='local_static'.")
        return LocalStaticResponseBackend.from_text_file(path=static_response_path, model_name=model_name)
    if backend == "api_todo":
        return ApiTodoModelBackend(model_name=model_name)
    if backend == "dmx_openai_compatible":
        config = DmxOpenAICompatibleConfig.from_mapping(api_config)
        resolved_model_name = model_name or config.model_id
        return DMXOpenAICompatibleBackend(config=config, model_name=resolved_model_name)
    raise ValueError(f"Unsupported backend '{backend}'.")


def supported_backends() -> list[str]:
    return ["local_deterministic", "local_static", "api_todo", "dmx_openai_compatible"]
