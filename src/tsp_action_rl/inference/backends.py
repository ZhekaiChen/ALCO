"""Concrete zero-shot model backends."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from dataclasses import dataclass, field

from tsp_action_rl.data import RolloutState, TSPInstance

from .interface import ModelBackend, ModelOutput

DEFAULT_LOCAL_VLLM_MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
DEFAULT_LOCAL_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"


@dataclass(frozen=True)
class DmxModelProfile:
    """Explicit DMX request-shaping profile defaults for known model families."""

    name: str
    model_id: str
    thinking_effort: str | None
    thinking_effort_field: str | None
    timeout_seconds: int
    omit_request_fields: tuple[str, ...] = ()
    extra_body: dict[str, Any] = field(default_factory=dict)


DMX_MODEL_PROFILES: dict[str, DmxModelProfile] = {
    "claude-opus-4-6-thinking": DmxModelProfile(
        name="claude-opus-4-6-thinking",
        model_id="claude-opus-4-6-thinking",
        thinking_effort="high",
        thinking_effort_field="reasoning_effort",
        timeout_seconds=180,
    ),
    "glm-5.1": DmxModelProfile(
        name="glm-5.1",
        model_id="glm-5.1",
        thinking_effort=None,
        thinking_effort_field=None,
        timeout_seconds=180,
    ),
    "gpt-5.4": DmxModelProfile(
        name="gpt-5.4",
        model_id="gpt-5.4",
        thinking_effort="high",
        thinking_effort_field="reasoning_effort",
        timeout_seconds=180,
    ),
}


def available_dmx_model_profiles() -> list[str]:
    """Return supported named DMX model profiles."""
    return sorted(DMX_MODEL_PROFILES.keys())


def _resolve_dmx_model_profile(source: Mapping[str, Any]) -> DmxModelProfile | None:
    profile_name_raw = source.get("model_profile")
    if profile_name_raw is None:
        inferred = source.get("model_id")
        if inferred is not None and str(inferred) in DMX_MODEL_PROFILES:
            profile_name_raw = inferred

    if profile_name_raw is None:
        return None

    profile_name = str(profile_name_raw).strip()
    if not profile_name:
        return None
    profile = DMX_MODEL_PROFILES.get(profile_name)
    if profile is None:
        supported = ", ".join(available_dmx_model_profiles())
        raise ValueError(f"Unsupported api_config.model_profile '{profile_name}'. Supported profiles: {supported}.")
    return profile


def _parse_omit_request_fields(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        fields = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
        return tuple(fields)
    if isinstance(raw, list):
        fields = [str(value).strip() for value in raw if str(value).strip()]
        return tuple(fields)
    raise ValueError("api_config.omit_request_fields must be a list or comma-separated string.")


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
    api_key: str | None = None
    require_api_key: bool = True
    endpoint_path: str = "/chat/completions"
    model_profile: str | None = None
    model_id: str = "claude-opus-4-6-thinking"
    thinking_effort: str | None = "high"
    thinking_effort_field: str = "reasoning_effort"
    max_tokens: int = 4096
    temperature: float | None = 0.2
    top_p: float | None = 1.0
    timeout_seconds: int = 180
    omit_request_fields: tuple[str, ...] = ()
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    debug_enabled: bool = False
    debug_output_root: str = "outputs/debug/dmxapi"
    dry_run: bool = False
    provider_name: str = "dmxapi"
    backend_name: str = "dmx_openai_compatible"
    base_url_label: str = "DMXAPI"
    api_key_label: str = "DMXAPI API key"
    local_model_path: str | None = None
    served_model_name: str | None = None
    server_host: str | None = None
    server_port: int | None = None
    server_tensor_parallel_size: int | None = None

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None) -> "DmxOpenAICompatibleConfig":
        source = raw or {}
        if not isinstance(source, Mapping):
            raise ValueError("api_config must be a mapping/object.")

        profile = _resolve_dmx_model_profile(source)
        extra_headers_raw = source.get("extra_headers", {})
        extra_body_raw = source.get("extra_body", {})
        debug_raw = source.get("debug", {})
        omit_fields_raw = source.get("omit_request_fields")
        if not isinstance(extra_headers_raw, Mapping):
            raise ValueError("api_config.extra_headers must be a mapping/object.")
        if not isinstance(extra_body_raw, Mapping):
            raise ValueError("api_config.extra_body must be a mapping/object.")
        if not isinstance(debug_raw, Mapping):
            raise ValueError("api_config.debug must be a mapping/object.")

        default_model_id = profile.model_id if profile is not None else "claude-opus-4-6-thinking"
        default_thinking_effort = profile.thinking_effort if profile is not None else "high"
        default_thinking_field = (
            profile.thinking_effort_field if profile is not None and profile.thinking_effort_field is not None else "reasoning_effort"
        )
        default_timeout = profile.timeout_seconds if profile is not None else 180
        if omit_fields_raw is None:
            omit_fields = () if profile is None else profile.omit_request_fields
        else:
            omit_fields = _parse_omit_request_fields(omit_fields_raw)

        merged_extra_body: dict[str, Any] = {}
        if profile is not None:
            merged_extra_body.update(profile.extra_body)
        merged_extra_body.update(dict(extra_body_raw))

        return DmxOpenAICompatibleConfig(
            base_url=source.get("base_url"),
            base_url_env=str(source.get("base_url_env", "DMXAPI_BASE_URL")),
            api_key_env=str(source.get("api_key_env", "DMXAPI_API_KEY")),
            api_key=None if source.get("api_key") is None else str(source.get("api_key")),
            require_api_key=bool(source.get("require_api_key", True)),
            endpoint_path=str(source.get("endpoint_path", "/chat/completions")),
            model_profile=None if profile is None else profile.name,
            model_id=str(source.get("model_id", default_model_id)),
            thinking_effort=(
                default_thinking_effort
                if "thinking_effort" not in source
                else (None if source.get("thinking_effort") is None else str(source.get("thinking_effort")))
            ),
            thinking_effort_field=str(source.get("thinking_effort_field", default_thinking_field)),
            max_tokens=int(source.get("max_tokens", 4096)),
            temperature=None if source.get("temperature") is None else float(source.get("temperature")),
            top_p=None if source.get("top_p") is None else float(source.get("top_p")),
            timeout_seconds=int(source.get("timeout_seconds", default_timeout)),
            omit_request_fields=omit_fields,
            extra_headers={str(k): str(v) for k, v in dict(extra_headers_raw).items()},
            extra_body=merged_extra_body,
            debug_enabled=bool(debug_raw.get("enabled", False)),
            debug_output_root=str(debug_raw.get("output_root", "outputs/debug/dmxapi")),
            dry_run=bool(debug_raw.get("dry_run", False)),
            provider_name=str(source.get("provider_name", "dmxapi")),
            backend_name=str(source.get("backend_name", "dmx_openai_compatible")),
            base_url_label=str(source.get("base_url_label", "DMXAPI")),
            api_key_label=str(source.get("api_key_label", "DMXAPI API key")),
            local_model_path=(
                None if source.get("local_model_path") is None else str(source.get("local_model_path"))
            ),
            served_model_name=(
                None if source.get("served_model_name") is None else str(source.get("served_model_name"))
            ),
            server_host=None if source.get("server_host") is None else str(source.get("server_host")),
            server_port=None if source.get("server_port") is None else int(source.get("server_port")),
            server_tensor_parallel_size=(
                None
                if source.get("server_tensor_parallel_size") is None
                else int(source.get("server_tensor_parallel_size"))
            ),
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
        request_start = time.monotonic()
        base_url = self._resolve_base_url()
        api_key = self._resolve_api_key()
        endpoint = base_url.rstrip("/") + "/" + self.config.endpoint_path.lstrip("/")
        omit_fields = set(self.config.omit_request_fields)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        if "max_tokens" not in omit_fields:
            payload["max_tokens"] = self.config.max_tokens
        if (
            self.config.thinking_effort
            and self.config.thinking_effort_field
            and self.config.thinking_effort_field not in omit_fields
        ):
            payload[self.config.thinking_effort_field] = self.config.thinking_effort
        if self.config.temperature is not None and "temperature" not in omit_fields:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None and "top_p" not in omit_fields:
            payload["top_p"] = self.config.top_p
        payload.update(self.config.extra_body)
        for field_name in omit_fields:
            payload.pop(field_name, None)

        headers = {"Content-Type": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
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
        latency_seconds = time.monotonic() - request_start
        metadata = {
            "backend": self.config.backend_name,
            "provider": self.config.provider_name,
            "model_id": self.model_name,
            "model_profile": self.config.model_profile,
            "endpoint": endpoint,
            "response_id": payload_json.get("id"),
            "usage": payload_json.get("usage"),
            "finish_reason": self._extract_finish_reason(payload_json),
            "latency_seconds": latency_seconds,
            "generation_settings": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "thinking_effort": self.config.thinking_effort,
                "thinking_effort_field": self.config.thinking_effort_field,
                "omit_request_fields": list(self.config.omit_request_fields),
            },
        }
        if self.config.local_model_path:
            metadata["local_model_path"] = self.config.local_model_path
        if self.config.served_model_name:
            metadata["served_model_name"] = self.config.served_model_name
        if self.config.server_host:
            metadata["server_host"] = self.config.server_host
        if self.config.server_port is not None:
            metadata["server_port"] = self.config.server_port
        if self.config.server_tensor_parallel_size is not None:
            metadata["server_tensor_parallel_size"] = self.config.server_tensor_parallel_size
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
            f"{self.config.base_url_label} base URL is missing. "
            f"Set env var {self.config.base_url_env} or api_config.base_url."
        )

    def _resolve_api_key(self) -> str | None:
        if self.config.api_key is not None and self.config.api_key.strip():
            return self.config.api_key.strip()
        value = os.getenv(self.config.api_key_env, "").strip()
        if value:
            return value
        if not self.config.require_api_key:
            return None
        raise RuntimeError(f"{self.config.api_key_label} is missing. Set env var {self.config.api_key_env}.")

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
            "backend": self.config.backend_name,
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
    if backend == "local_vllm_openai_compatible":
        local_config = dict(api_config or {})
        local_config.setdefault("base_url", DEFAULT_LOCAL_VLLM_BASE_URL)
        local_config.setdefault("base_url_env", "LOCAL_VLLM_BASE_URL")
        local_config.setdefault("api_key_env", "LOCAL_VLLM_API_KEY")
        local_config.setdefault("require_api_key", False)
        local_config.setdefault("endpoint_path", "/chat/completions")
        local_config.setdefault("model_id", DEFAULT_LOCAL_VLLM_MODEL_ID)
        local_config.setdefault("thinking_effort", None)
        local_config.setdefault("provider_name", "local_vllm")
        local_config.setdefault("backend_name", "local_vllm_openai_compatible")
        local_config.setdefault("base_url_label", "local vLLM")
        local_config.setdefault("api_key_label", "local OpenAI-compatible API key")
        local_config.setdefault("served_model_name", local_config.get("model_id"))
        config = DmxOpenAICompatibleConfig.from_mapping(local_config)
        resolved_model_name = model_name or config.model_id
        return DMXOpenAICompatibleBackend(config=config, model_name=resolved_model_name)
    raise ValueError(f"Unsupported backend '{backend}'.")


def supported_backends() -> list[str]:
    return [
        "local_deterministic",
        "local_static",
        "api_todo",
        "dmx_openai_compatible",
        "local_vllm_openai_compatible",
    ]
