"""Unit tests for DMX OpenAI-compatible backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from io import BytesIO

import pytest

from tsp_action_rl.data import build_initial_rollout_state, load_tsp_instance
from tsp_action_rl.inference import DMXAPIBackendError, DMXOpenAICompatibleBackend, DmxOpenAICompatibleConfig

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None


def test_dmx_backend_sends_openai_compatible_request(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    state = build_initial_rollout_state(instance, start_node=1)

    monkeypatch.setenv("DMXAPI_BASE_URL", "https://dmx.example/v1")
    monkeypatch.setenv("DMXAPI_API_KEY", "test-key")

    captured: dict[str, Any] = {}

    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "id": "resp_123",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Reasoning.\n\n<FINAL_NEXT_NODE>4</FINAL_NEXT_NODE>"},
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 7},
            }
        )

    monkeypatch.setattr("tsp_action_rl.inference.backends.urllib_request.urlopen", _fake_urlopen)

    backend = DMXOpenAICompatibleBackend(config=DmxOpenAICompatibleConfig())
    output = backend.generate("prompt", instance=instance, state=state)

    assert captured["url"] == "https://dmx.example/v1/chat/completions"
    assert captured["timeout"] == 180
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "claude-opus-4-6-thinking"
    assert captured["body"]["messages"] == [{"role": "user", "content": "prompt"}]
    assert captured["body"]["reasoning_effort"] == "high"
    assert output.raw_text.endswith("<FINAL_NEXT_NODE>4</FINAL_NEXT_NODE>")
    assert output.metadata["provider"] == "dmxapi"
    assert output.metadata["finish_reason"] == "stop"


def test_dmx_backend_extracts_text_from_content_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    state = build_initial_rollout_state(instance, start_node=1)

    monkeypatch.setenv("DMXAPI_BASE_URL", "https://dmx.example/v1")
    monkeypatch.setenv("DMXAPI_API_KEY", "test-key")

    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        del req, timeout
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "line 1"},
                                {"type": "text", "text": "line 2"},
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("tsp_action_rl.inference.backends.urllib_request.urlopen", _fake_urlopen)

    backend = DMXOpenAICompatibleBackend(config=DmxOpenAICompatibleConfig())
    output = backend.generate("prompt", instance=instance, state=state)
    assert output.raw_text == "line 1\nline 2"


def test_dmx_backend_requires_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    state = build_initial_rollout_state(instance, start_node=1)

    monkeypatch.setenv("DMXAPI_BASE_URL", "https://dmx.example/v1")
    monkeypatch.delenv("DMXAPI_API_KEY", raising=False)

    backend = DMXOpenAICompatibleBackend(config=DmxOpenAICompatibleConfig())
    with pytest.raises(RuntimeError, match="DMXAPI API key is missing"):
        backend.generate("prompt", instance=instance, state=state)


def test_dmx_backend_dry_run_writes_first_debug_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    state = build_initial_rollout_state(instance, start_node=1)

    monkeypatch.setenv("DMXAPI_BASE_URL", "https://dmx.example/v1")
    monkeypatch.setenv("DMXAPI_API_KEY", "test-key")

    backend = DMXOpenAICompatibleBackend(
        config=DmxOpenAICompatibleConfig(
            debug_enabled=True,
            debug_output_root=str(tmp_path),
            dry_run=True,
            extra_body={"api_key": "secret-value", "nested": {"token": "nested-secret", "ok": "value"}},
        )
    )

    with pytest.raises(DMXAPIBackendError, match="dry-run"):
        backend.generate("prompt for debug", instance=instance, state=state)

    debug_files = list(tmp_path.glob("dmx_first_request_*.json"))
    assert len(debug_files) == 1
    payload = json.loads(debug_files[0].read_text(encoding="utf-8"))
    assert payload["prompt_text"] == "prompt for debug"
    assert payload["request_body_redacted"]["api_key"] == "***REDACTED***"
    assert payload["request_body_redacted"]["nested"]["token"] == "***REDACTED***"
    assert payload["status"] == "failed"
    assert payload["failure"]["type"] == "dry_run"


def test_dmx_backend_http_error_includes_status_and_error_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = load_tsp_instance(FIXTURES_DIR / "tsp_instance_minimal.json")
    state = build_initial_rollout_state(instance, start_node=1)

    monkeypatch.setenv("DMXAPI_BASE_URL", "https://dmx.example/v1")
    monkeypatch.setenv("DMXAPI_API_KEY", "test-key")

    def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
        del req, timeout
        raise urllib_error.HTTPError(
            url="https://dmx.example/v1/chat/completions",
            code=504,
            msg="Gateway Timeout",
            hdrs=None,
            fp=BytesIO(b'{"error":"timeout"}'),
        )

    monkeypatch.setattr("tsp_action_rl.inference.backends.urllib_request.urlopen", _fake_urlopen)

    backend = DMXOpenAICompatibleBackend(
        config=DmxOpenAICompatibleConfig(
            debug_enabled=True,
            debug_output_root=str(tmp_path),
        )
    )
    with pytest.raises(DMXAPIBackendError) as exc_info:
        backend.generate("prompt", instance=instance, state=state)

    assert exc_info.value.metadata["http_status"] == 504
    assert "timeout" in exc_info.value.metadata["http_error_body"]
    debug_path = Path(exc_info.value.metadata["debug_record_path"])
    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert payload["failure"]["http_status"] == 504
    assert "timeout" in payload["failure"]["http_error_body"]
