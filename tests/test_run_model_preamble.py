"""Tests for run_model's tolerant structured-output handling.

The openai SDK's responses.parse() raised on any conversational preamble
before agent fallbacks could run. run_model now calls responses.create()
and validates a shared-extractor recovery of the JSON payload, so it must:
  * never raise on malformed/preambled content (output_parsed = None),
  * return the validated model on success,
  * preserve the raw output_text and the non-text_format return contract.
"""

import json
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest
from pydantic import BaseModel, Field

import worker_agents.model_runner as model_runner
from worker_agents.model_runner import run_model


class _ParseModel(BaseModel):
    is_supported: bool = False
    notes: List[str] = Field(default_factory=list)


class _FakeResponse:
    def __init__(self, text: str):
        self.output: List[Any] = []
        self.output_text = text
        self.usage = None


class _FakeResponses:
    def __init__(self, owner: "_FakeClient"):
        self._owner = owner

    def create(self, **kwargs: Any) -> _FakeResponse:
        self._owner.calls.append(kwargs)
        return self._owner.make_response()


class _FakeClient:
    def __init__(self, text: str):
        self.calls: List[dict] = []
        self._text = text
        self.responses = _FakeResponses(self)

    def make_response(self) -> _FakeResponse:
        return _FakeResponse(self._text)


def _install(monkeypatch, text: str) -> _FakeClient:
    client = _FakeClient(text)
    monkeypatch.setattr(model_runner, "get_client", lambda **kwargs: client)
    return client


def test_preambled_content_validates_into_model(monkeypatch):
    payload = {"is_supported": True, "notes": ["grounded"]}
    text = (
        "Let me carefully evaluate the report section by section.\n"
        + json.dumps(payload)
        + "\nOverall the draft is solid; see section 31. [W45] (GNS derived features"
    )
    client = _install(monkeypatch, text)
    response = run_model(
        instructions="critic", input_data="in", model="test-model", text_format=_ParseModel
    )
    assert isinstance(response, _FakeResponse)
    assert isinstance(response.output_parsed, _ParseModel)
    assert response.output_parsed.is_supported is True
    assert response.output_parsed.notes == ["grounded"]
    # Raw text preserved untouched for caller fallbacks.
    assert response.output_text == text
    # The strict schema is still sent for servers that honor text.format.
    fmt = client.calls[0]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True


def test_pure_prose_content_yields_none_not_exception(monkeypatch):
    _install(monkeypatch, "I cannot produce a report right now, sorry.")
    response = run_model(
        instructions="x", input_data="y", model="m", text_format=_ParseModel
    )
    assert response.output_parsed is None
    assert response.output_text.startswith("I cannot")


def test_truncated_json_yields_none_not_exception(monkeypatch):
    _install(monkeypatch, 'Here is my assessment:\n{"is_supported": true, "notes": ')
    response = run_model(
        instructions="x", input_data="y", model="m", text_format=_ParseModel
    )
    assert response.output_parsed is None


def test_fenced_content_validates_into_model(monkeypatch):
    text = "```json\n" + json.dumps({"is_supported": False, "notes": ["gap"]}) + "\n```"
    _install(monkeypatch, text)
    response = run_model(
        instructions="x", input_data="y", model="m", text_format=_ParseModel
    )
    assert isinstance(response.output_parsed, _ParseModel)
    assert response.output_parsed.is_supported is False


def test_non_text_format_call_unchanged(monkeypatch):
    client = _install(monkeypatch, "hello world")
    response = run_model(instructions="x", input_data="y", model="m")
    assert isinstance(response, _FakeResponse)
    assert response.output_text == "hello world"
    assert not hasattr(response, "output_parsed")
    # No text.format injected into plain calls.
    assert "text" not in client.calls[0]


def test_output_none_glitch_still_normalized(monkeypatch):
    client = _install(monkeypatch, "")
    client.make_response = lambda: SimpleNamespace(output=None, output_text=None)
    response = run_model(
        instructions="x", input_data="y", model="m", text_format=_ParseModel
    )
    assert response.output == []
    assert response.output_parsed is None
