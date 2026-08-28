"""Tests for the JSON output mode of worker_agents/writer_agent.py.

Plain pytest, no conftest: run from the repo root with
    ./venv/bin/python -m pytest tests/test_writer_json.py -q

Follows the repo idiom from tests/test_deep_pipeline.py: monkeypatch
wmod.run_model with a fake returning a canned output_text. No network,
no LLM calls.
"""

import importlib
import json
import re

import pytest

from models import (
    Metadata,
    QualityMetrics,
    Report,
    ResearchReport,
    Section,
    find_unresolvable_citations,
)

wmod = importlib.import_module("worker_agents.writer_agent")


class _FakeResponse:
    """Stand-in for the run_model response object (output_text plus the
    optional truncation signals the writer checks)."""

    def __init__(self, output_text: str, status=None, incomplete_details=None):
        self.output_text = output_text
        self.status = status
        self.incomplete_details = incomplete_details


class _FakeConfig:
    """Config stand-in with a small, known max_output_tokens for 2x asserts."""

    def get_reasoning_effort(self, agent_name: str) -> str:
        return "medium"

    def get_max_output_tokens(self, agent_name: str) -> int:
        return 1000


@pytest.fixture(autouse=True)
def _reset_writer_retry_budget():
    # The truncation-retry budget is module-global; keep tests isolated.
    wmod.reset_writer_retry_budget()
    yield
    wmod.reset_writer_retry_budget()


def _patch_run_model(monkeypatch, text: str):
    """Patch wmod.run_model to return _FakeResponse(text); record kwargs."""
    calls = []

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return _FakeResponse(text)

    monkeypatch.setattr(wmod, "run_model", fake)
    return calls


def _patch_run_model_seq(monkeypatch, responses: list):
    """Patch wmod.run_model to return the given responses in order; record
    kwargs per call. Use of a 2nd response is the retry under test."""
    calls = []
    it = iter(responses)

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return next(it)

    monkeypatch.setattr(wmod, "run_model", fake)
    return calls


def _section_kwargs(**overrides) -> dict:
    kwargs = dict(
        user_query="Explain fusion energy",
        outline="## Definition & Background\n## Synthesis",
        section_heading="Market",
        section_context="Cover market dynamics.",
        evidence_text="[D1] Evidence line one.",
        output_format="json",
    )
    kwargs.update(overrides)
    return kwargs


def _write_section(monkeypatch, text: str, **overrides):
    kwargs = dict(
        user_query="Explain fusion energy",
        outline="## Definition & Background\n## Synthesis",
        section_heading="Market",
        section_context="Cover market dynamics.",
        evidence_text="[D1] Evidence line one.",
    )
    kwargs.update(overrides)
    calls = _patch_run_model(monkeypatch, text)
    return wmod.write_section(**kwargs), calls


def _run_writer(monkeypatch, text: str, user_query: str = "Explain fusion energy", **overrides):
    calls = _patch_run_model(monkeypatch, text)
    return wmod.writer_agent(user_query=user_query, evidence_text="[D1] Evidence.", **overrides), calls


# ---------------------------------------------------------------------------
# Canned model outputs (one distinct text per test family)
# ---------------------------------------------------------------------------

SECTION_JSON = (
    '{"id": "market", "heading": "Market",'
    ' "blocks": [{"type": "paragraph",'
    ' "spans": [{"text": "Facts here.", "citations": ["D1"]}]}]}'
)

SECTION_JSON_FENCED = "```json\n" + SECTION_JSON + "\n```"

SECTION_JSON_NO_ID = (
    '{"heading": "Market Landscape",'
    ' "blocks": [{"type": "paragraph",'
    ' "spans": [{"text": "Facts here.", "citations": ["D1"]}]}]}'
)

SECTION_JSON_BAD_BLOCK = (
    '{"id": "market", "heading": "Market",'
    ' "blocks": [{"type": "bogus_block",'
    ' "spans": [{"text": "x", "citations": []}]}]}'
)

SECTION_JSON_INVENTED_KEY = (
    '{"id": "market", "heading": "Market",'
    ' "blocks": [{"type": "paragraph",'
    ' "spans": [{"text": "Half sourced, half invented.", "citations": ["D1", "D99"]}]}]}'
)

SYNTHESIS_JSON = (
    '{"heading": "Synthesis", "id": "synthesis",'
    ' "blocks": [{"type": "paragraph",'
    ' "spans": [{"text": "The sections agree.", "citations": ["D1"]}]}]}'
)

REPORT_JSON = (
    '{"title": "T", "executive_summary": ["p1"],'
    ' "sections": [{"id": "s", "heading": "S",'
    ' "blocks": [{"type": "paragraph",'
    ' "spans": [{"text": "x", "citations": []}]}]}]}'
)

REPORT_JSON_NO_TITLE = (
    '{"executive_summary": ["p1"],'
    ' "sections": [{"id": "s", "heading": "S",'
    ' "blocks": [{"type": "paragraph",'
    ' "spans": [{"text": "x", "citations": []}]}]}]}'
)

# ---------------------------------------------------------------------------
# A. _extract_json_object unit tests
# ---------------------------------------------------------------------------


def test_extract_plain_object():
    assert wmod._extract_json_object('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_extract_json_fenced():
    text = '```json\n{"a": 1}\n```'
    assert wmod._extract_json_object(text) == {"a": 1}


def test_extract_plain_fenced():
    text = '```\n{"a": 1}\n```'
    assert wmod._extract_json_object(text) == {"a": 1}


def test_extract_commentary_around_object():
    text = 'Here is the section you asked for:\n{"a": 1}\nHope that helps!'
    assert wmod._extract_json_object(text) == {"a": 1}


def test_extract_no_braces_returns_none():
    assert wmod._extract_json_object("no braces here") is None
    assert wmod._extract_json_object("not json at all") is None
    assert wmod._extract_json_object("") is None


def test_extract_invalid_json_returns_none():
    assert wmod._extract_json_object('{"a": ') is None


# ---------------------------------------------------------------------------
# B. write_section JSON mode
# ---------------------------------------------------------------------------


def test_write_section_json_returns_section(monkeypatch):
    section, _ = _write_section(monkeypatch, SECTION_JSON, output_format="json")
    assert isinstance(section, Section)
    assert section.heading == "Market"
    assert section.id == "market"
    assert section.blocks[0].spans[0].citations == ["D1"]


def test_write_section_json_uses_json_instructions(monkeypatch):
    _, calls = _write_section(monkeypatch, SECTION_JSON, output_format="json")
    assert "ONE JSON object" in calls[0]["instructions"]
    assert calls[0]["agent_name"] == "writer"


def test_write_section_json_fenced_still_parses(monkeypatch):
    section, _ = _write_section(monkeypatch, SECTION_JSON_FENCED, output_format="json")
    assert isinstance(section, Section)
    assert section.heading == "Market"
    assert section.blocks[0].spans[0].citations == ["D1"]


def test_write_section_json_malformed_falls_back_to_empty(monkeypatch):
    text = "I will write the section now, but I do not have any JSON on hand."
    section, _ = _write_section(monkeypatch, text, output_format="json")
    assert isinstance(section, Section)
    assert section.heading == "Market"
    assert section.blocks == []


def test_write_section_json_bad_block_type_falls_back(monkeypatch):
    section, _ = _write_section(monkeypatch, SECTION_JSON_BAD_BLOCK, output_format="json")
    assert isinstance(section, Section)
    assert section.blocks == []


def test_write_section_json_citations_detectable(monkeypatch):
    section, _ = _write_section(monkeypatch, SECTION_JSON_INVENTED_KEY, output_format="json")
    assert section.blocks[0].spans[0].citations == ["D1", "D99"]
    wrapper = ResearchReport(
        report=Report(
            metadata=Metadata(title="t", query="q", session_id="s", generated_at="g"),
            executive_summary=[],
            sections=[section],
            sources=[],
        ),
        quality=QualityMetrics(),
    )
    assert find_unresolvable_citations(wrapper, registry={"D1": {}}) == ["D99"]


def test_write_section_markdown_mode_unchanged(monkeypatch):
    text = "## Market\n\nBody [D1]."
    result, calls = _write_section(monkeypatch, text, output_format="markdown")
    assert isinstance(result, str)
    assert result.startswith("## Market")
    assert "ONE JSON object" not in calls[0]["instructions"]  # markdown prompt used


# ---------------------------------------------------------------------------
# C. write_synthesis JSON mode
# ---------------------------------------------------------------------------


def test_write_synthesis_json_valid(monkeypatch):
    calls = _patch_run_model(monkeypatch, SYNTHESIS_JSON)
    section = wmod.write_synthesis(
        "Explain fusion energy",
        [("Definition & Background", "## Definition & Background\n\nBody [D1].")],
        output_format="json",
    )
    assert isinstance(section, Section)
    assert section.heading == "Synthesis"
    assert "ONE JSON object" in calls[0]["instructions"]


def test_write_synthesis_json_malformed_falls_back(monkeypatch):
    _patch_run_model(monkeypatch, "Just prose, no object anywhere.")
    section = wmod.write_synthesis(
        "Explain fusion energy",
        [("Definition & Background", "## Definition & Background\n\nBody [D1].")],
        output_format="json",
    )
    assert isinstance(section, Section)
    assert section.heading == "Synthesis"
    assert section.blocks == []


# ---------------------------------------------------------------------------
# D. writer_agent JSON mode
# ---------------------------------------------------------------------------


def test_writer_agent_json_valid(monkeypatch):
    report, _ = _run_writer(monkeypatch, REPORT_JSON, output_format="json")
    assert isinstance(report, Report)
    assert report.metadata.title == "T"
    assert report.metadata.report_type == "standard"
    assert report.executive_summary == ["p1"]
    assert len(report.sections) == 1
    assert report.sections[0].heading == "S"
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", report.metadata.generated_at)


def test_writer_agent_json_missing_title_falls_back(monkeypatch):
    query = "How does fusion energy work in practice?"
    report, _ = _run_writer(monkeypatch, REPORT_JSON_NO_TITLE, user_query=query, output_format="json")
    assert isinstance(report, Report)
    assert report.metadata.title == query[:100]


def test_writer_agent_json_malformed_returns_empty_report(monkeypatch):
    query = "How does fusion energy work in practice?"
    report, _ = _run_writer(
        monkeypatch,
        "Sorry, I cannot output a JSON object right now.",
        user_query=query,
        output_format="json",
    )
    assert isinstance(report, Report)
    assert report.sections == []
    assert report.metadata.title == query[:100]


# ---------------------------------------------------------------------------
# E. Guardrails
# ---------------------------------------------------------------------------


def test_write_section_invalid_format_treated_as_markdown(monkeypatch):
    text = "## Market\n\nBody."
    result, _ = _write_section(monkeypatch, text, output_format="xml")
    assert isinstance(result, str)
    assert result.startswith("## Market")


def test_write_section_json_missing_id_gets_slug(monkeypatch):
    section, _ = _write_section(
        monkeypatch,
        SECTION_JSON_NO_ID,
        section_heading="Market Landscape",
        output_format="json",
    )
    assert isinstance(section, Section)
    assert section.heading == "Market Landscape"
    assert section.id == "market-landscape"


# ---------------------------------------------------------------------------
# F. Truncation, retry budget, extraction repair, heading dedup
# ---------------------------------------------------------------------------

TRUNCATED_SECTION = (
    '{"id": "market", "heading": "Market", "blocks": '
    '[{"type": "paragraph", "spans": [{"text": "The deployment climbed'
)


def test_truncated_response_retries_at_2x_and_recovers(monkeypatch):
    monkeypatch.setattr(wmod, "get_config", lambda: _FakeConfig())
    calls = _patch_run_model_seq(
        monkeypatch,
        [
            _FakeResponse(TRUNCATED_SECTION, status="incomplete"),
            _FakeResponse(SECTION_JSON),
        ],
    )
    section = wmod.write_section(**_section_kwargs())
    assert len(calls) == 2
    assert calls[0]["max_output_tokens"] == 1000
    assert calls[1]["max_output_tokens"] == 2000
    assert isinstance(section, Section)
    assert section.blocks  # recovered draft, not the soft-fail empty section


def test_truncated_response_budget_exhausted_no_retry(monkeypatch):
    monkeypatch.setattr(wmod, "get_config", lambda: _FakeConfig())
    wmod.reset_writer_retry_budget(0)
    calls = _patch_run_model_seq(
        monkeypatch, [_FakeResponse(TRUNCATED_SECTION, status="incomplete")]
    )
    section = wmod.write_section(**_section_kwargs())
    assert len(calls) == 1  # no second call when the budget is spent
    assert section.blocks == []  # soft-fail section
    assert section.heading == "Market"
    assert section.id == "market"


def test_non_truncated_parse_failure_no_retry(monkeypatch):
    monkeypatch.setattr(wmod, "get_config", lambda: _FakeConfig())
    # Balanced braces, unparseable body: NOT a truncation signal, so no retry.
    calls = _patch_run_model_seq(
        monkeypatch, [_FakeResponse('{"a": } oops, balanced but broken')]
    )
    section = wmod.write_section(**_section_kwargs())
    assert len(calls) == 1
    assert section.blocks == []
    assert section.heading == "Market"


def test_extract_prose_with_braces_before_valid_object():
    text = 'Intro with braces: {"x": 1} done. ' + SECTION_JSON
    assert wmod._extract_json_object(text) == {
        "id": "market",
        "heading": "Market",
        "blocks": [
            {"type": "paragraph", "spans": [{"text": "Facts here.", "citations": ["D1"]}]}
        ],
    }


def test_null_id_missing_heading_repaired_from_input(monkeypatch):
    text = (
        '{"id": null, "blocks": [{"type": "paragraph",'
        ' "spans": [{"text": "Body.", "citations": []}]}]}'
    )
    section, _ = _write_section(
        monkeypatch, text, section_heading="Deep Dive", output_format="json"
    )
    assert isinstance(section, Section)
    assert section.heading == "Deep Dive"
    assert section.id == "deep-dive"
    assert len(section.blocks) == 1


def test_duplicate_leading_heading_stripped_tolerant(monkeypatch):
    # blocks[0] re-emits the section title in a case/whitespace variant.
    text = (
        '{"id": "market", "heading": "Market", "blocks": ['
        '{"type": "heading", "level": 3, "text": "  market  "},'
        '{"type": "paragraph", "spans": [{"text": "Body.", "citations": []}]}]}'
    )
    section, _ = _write_section(monkeypatch, text, output_format="json")
    assert len(section.blocks) == 1
    assert section.blocks[0].type == "paragraph"


def test_different_first_subsection_heading_kept(monkeypatch):
    text = (
        '{"id": "market", "heading": "Market", "blocks": ['
        '{"type": "heading", "level": 3, "text": "Sub Section"},'
        '{"type": "paragraph", "spans": [{"text": "Body.", "citations": []}]}]}'
    )
    section, _ = _write_section(monkeypatch, text, output_format="json")
    assert len(section.blocks) == 2
    assert section.blocks[0].type == "heading"
    assert section.blocks[0].text == "Sub Section"


def test_extract_prefers_largest_keyed_dict_over_prose_decoy():
    real = json.dumps(
        {
            "id": "market",
            "heading": "Market",
            "blocks": [
                {"type": "paragraph", "spans": [{"text": "x" * 400, "citations": []}]}
            ],
        }
    )
    # A tiny decoy with a preferred key appears BEFORE the real section.
    text = 'Here is an example: {"heading": "wrong"}\n' + real
    assert wmod._extract_json_object(text) == json.loads(real)
    # Array-wrapped decoy (its dict is nested, so not a competing object;
    # the real section still wins on size).
    text2 = 'note: [{"heading": "decoy"}]\n' + real
    assert wmod._extract_json_object(text2) == json.loads(real)
    # Two real candidates: the bigger one wins.
    small = '{"id": "a", "heading": "A", "blocks": []}'
    text3 = small + "\n" + real
    assert wmod._extract_json_object(text3)["id"] == "market"
