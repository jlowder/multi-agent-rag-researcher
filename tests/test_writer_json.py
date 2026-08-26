"""Tests for the JSON output mode of worker_agents/writer_agent.py.

Plain pytest, no conftest: run from the repo root with
    ./venv/bin/python -m pytest tests/test_writer_json.py -q

Follows the repo idiom from tests/test_deep_pipeline.py: monkeypatch
wmod.run_model with a fake returning a canned output_text. No network,
no LLM calls.
"""

import importlib
import re

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
    """Stand-in for the run_model response object (only output_text is used)."""

    def __init__(self, output_text: str):
        self.output_text = output_text


def _patch_run_model(monkeypatch, text: str):
    """Patch wmod.run_model to return _FakeResponse(text); record kwargs."""
    calls = []

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return _FakeResponse(text)

    monkeypatch.setattr(wmod, "run_model", fake)
    return calls


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
