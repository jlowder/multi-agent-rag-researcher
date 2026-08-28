"""Unit tests for the P1-1 Decomposer (no live LLM; run_model is monkeypatched)."""

import json

from worker_agents import decomposition_agent
from worker_agents.decomposition_agent import (
    ResearchPlan,
    SubQuestion,
    _extract_json_object,
    decompose_query,
)


class _FakeResponse:
    """Mimics the Responses API parse result surface the agent consumes."""

    def __init__(self, parsed=None, output_text=""):
        self.output_parsed = parsed
        self.output_text = output_text
        self.parsed = None  # intentionally absent-style attr


def _valid_plan() -> ResearchPlan:
    return ResearchPlan(
        is_simple=False,
        sub_questions=[
            SubQuestion(
                id="sq1",
                question="What is genetic programming and how does it work?",
                angle="definition and core mechanics",
                expected_sources="doc",
                priority=1,
            ),
            SubQuestion(
                id="sq2",
                question="What are the major variants and recent advances?",
                angle="variants and state of the art",
                expected_sources="both",
                priority=2,
            ),
        ],
    )


class TestDecomposeQuery:
    def test_structured_plan_passes_through(self, monkeypatch):
        calls = []

        def fake_run_model(**kwargs):
            calls.append(kwargs)
            return _FakeResponse(parsed=_valid_plan())

        monkeypatch.setattr(decomposition_agent, "run_model", fake_run_model)
        plan = decompose_query(
            "genetic programming overview",
            [{"document_name": "a.pdf", "document_title": "Document A"}],
        )

        assert plan["source"] == "structured"
        assert plan["is_simple"] is False
        assert len(plan["sub_questions"]) == 2
        assert plan["sub_questions"][0]["id"] == "sq1"
        assert plan["sub_questions"][1]["expected_sources"] == "both"

        # Config wiring: structured call, decomposer agent, config values.
        assert calls[0]["text_format"] is ResearchPlan
        assert calls[0]["agent_name"] == "decomposer"
        assert calls[0]["reasoning_effort"] in ("low", "medium", "high")
        assert calls[0]["max_output_tokens"] > 0
        # The doc catalog is fed to the prompt.
        assert "Document A" in calls[0]["input_data"]
        assert "a.pdf" in calls[0]["input_data"]
        assert "genetic programming overview" in calls[0]["input_data"]

    def test_empty_catalog_is_allowed(self, monkeypatch):
        monkeypatch.setattr(
            decomposition_agent,
            "run_model",
            lambda **kw: _FakeResponse(parsed=_valid_plan()),
        )
        plan = decompose_query("some query", [])
        assert plan["source"] == "structured"

    def test_garbage_text_uses_final_fallback(self, monkeypatch):
        monkeypatch.setattr(
            decomposition_agent,
            "run_model",
            lambda **kw: _FakeResponse(
                parsed=None, output_text="I refuse to output JSON, sorry."
            ),
        )
        plan = decompose_query("genetic programming overview", [])
        assert plan["source"] == "fallback"
        assert plan["is_simple"] is True
        assert len(plan["sub_questions"]) == 1
        sq = plan["sub_questions"][0]
        assert sq["id"] == "sq1"
        assert sq["question"] == "genetic programming overview"
        assert sq["angle"] == "original query"
        assert sq["expected_sources"] == "both"
        assert sq["priority"] == 3

    def test_json_embedded_in_text_uses_json_fallback(self, monkeypatch):
        payload = json.dumps(
            {
                "is_simple": False,
                "sub_questions": [
                    {
                        "id": "sq1",
                        "question": "What is GP?",
                        "angle": "definition",
                        "expected_sources": "doc",
                        "priority": 1,
                    }
                ],
            }
        )
        monkeypatch.setattr(
            decomposition_agent,
            "run_model",
            lambda **kw: _FakeResponse(parsed=None, output_text=f"Sure, here you go:\n{payload}"),
        )
        plan = decompose_query("genetic programming", [])
        assert plan["source"] == "json-fallback"
        assert plan["is_simple"] is False
        assert len(plan["sub_questions"]) == 1
        assert plan["sub_questions"][0]["question"] == "What is GP?"

    def test_llm_exception_never_raises(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("server 507 overloaded")

        monkeypatch.setattr(decomposition_agent, "run_model", boom)
        plan = decompose_query("genetic programming", [])
        assert plan["source"] == "fallback"
        assert plan["is_simple"] is True
        assert len(plan["sub_questions"]) == 1

    def test_structured_exception_reasks_plain_text_and_extracts_json(self, monkeypatch):
        # Live Ornith failure mode: responses.parse() raises because the
        # model prepends preamble prose to its JSON; the plain re-ask then
        # yields "<preamble>\n{json}" which the extractor must recover.
        payload = json.dumps(
            {
                "is_simple": False,
                "sub_questions": [
                    {
                        "id": "sq1",
                        "question": "What is GP?",
                        "angle": "definition",
                        "expected_sources": "both",
                        "priority": 1,
                    }
                ],
            }
        )
        calls = []

        def fake_run_model(**kwargs):
            calls.append(kwargs)
            if kwargs.get("text_format") is not None:
                raise ValueError(
                    "ValidationError: Invalid JSON: expected value at line 1 column 1"
                )
            return _FakeResponse(
                parsed=None,
                output_text=(
                    "The user wants a comprehensive report. Here is the plan:\n"
                    + payload
                ),
            )

        monkeypatch.setattr(decomposition_agent, "run_model", fake_run_model)
        plan = decompose_query("genetic programming", [])
        assert plan["source"] == "json-fallback"
        assert len(plan["sub_questions"]) == 1
        assert plan["sub_questions"][0]["question"] == "What is GP?"
        assert len(calls) == 2  # one structured attempt + one plain re-ask


class TestExtractJsonObject:
    def test_trailing_data_after_json_is_ignored(self):
        payload = {"is_simple": True, "sub_questions": []}
        text = ("preamble prose\n" + json.dumps(payload)
                + "\nAnd that is the plan. {trailing note}")
        assert _extract_json_object(text) == payload

    def test_no_braces_or_broken_json_returns_none(self):
        assert _extract_json_object("no braces here") is None
        assert _extract_json_object("{broken json") is None
        assert _extract_json_object("") is None
        # A JSON array is not an object.
        assert _extract_json_object("[1, 2]") is None


class TestNormalizePlanCandidate:
    def test_dict_keyed_sub_questions_with_renamed_question(self, monkeypatch):
        # Live Ornith shape drift: sub_questions as a dict keyed by id with
        # `sub_question` instead of `question`, no angle/priority.
        payload = json.dumps(
            {
                "is_simple": False,
                "sub_questions": {
                    "sq1": {"sub_question": "What is GP?", "expected_sources": "doc"},
                    "sq2": {"sub_question": "Who are the pioneers?"},
                },
            }
        )
        monkeypatch.setattr(
            decomposition_agent,
            "run_model",
            lambda **kw: _FakeResponse(
                parsed=None, output_text="Plan:\n" + payload
            ),
        )
        plan = decompose_query("genetic programming", [])
        assert plan["source"] == "json-fallback"
        assert [sq["id"] for sq in plan["sub_questions"]] == ["sq1", "sq2"]
        assert plan["sub_questions"][0]["question"] == "What is GP?"
        assert plan["sub_questions"][0]["angle"] == "n/a"
        assert plan["sub_questions"][0]["priority"] == 3
        assert plan["sub_questions"][0]["expected_sources"] == "doc"
        assert plan["sub_questions"][1]["expected_sources"] == "both"

    def test_list_missing_angle_gets_default(self, monkeypatch):
        # Second live Ornith variant: proper list, `angle` omitted.
        payload = json.dumps(
            {
                "is_simple": False,
                "sub_questions": [
                    {"id": "sq1", "question": "What is GP?",
                     "heading": "Definition", "expected_sources": "both",
                     "priority": 1},
                ],
            }
        )
        monkeypatch.setattr(
            decomposition_agent,
            "run_model",
            lambda **kw: _FakeResponse(parsed=None, output_text=payload),
        )
        plan = decompose_query("genetic programming", [])
        assert plan["source"] == "json-fallback"
        assert plan["sub_questions"][0]["angle"] == "n/a"
        assert plan["sub_questions"][0]["priority"] == 1

    def test_structured_output_invalid_falls_back(self, monkeypatch):
        # Parsed value present but fails schema validation -> text fallback
        # path (no JSON in text) -> final fallback.
        monkeypatch.setattr(
            decomposition_agent,
            "run_model",
            lambda **kw: _FakeResponse(
                parsed={"is_simple": "not-a-bool", "sub_questions": "nope"},
                output_text="no json here either",
            ),
        )
        plan = decompose_query("my query", [])
        assert plan["source"] == "fallback"
        assert plan["is_simple"] is True
        assert plan["sub_questions"][0]["question"] == "my query"


class TestGenerateReportTitle:
    def test_strips_whitespace_and_quotes(self, monkeypatch):
        def fake(*args, **kwargs):
            return _FakeResponse(output_text="  Genetic Programming Report\n")

        monkeypatch.setattr(decomposition_agent, "run_model", fake)
        assert decomposition_agent.generate_report_title("some request") == "Genetic Programming Report"

    def test_call_failure_returns_empty(self, monkeypatch):
        def fake(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(decomposition_agent, "run_model", fake)
        assert decomposition_agent.generate_report_title("some request") == ""

    def test_empty_query_returns_empty_without_calling(self, monkeypatch):
        def fake(*args, **kwargs):
            raise AssertionError("run_model must not be called for an empty query")

        monkeypatch.setattr(decomposition_agent, "run_model", fake)
        assert decomposition_agent.generate_report_title("   ") == ""

    def test_uses_32_tokens_and_low_effort(self, monkeypatch):
        seen = {}

        def fake(*args, **kwargs):
            seen.update(kwargs)
            return _FakeResponse(output_text="T")

        monkeypatch.setattr(decomposition_agent, "run_model", fake)
        decomposition_agent.generate_report_title("req")
        assert seen["max_output_tokens"] == 32
        assert seen["reasoning_effort"] == "low"
        assert seen["agent_name"] == "decomposer"
