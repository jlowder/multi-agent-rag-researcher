"""Caller behavior once run_model's tolerant parse yields a model or None.

run_model no longer raises on preambled content: it returns the response
with output_parsed set to the validated model (success) or None. These
tests pin that each mapped text_format caller then behaves exactly as its
documented fallback always did:
  * verifier  : None + no JSON in text  -> neutral critic pass
  * sufficiency: None + no JSON in text -> insufficient, stop
  * decomposer : None                    -> embedded-JSON fallback, else
                                             single-sub-question fallback plan
and that a validated model instance is used directly on success.
"""

import importlib
import json
from dataclasses import fields
from types import SimpleNamespace

import worker_agents.decomposition_agent as decomposition_agent
# worker_agents/__init__ re-exports same-NAMED FUNCTIONS (retriever_agent,
# verifier_agent) that shadow the module attributes — import the real
# module objects via importlib so monkeypatch hits the right globals.
retriever_agent = importlib.import_module("worker_agents.retriever_agent")
verifier_agent = importlib.import_module("worker_agents.verifier_agent")


class _Cfg:
    default_model = "test-model"

    def get_reasoning_effort(self, agent_name: str) -> str:
        return "low"

    def get_max_output_tokens(self, agent_name: str) -> int:
        return 1000

    def get_agent_config(self, agent_name: str):
        return SimpleNamespace(endpoint="http://test", api_key="k", model="test-model")


def _resp(output_parsed=None, output_text: str = ""):
    return SimpleNamespace(
        output=[], output_parsed=output_parsed, output_text=output_text
    )


# ---------------------------------------------------------------------------
# verifier_agent.verification_critic
# ---------------------------------------------------------------------------


def test_critic_none_parse_and_prose_uses_neutral_pass(monkeypatch):
    monkeypatch.setattr(verifier_agent, "get_config", lambda: _Cfg())
    monkeypatch.setattr(
        verifier_agent, "run_model",
        lambda **kw: _resp(None, "I could not finish evaluating the report."),
    )
    out = verifier_agent.verification_critic("query", "draft", "evidence", ["s1", "s2"])
    assert out["source"] == "fallback"
    assert out["is_supported"] is True
    assert {e["section_id"] for e in out["per_section"]} == {"s1", "s2"}
    assert all(e["grounded"] for e in out["per_section"])


def test_critic_none_parse_but_json_in_text_uses_existing_fallback(monkeypatch):
    monkeypatch.setattr(verifier_agent, "get_config", lambda: _Cfg())
    payload = {
        "is_supported": False,
        "unsupported_claims": ["claim X"],
        "per_section": [
            {
                "section_id": "s1",
                "grounded": False,
                "depth_ok": True,
                "citation_density_ok": True,
                "gaps": ["thin"],
                "expand_queries": [],
            }
        ],
        "confidence_level": "low",
    }
    text = "Let me evaluate.\n" + json.dumps(payload) + "\nDone."
    monkeypatch.setattr(
        verifier_agent, "run_model", lambda **kw: _resp(None, text)
    )
    out = verifier_agent.verification_critic("query", "draft", "evidence", ["s1"])
    assert out["source"] == "json-fallback"
    assert out["is_supported"] is False
    assert out["per_section"][0]["gaps"] == ["thin"]


def test_critic_structured_model_instance_used_directly(monkeypatch):
    monkeypatch.setattr(verifier_agent, "get_config", lambda: _Cfg())
    report = verifier_agent.VerificationReport(
        is_supported=False,
        unsupported_claims=["claim X"],
        per_section=[
            verifier_agent.PerSectionReport(
                section_id="s1", grounded=False, gaps=["thin"]
            )
        ],
        confidence_level="low",
    )
    monkeypatch.setattr(verifier_agent, "run_model", lambda **kw: _resp(report))
    out = verifier_agent.verification_critic("query", "draft", "evidence", ["s1"])
    assert out["source"] == "structured"
    assert out["is_supported"] is False
    assert out["unsupported_claims"] == ["claim X"]


# ---------------------------------------------------------------------------
# retriever_agent._evaluate_sufficiency
# ---------------------------------------------------------------------------


def test_sufficiency_none_parse_and_prose_stops_insufficient(monkeypatch):
    monkeypatch.setattr(retriever_agent, "get_config", lambda: _Cfg())
    monkeypatch.setattr(
        retriever_agent, "run_model",
        lambda **kw: _resp(None, "I think the evidence looks okay-ish overall."),
    )
    out = retriever_agent._evaluate_sufficiency("goal", None, None)
    assert out == {
        "is_sufficient": False,
        "missing_aspects": [],
        "follow_up_queries": [],
        "source": "fallback",
    }


def test_sufficiency_structured_model_instance_used_directly(monkeypatch):
    monkeypatch.setattr(retriever_agent, "get_config", lambda: _Cfg())
    report = retriever_agent.SufficiencyReport(
        is_sufficient=True, follow_up_queries=[]
    )
    monkeypatch.setattr(retriever_agent, "run_model", lambda **kw: _resp(report))
    out = retriever_agent._evaluate_sufficiency("goal", None, None)
    assert out["is_sufficient"] is True
    assert out["source"] == "structured"


def test_sufficiency_prompt_carries_fill_in_skeleton():
    assert '"is_sufficient"' in retriever_agent.SUFFICIENCY_INSTRUCTIONS
    assert '"missing_aspects"' in retriever_agent.SUFFICIENCY_INSTRUCTIONS
    assert '"follow_up_queries"' in retriever_agent.SUFFICIENCY_INSTRUCTIONS


# ---------------------------------------------------------------------------
# decomposition_agent.decompose_query
# ---------------------------------------------------------------------------

PLAN_DICT = {
    "is_simple": False,
    "report_title": "T",
    "sub_questions": [
        {
            "id": "sq1",
            "question": "q1?",
            "angle": "a1",
            "expected_sources": "both",
            "priority": 2,
        }
    ],
}


def test_decomposer_none_parse_preambled_json_uses_json_fallback(monkeypatch):
    monkeypatch.setattr(decomposition_agent, "get_config", lambda: _Cfg())
    # A decoy object in the preamble must not steal the (larger) plan.
    text = (
        'Sure! A note: {"foo": 1} — the plan is:\n'
        + json.dumps(PLAN_DICT)
        + "\nHope that helps."
    )
    monkeypatch.setattr(
        decomposition_agent, "run_model", lambda **kw: _resp(None, text)
    )
    out = decomposition_agent.decompose_query("What is X?", [])
    assert out["source"] == "json-fallback"
    assert out["sub_questions"][0]["question"] == "q1?"


def test_decomposer_none_parse_and_prose_uses_fallback_plan(monkeypatch):
    monkeypatch.setattr(decomposition_agent, "get_config", lambda: _Cfg())
    monkeypatch.setattr(
        decomposition_agent, "run_model",
        lambda **kw: _resp(None, "I cannot decompose this, sorry."),
    )
    out = decomposition_agent.decompose_query("What is X?", [])
    assert out["source"] == "fallback"
    assert len(out["sub_questions"]) == 1


def test_decomposer_structured_model_instance_used_directly(monkeypatch):
    monkeypatch.setattr(decomposition_agent, "get_config", lambda: _Cfg())
    plan = decomposition_agent.ResearchPlan.model_validate(
        {"is_simple": False, "report_title": "T", "sub_questions": [
            {
                "id": "sq1",
                "question": "q1?",
                "angle": "a1",
                "expected_sources": "both",
                "priority": 2,
            }
        ]}
    )
    monkeypatch.setattr(
        decomposition_agent, "run_model", lambda **kw: _resp(plan)
    )
    out = decomposition_agent.decompose_query("What is X?", [])
    assert out["source"] == "structured"
    assert out["sub_questions"][0]["question"] == "q1?"


def test_sufficiency_token_cap_raised():
    from utils.config import Config

    cap = next(
        f for f in fields(Config) if f.name == "sufficiency_max_output_tokens"
    )
    assert cap.default == 2000
