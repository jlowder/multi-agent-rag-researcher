"""Unit tests for the P1-2 goal-driven investigator in retriever_agent.

All model/retrieval calls are monkeypatched — no live LLM, no Qdrant, no
Tavily. Covers: legacy path untouched (no sufficiency call), goal path
termination conditions (sufficiency, round cap, 0 new chunks, unparseable
sufficiency), and the pure _apply_budget helper.
"""

import importlib
import types

# NOTE: worker_agents/__init__.py re-exports the retriever_agent FUNCTION,
# shadowing the module name — import the module object explicitly so we can
# monkeypatch its internals (retrieve_document / web_search / run_model).
ra = importlib.import_module("worker_agents.retriever_agent")
from worker_agents.retriever_agent import (
    ResearchEvidencePack,
    SufficiencyReport,
    _apply_budget,
    retriever_agent,
)


def _legacy_response(output_text="legacy summary"):
    """Response shaped like a non-tool-call turn of the legacy loop."""
    return types.SimpleNamespace(
        id="resp_legacy",
        output=[],  # no function_call items -> legacy loop breaks immediately
        output_text=output_text,
        parsed=None,
        output_parsed=None,
    )


def _suff_response(report: dict):
    return types.SimpleNamespace(
        id="resp_suff",
        output=[],
        output_text="",
        parsed=None,
        output_parsed=report,
    )


def _doc_response(chunk_ids):
    return {
        "query": "q",
        "summary": "doc ok",
        "chunks": [
            {
                "document_name": "a.pdf",
                "document_title": "Doc A",
                "page_number": i + 1,
                "chunk_id": cid,
                "citation": f"[a.pdf p.{i + 1}]",
                "content": f"content {cid}",
                "score": round(0.4 + 0.01 * i, 2),
            }
            for i, cid in enumerate(chunk_ids)
        ],
    }


def _web_response(urls):
    return {
        "query": "q",
        "results": [
            {
                "title": f"T {i}",
                "url": u,
                "content": f"web content {u}",
                "score": round(0.3 + 0.01 * i, 2),
            }
            for i, u in enumerate(urls)
        ],
    }


def _patch_goal_machinery(monkeypatch, doc_chunks_fn, web_urls_fn, suff_fn):
    """doc_chunks_fn(query) -> list of chunk_ids; web_urls_fn(query) -> urls;
    suff_fn() -> response returned for the sufficiency call."""
    doc_queries = []
    web_queries = []
    suff_calls = []

    def fake_retrieve_document(query, per_doc_topk=8, score_threshold=0.2):
        doc_queries.append(query)
        return _doc_response(doc_chunks_fn(query))

    def fake_web_search(query, num_results=5):
        web_queries.append(query)
        return _web_response(web_urls_fn(query))

    def fake_run_model(**kwargs):
        if kwargs.get("text_format") is SufficiencyReport:
            suff_calls.append(kwargs)
            return suff_fn()
        raise AssertionError("unexpected non-sufficiency run_model call in goal mode")

    monkeypatch.setattr(ra, "retrieve_document", fake_retrieve_document)
    monkeypatch.setattr(ra, "web_search", fake_web_search)
    monkeypatch.setattr(ra, "run_model", fake_run_model)
    return doc_queries, web_queries, suff_calls


class TestLegacyPath:
    def test_empty_research_goal_uses_legacy_loop_without_sufficiency(self, monkeypatch):
        calls = []

        def fake_run_model(**kwargs):
            calls.append(kwargs)
            return _legacy_response()

        monkeypatch.setattr(ra, "run_model", fake_run_model)
        monkeypatch.setattr(ra, "get_indexed_document_catalog", lambda: [])

        pack = retriever_agent("standard query")

        assert isinstance(pack, ResearchEvidencePack)
        assert pack.summary == "legacy summary"
        assert pack.route_used == "none"
        # Standard-mode serialized shape unchanged: no "sufficiency" key.
        assert "sufficiency" not in pack.model_dump()

        # Exactly the legacy single tool-loop call: tools on, no text_format,
        # agent "retriever" — and no sufficiency call anywhere.
        assert len(calls) == 1
        assert calls[0]["tools"]
        assert calls[0].get("text_format") is None
        assert calls[0]["agent_name"] == "retriever"

    def test_default_kwargs_are_legacy(self):
        import inspect

        sig = inspect.signature(retriever_agent)
        assert sig.parameters["research_goal"].default == ""
        assert sig.parameters["max_rounds"].default == 3
        assert sig.parameters["budget_doc"].default == 10
        assert sig.parameters["budget_web"].default == 5


class TestGoalPath:
    def test_stops_on_sufficiency(self, monkeypatch):
        def doc_chunks(query):
            return ["c1", "c2"] if query == "the goal" else ["x1"]

        doc_queries, web_queries, suff_calls = _patch_goal_machinery(
            monkeypatch,
            doc_chunks,
            lambda q: ["https://a.org"],
            lambda: _suff_response(
                {"is_sufficient": True, "missing_aspects": [], "follow_up_queries": []}
            ),
        )

        pack = retriever_agent(
            "wrapper query", research_goal="the goal", max_rounds=3, budget_doc=10, budget_web=5
        )

        # Round 1 retrieves the goal via both routes; then sufficiency says
        # sufficient -> stop. No round 2.
        assert doc_queries == ["the goal"]
        assert web_queries == ["the goal"]
        assert len(suff_calls) == 1
        assert suff_calls[0]["agent_name"] == "sufficiency"
        assert pack.sufficiency["is_sufficient"] is True
        assert len(pack.document_evidence["chunks"]) == 2
        assert "the goal" in pack.summary

    def test_stops_at_round_cap(self, monkeypatch):
        doc_queries, web_queries, suff_calls = _patch_goal_machinery(
            monkeypatch,
            lambda q: [f"uniq-{q}-{1}", f"uniq-{q}-{2}"],  # always-new chunks
            lambda q: [],
            lambda: _suff_response(
                {
                    "is_sufficient": False,
                    "missing_aspects": ["more depth"],
                    "follow_up_queries": ["fu1", "fu2", "fu3"],
                }
            ),
        )

        pack = retriever_agent(
            "wrapper", research_goal="goal", max_rounds=3, budget_doc=10, budget_web=5
        )

        # Round 1: goal. Round 2: follow-ups capped at 2 (fu3 dropped).
        # Round 3: its follow-ups (same queries -> dedup to 0 new); final
        # round -> no sufficiency call. 6 unique chunks total (2 per query x
        # 3 distinct queries), under the doc budget.
        assert doc_queries[0] == "goal"
        assert doc_queries[1:3] == ["fu1", "fu2"]
        assert "fu3" not in doc_queries
        assert len(doc_queries) == 5  # 1 + 2 + 2
        assert len(suff_calls) == 2  # rounds 1 and 2 only
        # Follow-ups from the last sufficiency were never retrieved...
        assert pack.sufficiency["is_sufficient"] is False
        # ...and the accumulated chunks fit the budget.
        assert len(pack.document_evidence["chunks"]) == 6

    def test_stops_on_zero_new_chunks(self, monkeypatch):
        doc_queries, web_queries, suff_calls = _patch_goal_machinery(
            monkeypatch,
            lambda q: ["same1", "same2"],  # identical chunks every query
            lambda q: [],
            lambda: _suff_response(
                {
                    "is_sufficient": False,
                    "missing_aspects": ["more"],
                    "follow_up_queries": ["fu"],
                }
            ),
        )

        retriever_agent(
            "wrapper", research_goal="goal", max_rounds=3, budget_doc=10, budget_web=5
        )

        # Round 1 adds 2 chunks; round 2 (follow-up) adds 0 new after dedup
        # -> diminishing-returns stop even though sufficiency suggested a
        # follow-up. No round 3.
        assert doc_queries == ["goal", "fu"]
        assert len(suff_calls) == 2

    def test_unparseable_sufficiency_stops_after_first_round(self, monkeypatch):
        doc_queries, web_queries, suff_calls = _patch_goal_machinery(
            monkeypatch,
            lambda q: ["c1"],
            lambda q: [],
            lambda: types.SimpleNamespace(
                id="resp_suff", output=[], output_text="no json at all",
                parsed=None, output_parsed=None,
            ),
        )

        pack = retriever_agent(
            "wrapper", research_goal="goal", max_rounds=3, budget_doc=10, budget_web=5
        )

        # Parse failure -> is_sufficient=False, follow_ups=[] -> stop.
        assert doc_queries == ["goal"]
        assert len(suff_calls) == 1
        assert pack.sufficiency["is_sufficient"] is False
        assert pack.sufficiency["follow_up_queries"] == []
        assert pack.sufficiency["source"] == "fallback"

    def test_budget_truncates_goal_pack(self, monkeypatch):
        _patch_goal_machinery(
            monkeypatch,
            lambda q: [f"chunk-{i}" for i in range(12)],  # 12 chunks > budget 10
            lambda q: [f"https://w{i}.org" for i in range(7)],  # 7 > budget 5
            lambda: _suff_response(
                {"is_sufficient": True, "missing_aspects": [], "follow_up_queries": []}
            ),
        )

        pack = retriever_agent(
            "wrapper", research_goal="goal", max_rounds=1, budget_doc=10, budget_web=5
        )

        assert len(pack.document_evidence["chunks"]) == 10
        assert len(pack.web_evidence["results"]) == 5
        # Top-scored kept: scores ascend with index in the fakes.
        doc_scores = [c["score"] for c in pack.document_evidence["chunks"]]
        assert doc_scores == sorted(doc_scores, reverse=True)
        web_scores = [r["score"] for r in pack.web_evidence["results"]]
        assert web_scores == sorted(web_scores, reverse=True)
        # max_rounds=1: single round, no sufficiency call at all.
        assert "sufficiency" not in pack.model_dump()


class TestApplyBudget:
    def test_truncates_and_orders_by_score_desc(self):
        pack = {
            "document_evidence": {
                "query": "q",
                "summary": "s",
                "chunks": [
                    {"chunk_id": "a", "score": 0.2},
                    {"chunk_id": "b", "score": 0.9},
                    {"chunk_id": "c", "score": 0.5},
                    {"chunk_id": "d", "score": 0.7},
                ],
            },
            "web_evidence": {
                "query": "q",
                "summary": "s",
                "results": [
                    {"url": "u1", "score": 0.1},
                    {"url": "u2", "score": 0.8},
                    {"url": "u3", "score": 0.3},
                ],
            },
        }
        out = _apply_budget(pack, budget_doc=2, budget_web=5)
        assert [c["chunk_id"] for c in out["document_evidence"]["chunks"]] == ["b", "d"]
        assert [r["url"] for r in out["web_evidence"]["results"]] == ["u2", "u3", "u1"]
        # Pure: input untouched.
        assert len(pack["document_evidence"]["chunks"]) == 4
        assert [c["chunk_id"] for c in pack["document_evidence"]["chunks"]] == [
            "a",
            "b",
            "c",
            "d",
        ]

    def test_tolerates_missing_or_empty_evidence(self):
        assert _apply_budget({}, 10, 5) == {}
        out = _apply_budget({"document_evidence": None, "web_evidence": {}}, 10, 5)
        assert out["document_evidence"] is None
        assert out["web_evidence"] == {}
        # Zero budgets empty the lists.
        out2 = _apply_budget(
            {
                "document_evidence": {"chunks": [{"chunk_id": "a", "score": 1.0}]},
                "web_evidence": {"results": [{"url": "u", "score": 1.0}]},
            },
            budget_doc=0,
            budget_web=0,
        )
        assert out2["document_evidence"]["chunks"] == []
        assert out2["web_evidence"]["results"] == []
