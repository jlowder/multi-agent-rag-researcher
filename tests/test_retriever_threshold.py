"""
Tests for the config-driven document retrieval floor (C3):
retrieve_document resolves score_threshold=None to
get_config().doc_score_threshold instead of the old hard-coded 0.2.
"""

import importlib
import types

# NOTE: worker_agents/__init__.py re-exports the retriever_agent FUNCTION,
# shadowing the module name — import the module object explicitly so we can
# monkeypatch its internals (similarity_search / get_config).
ra = importlib.import_module("worker_agents.retriever_agent")


def _fake_config(threshold: float):
    return types.SimpleNamespace(doc_score_threshold=threshold)


def test_retrieve_document_uses_config_threshold_when_unspecified(
    monkeypatch,
):
    calls = {}

    def fake_similarity_search(query, per_doc_topk=3, max_results=None,
                               score_threshold=None):
        calls["score_threshold"] = score_threshold
        return []

    monkeypatch.setattr(ra, "similarity_search", fake_similarity_search)
    monkeypatch.setattr(ra, "get_config", lambda: _fake_config(0.7))

    result = ra.retrieve_document("some query")

    assert calls["score_threshold"] == 0.7
    assert result["chunks"] == []


def test_retrieve_document_defaults_to_0_2_when_config_is_default(
    monkeypatch,
):
    calls = {}

    def fake_similarity_search(query, per_doc_topk=3, max_results=None,
                               score_threshold=None):
        calls["score_threshold"] = score_threshold
        return []

    monkeypatch.setattr(ra, "similarity_search", fake_similarity_search)
    monkeypatch.setattr(ra, "get_config", lambda: _fake_config(0.2))

    ra.retrieve_document("some query")
    assert calls["score_threshold"] == 0.2


def test_retrieve_document_explicit_threshold_wins_over_config(
    monkeypatch,
):
    calls = {}

    def fake_similarity_search(query, per_doc_topk=3, max_results=None,
                               score_threshold=None):
        calls["score_threshold"] = score_threshold
        return []

    monkeypatch.setattr(ra, "similarity_search", fake_similarity_search)
    monkeypatch.setattr(ra, "get_config", lambda: _fake_config(0.7))

    ra.retrieve_document("some query", score_threshold=0.5)
    assert calls["score_threshold"] == 0.5
