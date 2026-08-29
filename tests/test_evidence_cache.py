"""
Tests for P2-2: per-sub-question evidence cache (memory/evidence_cache.py)
and its deep-pipeline integration (stage-2 reuse, stats, flag off).

No network, Qdrant, Tavily, or LLM calls: unit tests use a tmp DB (the
module's DB path is monkeypatched) and the pipeline tests reuse the stub
machinery from tests/test_deep_pipeline.py.
"""

import json
import shutil
from datetime import datetime, timedelta, timezone

import pytest

import deep_research_orchestrator as dpo
import memory.evidence_cache as ecache
import utils.config as uconfig
from test_deep_pipeline import (
    _CACHE_TMP_DIRS,
    _basic_env,
    _doc,
    _install_stubs,
    _web,
)
from worker_agents.retriever_agent import ResearchEvidencePack


@pytest.fixture(autouse=True)
def _cleanup_evidence_cache_tmp_dirs():
    # _install_stubs appends its temp DB dirs to test_deep_pipeline's
    # registry; that module's autouse fixture only covers its own tests.
    for d in _CACHE_TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _CACHE_TMP_DIRS.clear()
    yield


def _fresh_db(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ecache, "EVIDENCE_CACHE_DB_PATH", tmp_path / "evidence_cache.db"
    )
    monkeypatch.setattr(ecache, "_purged_this_process", False)


def _days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _pack(marker: str = "pack") -> dict:
    """The exact pack dict shape the pipeline stores (model_dump form)."""
    return {
        "query": marker,
        "route_used": "web",
        "summary": f"summary for {marker}",
        "document_evidence": {"chunks": [_doc("c1", "a.pdf", 0.9)]},
        "web_evidence": {"results": [_web("https://ex/u1", marker, 0.9)]},
    }


def _insert(session_id: str, question: str, retrieved_at: str,
            evidence: dict, sufficiency_ok: int = 1) -> None:
    with ecache.get_evidence_cache_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO evidence_cache
                (session_id, sub_topic, question, retrieved_at, evidence_json,
                 sufficiency_ok)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                ecache.normalize_question(question),
                question,
                retrieved_at,
                json.dumps(evidence),
                sufficiency_ok,
            ),
        )


def _row_count() -> int:
    conn = ecache.get_evidence_cache_connection()
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM evidence_cache").fetchone()
    finally:
        conn.close()
    return row["n"]


# ---------------------------------------------------------------------------
# Unit — save/lookup round-trip, normalization, threshold, TTL, sufficiency,
# tie-break, purge, replace
# ---------------------------------------------------------------------------

def test_save_and_lookup_exact_round_trips_pack(monkeypatch, tmp_path):
    _fresh_db(monkeypatch, tmp_path)
    pack = _pack("exact")
    ecache.save_evidence("s1", "What are the applications of GP?", pack, True)
    got = ecache.lookup_evidence("what are the applications of gp?", 30)
    assert got == pack  # json round-trips to the exact pack dict shape


def test_normalization_equivalence_hits(monkeypatch, tmp_path):
    _fresh_db(monkeypatch, tmp_path)
    assert ecache.normalize_question("What   ARE the applications, of GP?") == (
        ecache.normalize_question("what are the applications of gp")
    )
    ecache.save_evidence(
        "s1", "What   ARE the applications, of GP?", _pack("norm"), True
    )
    assert ecache.lookup_evidence("what are the applications of gp", 30) is not None
    assert ecache.jaccard(
        "What   ARE the applications, of GP?", "what are the applications of gp"
    ) == 1.0


def test_jaccard_threshold_boundaries(monkeypatch, tmp_path):
    _fresh_db(monkeypatch, tmp_path)
    base = "what are the main applications of genetic programming in scheduling"
    ecache.save_evidence("s1", base, _pack("base"), True)

    # 9/11 ≈ 0.818 token overlap → hit (≥ 0.8)
    near = "what are the main applications of genetic programming in timetabling"
    assert ecache.jaccard(base, near) == pytest.approx(9 / 11)
    assert ecache.lookup_evidence(near, 30) is not None

    # 8/13 ≈ 0.615 token overlap → miss (~0.6-0.7)
    far = ("what are the main applications of genetic programming "
           "for medical diagnosis")
    assert ecache.jaccard(base, far) == pytest.approx(8 / 13)
    assert ecache.lookup_evidence(far, 30) is None


def test_ttl_expiry(monkeypatch, tmp_path):
    _fresh_db(monkeypatch, tmp_path)
    _insert("s1", "stale question", _days_ago(31), _pack("stale"))
    _insert("s1", "fresh question", _days_ago(29), _pack("fresh"))
    # 31 days old → outside the 30-day TTL → miss; 29 days old → hit.
    assert ecache.lookup_evidence("stale question", 30) is None
    assert ecache.lookup_evidence("fresh question", 30) is not None


def test_insufficient_pack_not_reused(monkeypatch, tmp_path):
    _fresh_db(monkeypatch, tmp_path)
    ecache.save_evidence("s1", "question x", _pack("insufficient"), False)
    # sufficiency_ok = 0 → miss even on an exact question match.
    assert ecache.lookup_evidence("question x", 30) is None


def test_tie_break_most_recent_retrieved_at_wins(monkeypatch, tmp_path):
    _fresh_db(monkeypatch, tmp_path)
    # Same question text (jaccard 1.0 for both) from two different sessions.
    _insert("sess-old", "same question", _days_ago(2), _pack("old"))
    _insert("sess-new", "same question", _days_ago(1), _pack("new"))
    got = ecache.lookup_evidence("same question", 30)
    assert got is not None
    assert got["web_evidence"]["results"][0]["title"] == "new"


def test_purge_stale_removes_only_old_rows(monkeypatch, tmp_path):
    _fresh_db(monkeypatch, tmp_path)
    _insert("s1", "old topic", _days_ago(35), _pack("old"))
    _insert("s1", "new topic", _days_ago(5), _pack("new"))
    ecache.purge_stale(30)
    conn = ecache.get_evidence_cache_connection()
    try:
        rows = conn.execute(
            "SELECT question FROM evidence_cache"
        ).fetchall()
    finally:
        conn.close()
    assert [r["question"] for r in rows] == ["new topic"]


def test_insert_or_replace_same_session_and_topic(monkeypatch, tmp_path):
    _fresh_db(monkeypatch, tmp_path)
    # "q same?" and "q same" normalize to the same sub_topic.
    ecache.save_evidence("s1", "q same?", _pack("first"), True)
    ecache.save_evidence("s1", "q same", _pack("second"), True)
    assert _row_count() == 1
    got = ecache.lookup_evidence("q same", 30)
    assert got["query"] == "second"


def test_empty_question_returns_none(monkeypatch, tmp_path):
    _fresh_db(monkeypatch, tmp_path)
    assert ecache.lookup_evidence("", 30) is None
    assert ecache.lookup_evidence("   ", 30) is None
    ecache.save_evidence("s1", "   ", _pack("empty"), True)  # guarded no-op
    assert _row_count() == 0


def test_clear_evidence_cache_drops_every_row(monkeypatch, tmp_path):
    _fresh_db(monkeypatch, tmp_path)
    _insert("s1", "topic one", _days_ago(1), _pack("one"))
    _insert("s2", "topic two", _days_ago(1), _pack("two"))
    assert ecache.clear_evidence_cache() == 2
    assert _row_count() == 0
    assert ecache.lookup_evidence("topic one", 30) is None
    # Idempotent: a second clear on the empty table reports 0.
    assert ecache.clear_evidence_cache() == 0


def test_clear_evidence_cache_creates_table_when_db_absent(
    monkeypatch, tmp_path
):
    _fresh_db(monkeypatch, tmp_path)  # points at a not-yet-existing file
    assert ecache.clear_evidence_cache() == 0
    assert (tmp_path / "evidence_cache.db").exists()


# ---------------------------------------------------------------------------
# Pipeline integration — stage-2 reuse, stats, fresh-pack storage
# ---------------------------------------------------------------------------

PLAN_CACHE_JSON = json.dumps(
    {
        "is_simple": False,
        "sub_questions": [
            {
                "question": "Q1 how does the evolutionary loop work?",
                "angle": "operators",
                "expected_sources": "web",
                "priority": 1,
                "heading": "Section One",
            },
            {
                "question": "What is the core algorithm behind genetic programming?",
                "angle": "foundations",
                "expected_sources": "web",
                "priority": 2,
                "heading": "Section Two",
            },
        ],
    }
)


def _fake_research_pack(goal: str) -> ResearchEvidencePack:
    return ResearchEvidencePack(
        query=goal,
        route_used="web",
        summary=f"summary for {goal}",
        document_evidence={"chunks": []},
        web_evidence={"results": [_web("https://ex/u1", "Fresh", 0.9)]},
        sufficiency={
            "is_sufficient": True,
            "summary": "sufficient",
            "missing_aspects": [],
            "follow_up_queries": [],
        },
    )


def test_pipeline_reuses_cached_evidence_for_near_identical_sq(
    monkeypatch, tmp_path
):
    _fresh_db(monkeypatch, tmp_path)
    env = _basic_env()
    env["plan_json"] = PLAN_CACHE_JSON
    _install_stubs(monkeypatch, env)

    # Pre-seed a fresh, sufficient pack from a PRIOR session for the second
    # sub-question's topic (jaccard 1.0 after normalization).
    seeded = _pack("seeded")
    ecache.save_evidence(
        "seeded-session",
        "What is the core algorithm behind genetic programming?",
        seeded,
        True,
    )

    retriever_goals = []

    def fake_retriever(user_query, research_goal="", **kw):
        retriever_goals.append(research_goal)
        if "core algorithm behind genetic programming" in research_goal:
            raise AssertionError(
                "retriever must not be called for the cached sub-question"
            )
        return _fake_research_pack(research_goal)

    monkeypatch.setattr(dpo, "retriever_agent", fake_retriever)

    result = dpo.deep_research("test research query", verbose=False, max_rounds=3)

    assert result["stats"]["cache_hits"] == 1
    assert result["stats"]["cache_misses"] == 1
    # Only the fresh sub-question hit the retrieval path.
    assert len(retriever_goals) == 1
    assert "evolutionary loop" in retriever_goals[0]
    # The cached pack was used verbatim for one of the two sections.
    packs = result["state"]["sub_question_evidence"].values()
    assert any(
        (p.get("web_evidence") or {}).get("results", [{}])[0].get("url")
        == "https://ex/u1"
        and p.get("query") == "seeded"
        for p in packs
    )
    # The fresh pack was stored for future runs (near-identical lookup).
    assert (
        ecache.lookup_evidence("how does the evolutionary loop work", 30) is not None
    )
    assert result["stats"]["sections"] == 2


def test_pipeline_cache_flag_off_skips_lookup_and_writes(
    monkeypatch, tmp_path
):
    _fresh_db(monkeypatch, tmp_path)
    # Flag off: rebuild the global config from the patched env, restoring the
    # original cached config afterwards (same pattern as
    # tests/test_config_decomposer_override.py).
    original_config = uconfig._config
    monkeypatch.setenv("EVIDENCE_CACHE_ENABLED", "false")
    uconfig.reset_config()
    try:
        env = _basic_env()
        env["plan_json"] = PLAN_CACHE_JSON
        _install_stubs(monkeypatch, env)

        # Pre-seed a matching row: it must be IGNORED with the flag off.
        ecache.save_evidence(
            "seeded-session",
            "What is the core algorithm behind genetic programming?",
            _pack("seeded"),
            True,
        )

        retriever_goals = []

        def fake_retriever(user_query, research_goal="", **kw):
            retriever_goals.append(research_goal)
            return _fake_research_pack(research_goal)

        monkeypatch.setattr(dpo, "retriever_agent", fake_retriever)

        result = dpo.deep_research(
            "test research query", verbose=False, max_rounds=3
        )

        # Both sub-questions were retrieved; the seeded row was ignored.
        assert len(retriever_goals) == 2
        assert result["stats"]["cache_hits"] == 0
        assert result["stats"]["cache_misses"] == 0
        # No writes with the flag off: the seeded row is still the only row.
        assert _row_count() == 1
    finally:
        uconfig._config = original_config
