"""
Tests for the deep-research pipeline (P1-3/P1-4).

Monkeypatches every LLM surface (run_model in each agent module plus
deep_research_orchestrator's own import) and the retrieval primitives
(retrieve_document / web_search), so no network, Qdrant, or Tavily call is
made. Covers the five required scenarios:

  1. Global citation key assignment across merged per-sub-question packs
     (unique keys, dedupe keeps highest score, re-retrieve merge continues
     the key sequence).
  2. References rendered in first-appearance order ([D2]/[W2] cited before
     [D1]/[W1] becomes reference 1).
  3. The write_section revision path is invoked with the critic's gaps.
  4. Critic garbage output falls back to the neutral pass (no revisions).
  5. MAX_LLM_CALLS exhaustion stops the pipeline (warning) but still
     assembles a final answer from what exists.
"""

import json
from pathlib import Path

import importlib
import re
import shutil
import tempfile
import pytest

import deep_research_orchestrator as dpo
import run_orchestrator as ro
# importlib (not `import ... as`): worker_agents/__init__.py re-exports the
# retriever_agent/verifier_agent functions, shadowing the same-named module
# attributes in the package namespace.
dmod = importlib.import_module("worker_agents.decomposition_agent")
rmod = importlib.import_module("worker_agents.retriever_agent")
wmod = importlib.import_module("worker_agents.writer_agent")
vmod = importlib.import_module("worker_agents.verifier_agent")
ra = importlib.import_module("worker_agents.retriever_agent")

# Temp directories holding per-test evidence-cache DBs (P2-2 isolation);
# cleaned up by the autouse fixture below.
_CACHE_TMP_DIRS: list = []


@pytest.fixture(autouse=True)
def _cleanup_evidence_cache_tmp_dirs():
    for d in _CACHE_TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _CACHE_TMP_DIRS.clear()
    yield


class _FakeResponse:
    """Minimal stand-in for ModelResponse: the decomposer, sufficiency,
    critic, writer, and exec-summary code paths read .output_text and
    .output_parsed."""

    def __init__(self, text: str = "", parsed=None):
        self.output_text = text
        self.output_parsed = parsed


def _web(url: str, title: str, score: float, content: str = "") -> dict:
    return {
        "title": title,
        "url": url,
        "content": content or f"content for {title}",
        "score": score,
    }


def _doc(chunk_id: str, name: str, score: float, content: str = "") -> dict:
    return {
        "document_name": name,
        "document_title": name.replace(".pdf", " Title"),
        "chunk_id": chunk_id,
        "content": content or f"content for {chunk_id}",
        "score": score,
    }


PLAN_JSON = json.dumps(
    {
        "is_simple": False,
        "sub_questions": [
            {
                "question": "Q1 what is the core algorithm?",
                "angle": "operators and search",
                "expected_sources": "both",
                "priority": 1,
                "heading": "Section One",
            },
            {
                "question": "Q2 who were the pioneers?",
                "angle": "history",
                "expected_sources": "web",
                "priority": 2,
                "heading": "Section Two",
            },
        ],
    }
)

SUFFICIENT_JSON = json.dumps(
    {
        "is_sufficient": True,
        "summary": "sufficient",
        "missing_aspects": [],
        "follow_up_queries": [],
    }
)

CRITIC_OK_JSON = json.dumps(
    {
        "confidence_level": "high",
        "overall_summary": "solid",
        "hallucinated_claims": [],
        "unsupported_claims": [],
        "per_section": [
            {"section_id": "sq1", "grounded": True, "depth_ok": True, "gaps": []},
            {"section_id": "sq2", "grounded": True, "depth_ok": True, "gaps": []},
        ],
        "re_retrieve_suggested": False,
        "specific_queries": [],
    }
)


def _default_web_results(query: str):
    if "specific combined query" in query:
        return [_web("https://ex/u4", "New Finding", 0.85), _web("https://ex/u1", "One", 0.9)]
    if "Q1" in query:
        return [_web("https://ex/u1", "One", 0.9), _web("https://ex/u2", "Two", 0.8)]
    return [_web("https://ex/u1", "One", 0.4), _web("https://ex/u3", "Three", 0.7)]


def _install_stubs(monkeypatch, env: dict) -> dict:
    """Install every fake; env keys: plan_json, sufficiency_json, critic_text,
    writer_text(index, kwargs) -> str, web_results(query) -> list,
    exec_text. Returns the writer call log."""
    monkeypatch.setattr(dpo, "_read_doc_catalog", lambda: [])
    # P2-2: keep the deep-mode evidence cache off the real DB — fresh temp
    # DB per test (lookups miss, saves land in a throwaway file).
    ecache_mod = importlib.import_module("memory.evidence_cache")
    cache_tmp_dir = tempfile.mkdtemp(prefix="evidence_cache_test_")
    _CACHE_TMP_DIRS.append(cache_tmp_dir)
    monkeypatch.setattr(
        ecache_mod, "EVIDENCE_CACHE_DB_PATH",
        Path(cache_tmp_dir) / "evidence_cache_test.db",
    )
    monkeypatch.setattr(ecache_mod, "_purged_this_process", False)
    monkeypatch.setattr(
        rmod, "retrieve_document",
        lambda *a, **k: {"query": a[0] if a else "", "chunks": []},
    )
    monkeypatch.setattr(
        rmod, "web_search",
        lambda query: {"query": query, "results": env["web_results"](query)},
    )

    writer_calls = []

    def writer_stub(*a, **k):
        writer_calls.append(k)
        text = env["writer_text"](len(writer_calls) - 1, k)
        # The writer contract requires ~300+ words of substance; pad the
        # canned stubs up to that floor so the deterministic must-revise
        # (empty / <300-word section) does not fire in tests that are not
        # about it. (env may opt out with "pad_writer": False.)
        if env.get("pad_writer", True) and len(text.split()) < 300:
            text = text + " " + " ".join(
                f"token{i}" for i in range(320 - len(text.split()))
            )
        return _FakeResponse(text=text)

    monkeypatch.setattr(dmod, "run_model", lambda *a, **k: _FakeResponse(text=env["plan_json"]))
    monkeypatch.setattr(rmod, "run_model", lambda *a, **k: _FakeResponse(text=env["sufficiency_json"]))
    monkeypatch.setattr(wmod, "run_model", writer_stub)
    monkeypatch.setattr(vmod, "run_model", lambda *a, **k: _FakeResponse(text=env["critic_text"]))
    monkeypatch.setattr(
        dpo, "run_model", lambda *a, **k: _FakeResponse(text=env.get("exec_text", "Exec summary prose."))
    )
    return writer_calls


def _basic_env(critic_text: str = CRITIC_OK_JSON, writer_text=None) -> dict:
    if writer_text is None:
        def writer_text(i, k):  # noqa: F811
            if i == 0:
                return "## Section One body citing [W1]."
            if i == 1:
                return "## Section Two body citing [W3]."
            return "## Section body revised [W4]."

    return {
        "plan_json": PLAN_JSON,
        "sufficiency_json": SUFFICIENT_JSON,
        "critic_text": critic_text,
        "writer_text": writer_text,
        "web_results": _default_web_results,
        "exec_text": "Synthesized executive summary prose.",
    }


def _json_writer(i: int) -> str:
    """Contract-compliant JSON section for json-mode pipeline tests: a
    valid Section with 300+ words, so the deterministic must-revise
    backstop (which rewrites empty / <300-word sections) stays silent."""
    return json.dumps(
        {
            "id": f"section-{i + 1}",
            "heading": f"Section {i + 1}",
            "blocks": [
                {
                    "type": "paragraph",
                    "spans": [
                        {
                            "text": "Body for section "
                            + str(i + 1)
                            + ". "
                            + " ".join(f"word{j}" for j in range(310)),
                            "citations": [],
                        }
                    ],
                }
            ],
        }
    )


def _run(monkeypatch, env, **kw) -> dict:
    _install_stubs(monkeypatch, env)
    return dpo.deep_research(
        "test research query", verbose=False, max_rounds=3, **kw
    )


# ---------------------------------------------------------------------------
# Scenario 1 — global key assignment across merged packs
# ---------------------------------------------------------------------------

def test_merge_evidence_dedup_keeps_highest_score():
    low = _doc("c1", "a.pdf", 0.4)
    high = _doc("c1", "a.pdf", 0.9)
    all_doc, all_web = dpo._merge_evidence(
        [
            {"document_evidence": {"chunks": [low]}, "web_evidence": {"results": []}},
            {"document_evidence": {"chunks": [high]}, "web_evidence": {"results": []}},
        ]
    )
    assert len(all_doc) == 1
    assert all_doc[0]["score"] == 0.9


def test_merge_new_evidence_continues_key_sequences():
    registry = {
        "W1": {"kind": "web", "title": "One", "url": "https://ex/u1",
               "published_date": None, "score": 0.9},
    }
    web_key_map = {"https://ex/u1": "W1"}
    # Global list after a re-retrieval: the existing u1 (now seen again with a
    # lower score) plus two genuinely new results.
    all_web = [
        _web("https://ex/u1", "One", 0.9),
        _web("https://ex/u2", "Two", 0.7),
        _web("https://ex/u3", "Three", 0.6),
    ]
    registry2, doc_map, web_map, added_docs, added_webs = dpo._merge_new_evidence(
        registry, {}, web_key_map, [], all_web
    )
    # Existing key/entry untouched; continuation keys W2, W3 assigned in
    # score-descending order; the already-keyed u1 gets no new key.
    assert web_map["https://ex/u1"] == "W1"
    assert web_map["https://ex/u2"] == "W2"
    assert web_map["https://ex/u3"] == "W3"
    assert registry2["W1"]["title"] == "One"
    assert {r["url"] for r in added_webs} == {"https://ex/u2", "https://ex/u3"}
    assert added_docs == []


def test_pipeline_global_keys_across_merged_packs_and_re_retrieve(monkeypatch, capsys):
    env = _basic_env(critic_text=json.dumps(
        {
            "confidence_level": "medium",
            "overall_summary": "ok",
            "hallucinated_claims": [],
            "unsupported_claims": [],
            "per_section": [
                {"section_id": "sq1", "grounded": True, "depth_ok": True, "gaps": []},
                {"section_id": "sq2", "grounded": True, "depth_ok": True, "gaps": []},
            ],
            "re_retrieve_suggested": True,
            "specific_queries": ["specific combined query"],
        }
    ))

    def writer_text(i, k):
        if i == 0:
            return "## Section One body [W1]."
        if i == 1:
            return "## Section Two body [W3]."
        return "## Revised [W4]."

    env["writer_text"] = writer_text
    result = _run(monkeypatch, env, output_format="markdown")

    registry = result["state"]["citation_registry"]
    web_map = result["state"]["web_key_map"]

    # Four unique keys across the merged packs; the re-retrieval's new result
    # continues the sequence at W4.
    assert sorted(registry) == ["W1", "W2", "W3", "W4"]
    assert len(web_map) == 4
    assert web_map["https://ex/u4"] == "W4"
    # Dedupe kept the highest-score u1 (0.9), not the re-retrieval dup.
    assert registry["W1"]["score"] == 0.9
    assert result["stats"]["re_retrieves"] == 1
    # No revisions (critic found everything strong).
    assert result["stats"]["revisions"] == 0
    assert "## References" in result["final_answer"]
    out = capsys.readouterr().out
    assert "stage 5 ASSEMBLY" in out


# ---------------------------------------------------------------------------
# Scenario 2 — references in first-appearance order
# ---------------------------------------------------------------------------

def test_references_first_appearance_order(monkeypatch):
    def writer_text(i, k):
        if i == 0:
            # [W2] appears before [W1] in the body.
            return "## Section One body: [W2] first, then [W1] second."
        return "## Section Two body citing [W3]."

    result = _run(monkeypatch, _basic_env(writer_text=writer_text), output_format="markdown")
    final = result["final_answer"]

    refs = final.split("## References", 1)[1]
    one = refs.index("[1] Two")
    two = refs.index("[2] One")
    three = refs.index("[3] Three")
    assert one < two < three
    # Internal keys are renumbered to the reference numbers in the body.
    body_only = final.split("## References", 1)[0]
    assert "[1] first, then [2] second." in body_only
    assert "citing [3]." in body_only
    assert not re.search(r"\[[DW]\d+\]", final)


def test_collect_cited_keys_first_appearance_and_multi_digit():
    registry = {"D1": {}, "D10": {}, "W1": {}}
    keys = dpo._collect_cited_keys(
        "start [D10] then [D1] and [W1] again [D10]", registry
    )
    assert keys == ["D10", "D1", "W1"]


def test_strip_invented_keys_partial_and_full():
    registry = {"D1": {}, "D22": {}, "W1": {}}
    cleaned, invented = dpo._strip_invented_keys(
        "a [D22, D103] b [D103] c [W1] d [D1]", registry
    )
    assert cleaned == "a [D22] b c [W1] d [D1]"
    assert invented == ["D103"]
    # All-resolving brackets are untouched.
    cleaned2, invented2 = dpo._strip_invented_keys("x [D1, W1] y", registry)
    assert cleaned2 == "x [D1, W1] y"
    assert invented2 == []
    # Mixed brackets: bare-number items are kept, invented keys dropped.
    cleaned3, invented3 = dpo._strip_invented_keys(
        "a [69, D103] b [D22, 7] c [W1, 3]", registry
    )
    assert cleaned3 == "a [69] b [D22, 7] c [W1, 3]"
    assert invented3 == ["D103"]
    # Pure-numeric brackets are never touched.
    cleaned4, invented4 = dpo._strip_invented_keys("x [69, 71] y", registry)
    assert cleaned4 == "x [69, 71] y"
    assert invented4 == []


# ---------------------------------------------------------------------------
# Fix 1 — inline [D#]/[W#] keys renumbered to [1..N] at assembly
# ---------------------------------------------------------------------------

def test_key_number_map_mirrors_reference_numbering():
    registry = {"D1": {"kind": "doc"}, "W2": {"kind": "web"}, "W1": {"kind": "web"}}
    # D9 is not in the registry (format_references would skip it); the map
    # must skip it too and stay contiguous 1..N in first-appearance order.
    assert dpo._key_number_map(["W2", "D1", "D9", "W1", "W2"], registry) == {
        "W2": 1,
        "D1": 2,
        "W1": 3,
    }
    assert dpo._key_number_map([], registry) == {}


def test_renumber_inline_keys_single_multi_and_unknown():
    mapping = {"W2": 1, "W1": 2}
    out, removed = dpo._renumber_inline_keys(
        "a [W2] b [W1, W2] c [D9] d [W1].", mapping
    )
    assert out == "a [1] b [2, 1] c d [2]."
    assert removed == ["D9"]
    # Keys absent from the map inside a multi-key bracket: entry stripped.
    out2, removed2 = dpo._renumber_inline_keys("x [D9, W2] y", mapping)
    assert out2 == "x [1] y"
    assert removed2 == ["D9"]
    # Mixed brackets: number-first items pass through, keys renumbered.
    out3, removed3 = dpo._renumber_inline_keys(
        "a [69, W2, D9] b [W1, 81] c [102, W2]", mapping
    )
    assert out3 == "a [69, 1] b [2, 81] c [102, 1]"
    assert removed3 == ["D9"]
    # No brackets at all is a no-op.
    assert dpo._renumber_inline_keys("plain prose", mapping) == ("plain prose", [])


def test_drop_unresolved_numbers():
    out, dropped = dpo._drop_unresolved_numbers(
        "x [101] y [69, 41] z [41] w [44, 45] v [1, 2].", valid={1, 2, 41}
    )
    assert out == "x y [41] z [41] w v [1, 2]."
    assert dropped == ["101", "69", "44", "45"]
    # Everything resolvable: untouched.
    out2, dropped2 = dpo._drop_unresolved_numbers("a [1, 2] b", valid={1, 2})
    assert out2 == "a [1, 2] b"
    assert dropped2 == []


def test_pipeline_renumbers_whole_body_including_exec_summary(monkeypatch):
    def writer_text(i, k):
        if i == 0:
            # Includes a mixed bracket (number first) and a hallucinated
            # numeric citation [49] that must be dropped (refs are 1..3).
            return (
                "## Section One body: [W2] first, [W1, W2] multi, "
                "[49, W3] mixed, and [49] dead."
            )
        return "## Section Two body citing [W3]."

    result = _run(monkeypatch, _basic_env(writer_text=writer_text), output_format="markdown")
    final = result["final_answer"]
    body_only = final.split("## References", 1)[0]
    # No internal key survives anywhere in the final answer...
    assert not re.search(r"\[[DW]\d+\]", final)
    # ...and the hallucinated number is gone too.
    assert "[49]" not in body_only
    # ...and every inline [n] resolves to a rendered reference entry.
    used = set()
    for group in re.findall(r"\[([^\]]*)\]", body_only):
        for tok in group.split(","):
            if tok.strip().isdigit():
                used.add(tok.strip())
    refs_block = final.split("## References", 1)[1]
    rendered = set(re.findall(r"^\[(\d+)\]", refs_block, re.M))
    assert used == rendered == {"1", "2", "3"}
    # Exec summary (no citations by instruction) sits above the renumbered
    # sections, and mixed brackets are renumbered per surviving item.
    assert body_only.index("Synthesized executive summary prose.") < body_only.index(
        "[1] first, [2, 1] multi, [3] mixed, and dead."
    )


# ---------------------------------------------------------------------------
# Fix 2 — decomposition retry on fallback
# ---------------------------------------------------------------------------

def _run_with_decompose_stub(monkeypatch, env, decompose_stub, verbose=False) -> dict:
    _install_stubs(monkeypatch, env)
    monkeypatch.setattr(dmod, "run_model", decompose_stub)
    return dpo.deep_research("test research query", verbose=verbose, max_rounds=3)


def test_decompose_fallback_retries_once_and_keeps_bigger_plan(monkeypatch, capsys):
    calls = []

    def decompose_stub(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return _FakeResponse(text="total garbage, no JSON in sight")
        return _FakeResponse(text=PLAN_JSON)

    env = _basic_env()
    # json-mode run: the markdown stubs would soft-fail to empty Sections,
    # which the must-revise backstop would then rewrite (changing call counts).
    env["writer_text"] = lambda i, k: _json_writer(i)
    result = _run_with_decompose_stub(
        monkeypatch, env, decompose_stub, verbose=True
    )
    out = capsys.readouterr().out

    assert len(calls) == 2  # exactly one retry, no more
    assert "retrying once" in out
    plan = result["state"]["plan"]
    assert plan["source"] == "json-fallback"  # the retry's plan was kept
    assert len(plan["sub_questions"]) == 2
    assert result["stats"]["sections"] == 2
    # The retry counted against the budget like any other call:
    # decompose x2 + sufficiency x2 + drafts x2 + critic + exec summary.
    assert result["stats"]["llm_calls"] == 8


def test_decompose_retry_tie_keeps_first_plan_and_retries_at_most_once(monkeypatch):
    calls = []

    def decompose_stub(*a, **k):
        calls.append(1)
        # Both attempts are garbage -> both fall back with 1 sub-question.
        return _FakeResponse(text=f"garbage attempt {len(calls)}")

    result = _run_with_decompose_stub(monkeypatch, _basic_env(), decompose_stub)

    assert len(calls) == 2  # the tie did not trigger a second retry
    plan = result["state"]["plan"]
    assert plan["source"] == "fallback"  # first (original) plan kept on tie
    assert len(plan["sub_questions"]) == 1
    assert result["stats"]["sections"] == 1


# ---------------------------------------------------------------------------
# Scenario 3 — writer revision path receives the critic's gaps
# ---------------------------------------------------------------------------

def test_writer_revision_path_called_with_expansion_gaps(monkeypatch):
    critic = json.dumps(
        {
            "confidence_level": "medium",
            "overall_summary": "one gap",
            "hallucinated_claims": [],
            "unsupported_claims": [],
            "per_section": [
                {"section_id": "sq1", "grounded": True, "depth_ok": False,
                 "gaps": ["Add concrete numbers"]},
                {"section_id": "sq2", "grounded": True, "depth_ok": True, "gaps": []},
            ],
            "re_retrieve_suggested": False,
            "specific_queries": [],
        }
    )
    writer_calls = _install_stubs(
        monkeypatch, _basic_env(critic_text=critic)
    )
    result = dpo.deep_research("test query", verbose=False, max_rounds=3, output_format="markdown")

    # 2 drafts + exactly 1 revision (only sq1 was flagged with gaps).
    assert len(writer_calls) == 3
    assert result["stats"]["revisions"] == 1
    revision_input = writer_calls[2].get("input_data") or ""
    assert "REVISION REQUIRED" in revision_input
    assert "Add concrete numbers" in revision_input
    # Drafts did not carry the revision block.
    for call in writer_calls[:2]:
        assert "REVISION REQUIRED" not in (call.get("input_data") or "")
    # The revised section replaced the original in the final answer.
    assert "Section body revised" in result["final_answer"]
    # [W4] was never keyed in this run (no re-retrieval): the invented key is
    # stripped from the body, a footer note is added, and no reference entry
    # exists for it.
    body_only = result["final_answer"].split("## References", 1)[0]
    assert "[W4]" not in body_only
    assert "were cited without a matching evidence item" in result["final_answer"]
    refs = result["final_answer"].split("## References", 1)[1]
    assert "New Finding" not in refs


# ---------------------------------------------------------------------------
# Scenario 4 — critic garbage → neutral pass, no revisions
# ---------------------------------------------------------------------------

def test_critic_garbage_falls_back_to_neutral_no_revisions(monkeypatch):
    writer_calls = _install_stubs(
        monkeypatch, _basic_env(critic_text="This is not JSON at all ###")
    )
    result = dpo.deep_research("test query", verbose=False, max_rounds=3, output_format="markdown")

    assert result["state"]["critic"]["source"] == "fallback"
    assert result["stats"]["revisions"] == 0
    assert len(writer_calls) == 2  # drafts only, no expansion
    # Neutral pass: every per_section entry carries citation_density_ok=True
    # (no spurious density revisions from the fallback).
    assert result["state"]["critic"]["per_section"]
    assert all(
        entry["citation_density_ok"] is True
        for entry in result["state"]["critic"]["per_section"]
    )
    # Report still assembles with both sections + references.
    assert "## Section One" in result["final_answer"]
    assert "## Section Two" in result["final_answer"]
    assert "## References" in result["final_answer"]
    assert result["state"]["verification_status"]["coverage"] == "moderate"


# ---------------------------------------------------------------------------
# Scenario 5 — budget exhaustion stops the pipeline but assembles
# ---------------------------------------------------------------------------

def test_budget_exhaustion_stops_and_assembles(monkeypatch, capsys):
    monkeypatch.setattr(dpo, "MAX_LLM_CALLS", 5)
    # Cost model: decomposer(1) + sufficiency x2 (2,3) + 2 section drafts
    # (4,5) → exhausted before the critic and the executive summary.
    result = _run(monkeypatch, _basic_env(), output_format="markdown")

    assert result["stats"]["llm_calls"] == 5
    assert result["stats"]["revisions"] == 0
    final = result["final_answer"]
    assert "## Section One" in final
    assert "## Section Two" in final
    assert "## Executive Summary" not in final
    # References are deterministic — no LLM needed, so they still render.
    assert "## References" in final
    # A warning was logged when the budget ran out.
    out = capsys.readouterr().out
    assert "WARNING" in out and "budget exhausted" in out


# ---------------------------------------------------------------------------
# FIX 2 — transient LLM failures must not kill a deep run
# ---------------------------------------------------------------------------

def test_pipeline_survives_section_writer_failure(monkeypatch, capsys):
    # T1: the writer raises on section 2 of 2 -> deep_research completes,
    # keeps section 1, skips section 2 (a gap, never a crash).
    def writer_text(i, k):
        if i == 0:
            return "## Section One body citing [W1]."
        if i == 1:
            raise RuntimeError("simulated 500 from the LLM server")
        return "## Section Two body citing [W3]."

    result = _run(monkeypatch, _basic_env(writer_text=writer_text), output_format="markdown")

    final = result["final_answer"]
    assert "## Section One" in final
    assert "Section Two body" not in final  # the failed section is a gap
    assert "## References" in final
    assert result["stats"]["section_failures"] == 1
    assert result["stats"]["sections"] == 1
    assert result["stats"]["exec_summary_failed"] is False
    out = capsys.readouterr().out
    assert "write_section failed" in out
    # Stage 3's log line surfaces the failure count when nonzero.
    assert "stage 3 PER-SECTION DRAFT done" in out
    assert "section_failures=1" in out


def test_pipeline_survives_exec_summary_failure(monkeypatch, capsys):
    # T2: only the exec-summary call raises (detected via its instructions
    # marker) -> final answer has sections + references, no exec summary.
    _install_stubs(monkeypatch, _basic_env())

    def exec_stub(*a, **k):
        if "executive summary" in (k.get("instructions") or "").lower():
            raise TimeoutError("simulated timeout on the exec summary call")
        return _FakeResponse(text="unreachable")

    monkeypatch.setattr(dpo, "run_model", exec_stub)
    result = dpo.deep_research("test research query", verbose=False, max_rounds=3, output_format="markdown")

    final = result["final_answer"]
    assert "## Executive Summary" not in final
    assert "Synthesized executive summary prose." not in final  # prose absent
    assert "## Section One" in final
    assert "## Section Two" in final
    assert "## References" in final
    assert result["stats"]["exec_summary_failed"] is True
    assert result["stats"]["section_failures"] == 0
    out = capsys.readouterr().out
    assert "executive summary call failed" in out
    assert "stage 5 ASSEMBLY done" in out
    assert "exec_summary_failed=1" in out


# ---------------------------------------------------------------------------
# FIX 1 — deep envelope unwrapping for the save flow
# ---------------------------------------------------------------------------

def test_deep_result_envelope_shape_and_top_level_state(monkeypatch):
    # T3a: deep_research returns {final_answer, state, stats} and the
    # save_report-consumed fields live at the TOP level of result["state"].
    result = _run(monkeypatch, _basic_env())

    assert set(result.keys()) == {"final_answer", "state", "stats"}
    state = result["state"]
    assert isinstance(state["evidence_json"], str) and state["evidence_json"]
    assert "verification_status" in state
    assert "verification" in state
    parsed = json.loads(state["evidence_json"])
    # The nested evidence pack is well-formed and non-empty in this run.
    assert parsed["web_evidence"]["results"]
    assert parsed["document_evidence"]["chunks"] == []  # web-only run


def test_deep_state_unwrap_helper():
    # T3b: run_orchestrator._deep_state unwraps the envelope to the nested
    # state (identity); a result without a usable "state" dict yields {} so
    # save_report degrades to plain markdown (documented in the helper).
    inner = {
        "evidence_json": "e",
        "verification_status": {"confidence": "high"},
        "verification": "v",
    }
    envelope = {"final_answer": "a", "state": inner, "stats": {}}
    assert ro._deep_state(envelope) is inner
    assert ro._deep_state({"final_answer": "a"}) == {}  # no "state" key
    assert ro._deep_state({"final_answer": "a", "state": None}) == {}
    assert ro._deep_state({"final_answer": "a", "state": "not-a-dict"}) == {}
    assert ro._deep_state(None) == {}


# ---------------------------------------------------------------------------
# P2-4a — cross-section synthesizer + deterministic citation density
# ---------------------------------------------------------------------------

PLAN_JSON_3SQ = json.dumps(
    {
        "is_simple": False,
        "sub_questions": [
            {
                "question": "Q1 what is the core algorithm?",
                "angle": "operators and search",
                "expected_sources": "both",
                "priority": 1,
                "heading": "Section One",
            },
            {
                "question": "Q2 who were the pioneers?",
                "angle": "history",
                "expected_sources": "web",
                "priority": 2,
                "heading": "Section Two",
            },
            {
                "question": "Q3 what tools exist?",
                "angle": "ecosystem",
                "expected_sources": "web",
                "priority": 3,
                "heading": "Section Three",
            },
        ],
    }
)


def _synth_calls(writer_calls: list) -> list:
    return [
        c for c in writer_calls
        if "SYNTHESIS" in (c.get("instructions") or "")
    ]


def test_synthesis_appended_last_and_keys_resolve(monkeypatch):
    # T1: 3+ sections → write_synthesis called once; its heading is the
    # LAST content H2 (before References); its [D2] resolves in References.
    def writer_text(i, k):
        if "SYNTHESIS" in (k.get("instructions") or ""):
            return (
                "## Synthesis\n\n"
                "Section One and Section Two agree [D2], "
                "while Section Three contrasts them [W1]."
            )
        return {
            0: "## Section One\n\nBody [D1].",
            1: "## Section Two\n\nBody [D1, W1].",
            2: "## Section Three\n\nBody [W1].",
        }[i]

    env = _basic_env(writer_text=writer_text)
    env["plan_json"] = PLAN_JSON_3SQ
    writer_calls = _install_stubs(monkeypatch, env)
    # Non-empty doc catalog (defeats the stub's web-only mode) plus doc
    # evidence, so [D1]/[D2] are real registry keys (not invented).
    monkeypatch.setattr(
        dpo, "_read_doc_catalog",
        lambda: [{"document_name": "a.pdf", "document_title": "a.pdf Title"}],
    )
    monkeypatch.setattr(
        rmod, "retrieve_document",
        lambda *a, **k: {
            "query": a[0] if a else "",
            "chunks": [_doc("c1", "a.pdf", 0.9), _doc("c2", "b.pdf", 0.8)],
        },
    )
    result = dpo.deep_research("test query", verbose=False, max_rounds=3, output_format="markdown")

    assert len(_synth_calls(writer_calls)) == 1
    assert result["stats"]["synthesis_words"] > 0
    assert result["stats"]["synthesis_failed"] is False
    assert result["stats"]["synthesis_skipped"] is None

    # Heading order: content sections, then Synthesis, then References.
    h2s = re.findall(r"^## (.+)$", result["final_answer"], re.M)
    assert h2s[-1] == "References"
    assert h2s[-2] == "Synthesis"
    assert h2s[:3] == ["Section One", "Section Two", "Section Three"]

    # The synthesis's [D2] was renumbered (not dropped as invented) and its
    # reference entry (b.pdf) is rendered; every inline [n] resolves.
    body, refs = result["final_answer"].split("## References", 1)
    assert "[D2]" not in body and "[D1]" not in body
    assert "b.pdf" in refs
    nums = {
        int(n) for grp in re.findall(r"\[([\d,\s]+)\]", body)
        for n in grp.split(",") if n.strip().isdigit()
    }
    assert nums
    for n in nums:
        assert re.search(rf"^\[{n}\] ", refs, re.M), f"[{n}] has no reference"


def test_synthesis_skipped_on_budget_exec_summary_still_writes(monkeypatch):
    # T2: 8 tracked calls before stage 5 (decompose + 3 sufficiency +
    # 3 drafts + critic); limit 9 leaves exactly 1 call → synthesis (needs
    # 2: itself + exec summary) is skipped, exec summary still writes.
    def writer_text(i, k):
        return _json_writer(i)

    env = _basic_env(writer_text=writer_text)
    env["plan_json"] = PLAN_JSON_3SQ
    writer_calls = _install_stubs(monkeypatch, env)
    monkeypatch.setattr(dpo, "MAX_LLM_CALLS", 9)
    result = dpo.deep_research("test query", verbose=False, max_rounds=3)

    assert not _synth_calls(writer_calls)
    assert result["stats"]["synthesis_skipped"] == "budget"
    assert "## Synthesis" not in result["final_answer"]
    # Exec summary wrote with the final remaining call.
    assert "Synthesized executive summary prose." in result["final_answer"]
    assert result["stats"]["llm_calls"] == 9


def test_synthesis_failure_leaves_report_intact(monkeypatch):
    # T3: write_synthesis raises → synthesis_failed=True, report assembles
    # WITHOUT a Synthesis section, no crash.
    def writer_text(i, k):
        if "SYNTHESIS" in (k.get("instructions") or ""):
            raise RuntimeError("boom: transient synthesis failure")
        return {
            0: "## Section One body [W1].",
            1: "## Section Two body [W1].",
            2: "## Section Three body [W1].",
        }[i]

    env = _basic_env(writer_text=writer_text)
    env["plan_json"] = PLAN_JSON_3SQ
    _install_stubs(monkeypatch, env)
    result = dpo.deep_research("test query", verbose=False, max_rounds=3, output_format="markdown")

    assert result["stats"]["synthesis_failed"] is True
    assert "## Synthesis" not in result["final_answer"]
    for h in ("Section One", "Section Two", "Section Three"):
        assert f"## {h}" in result["final_answer"]
    assert "## References" in result["final_answer"]


def test_synthesis_skipped_when_too_few_sections(monkeypatch):
    # T4: default plan = 2 sub-questions → synthesis never called.
    env = _basic_env()
    env["writer_text"] = lambda i, k: _json_writer(i)
    writer_calls = _install_stubs(monkeypatch, env)
    result = dpo.deep_research("test query", verbose=False, max_rounds=3)

    assert not _synth_calls(writer_calls)
    assert result["stats"]["synthesis_skipped"] == "too_few_sections"
    assert "## Synthesis" not in result["final_answer"]
    assert result["stats"]["llm_calls"] == 7  # nothing charged for synthesis


def test_citation_density_state_is_deterministic(monkeypatch):
    # T5: state["citation_density"] = {overall, per_section}; overall
    # matches an independent recomputation on the final body.
    result = _run(monkeypatch, _basic_env(), output_format="markdown")
    cd = result["state"]["citation_density"]
    assert set(cd) == {"overall", "per_section"}
    assert cd["per_section"].keys() == {"Section One", "Section Two"}

    body = result["final_answer"].split("## References", 1)[0]
    brackets = re.findall(r"\[\d+(?:,\s*\d+)*\]", body)
    words = re.findall(r"\b\w+\b", body)
    expected = round(len(brackets) / len(words) * 100, 2)
    assert abs(cd["overall"] - expected) < 0.01
    assert cd["overall"] > 0  # the stubbed sections do carry citations


def test_evidence_cache_ttl_bad_value_falls_back(monkeypatch):
    # T6: a non-int EVIDENCE_CACHE_TTL_DAYS must not crash get_config.
    import utils.config as config_mod

    config_mod.reset_config()
    try:
        monkeypatch.setenv("EVIDENCE_CACHE_TTL_DAYS", "abc")
        cfg = config_mod.get_config()
        assert cfg.evidence_cache_ttl_days == 30
    finally:
        config_mod.reset_config()


# ---------------------------------------------------------------------------
# P2-3 — citation-density tightening (writer contract + critic field/trigger)
# ---------------------------------------------------------------------------

def test_write_section_prompt_has_quantitative_citation_contract():
    wmod = importlib.import_module("worker_agents.writer_agent")
    prompt = wmod.WRITE_SECTION_INSTRUCTIONS.replace(
        "{SECTION_HEADING}", "Test Heading"
    )
    assert "4 citations per 100 words" in prompt
    assert "every factual sentence" in prompt
    # Reusing a provided key is expected; inventing keys stays banned.
    assert "reusing the same key" in prompt
    assert "NEVER invent a key" in prompt
    # Uncitable (synthesis/transition) sentences stay a minority.
    assert "minority of the section" in prompt


def test_critic_prompt_has_density_rule():
    vmod_critic = importlib.import_module("worker_agents.verifier_agent")
    assert "4 per 100 words" in vmod_critic.CRITIC_INSTRUCTIONS
    assert "citation_density_ok" in vmod_critic.CRITIC_INSTRUCTIONS
    assert "under-cited: add [D#]/[W#] citations" in vmod_critic.CRITIC_INSTRUCTIONS


def test_neutral_critic_report_passes_density():
    vmod_critic = importlib.import_module("worker_agents.verifier_agent")
    report = vmod_critic._neutral_critic_report(["sq1", "sq2"])
    assert [e["section_id"] for e in report["per_section"]] == ["sq1", "sq2"]
    assert all(e["citation_density_ok"] is True for e in report["per_section"])


def test_critic_low_citation_density_triggers_revision(monkeypatch):
    under_cited_gap = (
        "under-cited: add [D#]/[W#] citations to the uncited factual sentences"
    )
    critic = json.dumps(
        {
            "is_supported": True,
            "overall_summary": "one under-cited section",
            "hallucinated_claims": [],
            "unsupported_claims": [],
            "per_section": [
                # grounded + depth_ok, but under-cited with a concrete gap:
                # revision-eligible via the citation_density_ok branch.
                {"section_id": "sq1", "grounded": True, "depth_ok": True,
                 "citation_density_ok": False, "gaps": [under_cited_gap]},
                {"section_id": "sq2", "grounded": True, "depth_ok": True,
                 "citation_density_ok": True, "gaps": []},
            ],
            "re_retrieve_suggested": False,
            "specific_queries": [],
        }
    )
    writer_calls = _install_stubs(
        monkeypatch, _basic_env(critic_text=critic)
    )
    result = dpo.deep_research("test query", verbose=False, max_rounds=3, output_format="markdown")

    # 2 drafts + exactly 1 revision (only sq1 was flagged for density).
    assert len(writer_calls) == 3
    assert result["stats"]["revisions"] == 1
    revision_input = writer_calls[2].get("input_data") or ""
    assert "REVISION REQUIRED" in revision_input
    assert under_cited_gap in revision_input
    # The density flag normalized through the report intact.
    by_id = {e["section_id"]: e for e in result["state"]["critic"]["per_section"]}
    assert by_id["sq1"]["citation_density_ok"] is False
    assert by_id["sq2"]["citation_density_ok"] is True


# ---------------------------------------------------------------------------
# Deterministic must-revise + critic draft boundaries (stage 4)
# ---------------------------------------------------------------------------


def test_must_revise_empty_section_even_when_critic_all_pass(monkeypatch):
    # One stage-3 draft soft-fails to an EMPTY Section (json mode); the
    # critic says everything is fine. The deterministic must-revise pass
    # must still rewrite the empty section, and the revision ships.
    def writer_text(i, k):
        if "REVISION REQUIRED" in (k.get("input_data") or ""):
            return json.dumps(
                {
                    "id": "section-two",
                    "heading": "Section Two",
                    "blocks": [
                        {
                            "type": "paragraph",
                            "spans": [
                                {
                                    "text": "Revised full body with substance. "
                                    + " ".join(
                                        f"fact{i}" for i in range(40)
                                    ),
                                    "citations": [],
                                }
                            ],
                        }
                    ],
                }
            )
        if i == 1:
            return "garbage, no JSON object here"
        return _json_writer(i)

    env = _basic_env(writer_text=writer_text)
    env["pad_writer"] = False
    writer_calls = _install_stubs(monkeypatch, env)
    result = dpo.deep_research("test query", verbose=False, max_rounds=3)

    assert result["stats"]["revisions"] == 1
    revision = [c for c in writer_calls if "REVISION REQUIRED" in (c.get("input_data") or "")]
    assert len(revision) == 1
    assert "Section Two" in revision[0]["input_data"]
    # The must-revise output is what ships.
    assert "Revised full body with substance." in result["final_answer"]


def test_critic_draft_shows_boundary_for_empty_section(monkeypatch):
    # The critic's draft text carries a "## heading (id: sid)" boundary per
    # section — including for a section whose draft is empty.
    critic_inputs = []

    def writer_text(i, k):
        if i == 1:
            return "garbage, no JSON object here"
        return _json_writer(i)

    env = _basic_env(writer_text=writer_text)
    env["pad_writer"] = False
    _install_stubs(monkeypatch, env)
    monkeypatch.setattr(
        vmod,
        "run_model",
        lambda *a, **k: (critic_inputs.append(k), _FakeResponse(text=CRITIC_OK_JSON))[1],
    )
    dpo.deep_research("test query", verbose=False, max_rounds=3)

    assert critic_inputs, "critic was never called"
    draft = critic_inputs[0].get("input_data") or ""
    assert "## Section One (id: sq1)" in draft
    assert "## Section Two (id: sq2)" in draft


def test_tracked_run_model_stack_reentrant():
    # Two install/restore cycles (as nested deep runs would under the lock)
    # must unwind back to each agent module's original run_model, with no
    # stack residue.
    agents = (dmod, rmod, wmod, vmod)
    originals = {id(m): m.run_model for m in agents}
    s1 = dpo._install_tracked_run_models(dpo._LLMBudget(40), False)
    try:
        s2 = dpo._install_tracked_run_models(dpo._LLMBudget(40), False)
        assert all(m.run_model is not originals[id(m)] for m in agents)
        dpo._restore_tracked_run_models(s2)  # inner run unwinds first
    finally:
        dpo._restore_tracked_run_models(s1)
    for m in agents:
        assert m.run_model is originals[id(m)]
        assert not dpo._run_model_stacks.get(m)
    # A second sequential cycle also ends clean.
    s3 = dpo._install_tracked_run_models(dpo._LLMBudget(40), False)
    dpo._restore_tracked_run_models(s3)
    for m in agents:
        assert m.run_model is originals[id(m)]
        assert not dpo._run_model_stacks.get(m)


def test_must_revise_short_json_section_even_when_critic_all_pass(monkeypatch):
    # A NON-empty but <300-word Section (json mode) must also be queued for
    # must-revise: the 30-word ship-guard floor is not the writer contract.
    def writer_text(i, k):
        if "REVISION REQUIRED" in (k.get("input_data") or ""):
            return json.dumps(
                {
                    "id": f"section-{i + 1}",
                    "heading": f"Section {i + 1}",
                    "blocks": [
                        {
                            "type": "paragraph",
                            "spans": [
                                {
                                    "text": "Revised substantive draft. "
                                    + " ".join(f"fact{j}" for j in range(320)),
                                    "citations": [],
                                }
                            ],
                        }
                    ],
                }
            )
        return json.dumps(
            {
                "id": f"section-{i + 1}",
                "heading": f"Section {i + 1}",
                "blocks": [
                    {
                        "type": "paragraph",
                        "spans": [
                            {
                                "text": " ".join(f"word{j}" for j in range(100)),
                                "citations": [],
                            }
                        ],
                    }
                ],
            }
        )

    env = _basic_env(writer_text=writer_text)
    env["pad_writer"] = False
    writer_calls = _install_stubs(monkeypatch, env)
    result = dpo.deep_research("test query", verbose=False, max_rounds=3)

    # Both ~100-word sections were rewritten (~320-word revisions ship).
    revisions = [
        c for c in writer_calls if "REVISION REQUIRED" in (c.get("input_data") or "")
    ]
    assert len(revisions) == 2
    assert result["stats"]["revisions"] == 2
    assert "Revised substantive draft." in result["final_answer"]
