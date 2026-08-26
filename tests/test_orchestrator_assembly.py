"""Phase 3 tests: structured assembly helpers + JSON-mode deep pipeline.

Covers registry_to_sources (plan section 8.1), assemble_structured_report
renumbering/flagging (plan section 6.3), parse_exec_summary salvage,
sections_plain_text, and one end-to-end deep_research(output_format="json")
run with every LLM call monkeypatched (plan sections 9.3 / 9.4).
"""

import importlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest

import deep_research_orchestrator as dpo
from deep_research_structured import (
    assemble_structured_report,
    parse_exec_summary,
    sections_plain_text,
)
from memory.helpers import registry_to_sources
from models.report_schema import (
    BlockType,
    QualityMetrics,
    Metadata,
    Report,
    ReportBlock,
    ResearchReport,
    Section,
    Span,
)

# importlib (not `import ... as`): worker_agents/__init__.py re-exports the
# writer_agent function, shadowing the module name in the package namespace.
dmod = importlib.import_module("worker_agents.decomposition_agent")
rmod = importlib.import_module("worker_agents.retriever_agent")
wmod = importlib.import_module("worker_agents.writer_agent")
vmod = importlib.import_module("worker_agents.verifier_agent")

_CACHE_TMP_DIRS: list = []


class _FakeResponse:
    """Minimal stand-in for ModelResponse: .output_text / .output_parsed."""

    def __init__(self, text: str = "", parsed=None):
        self.output_text = text
        self.output_parsed = parsed


@pytest.fixture(autouse=True)
def _cleanup_evidence_cache_tmp_dirs():
    yield
    for d in _CACHE_TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _CACHE_TMP_DIRS.clear()


DOC_A = {
    "kind": "doc",
    "title": "Alpha Study",
    "document_name": "alpha.pdf",
    "page_number": 3,
    "citation": "Alpha Study, p. 3",
    "score": 0.9,
}
DOC_B = {
    "kind": "doc",
    "title": "Beta Report",
    "document_name": "beta.pdf",
    "page_number": 1,
    "citation": "Beta Report, p. 1",
    "score": 0.8,
}
WEB_A = {
    "kind": "web",
    "title": "Web A",
    "url": "https://example.com/a",
    "published_date": "2023",
    "score": 0.8,
}


def _make_report(*sections) -> ResearchReport:
    report = Report(
        metadata=Metadata(
            title="T",
            query="q",
            session_id="s1",
            generated_at="2025-01-01T00:00:00Z",
        ),
        executive_summary=[],
        sections=list(sections),
        sources=[],
    )
    return ResearchReport(report=report, quality=QualityMetrics())


def _para(text_cits) -> ReportBlock:
    return ReportBlock(
        type=BlockType.paragraph,
        spans=[Span(text=t, citations=list(c)) for t, c in text_cits],
    )


def _all_citations(rep: ResearchReport):
    return [
        c
        for s in rep.report.sections
        for b in s.blocks
        for sp in b.spans
        for c in sp.citations
    ]


class TestRegistryToSources:
    def test_web_record(self):
        recs = registry_to_sources({"W1": dict(WEB_A)}, ["W1"])
        assert len(recs) == 1
        r = recs[0]
        assert r["id"] == "source-w1"
        assert r["type"] == "webpage"
        assert r["title"] == "Web A"
        assert r["URL"] == "https://example.com/a"
        assert r["issued"] == {"date-parts": [[2023]]}
        assert r["citation_key"] == "W1"
        assert all(v is not None for v in r.values())

    def test_doc_record(self):
        recs = registry_to_sources({"D1": dict(DOC_A)}, ["D1"])
        assert len(recs) == 1
        r = recs[0]
        assert r["id"] == "source-d1"
        assert r["type"] == "report"
        assert r["title"] == "Alpha Study"
        assert r["citation_key"] == "D1"
        assert r["URL"] == ""

    def test_only_cited_keys(self):
        reg = {"D1": dict(DOC_A), "D2": dict(DOC_B)}
        recs = registry_to_sources(reg, ["D2"])
        assert [r["citation_key"] for r in recs] == ["D2"]

    def test_doc_dedupe_keeps_primary(self):
        reg = {
            "D1": dict(DOC_A),
            "D2": dict(DOC_B),
            "D3": dict(DOC_A, page_number=7),
        }
        recs = registry_to_sources(reg, ["D1", "D3", "D2"])
        assert len(recs) == 2
        assert recs[0]["citation_key"] == "D1"
        assert recs[1]["citation_key"] == "D2"

    def test_missing_fields_empty_not_null(self):
        web = {"kind": "web", "title": "NoDate", "url": "https://x.test/n"}
        recs = registry_to_sources({"W2": web}, ["W2"])
        r = recs[0]
        assert not r.get("issued")
        assert r.get("author", []) == []
        assert all(v is not None for v in r.values())


class TestAssemble:
    def _assemble(self, *sections, registry):
        return assemble_structured_report(
            sections=list(sections),
            registry=registry,
            user_query="q",
            session_id="s1",
            exec_paragraphs=["Ex."],
            verification_status={"confidence": "high"},
            title="T",
        )

    def test_renumber_first_appearance_order(self):
        s1 = Section(id="s1", heading="One", blocks=[_para([("A.", ["W1", "D1"]), ("B.", [])])])
        s2 = Section(id="s2", heading="Two", blocks=[_para([("C.", ["D2"]), ("D.", ["W1"])])])
        reg = {"D1": dict(DOC_A), "D2": dict(DOC_B), "W1": dict(WEB_A)}
        rep = self._assemble(s1, s2, registry=reg)
        assert _all_citations(rep) == ["1", "2", "3", "1"]
        types = [s.type for s in rep.report.sources]
        assert types == ["webpage", "report", "report"]
        assert all(s.citation_key for s in rep.report.sources)

    def test_invented_key_dropped_and_flagged(self):
        s1 = Section(id="s1", heading="One", blocks=[_para([("A.", ["W1"]), ("B.", ["D9"])])])
        rep = self._assemble(s1, registry={"W1": dict(WEB_A)})
        assert _all_citations(rep) == ["1"]
        assert rep.quality.verification.get("unresolvable_citations") == ["D9"]

    def test_bare_numeric_out_of_range_dropped(self):
        block = ReportBlock(type=BlockType.paragraph, text="Bare.", citations=["7", "W1"])
        s1 = Section(id="s1", heading="One", blocks=[block])
        rep = self._assemble(s1, registry={"W1": dict(WEB_A)})
        assert _all_citations(rep) == ["1"]
        assert "7" in rep.quality.verification.get("dropped_bare_citations", [])

    def test_quality_metrics(self):
        s1 = Section(id="s1", heading="One", blocks=[_para([("A fact here.", ["W1"])])])
        rep = self._assemble(s1, registry={"W1": dict(WEB_A)})
        assert rep.report.metadata.title == "T"
        assert rep.report.executive_summary == ["Ex."]
        assert rep.quality.sources_count == {"documents": 0, "web": 1}
        assert rep.quality.total_words > 0
        assert rep.quality.verification.get("confidence") == "high"
        assert isinstance(rep.quality.citation_density, dict)

    def test_empty_section_soft(self):
        rep = self._assemble(
            Section(id="s1", heading="Empty", blocks=[]), registry={}
        )
        assert rep.report.sections[0].blocks == []
        assert rep.report.sources == []
        assert rep.quality.total_words == 1  # the section heading counts
        assert isinstance(rep.quality.citation_density, dict)


class TestParseExecSummary:
    def test_json_array(self):
        assert parse_exec_summary('["A.", "B." ]') == ["A.", "B."]
        assert parse_exec_summary('["A.", "", "B." ]') == ["A.", "B."]

    def test_salvage_prose(self):
        assert parse_exec_summary("First para.\n\nSecond para.") == [
            "First para.",
            "Second para.",
        ]

    def test_garbage_never_raises(self):
        assert parse_exec_summary("{not json")  # list, no exception
        assert parse_exec_summary("") == []


class TestSectionsPlainText:
    def test_markers_rendered(self):
        s = Section(
            id="s1",
            heading="H",
            blocks=[_para([("abc", ["D1", "W2"]), ("def", [])])],
        )
        assert sections_plain_text(s) == "abc [D1, W2] def"


PLAN_JSON = json.dumps(
    {
        "is_simple": False,
        "report_title": "Test Report Title",
        "sub_questions": [
            {
                "id": "sq1",
                "question": "What is X?",
                "angle": "definition",
                "expected_sources": "both",
                "priority": 1,
                "heading": "Section One",
            },
            {
                "id": "sq2",
                "question": "How does X work?",
                "angle": "mechanics",
                "expected_sources": "both",
                "priority": 2,
                "heading": "Section Two",
            },
        ],
    }
)
SUFFICIENT_JSON = json.dumps(
    {
        "is_sufficient": True,
        "summary": "enough evidence",
        "missing_aspects": [],
        "follow_up_queries": [],
    }
)
CRITIC_JSON = json.dumps(
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


def _json_section(i):
    if i == 0:
        return json.dumps(
            {
                "id": "section-one",
                "heading": "Section One",
                "blocks": [
                    {
                        "type": "paragraph",
                        "spans": [
                            {"text": "First fact.", "citations": ["W1"]},
                            {"text": "Doc fact.", "citations": ["D1"]},
                        ],
                    }
                ],
            }
        )
    if i == 1:
        return json.dumps(
            {
                "id": "section-two",
                "heading": "Section Two",
                "blocks": [
                    {
                        "type": "paragraph",
                        "spans": [
                            {"text": "Second doc fact.", "citations": ["D2"]},
                            {"text": "Invented key fact.", "citations": ["D9"]},
                            {"text": "Back to web.", "citations": ["W1"]},
                        ],
                    }
                ],
            }
        )
    raise AssertionError(f"unexpected writer call {i}")


def _install_json_stubs(monkeypatch):
    monkeypatch.setattr(
        dpo, "_read_doc_catalog",
        lambda: [
            {"document_name": "alpha.pdf", "document_title": "Alpha Study"},
            {"document_name": "beta.pdf", "document_title": "Beta Report"},
        ],
    )
    ecache_mod = importlib.import_module("memory.evidence_cache")
    cache_tmp_dir = tempfile.mkdtemp(prefix="evidence_cache_asm_")
    _CACHE_TMP_DIRS.append(cache_tmp_dir)
    monkeypatch.setattr(
        ecache_mod, "EVIDENCE_CACHE_DB_PATH",
        Path(cache_tmp_dir) / "evidence_cache_test.db",
    )
    monkeypatch.setattr(ecache_mod, "_purged_this_process", False)
    monkeypatch.setattr(
        rmod, "retrieve_document",
        lambda *a, **k: {
            "query": a[0] if a else "",
            "chunks": [
                {
                    "document_name": "alpha.pdf",
                    "document_title": "Alpha Study",
                    "chunk_id": "c1",
                    "content": "alpha chunk",
                    "score": 0.9,
                },
                {
                    "document_name": "beta.pdf",
                    "document_title": "Beta Report",
                    "chunk_id": "c2",
                    "content": "beta chunk",
                    "score": 0.8,
                },
            ],
        },
    )
    monkeypatch.setattr(
        rmod, "web_search",
        lambda query: {
            "query": query,
            "results": [
                {
                    "title": "Web A",
                    "url": "https://example.com/a",
                    "content": "web content",
                    "score": 0.9,
                }
            ],
        },
    )
    monkeypatch.setattr(dmod, "run_model", lambda *a, **k: _FakeResponse(text=PLAN_JSON))
    monkeypatch.setattr(
        rmod, "run_model", lambda *a, **k: _FakeResponse(text=SUFFICIENT_JSON)
    )
    calls = []

    def writer_stub(*a, **k):
        calls.append(k)
        return _FakeResponse(text=_json_section(len(calls) - 1))

    monkeypatch.setattr(wmod, "run_model", writer_stub)
    monkeypatch.setattr(
        vmod, "run_model", lambda *a, **k: _FakeResponse(text=CRITIC_JSON)
    )
    monkeypatch.setattr(
        dpo, "run_model",
        lambda *a, **k: _FakeResponse(text=json.dumps(["Para one.", "Para two."])),
    )
    return calls


def test_deep_research_json_mode_e2e(monkeypatch):
    calls = _install_json_stubs(monkeypatch)
    result = dpo.deep_research(
        "test research query", verbose=False, max_rounds=3, output_format="json"
    )

    # JSON mode: no Markdown final_answer; report_json carries the document.
    assert result["final_answer"] == ""
    state = result["state"]
    assert "report_json" in state
    rep = ResearchReport.model_validate_json(state["report_json"])
    assert rep.schema_version == "1.0"

    # Title comes from the decomposer plan (report_title field).
    assert rep.report.metadata.title == "Test Report Title"
    assert rep.report.metadata.query == "test research query"

    # Executive summary parsed from the JSON array.
    assert rep.report.executive_summary == ["Para one.", "Para two."]

    # Sections preserved in order.
    assert [s.heading for s in rep.report.sections] == [
        "Section One",
        "Section Two",
    ]

    # Every citation renumbered to a digit; 3 unique sources (W1, D1, D2).
    cits = _all_citations(rep)
    assert all(c.isdigit() for c in cits)
    assert set(cits) == {"1", "2", "3"}
    assert "D9" not in cits
    assert len(rep.report.sources) == 3
    types = sorted(s.type for s in rep.report.sources)
    assert types == ["report", "report", "webpage"]

    # Invented key flagged; no bare numerics dropped (none were present).
    q = rep.quality
    assert q.verification.get("unresolvable_citations") == ["D9"]
    assert q.verification.get("dropped_bare_citations") == []
    assert q.sources_count == {"documents": 2, "web": 1}
    assert q.total_words > 0
    assert isinstance(q.citation_density, dict)

    # Legacy state compatibility for the UI / standard handlers.
    assert state["citation_density"] == q.citation_density
    assert state["verification"] == ""
    assert state["draft"] == ""
    secs = state["sections"]
    assert [s["id"] for s in secs] == ["sq1", "sq2"]
    assert all(set(s) == {"id", "heading", "text"} for s in secs)
    assert all(s["text"] for s in secs)

    # Writer JSON mode was actually used (two section calls, no synthesis).
    assert len(calls) == 2
