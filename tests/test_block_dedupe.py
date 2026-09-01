"""Tests for degenerate duplicate-block defenses (writer salvage + assembly).

Two layers:
1. worker_agents.writer_agent._recover_trailing_blocks drops recovered blocks
   that are EXACT duplicates of a block already in the section (normalized
   via model_dump_json) or of an earlier recovery in the same remainder,
   while still salvaging genuinely novel blocks.
2. deep_research_structured.assemble_structured_report collapses runs of
   ADJACENT byte-identical blocks to one (conservative: non-adjacent and
   different blocks are kept).

Plain pytest unit tests, no network, no LLM calls: run from the repo root
with ./venv/bin/python -m pytest tests/test_block_dedupe.py -q
"""

import copy
import importlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest

import deep_research_orchestrator as dpo
import deep_research_structured as dds
from deep_research_structured import assemble_structured_report
from models.report_schema import ReportBlock, ResearchReport, Section

wmod = importlib.import_module("worker_agents.writer_agent")
dmod = importlib.import_module("worker_agents.decomposition_agent")
rmod = importlib.import_module("worker_agents.retriever_agent")
vmod = importlib.import_module("worker_agents.verifier_agent")

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
    """Minimal stand-in for ModelResponse: .output_text / .output_parsed."""

    def __init__(self, text: str = "", parsed=None):
        self.output_text = text
        self.output_parsed = parsed


def _para(text: str, cites=()) -> dict:
    return {"type": "paragraph", "spans": [{"text": text, "citations": list(cites)}]}


def _note(text: str, cites=(), ctype: str = "note") -> dict:
    return {
        "type": "callout",
        "callout_type": ctype,
        "spans": [{"text": text, "citations": list(cites)}],
    }


def _list_item(text: str) -> dict:
    return {"type": "unordered_list", "items": [{"text": text, "citations": []}]}


# 38 words: long enough that a section of deduped blocks survives the
# assembly 30-word empty-section guard.
_PAD = (
    "The field disagrees most sharply over the deployment timeframe: the "
    "roadmap literature sketches fault tolerance within the 2029 to 2033 "
    "window while the applications literature rewards only quantum-native "
    "advantages right now, and both bodies of work stay internally "
    "consistent."
)
_NOTE_X = "Where the field disagrees most is over timeframe [15, 32]."


def _synth_text(trailing_count: int, last_note_text: str = _NOTE_X) -> str:
    """The user's incident shape: a 6-block synthesis object that closes,
    then `trailing_count` copies of its final note as trailing JSON."""
    blocks = [
        _para("intro"),
        _note("warning text", ctype="warning"),
        _list_item("one item"),
        _note("info text", ctype="info"),
        _note("distinct note one"),
        _note(last_note_text, cites=("15", "32")),
    ]
    closed = json.dumps({"id": "synthesis", "heading": "Synthesis", "blocks": blocks})
    tail = "".join(json.dumps(_note(last_note_text, cites=("15", "32"))) for _ in range(trailing_count))
    return closed + tail


def _recover_like_call_site(text: str):
    """Mirror the writer's call site: extract the object, recover the
    trailing blocks against the section's existing blocks, extend."""
    obj, obj_end = wmod._extract_json_object_span(text)
    section = Section.model_validate(obj)
    recovered = wmod._recover_trailing_blocks(text, obj_end, section.blocks)
    section.blocks.extend(recovered)
    return section, recovered


# ---------------------------------------------------------------------------
# Layer 1: _recover_trailing_blocks duplicate guard
# ---------------------------------------------------------------------------


def test_recover_drops_trailing_copies_of_section_tail():
    # Section ends with note X; the model re-emits X five times after the
    # premature close. None may be recovered; the section stays at 6 blocks.
    text = _synth_text(5)
    section, recovered = _recover_like_call_site(text)
    assert recovered == []
    assert len(section.blocks) == 6
    keys = [b.model_dump_json() for b in section.blocks]
    assert len(set(keys)) == 6  # the section itself was never duplicated


def test_recover_keeps_novel_drops_duplicate_within_trailing():
    # Trailing: novel paragraph Y, then a duplicate Y, then novel note Z:
    # exactly one Y plus Z survive (internal dedupe, cap intact).
    y = _para("A genuinely novel paragraph that was not in the object.")
    z = _note("Another novel note that was not in the object.")
    para = _para("intro")
    closed = json.dumps({"id": "s", "heading": "S", "blocks": [para]})
    text = closed + json.dumps(y) + json.dumps(y) + json.dumps(z)
    section, recovered = _recover_like_call_site(text)
    assert [b.type.value for b in section.blocks] == ["paragraph", "paragraph", "callout"]
    assert len(recovered) == 2


def test_recover_still_salvages_novel_blocks_regardless_of_existing():
    # Regression of the salvage purpose (mirrors
    # test_writer_json.test_premature_close_recovers_trailing_blocks at the
    # unit level): distinct novel blocks are all recovered.
    para = _para("first block text", cites=("D1",))
    heading = {"type": "heading", "level": 3, "text": "More"}
    table = {
        "type": "comparison_table",
        "caption": "c",
        "columns": ["A"],
        "rows": [["cell"]],
    }
    closed = json.dumps({"id": "market", "heading": "Market", "blocks": [para]})
    text = closed + "," + json.dumps(heading) + "," + json.dumps(table)
    section, recovered = _recover_like_call_site(text)
    assert [b.type.value for b in section.blocks] == [
        "paragraph",
        "heading",
        "comparison_table",
    ]
    assert len(recovered) == 2


def test_user_case_14_identical_trailing_notes_yields_6_blocks():
    # The exact incident: 6-block section + 14x identical trailing note
    # (well under the _MAX_RECOVERED_BLOCKS=20 cap) -> final 6 blocks.
    section, recovered = _recover_like_call_site(_synth_text(14))
    assert recovered == []
    assert len(section.blocks) == 6
    assert [b.type.value for b in section.blocks] == [
        "paragraph",
        "callout",
        "unordered_list",
        "callout",
        "callout",
        "callout",
    ]


# ---------------------------------------------------------------------------
# Layer 2: assemble_structured_report adjacent-identical collapse
# ---------------------------------------------------------------------------


def _assemble(*sections: Section) -> Section:
    rep = assemble_structured_report(
        sections=list(sections),
        registry={},
        user_query="q",
        session_id="s1",
        exec_paragraphs=[],
        verification_status={"confidence": "high"},
        title="T",
    )
    return rep.report.sections[0]


def _section(blocks: list) -> Section:
    # id != "synthesis" so the empty-guard semantics stay the documented
    # gap-notice path; every block carries 30+ words of substance so the
    # deduped section always clears _MIN_SECTION_WORDS.
    validated = [
        b if isinstance(b, ReportBlock) else ReportBlock.model_validate(b)
        for b in blocks
    ]
    return Section(id="s1", heading="S", blocks=validated)


def _pad(text: str) -> str:
    """Ensure a block's text is 30+ words (the assembly empty-section guard)."""
    words = text.split()
    if len(words) < 31:
        return text + " " + " ".join(["substance"] * (31 - len(words)))
    return text


def test_assemble_collapses_adjacent_identical_callouts():
    a_text = _pad("Opening paragraph.")
    n_text = _pad(_NOTE_X)
    b_text = _pad("Closing paragraph.")
    a = _para(a_text)
    n = _note(n_text, ("15", "32"))
    b = _para(b_text)
    out = _assemble(_section([a, n, n, n, b]))
    assert [blk.type.value for blk in out.blocks] == ["paragraph", "callout", "paragraph"]
    # Renumbering (empty registry) drops citations, so compare span TEXT —
    # the dedupe decision is content-identity, which the text reflects.
    assert [s.text for blk in out.blocks for s in blk.spans] == [a_text, n_text, b_text]


def test_assemble_collapses_adjacent_identical_non_callout():
    x = _para(_pad("Doubled prose paragraph."))
    out = _assemble(_section([x, x]))
    assert len(out.blocks) == 1
    assert out.blocks[0].type.value == "paragraph"


def test_assemble_keeps_distinct_adjacent_callouts():
    n1 = _note(_pad("First disagreement note."))
    n2 = _note(_pad("Second, different disagreement note."))
    out = _assemble(_section([n1, n2]))
    assert len(out.blocks) == 2


def test_assemble_keeps_identical_non_adjacent_blocks():
    n_text = _pad(_NOTE_X)
    p_text = _pad("Interposing prose between the duplicates.")
    n = _note(n_text)
    p = _para(p_text)
    out = _assemble(_section([n, p, n]))
    assert len(out.blocks) == 3
    assert [s.text for blk in out.blocks for s in blk.spans] == [n_text, p_text, n_text]


# ---------------------------------------------------------------------------
# Pipeline-level regression: the 2026-08 incident through the live path
# ---------------------------------------------------------------------------


def _install_pipeline_stubs(monkeypatch, env: dict) -> list:
    """Install the same fake set tests/test_deep_pipeline.py uses: every
    LLM surface + retrieval stubbed, evidence cache redirected to a temp
    DB, writer responses padded to the 300-word contract floor. Returns
    the writer call log."""
    monkeypatch.setattr(dpo, "_read_doc_catalog", lambda: [])
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


_PIPELINE_NOTE = (
    "Where the field disagrees most is over timeframe: the roadmap "
    "literature sketches fault tolerance in the 2029 to 2033 window "
    "while the applications sections reward only quantum-native "
    "advantages now."
)


def _degenerate_synth_json() -> str:
    """The incident shape: a properly CLOSED 6-block synthesis object
    (last block = a "note" callout) followed by a second note callout
    re-emitted 14x as trailing JSON (the degenerate loop). The looped note
    is deliberately DISTINCT from the object's final note, so the one
    surviving copy is a genuine addition: 6 originals + 1 = 7 blocks."""
    note_x = {
        "type": "callout",
        "callout_type": "note",
        "spans": [{"text": _PIPELINE_NOTE, "citations": ["15", "32"]}],
    }
    note_d = {
        "type": "callout",
        "callout_type": "note",
        "spans": [{"text": "Distinct note " + " ".join(f"d{j}" for j in range(60)), "citations": []}],
    }
    note_y = {
        "type": "callout",
        "callout_type": "note",
        "spans": [{"text": "Looped note " + " ".join(f"y{j}" for j in range(60)), "citations": []}],
    }

    def _callout(prefix: str, ctype: str) -> dict:
        return {
            "type": "callout",
            "callout_type": ctype,
            "spans": [{"text": prefix + " " + " ".join(f"tok{j}" for j in range(60)), "citations": []}],
        }

    blocks = [
        {
            "type": "paragraph",
            "spans": [{"text": "Opening synthesis connecting the sections. " + " ".join(f"conn{j}" for j in range(40)), "citations": []}],
        },
        _callout("Warning", "warning"),
        {
            "type": "unordered_list",
            "items": [
                {"text": "Implication one " + " ".join(f"t{j}" for j in range(30)), "citations": []},
                {"text": "Implication two " + " ".join(f"u{j}" for j in range(30)), "citations": []},
            ],
        },
        _callout("Info", "info"),
        note_d,
        note_x,
    ]
    closed = json.dumps({"id": "synthesis", "heading": "Synthesis", "blocks": blocks})
    tail = "".join(json.dumps(note_y) for _ in range(14))
    return closed + tail


def _assert_deduped(section: Section, expected_total: int) -> None:
    """The incident note and the looped note must each appear exactly once,
    the section must hold the expected block count, and no adjacent blocks
    may be byte-identical."""
    for marker in ("Where the field disagrees most is over timeframe", "Looped note"):
        hits = [
            b
            for b in section.blocks
            if b.type.value == "callout"
            and any(sp.text.startswith(marker) for sp in b.spans)
        ]
        assert len(hits) == 1, (
            f"{section.heading}: callout {marker!r} appears {len(hits)}x — "
            "degenerate duplicate blocks survived dedupe"
        )
    assert len(section.blocks) == expected_total, (
        f"{section.heading}: {len(section.blocks)} blocks, expected {expected_total}"
    )
    keys = [b.model_dump_json() for b in section.blocks]
    for a, b in zip(keys, keys[1:]):
        assert a != b, f"{section.heading}: adjacent byte-identical blocks survived"


def test_pipeline_synthesis_note_loop_deduped_end_to_end(monkeypatch):
    """Pipeline-level guard for the 2026-08 incident, driven through the
    live path (real write_synthesis → _recover_trailing_blocks →
    assemble_structured_report): a synthesis draft that closed its 6-block
    section JSON then degenerate-looped, re-emitting a note callout 14x as
    trailing JSON, must still ship each note exactly once — 7 total blocks
    (the 6 originals plus one surviving copy), with no adjacent duplicates.
    The writer-stage snapshot (taken by copying the returned Section, since
    assembly mutates it in place) catches the salvage guard; the final
    report catches the assembly guard. A refactor that removes either dedupe
    breaks this test loudly."""
    # Sentinel: the two dedupe guards must still exist in their modules.
    assert hasattr(wmod, "_recover_trailing_blocks"), (
        "writer salvage duplicate guard (_recover_trailing_blocks) is gone — "
        "the 2026-08 note-loop pipeline guard cannot protect the pipeline"
    )
    assert hasattr(dds, "_collapse_adjacent_duplicate_blocks"), (
        "assembly adjacent-duplicate collapse (_collapse_adjacent_duplicate_blocks) "
        "is gone — the 2026-08 note-loop pipeline guard cannot protect the pipeline"
    )

    def writer_text(i, k):
        if "SYNTHESIS" in (k.get("instructions") or ""):
            return _degenerate_synth_json()
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

    env = {
        "plan_json": json.dumps(
            {
                "is_simple": False,
                "report_title": "Test Report",
                "sub_questions": [
                    {
                        "question": f"Q{i} what is the core finding?",
                        "angle": "angle",
                        "expected_sources": "web",
                        "priority": i,
                        "heading": f"Section {i}",
                    }
                    for i in (1, 2, 3)  # synthesis requires >= 3 sections
                ],
            }
        ),
        "sufficiency_json": json.dumps(
            {"is_sufficient": True, "summary": "sufficient", "missing_aspects": [], "follow_up_queries": []}
        ),
        "critic_text": json.dumps(
            {
                "confidence_level": "high",
                "overall_summary": "solid",
                "hallucinated_claims": [],
                "unsupported_claims": [],
                "per_section": [
                    {"section_id": f"sq{i}", "grounded": True, "depth_ok": True, "gaps": []}
                    for i in (1, 2, 3)
                ],
                "re_retrieve_suggested": False,
                "specific_queries": [],
            }
        ),
        "writer_text": writer_text,
        "web_results": lambda query: [
            {"title": "Web One", "url": "https://ex/u1", "content": "content", "score": 0.9}
        ],
        "exec_text": "Synthesized executive summary prose.",
    }
    _install_pipeline_stubs(monkeypatch, env)

    # Spy on the REAL write_synthesis (the salvage dedupe runs inside it)
    # to observe the writer-stage section; deepcopy it, because assembly's
    # collapse mutates the Section in place (pydantic keeps object identity).
    synth_returns = []
    orig_write_synthesis = dpo.write_synthesis

    def spy(*a, **k):
        out = orig_write_synthesis(*a, **k)
        synth_returns.append(copy.deepcopy(out))
        return out

    monkeypatch.setattr(dpo, "write_synthesis", spy)

    result = dpo.deep_research("test research query", verbose=False, max_rounds=3, output_format="json")

    # Writer stage: the 13 trailing copies of the looped note are exact
    # duplicates of the one surviving copy, so salvage appends exactly one
    # new block — 7 total, each note once.
    assert len(synth_returns) == 1
    _assert_deduped(synth_returns[0], 7)

    # Final structured report: same — no duplicate may ship.
    assert result["stats"].get("synthesis_failed") is False
    rep = ResearchReport.model_validate_json(result["state"]["report_json"])
    synth = [s for s in rep.report.sections if s.heading == "Synthesis"]
    assert len(synth) == 1
    _assert_deduped(synth[0], 7)

