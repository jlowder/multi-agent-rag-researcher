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

import importlib
import json

from deep_research_structured import assemble_structured_report
from models.report_schema import ReportBlock, Section

wmod = importlib.import_module("worker_agents.writer_agent")


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
