"""Orphan-subheading hygiene in assemble_structured_report.

Weak models occasionally emit a subsection heading and then write no body
for it (the PDF shows one heading stacked on the next). The section-level
30-word guard cannot see this when the rest of the section is full, so the
assembly drops heading blocks whose content zone has no renderable content.
"""

from deep_research_structured import (
    _drop_orphan_subheadings,
    assemble_structured_report,
)
from models.report_schema import (
    BlockType,
    ResearchReport,
    ReportBlock,
    Section,
    Span,
)

METADATA = {
    "title": "T",
    "query": "q",
    "session_id": "s1",
    "generated_at": "2026-01-01T00:00:00Z",
}


def _h(level, text):
    return ReportBlock(type=BlockType.heading, level=level, text=text)


def _para(text, words=1):
    t = text if words == 1 else " ".join(f"word{i}" for i in range(words))
    return ReportBlock(
        type=BlockType.paragraph,
        spans=[Span(text=t, citations=[])],
    )


def _section(*blocks):
    return Section(id="s1", heading="S", blocks=list(blocks))


def _report(section):
    return ResearchReport.model_validate(
        {
            "schema_version": "1.0",
            "report": {
                "metadata": METADATA,
                "sections": [section.model_dump()],
                "sources": [],
            },
            "quality": {},
        }
    )


def test_back_to_back_h3s_then_content_drops_only_the_empty_ones():
    rep = _report(
        _section(
            _h(3, "Readout Error Mitigation"),
            _h(3, "GHZ Benchmarking"),
            _h(3, "Entanglement via Schmidt Decomposition"),
            _para("The benchmark results follow here.", words=40),
        )
    )
    gaps = _drop_orphan_subheadings(rep)
    heads = [b.text for b in rep.report.sections[0].blocks if b.type == BlockType.heading]
    # The two unwritten H3s go; the one whose zone reaches the paragraph stays.
    assert heads == ["Entanglement via Schmidt Decomposition"]
    assert gaps == [
        "orphan_subheading: Readout Error Mitigation",
        "orphan_subheading: GHZ Benchmarking",
    ]


def test_h3_followed_by_paragraph_is_kept():
    rep = _report(_section(_h(3, "Methods"), _para("We measured error rates.", words=35)))
    assert _drop_orphan_subheadings(rep) == []
    assert [b.type for b in rep.report.sections[0].blocks] == [
        BlockType.heading,
        BlockType.paragraph,
    ]


def test_h2_whose_zone_is_only_unwritten_h3s_is_orphan_too():
    rep = _report(_section(_h(2, "Part I"), _h(3, "A"), _h(3, "B")))
    gaps = _drop_orphan_subheadings(rep)
    assert rep.report.sections[0].blocks == []
    assert len(gaps) == 3  # the H2 and both unwritten H3s


def test_h2_kept_when_a_deeper_subsection_has_content():
    rep = _report(
        _section(_h(2, "Part I"), _h(3, "A"), _h(3, "B"), _para("Body for B.", words=35))
    )
    _drop_orphan_subheadings(rep)
    # A's zone ends at the next same-level heading B, and B is a heading (not
    # content) -> A is orphan and dropped. B reaches the paragraph -> kept.
    # Part I's zone contains the paragraph -> kept.
    assert [b.text for b in rep.report.sections[0].blocks if b.type == BlockType.heading] == [
        "Part I",
        "B",
    ]


def test_idempotent():
    rep = _report(_section(_h(3, "A"), _h(3, "B"), _para("x", words=35)))
    first = _drop_orphan_subheadings(rep)
    second = _drop_orphan_subheadings(rep)
    assert first != [] and second == []


def test_non_heading_blocks_never_touched():
    para = _para("Just prose.", words=35)
    before = para.model_dump_json()
    rep = _report(_section(para))
    assert _drop_orphan_subheadings(rep) == []
    assert rep.report.sections[0].blocks[0].model_dump_json() == before


def test_real_report_shape_from_hilbert_file():
    # [para, H3 Readout, H3 GHZ, H3 Entanglement, para, H3 Trapped, para, para]
    rep = _report(
        _section(
            _para("Intro paragraph.", words=35),
            _h(3, "Readout Error Mitigation"),
            _h(3, "GHZ Benchmarking"),
            _h(3, "Entanglement via Schmidt Decomposition"),
            _para("Schmidt content.", words=35),
            _h(3, "Trapped-Ion Control Stack"),
            _para("Ion content one.", words=35),
            _para("Ion content two.", words=35),
        )
    )
    gaps = _drop_orphan_subheadings(rep)
    heads = [b.text for b in rep.report.sections[0].blocks if b.type == BlockType.heading]
    assert heads == ["Entanglement via Schmidt Decomposition", "Trapped-Ion Control Stack"]
    assert sorted(gaps) == [
        "orphan_subheading: GHZ Benchmarking",
        "orphan_subheading: Readout Error Mitigation",
    ]
    assert len(rep.report.sections[0].blocks) == 6  # 2 headings + 4 paragraphs


def test_end_to_end_assemble_records_gaps():
    rep = assemble_structured_report(
        sections=[
            Section(
                id="s1",
                heading="Experimental Realizations",
                blocks=[
                    _para("evidence", words=40),
                    _h(3, "Orphan One"),
                    _h(3, "Orphan Two"),
                    _h(3, "Real"),
                    _para("real content", words=40),
                ],
            )
        ],
        registry={},
        user_query="q",
        session_id="s1",
        exec_paragraphs=["Summary."],
        verification_status={"gaps": ["pre-existing"]},
        title="T",
    )
    heads = [b.text for b in rep.report.sections[0].blocks if b.type == BlockType.heading]
    assert heads == ["Real"]
    assert rep.quality.verification["gaps"] == [
        "pre-existing",
        "orphan_subheading: Orphan One",
        "orphan_subheading: Orphan Two",
    ]
