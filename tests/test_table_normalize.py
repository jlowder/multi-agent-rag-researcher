"""Table-width normalization in assemble_structured_report.

Ragged comparison tables (rows whose cell count differs from len(columns))
pass the producer schema but crash downstream consumers that index rows up
to the header width (paperbot's normalizer: ``row[i]`` on a 3-column table
with 2-cell rows). The assembly pass must make every row exactly as wide as
its header, padded/truncated, at the source.
"""

import pytest  # noqa: F401  (kept for symmetry with sibling test modules)

from deep_research_structured import (
    _normalize_comparison_table_widths,
    assemble_structured_report,
)
from models.report_schema import (
    BlockType,
    ResearchReport,
    ReportBlock,
    Section,
    Span,
)


def _table(columns, rows):
    return ReportBlock(
        type=BlockType.comparison_table, columns=list(columns), rows=rows
    )


def _report(*blocks):
    return ResearchReport.model_validate(
        {
            "schema_version": "1.0",
            "report": {
                "metadata": {
                    "title": "T",
                    "query": "q",
                    "session_id": "s1",
                    "generated_at": "2026-01-01T00:00:00Z",
                },
                "sections": [
                    {"id": "s1", "heading": "S", "blocks": [b.model_dump() for b in blocks]}
                ],
                "sources": [],
            },
            "quality": {},
        }
    )


def test_ragged_rows_are_padded_to_header_width():
    rep = _report(
        _table(
            ["Property", "General MPS/PEPS", "MERA"],
            [
                ["2D / 3D tensor train", "Branching tree"],
                ["Bond dim", "Shared across bonds"],
            ],
        )
    )
    changed = _normalize_comparison_table_widths(rep)
    table = rep.report.sections[0].blocks[0]
    assert changed == 2
    for row in table.rows:
        assert len(row) == 3
    # original cell text is preserved; the pad cell is an empty span
    assert [c.text for c in table.rows[0]] == ["2D / 3D tensor train", "Branching tree", ""]
    assert [c.text for c in table.rows[1]] == ["Bond dim", "Shared across bonds", ""]
    assert all(isinstance(c, Span) for row in table.rows for c in row)


def test_overlong_rows_are_truncated_to_header_width():
    rep = _report(
        _table(
            ["A", "B"],
            [["one", "two", "extra-cell-that-must-go"]],
        )
    )
    _normalize_comparison_table_widths(rep)
    row = rep.report.sections[0].blocks[0].rows[0]
    assert [c.text for c in row] == ["one", "two"]


def test_non_string_cells_are_coerced_to_spans_with_str_text():
    rep = _report(_table(["A", "B", "C"], [["x", "y", "z"]]))
    # Simulate weak-model residue mutating the built model in place
    # (no validate_assignment): a bare number replaces a cell.
    rep.report.sections[0].blocks[0].rows[0][1] = 123
    _normalize_comparison_table_widths(rep)
    row = rep.report.sections[0].blocks[0].rows[0]
    assert [c.text for c in row] == ["x", "123", "z"]
    assert all(isinstance(c, Span) for c in row)


def test_table_without_columns_is_left_intact():
    rep = _report(
        _table([], [["a", "b"], ["c", "d"]])
    )
    changed = _normalize_comparison_table_widths(rep)
    assert changed == 0
    row = rep.report.sections[0].blocks[0].rows[0]
    assert [c.text for c in row] == ["a", "b"]  # untouched, still 2 cells


def test_non_table_blocks_are_untouched():
    para = ReportBlock(
        type=BlockType.paragraph,
        spans=[Span(text="Some text.", citations=["D1"])],
    )
    rep = _report(para)
    before = para.model_dump_json()
    assert _normalize_comparison_table_widths(rep) == 0
    assert rep.report.sections[0].blocks[0].model_dump_json() == before


def test_well_formed_table_needs_no_change():
    rep = _report(
        _table(
            ["A", "B"],
            [["1", "2"], ["3", "4"]],
        )
    )
    assert _normalize_comparison_table_widths(rep) == 0


def _paragraph(words=40):
    return ReportBlock(
        type=BlockType.paragraph,
        spans=[Span(text=" ".join(f"word{i}" for i in range(words)), citations=[])],
    )


def test_assemble_end_to_end_ragged_tables_are_well_formed():
    ragged_s1 = _table(
        ["Framework", "Primitive object", "Hilbert space is", "Strengths"],
        [
            ["C*-algebras", "Operators", "GNS construction"],  # 3 of 4
            ["von Neumann", "Bounded operators", "GNS", "Rigorous"],  # ok
            ["Quantum info", "Qubits"],  # 2 of 4
        ],
    )
    ragged_synth = _table(
        ["Property", "General MPS/PEPS", "MERA"],
        [
            ["Geometry", "Tensor train"],  # 2 of 3, like the hilbert report
        ],
    )
    rep = assemble_structured_report(
        sections=[
            # 30+ words each so the empty-section guard keeps the tables.
            Section(id="s1", heading="Beyond standard Hilbert spaces",
                    blocks=[_paragraph(), ragged_s1]),
            Section(id="synthesis", heading="Synthesis", blocks=[_paragraph(), ragged_synth]),
        ],
        registry={},
        user_query="q",
        session_id="s1",
        exec_paragraphs=["Summary."],
        verification_status={},
        title="T",
    )
    tables = [
        b
        for s in rep.report.sections
        for b in s.blocks
        if b.type == BlockType.comparison_table
    ]
    assert len(tables) == 2
    for t in tables:
        for row in t.rows:
            assert len(row) == len(t.columns), f"ragged row {row!r} vs {t.columns!r}"
    # the padded third/fourth cells are empty text
    assert tables[0].rows[0][3].text == ""
    assert tables[1].rows[0][2].text == ""
