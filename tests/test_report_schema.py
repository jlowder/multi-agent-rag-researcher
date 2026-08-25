"""Tests for the structured report schema (models/report_schema.py)."""
import copy

import pytest
from pydantic import ValidationError

from models import (
    BlockType,
    ReportBlock,
    ResearchReport,
    Section,
    Span,
    compute_citation_density,
    count_total_words,
    drop_bare_numeric_citations,
    find_unresolvable_citations,
    to_json_schema,
)

SECTION = {
    "id": "fundamentals",
    "heading": "Diffusion Model Fundamentals",
    "blocks": [
        {
            "type": "paragraph",
            "spans": [
                {"text": "Diffusion models are a class of ", "citations": []},
                {"text": "latent variable generative models", "citations": ["D1"]},
                {
                    "text": "that synthesize data by learning a stochastic transformation.",
                    "citations": ["D1", "D2"],
                },
            ],
        },
        {"type": "heading", "level": 3, "text": "Forward Process"},
        {
            "type": "paragraph",
            "spans": [
                {"text": "The forward process gradually adds Gaussian noise.", "citations": ["D1"]},
            ],
        },
    ],
}

SOURCES = [
    {
        "id": "source-d1",
        "type": "report",
        "title": "A Mathematical Introduction to Diffusion Models",
        "citation_key": "D1",
    },
    {"id": "source-d2", "type": "report", "title": "TESS 2", "citation_key": "D2"},
]


def _report_data(sections=None, executive_summary=None, sources=None):
    return {
        "report": {
            "metadata": {
                "title": "Diffusion Models: A Research Survey",
                "query": "How do diffusion models work?",
                "session_id": "sess-123",
                "generated_at": "2026-02-03T12:00:00Z",
            },
            "executive_summary": (
                executive_summary
                if executive_summary is not None
                else ["Diffusion models are latent variable generative models."]
            ),
            "sections": sections if sections is not None else [copy.deepcopy(SECTION)],
            "sources": sources if sources is not None else copy.deepcopy(SOURCES),
        },
        "quality": {
            "citation_density": {"overall": 12.5},
            "verification": {"verified": 5, "unverified": 0},
            "sources_count": {"web": 0, "report": 2},
            "total_words": 500,
        },
    }


def _single_span_section(citations, heading="S1", sid="s1"):
    return {
        "id": sid,
        "heading": heading,
        "blocks": [
            {"type": "paragraph", "spans": [{"text": "some text", "citations": list(citations)}]},
        ],
    }


def test_round_trip_research_report():
    report = ResearchReport.model_validate(_report_data())
    dumped = report.model_dump()
    assert dumped["schema_version"] == "1.0"
    section = dumped["report"]["sections"][0]
    assert section["id"] == "fundamentals"
    assert section["heading"] == "Diffusion Model Fundamentals"
    assert [s["citations"] for s in section["blocks"][0]["spans"]] == [[], ["D1"], ["D1", "D2"]]
    assert dumped["report"]["metadata"]["title"] == "Diffusion Models: A Research Survey"


def test_section_round_trip():
    section = Section.model_validate(copy.deepcopy(SECTION))
    assert section.heading == "Diffusion Model Fundamentals"
    assert section.id == "fundamentals"
    assert len(section.blocks) == 3


def test_invented_keys_flagged():
    sections = [
        _single_span_section(["D1", "D99"], heading="A", sid="a"),
        _single_span_section(["D99", "W7"], heading="B", sid="b"),
    ]
    report = ResearchReport.model_validate(_report_data(sections=sections))
    assert find_unresolvable_citations(report, {"D1": {"title": "t"}}) == ["D99", "W7"]


def test_empty_sources_valid():
    report = ResearchReport.model_validate(_report_data(sources=[]))
    assert report.report.sources == []


def test_missing_heading_raises():
    with pytest.raises(ValidationError):
        Section.model_validate({"id": "x"})


def test_missing_title_raises():
    data = _report_data()
    del data["report"]["metadata"]["title"]
    with pytest.raises(ValidationError):
        ResearchReport.model_validate(data)


def test_citations_shorthand_merges():
    block = ReportBlock(type="paragraph", text="Some words here.", citations=["D1"])
    assert block.spans == [Span(text="Some words here.", citations=["D1"])]


def test_density_hand_computed():
    words = [f"w{i:02d}" for i in range(100)]
    spans = [
        Span(text=" ".join(words[0:40]), citations=["D1", "D2"]),
        Span(text=" ".join(words[40:75]), citations=["W1"]),
        Span(text=" ".join(words[75:100]), citations=["D3"]),
    ]
    section = {
        "id": "density",
        "heading": "Density Check",
        "blocks": [{"type": "paragraph", "spans": [s.model_dump() for s in spans]}],
    }
    report = ResearchReport.model_validate(
        _report_data(
            sections=[section],
            executive_summary=["one two three four five six seven eight nine ten"],
        )
    )
    density = compute_citation_density(report)
    assert density["per_section"] == {"Density Check": 4.0}
    assert density["overall"] == round(4 / 110 * 100, 1)
    assert count_total_words(report) == 110


def test_density_zero_words():
    report = ResearchReport.model_validate(
        _report_data(
            sections=[{"id": "empty", "heading": "Empty", "blocks": []}],
            executive_summary=[],
        )
    )
    density = compute_citation_density(report)
    assert density["per_section"] == {"Empty": 0.0}
    assert density["overall"] == 0.0


def test_drop_bare_numeric_with_keys():
    report = ResearchReport.model_validate(_report_data(sections=[_single_span_section(["D1", "W2", "9"])]))
    dropped = drop_bare_numeric_citations(report, {"D1": {}, "W2": {}})
    assert dropped == ["9"]
    assert report.report.sections[0].blocks[0].spans[0].citations == ["D1", "W2"]


def test_drop_bare_numeric_renumbered():
    report = ResearchReport.model_validate(
        _report_data(sections=[_single_span_section(["1", "5", "99"])])
    )
    registry = {f"D{i}": {} for i in range(1, 11)}
    dropped = drop_bare_numeric_citations(report, registry)
    assert dropped == ["99"]
    assert report.report.sections[0].blocks[0].spans[0].citations == ["1", "5"]


MARKDOWN = """## Market Landscape

### Sub A

First fact [D1].
Second fact [W2, D1].

- bullet one [D1]
- bullet two

1. first item
2. second item [W2]

```python
x = 1
```

| Col A | Col B |
|---|---|
| a [D1] | b |

> Note text [W2]
"""


def test_from_markdown_basic():
    section = Section.from_markdown(MARKDOWN, {"D1": {}, "W2": {}})
    assert section.heading == "Market Landscape"
    assert section.id == "market-landscape"
    assert [b.type for b in section.blocks] == [
        BlockType.heading,
        BlockType.paragraph,
        BlockType.unordered_list,
        BlockType.ordered_list,
        BlockType.code_block,
        BlockType.comparison_table,
        BlockType.callout,
    ]
    assert section.blocks[0].level == 3
    assert section.blocks[0].text == "Sub A"
    para = section.blocks[1]
    assert para.spans[0].text == "First fact"
    assert para.spans[0].citations == ["D1"]
    assert para.spans[1].text == ". Second fact"
    assert para.spans[1].citations == ["W2", "D1"]
    assert section.blocks[2].items[0].citations == ["D1"]
    assert section.blocks[3].items[1].citations == ["W2"]
    code = section.blocks[4]
    assert code.language == "python"
    assert "x = 1" in code.text
    assert section.blocks[5].columns == ["Col A", "Col B"]
    assert section.blocks[6].callout_type == "note"
    assert section.blocks[6].spans[0].citations == ["W2"]


def test_from_markdown_no_heading():
    section = Section.from_markdown("just text", {})
    assert section.heading == ""
    assert section.id == ""
    assert section.blocks[0].type == BlockType.paragraph
    assert section.blocks[0].spans == [Span(text="just text", citations=[])]


def test_to_json_schema():
    schema = to_json_schema()
    assert isinstance(schema, dict)
    assert schema["title"] == "ResearchReport"
    assert "$defs" in schema or "definitions" in schema
