"""Structured report schema — canonical JSON output model (plan §3-4)."""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

_CITATION_MARKER_RE = re.compile(r"\[((?:[DW]\d+|\d+)(?:\s*,\s*(?:[DW]\d+|\d+))*)\]")
_INTERNAL_KEY_RE = re.compile(r"^[DW]\d+$")
_BULLET_RE = re.compile(r"^[-*]\s+(.*)$")
_ORDERED_ITEM_RE = re.compile(r"^\d+\.\s+")


class BlockType(str, Enum):
    heading = "heading"
    paragraph = "paragraph"
    ordered_list = "ordered_list"
    unordered_list = "unordered_list"
    callout = "callout"
    comparison_table = "comparison_table"
    code_block = "code_block"
    page_break = "page_break"
    citation_note = "citation_note"


class Span(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)


def _coerce_span(value: Any) -> Any:
    """Lenient cell coercion: a bare string becomes one uncited Span; dicts
    and Span instances pass through (missing 'citations' already defaults)."""
    if isinstance(value, str):
        return Span(text=value, citations=[])
    return value


class ReportBlock(BaseModel):
    type: BlockType
    text: str = ""
    spans: list[Span] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    level: int = 2
    items: list[Span] = Field(default_factory=list)
    caption: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Span]] = Field(default_factory=list)
    callout_type: Literal["note", "warning", "info"] = "note"
    callout_title: str = ""
    language: str = ""

    @field_validator("spans", "items", mode="before")
    @classmethod
    def _coerce_span_lists(cls, value: Any) -> Any:
        """Weak models emit bare strings where spans/items are expected
        ("\"rows\": [[\"a\", \"b\"]] style); coerce each str entry to a Span.
        A single bare string is treated as a one-item list."""
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return [_coerce_span(v) for v in value]
        return value

    @field_validator("rows", mode="before")
    @classmethod
    def _coerce_rows(cls, value: Any) -> Any:
        """Table cells arrive as plain strings from weak models; coerce each
        cell to a Span. Non-list rows (dicts etc.) pass through untouched so
        they still fail validation as before."""
        if not isinstance(value, list):
            return value
        rows: list[Any] = []
        for row in value:
            if isinstance(row, str):
                rows.append([_coerce_span(row)])
            elif isinstance(row, list):
                rows.append([_coerce_span(cell) for cell in row])
            else:
                rows.append(row)
        return rows

    @model_validator(mode="after")
    def _merge_citation_shorthand(self) -> "ReportBlock":
        if not self.spans and self.citations:
            self.spans = [Span(text=self.text, citations=list(self.citations))]
        return self


def _slugify(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def _split_citation_keys(inner: str) -> list[str]:
    return [entry.strip() for entry in inner.split(",") if entry.strip()]


def _split_spans(line: str) -> list[Span]:
    """Split `line` into Spans at bracketed citation markers.

    The text run before each marker becomes a span carrying that marker's
    citation keys (comma-split, stripped); a new span then starts with
    empty citations. Spans are whitespace-stripped; empty-text spans
    without citations are dropped, the final span is kept even if empty
    with citations.
    """
    spans: list[Span] = []
    pos = 0
    for m in _CITATION_MARKER_RE.finditer(line):
        spans.append(
            Span(text=line[pos:m.start()].strip(), citations=_split_citation_keys(m.group(1)))
        )
        pos = m.end()
    spans.append(Span(text=line[pos:].strip(), citations=[]))
    out = [s for s in spans[:-1] if s.text or s.citations]
    if spans and (spans[-1].text or spans[-1].citations):
        out.append(spans[-1])
    return out


def _single_span(line: str) -> Span:
    """Collapse a list item or table cell into one span: marker brackets
    are removed (surrounding whitespace collapsed) and every citation key
    on the line lands on the span, in order."""
    citations: list[str] = []

    def _repl(m: re.Match) -> str:
        citations.extend(_split_citation_keys(m.group(1)))
        return " "

    text = re.sub(r"\s{2,}", " ", _CITATION_MARKER_RE.sub(_repl, line)).strip()
    return Span(text=text, citations=citations)


def _table_block(rows: list[str]) -> ReportBlock:
    def cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    parsed = [cells(r) for r in rows]
    return ReportBlock(
        type=BlockType.comparison_table,
        columns=parsed[0],
        rows=[[_single_span(c) for c in row] for row in parsed[1:]],
    )


class Section(BaseModel):
    id: str = ""
    heading: str
    blocks: list[ReportBlock] = Field(default_factory=list)

    @classmethod
    def from_markdown(cls, text: str, registry: dict) -> "Section":
        """Parse one Markdown section into a Section of ReportBlocks.

        Deterministic, stdlib-only (plan §8.2). The first "## " line sets
        the section heading (id = slugified heading; content before it is
        ignored and later "## " lines become level-2 heading blocks).
        "### " lines become level-3 heading blocks; consecutive "- "/"* "
        bullets form one unordered_list and consecutive "N. " lines one
        ordered_list (one span per line); ``` fences become code_block
        (language = fence label, text = joined inner lines); consecutive
        "| a | b |" rows become a comparison_table (first row = columns);
        consecutive "> " lines form one callout (callout_type="note");
        other non-blank lines accumulate into a paragraph joined with " ".
        Citation markers ([D1], [W2], [D1, W2], [4]) are split into spans;
        keys are kept as written — registry is accepted for API
        compatibility, use find_unresolvable_citations to flag invented
        keys.
        """
        lines = text.splitlines()
        heading = ""
        start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("## "):
                heading = line.strip()[3:].strip()
                start = i + 1
                break

        blocks: list[ReportBlock] = []
        para: list[str] = []
        bullets: list[str] = []
        ordered: list[str] = []
        table: list[str] = []
        callout: list[str] = []
        code: list[str] | None = None
        code_lang = ""

        def flush_para() -> None:
            nonlocal para
            if para:
                blocks.append(
                    ReportBlock(type=BlockType.paragraph, spans=_split_spans(" ".join(para)))
                )
                para = []

        def flush_bullets() -> None:
            nonlocal bullets
            if bullets:
                blocks.append(
                    ReportBlock(
                        type=BlockType.unordered_list,
                        items=[_single_span(b) for b in bullets],
                    )
                )
                bullets = []

        def flush_ordered() -> None:
            nonlocal ordered
            if ordered:
                blocks.append(
                    ReportBlock(
                        type=BlockType.ordered_list,
                        items=[_single_span(o) for o in ordered],
                    )
                )
                ordered = []

        def flush_table() -> None:
            nonlocal table
            if table:
                blocks.append(_table_block(table))
                table = []

        def flush_callout() -> None:
            nonlocal callout
            if callout:
                blocks.append(
                    ReportBlock(
                        type=BlockType.callout,
                        callout_type="note",
                        spans=_split_spans(" ".join(callout)),
                    )
                )
                callout = []

        def flush_all() -> None:
            flush_para()
            flush_bullets()
            flush_ordered()
            flush_table()
            flush_callout()

        for line in lines[start:]:
            stripped = line.strip()
            if code is not None:
                if stripped.startswith("```"):
                    blocks.append(
                        ReportBlock(
                            type=BlockType.code_block,
                            language=code_lang,
                            text="\n".join(code),
                        )
                    )
                    code = None
                else:
                    code.append(line)
                continue
            if not stripped:
                flush_all()
                continue
            if stripped.startswith("```"):
                flush_all()
                code = []
                code_lang = stripped[3:].strip()
                continue
            if stripped.startswith("### "):
                flush_all()
                blocks.append(
                    ReportBlock(
                        type=BlockType.heading, level=3, text=stripped[4:].strip()
                    )
                )
                continue
            if stripped.startswith("## "):
                flush_all()
                blocks.append(
                    ReportBlock(
                        type=BlockType.heading, level=2, text=stripped[3:].strip()
                    )
                )
                continue
            m = _BULLET_RE.match(stripped)
            if m:
                flush_para()
                flush_ordered()
                flush_table()
                flush_callout()
                bullets.append(m.group(1).strip())
                continue
            m = _ORDERED_ITEM_RE.match(stripped)
            if m:
                flush_para()
                flush_bullets()
                flush_table()
                flush_callout()
                ordered.append(stripped[m.end():].strip())
                continue
            if (
                stripped.startswith("|")
                and stripped.endswith("|")
                and stripped.count("|") >= 2
            ):
                flush_para()
                flush_bullets()
                flush_ordered()
                flush_callout()
                table.append(stripped)
                continue
            if stripped.startswith("> ") or stripped == ">":
                flush_para()
                flush_bullets()
                flush_ordered()
                flush_table()
                callout.append(stripped[1:].strip())
                continue
            flush_bullets()
            flush_ordered()
            flush_table()
            flush_callout()
            para.append(stripped)

        if code is not None:
            blocks.append(
                ReportBlock(
                    type=BlockType.code_block, language=code_lang, text="\n".join(code)
                )
            )
        flush_all()
        return cls(id=_slugify(heading), heading=heading, blocks=blocks)


class Source(BaseModel):
    id: str
    type: str = "web"
    title: str = ""
    author: list[dict] = Field(default_factory=list)
    issued: dict = Field(default_factory=dict)
    URL: str = ""
    publisher: str = ""
    DOI: str = ""
    citation_key: str = ""
    accessed: str = ""


class Metadata(BaseModel):
    title: str
    subtitle: str = ""
    query: str
    session_id: str
    generated_at: str
    author: str = "Research Assistant"
    report_type: Literal["standard", "deep_research"] = "deep_research"


class QualityMetrics(BaseModel):
    citation_density: dict = Field(default_factory=dict)
    verification: dict = Field(default_factory=dict)
    sources_count: dict = Field(default_factory=dict)
    total_words: int = 0


class Report(BaseModel):
    metadata: Metadata
    executive_summary: list[str] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


class ResearchReport(BaseModel):
    schema_version: str = "1.0"
    report: Report
    quality: QualityMetrics


def _iter_block_spans(report: ResearchReport) -> list[Span]:
    spans: list[Span] = []
    for section in report.report.sections:
        for block in section.blocks:
            spans.extend(block.spans)
            spans.extend(block.items)
            for row in block.rows:
                spans.extend(row)
    return spans


def find_unresolvable_citations(report: ResearchReport, registry: dict) -> list[str]:
    """Return internal citation keys (^[DW]\\d+$) cited in the report that
    are absent from registry, in first-appearance order, deduped. Never
    raises: validation problems yield whatever was collected so far."""
    out: list[str] = []
    seen: set[str] = set()
    try:
        known = set((registry or {}).keys())
        for span in _iter_block_spans(report):
            for c in span.citations:
                if _INTERNAL_KEY_RE.match(c) and c not in known and c not in seen:
                    seen.add(c)
                    out.append(c)
    except Exception:
        pass
    return out


def drop_bare_numeric_citations(report: ResearchReport, registry: dict) -> list[str]:
    """Drop bare-number citation entries that do not correspond to a
    citation key.

    Semantics (deterministic, in place):
    - cited_keys = all entries matching ^[DW]\\d+$ in block citation lists
      (spans, items, rows cells), in first-appearance order, deduped.
    - valid = {str(i) for i in 1..len(cited_keys)} when cited_keys is
      non-empty, else {str(i) for i in 1..len(registry)}.
    - Every all-digit citation entry outside valid is removed from
      spans/items/rows in place.
    - Returns the removed entries, deduped, in first-appearance order.
    """
    cited_keys: list[str] = []
    seen: set[str] = set()
    for span in _iter_block_spans(report):
        for c in span.citations:
            if _INTERNAL_KEY_RE.match(c) and c not in seen:
                seen.add(c)
                cited_keys.append(c)
    n = len(cited_keys) if cited_keys else len(registry or {})
    valid = {str(i) for i in range(1, n + 1)}

    dropped: list[str] = []
    dropped_seen: set[str] = set()
    for span in _iter_block_spans(report):
        kept: list[str] = []
        for c in span.citations:
            if c.isdigit() and c not in valid:
                if c not in dropped_seen:
                    dropped_seen.add(c)
                    dropped.append(c)
            else:
                kept.append(c)
        span.citations = kept
    return dropped


def _section_words(section: Section) -> int:
    n = len("".join(b.text for b in section.blocks).split())
    for span in (
        s
        for b in section.blocks
        for s in [
            *b.spans,
            *b.items,
            *(c for row in b.rows for c in row),
        ]
    ):
        n += len(span.text.split())
    return n


def _section_citation_count(section: Section) -> int:
    n = 0
    for block in section.blocks:
        n += sum(len(s.citations) for s in block.spans)
        n += sum(len(s.citations) for s in block.items)
        n += sum(len(s.citations) for row in block.rows for s in row)
    return n


def compute_citation_density(report: ResearchReport) -> dict:
    """Citations per 100 words.

    per_section: {heading: round(citations / words * 100, 1)} where words
    are whitespace-split words over block text, span texts, item texts,
    and row cell texts (no dedup); 0.0 when a section has no words.
    overall: same ratio over all sections' words plus executive_summary
    paragraph words; citations count over all sections only.
    """
    per_section: dict[str, float] = {}
    total_words = 0
    total_citations = 0
    for section in report.report.sections:
        words = _section_words(section)
        citations = _section_citation_count(section)
        per_section[section.heading] = round(citations / words * 100, 1) if words else 0.0
        total_words += words
        total_citations += citations
    total_words += sum(len(p.split()) for p in report.report.executive_summary)
    overall = round(total_citations / total_words * 100, 1) if total_words else 0.0
    return {"overall": overall, "per_section": per_section}


def count_total_words(report: ResearchReport) -> int:
    """Whitespace-split word count over executive_summary paragraphs plus
    all section text content (block text, span texts, item texts, row
    cell texts)."""
    n = sum(len(p.split()) for p in report.report.executive_summary)
    for section in report.report.sections:
        n += _section_words(section)
    return n


def to_json_schema() -> dict:
    return ResearchReport.model_json_schema()
