"""Deterministic structured-report assembly (plan §6, §7). No LLM calls in this module."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from models import (
    Report,
    ResearchReport,
    ReportBlock,
    Section,
    Source,
    Span,
    Metadata,
    QualityMetrics,
    compute_citation_density,
    count_total_words,
    drop_bare_numeric_citations,
    find_unresolvable_citations,
)
from memory.helpers import registry_to_sources

EXEC_SUMMARY_JSON_INSTRUCTIONS = """
Write the executive summary of the research report.
CONTRACT:
1. Output EXACTLY ONE JSON array, no Markdown, no code fences, no
   commentary.
2. The array holds 2-4 self-contained prose paragraphs (strings), each
   60-120 words, synthesizing the key findings across the provided
   sections — findings, not a section index.
3. No citation markers inside the summary text.
4. Do not invent facts beyond the provided sections.
"""

_CITATION_KEY_RE = re.compile(r"[DW]\d+")


def _extract_json_array(text: str) -> list | None:
    """Extract a single JSON array from a model response (best-effort).

    Mirror of worker_agents.writer_agent._extract_json_object for arrays:
    strip ```json/``` code fences when the whole response is fenced, then
    take the substring from the first "[" to the last "]" and parse it.
    Returns the parsed list, or None on any error (soft-fail).
    """
    if not text:
        return None
    candidate = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(candidate[start:end + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, list) else None


def parse_exec_summary(text: str) -> list[str]:
    """Parse an executive-summary model response into prose paragraphs.

    Prefers a JSON array of strings (contract per
    EXEC_SUMMARY_JSON_INSTRUCTIONS). On any parse failure, salvages the
    raw text as blank-line-separated paragraphs (plan soft-fail). Returns
    [] when nothing usable is present; never raises.
    """
    items = _extract_json_array(text)
    if items is not None:
        return [str(p) for p in items if str(p).strip()]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return paragraphs


def _iter_citation_holders(block: ReportBlock):
    """Yield every object in a block that owns a .citations list:
    paragraph/callout/citation_note spans, list items, and table cells."""
    for holder in block.spans or []:
        yield holder
    for holder in block.items or []:
        yield holder
    for row in block.rows or []:
        for holder in row or []:
            yield holder


def _collect_cited_keys(report: ResearchReport, registry: dict) -> list[str]:
    """First-appearance order of [DW]\\d+ keys cited in the report's blocks
    that exist in the registry (deduped). Deterministic; never raises."""
    known = set((registry or {}).keys())
    out: list[str] = []
    seen: set[str] = set()
    try:
        for section in report.report.sections:
            for block in section.blocks:
                for holder in _iter_citation_holders(block):
                    for entry in holder.citations or []:
                        if (
                            _CITATION_KEY_RE.fullmatch(entry)
                            and entry in known
                            and entry not in seen
                        ):
                            seen.add(entry)
                            out.append(entry)
    except Exception:
        pass
    return out


def _renumber_json_citations(report: ResearchReport, registry: dict, cited_keys: list) -> dict:
    """Replace every citation-key reference in the report's blocks with its
    [n] number (plan §6.3).

    key_to_number maps each key in `cited_keys` (first-appearance order,
    registry-present only) to 1..N. Citation entries not in the map —
    invented keys and bare numbers — are DROPPED (they were already
    pruned/flagged upstream; be safe). Mutates `report` in place and
    returns the key_to_number map.
    """
    if not cited_keys:
        cited_keys = _collect_cited_keys(report, registry)
    known = set((registry or {}).keys())
    key_to_number = {key: i + 1 for i, key in enumerate(cited_keys) if key in known}
    for section in report.report.sections:
        for block in section.blocks:
            for holder in _iter_citation_holders(block):
                holder.citations = [
                    str(key_to_number[k])
                    for k in (holder.citations or [])
                    if k in key_to_number
                ]
    return key_to_number


def assemble_structured_report(
    *,
    sections: list,
    registry: dict,
    user_query: str,
    session_id: str,
    exec_paragraphs: list,
    verification_status: dict,
    title: str,
    subtitle: str = "",
    evidence_json: str = "",
) -> ResearchReport:
    """Deterministically assemble a validated ResearchReport from written
    sections and the citation registry (plan §6). No LLM calls.

    Steps: flag unresolvable citations, drop bare-numeric citations,
    renumber [DW]# keys to 1..N in first-appearance order, map the cited
    registry entries to Source records (plan §8.1), and compute quality
    metrics. `evidence_json` is accepted for interface stability and
    reserved for future provenance fields.
    """
    report = ResearchReport(
        schema_version="1.0",
        report=Report(
            metadata=Metadata(
                title=title,
                subtitle=subtitle,
                query=user_query,
                session_id=session_id,
                generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                author="Research Assistant",
                report_type="deep_research",
            ),
            executive_summary=list(exec_paragraphs or []),
            sections=list(sections or []),
            sources=[],
        ),
        quality=QualityMetrics(),
    )

    unresolvable = find_unresolvable_citations(report, registry)
    dropped = drop_bare_numeric_citations(report, registry)

    cited_keys = _collect_cited_keys(report, registry)
    _renumber_json_citations(report, registry, cited_keys)

    source_dicts = registry_to_sources(registry, cited_keys)
    report.report.sources = [Source.model_validate(d) for d in source_dicts]

    report.quality.citation_density = compute_citation_density(report)
    report.quality.verification = {
        **(verification_status or {}),
        "unresolvable_citations": unresolvable,
        "dropped_bare_citations": dropped,
    }

    doc_count = sum(1 for s in report.report.sources if s.type == "report")
    web_count = sum(1 for s in report.report.sources if s.type == "webpage")
    report.quality.sources_count = {"documents": doc_count, "web": web_count}

    report.quality.total_words = count_total_words(report)

    return report


def sections_plain_text(section: Section) -> str:
    """Flatten a Section to plain text for legacy state["sections"]
    compatibility and on_section streaming previews.

    Concatenates block text (headings, code blocks), span texts, list-item
    texts, and table-cell texts with single spaces. Spans, list items, and
    table cells with non-empty citations render their text followed by a
    bracketed marker, e.g. "abc [D1, W2]" (legacy Markdown-section
    semantics). Deterministic; never raises.
    """
    parts: list[str] = []
    try:
        for block in section.blocks or []:
            if block.text:
                parts.append(block.text)
            for span in block.spans or []:
                text = span.text
                if span.citations:
                    text = f"{text} [{', '.join(span.citations)}]"
                parts.append(text)
            for item in block.items or []:
                text = item.text
                if item.citations:
                    text = f"{text} [{', '.join(item.citations)}]"
                parts.append(text)
            for row in block.rows or []:
                for cell in row or []:
                    text = cell.text
                    if cell.citations:
                        text = f"{text} [{', '.join(cell.citations)}]"
                    parts.append(text)
    except Exception:
        pass
    return " ".join(part.strip() for part in parts if part and part.strip())
