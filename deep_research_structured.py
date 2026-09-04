"""Deterministic structured-report assembly (plan §6, §7). No LLM calls in this module."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from models import (
    BlockType,
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

logger = logging.getLogger(__name__)

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

# Final ship-guard threshold: targets genuinely empty output only (the
# writer-side 300-word contract and the orchestrator's must-revise pass are
# the first lines of defense; this catches what slips past both).
_MIN_SECTION_WORDS = 30
_NOT_GENERATED_TEXT = (
    "The available evidence does not cover this area in sufficient "
    "detail, so no content was generated for this section."
)


def _content_word_count(section: Section) -> int:
    """Whitespace word count over a section's text content (block text plus
    span/item/cell texts). Never raises."""
    try:
        n = len("".join(b.text or "" for b in section.blocks).split())
        for span in (
            s
            for b in section.blocks
            for s in [*b.spans, *b.items, *(c for row in b.rows for c in row)]
        ):
            n += len((span.text or "").split())
        return n
    except Exception:
        return 0


def _apply_empty_section_guards(report: ResearchReport) -> list:
    """Replace (near-)empty content sections with a gap notice and drop a
    (near-)empty synthesis, in place. Returns the "not_generated: ..." gap
    entries for the quality verification dict.

    An empty synthesis ships as NO synthesis section — the same shape as the
    orchestrator's synthesis_failed degradation path, which render_markdown
    and save_structured_report already handle.
    """
    keep = []
    not_generated = []
    sections = list(report.report.sections)
    for section in sections:
        # The writer always slugs the synthesis id to "synthesis" — that is
        # authoritative. A heading match is a fallback for hand-built
        # reports, but ONLY for the LAST section: a content section titled
        # "Synthesis" (e.g. a chemistry topic) earlier in the report must
        # get a gap notice, never a silent drop.
        is_synthesis = section.id == "synthesis" or (
            section is sections[-1]
            and (section.heading or "").strip().casefold() == "synthesis"
        )
        if _content_word_count(section) < _MIN_SECTION_WORDS:
            if is_synthesis:
                continue
            section.blocks = [
                ReportBlock(
                    type="paragraph",
                    spans=[Span(text=_NOT_GENERATED_TEXT, citations=[])],
                )
            ]
            not_generated.append(f"not_generated: {section.heading}")
        keep.append(section)
    report.report.sections = keep
    return not_generated


def _block_has_renderable_content(block) -> bool:
    """True if a block carries any non-empty text anywhere (its own text,
    spans, list items, table cells, caption, or callout title)."""
    if (block.text or "").strip():
        return True
    if any((s.text or "").strip() for s in block.spans or []):
        return True
    if any((i.text or "").strip() for i in block.items or []):
        return True
    if any((c.text or "").strip() for row in block.rows or [] for c in row or []):
        return True
    return bool((block.caption or "").strip() or (block.callout_title or "").strip())


def _drop_orphan_subheadings(report: ResearchReport) -> list:
    """Drop heading blocks whose content zone carries no renderable content,
    in place. Returns "orphan_subheading: ..." gap entries; never raises.

    Weak models occasionally emit a subsection heading and then write no
    body (the PDF shows one heading stacked on the next). The section-level
    word-count guard cannot see this: the surrounding section is full. So
    work at block level. Level semantics: a heading of level L is "orphan"
    when the blocks between it and the next heading of level <= L (or end of
    the block list) contain ZERO blocks with renderable content. Headings
    inside the zone do NOT count as content — so a level-2 heading whose
    zone holds only deeper, also-unwritten subheadings is orphan and is
    dropped along with them. A heading is kept as soon as ANY subsequent
    block in its zone (at any depth below L) has content. Idempotent.
    """
    gaps: list = []
    try:
        for section in report.report.sections:
            blocks = section.blocks or []
            drop = set()
            for i, block in enumerate(blocks):
                if block.type != BlockType.heading:
                    continue
                level = int(block.level or 3)
                j = len(blocks)
                for k in range(i + 1, len(blocks)):
                    if blocks[k].type == BlockType.heading and int(blocks[k].level or 3) <= level:
                        j = k
                        break
                if not any(
                    blocks[k].type != BlockType.heading
                    and _block_has_renderable_content(blocks[k])
                    for k in range(i + 1, j)
                ):
                    drop.add(i)
                    text = (block.text or "").strip() or f"(level {level})"
                    logger.warning(
                        "dropping orphan subheading with no content zone: %s",
                        text[:80],
                    )
                    gaps.append(f"orphan_subheading: {text[:120]}")
            if drop:
                section.blocks = [b for i, b in enumerate(blocks) if i not in drop]
    except Exception:
        pass
    return gaps

_JSON_DECODER = json.JSONDecoder()


def _extract_json_array(text: str) -> list | None:
    """Extract a single JSON array from a model response (best-effort).

    Mirror of worker_agents.writer_agent._extract_json_object for arrays:
    strip ```json/``` code fences when the whole response is fenced, then
    raw_decode at EVERY "[" index and collect the lists that parse. (The
    old first-"["→last-"]" slice silently returned None when prose around
    the JSON contained brackets, or when two arrays appeared.) If any
    parsed list contains str or dict items the FIRST such list (in order of
    appearance — matching the exec-summary contract of an
    array of paragraph strings, while tolerating dict entries; when several
    qualify the LARGEST (by decoded span) wins, so a small residue array
    cannot steal the parse. Lists nested inside an earlier-parsed container
    are skipped. Returns None when nothing parses (soft-fail).
    """
    if not text:
        return None
    candidate = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    parsed = []
    spans: list[tuple[int, int]] = []
    for i, ch in enumerate(candidate):
        if ch != "[":
            continue
        try:
            obj, end = _JSON_DECODER.raw_decode(candidate, i)
        except ValueError:
            continue
        if not isinstance(obj, list):
            continue
        # Skip arrays nested INSIDE an earlier-parsed container: when the
        # response holds one top-level array, its inner arrays are parts of
        # it, not competing arrays.
        if any(start <= i < e for start, e in spans):
            continue
        parsed.append(obj)
        spans.append((i, end))
    # Prefer the LARGEST str/dict-item list (decoded span), not the first:
    # a small residue array may appear before the real 2-4 paragraph array
    # and first-wins would let it dead-end the parse. A LONE single-item
    # list is almost certainly residue (the contract is 2-4 paragraphs of
    # ~60-120 words each): yield None so the prose fallback can rescue.
    keyed = [
        (obj, end - i)
        for obj, (i, end) in zip(parsed, spans)
        if any(isinstance(item, (str, dict)) for item in obj)
    ]
    if keyed:
        best, _size = max(keyed, key=lambda pair: pair[1])
        if len(best) == 1 and isinstance(best[0], str) and len(best[0].split()) < 30:
            return None
        return best
    return parsed[0] if parsed else None


def parse_exec_summary(text: str) -> list[str]:
    """Parse an executive-summary model response into prose paragraphs.

    Prefers a JSON array of strings (contract per
    EXEC_SUMMARY_JSON_INSTRUCTIONS), keeping only str items. On any parse
    failure, salvages the raw text as blank-line-separated paragraphs,
    stripping JSON residue (lines that are JSON-ish or contain a JSON array
    literal — apologies + raw JSON must not ship as the summary; plan
    soft-fail). Returns [] when nothing usable is present; never raises.
    """
    items = _extract_json_array(text)
    if items is not None and any(isinstance(p, str) and p.strip() for p in items):
        # Keep only real prose paragraphs; dict/list entries are structural
        # residue, not summary text (the contract is an array of strings).
        return [p for p in items if isinstance(p, str) and p.strip()]
    # An array with no str items (e.g. "[1]" — a citation that happens to
    # be valid JSON) is not a summary: fall through to the prose salvage.
    out_lines = []
    for line in (text or "").splitlines():
        s = line.strip()
        # Residue stripping: drop lines that are themselves JSON-ish or that
        # contain a JSON array literal. A leading bracket run that is a
        # SHORT citation token (digits/letters, e.g. [1] or [W2, D3] — no
        # quote/brace/comma inside, not empty) is prose, not residue.
        if s.startswith("{"):
            out_lines.append("")
        elif s.startswith("["):
            m = re.match(r"\[([^\[\]]{0,8})\]", s)
            if m and re.fullmatch(r"[A-Za-z0-9]+(?:\s*,\s*[A-Za-z0-9]+)*", m.group(1).strip()):
                out_lines.append(line)  # citation token lead-in
            else:
                out_lines.append("")
        elif re.search(r"\[\s*[\"'{\[]", s):
            out_lines.append("")
        else:
            out_lines.append(line)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(out_lines)) if p.strip()]
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


_CITATION_BRACKET_RE = re.compile(
    r"\[((?:[DW]\d+|\d+)(?:\s*,\s*(?:[DW]\d+|\d+))*)\]"
)


def _source_dedupe_unit(entry: dict) -> tuple:
    """Dedupe unit of a registry entry. MUST mirror registry_to_sources
    exactly: docs dedupe by document_name (fallback title), web by url
    (fallback title); empty names/urls never collide (the or-"" guard).
    """
    entry = entry or {}
    if entry.get("kind") == "doc":
        return ("doc", (entry.get("document_name") or entry.get("title") or "").strip().lower())
    return ("web", (entry.get("url") or "").strip().lower() or (entry.get("title") or "").strip().lower())


def _build_key_to_source_map(registry: dict, sources: list) -> dict:
    """Map every registry citation_key to the surviving Source record its
    dedupe unit collapsed into: {key: (surviving_citation_key, position)}
    where position is 1-based into `sources`. Keys whose unit (document / url)
    has no surviving record are absent."""
    unit_to_source: dict[tuple, tuple] = {}
    for i, source in enumerate(sources or []):
        entry = (registry or {}).get(source.citation_key) or {}
        unit_to_source.setdefault(_source_dedupe_unit(entry), (source.citation_key, i + 1))
    key_map: dict[str, tuple] = {}
    for key, entry in (registry or {}).items():
        target = unit_to_source.get(_source_dedupe_unit(entry))
        if target is not None:
            key_map[key] = target
    return key_map


def _remap_citations_to_final_sources(report: ResearchReport, registry: dict) -> dict:
    """Renumber citation arrays AND rewrite [D#]/[W#] text markers onto the
    FINAL deduped sources array (plan §6.3, after §8.1 dedupe).

    `report.report.sources` must already hold the deduped records: numeric
    entries are positions into that array, and every registry key —
    including deduped non-primary keys — resolves to its surviving record's
    citation_key and position, so both the `citations` arrays and the inline
    bracket markers in span/item/cell text stay resolvable post-dedupe.
    Citation entries that are not registry keys (invented keys, stale
    numbers) are dropped; markers whose key has no surviving record are left
    untouched. Mutates `report` in place; returns the key -> (citation_key,
    position) map.
    """
    key_map = _build_key_to_source_map(registry, report.report.sources)

    def _rewrite_markers(text: str) -> str:
        def _repl(match: re.Match) -> str:
            out_items: list[str] = []
            for item in re.findall(r"[DW]\d+|\d+", match.group(1)):
                if item[0] in "DW":
                    target = key_map.get(item)
                    key = target[0] if target is not None else item
                else:
                    key = item  # bare number: pass through for paperbot
                if key not in out_items:
                    out_items.append(key)
            return "[" + ", ".join(out_items) + "]"

        out = _CITATION_BRACKET_RE.sub(_repl, text or "")
        out = re.sub(r" {2,}", " ", out)
        out = re.sub(r" \.", ".", out)
        return out

    for section in report.report.sections:
        for block in section.blocks:
            for holder in _iter_citation_holders(block):
                if holder.citations:
                    out_cits: list[str] = []
                    for c in holder.citations:
                        if c in key_map:
                            pos = str(key_map[c][1])
                            if pos not in out_cits:
                                out_cits.append(pos)
                    holder.citations = out_cits
                if holder.text:
                    holder.text = _rewrite_markers(holder.text)
    return key_map


# Display equation emitted without delimiters: a LaTeX control sequence,
# no math delimiters, and no prose words once all \command{...} groups,
# {...} groups, and bare \command tokens are stripped (single-letter
# variables and ALL-CAPS labels survive; a lowercase run of 2+ vetoes).
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\b")
_COMMAND_GROUP_RE = re.compile(r"\\[a-zA-Z]+\{[^{}]*\}")
_BRACE_GROUP_RE = re.compile(r"\{[^{}]*\}")
_BARE_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_LOWERCASE_RUN_RE = re.compile(r"[a-z]{2,}")


def _is_bare_equation(text: str) -> bool:
    """True when a span's text is a display equation the writer emitted
    without ANY math delimiters (no $ / $ / \\( / \[): it contains a LaTeX
    control sequence, and after stripping \command{...} groups, {...}
    groups, and bare \command tokens, no lowercase run of 2+ remains
    (real prose would). Never raises."""
    t = (text or "").strip()
    if not t or "$" in t or "\\(" in t or "\\[" in t:
        return False
    if not _LATEX_CMD_RE.search(t):
        return False
    s = t
    while True:
        stripped = _COMMAND_GROUP_RE.sub(" ", s)
        stripped = _BRACE_GROUP_RE.sub(" ", stripped)
        stripped = _BARE_LATEX_CMD_RE.sub(" ", stripped)
        if stripped == s:
            break
        s = stripped
    return not _LOWERCASE_RUN_RE.search(s)


def _promote_bare_equation_spans(report: ResearchReport) -> int:
    """Promote paragraph spans that are undelimited display equations to
    standalone equation blocks so downstream renderers (paperbot/KaTeX)
    typeset them as math instead of printing raw TeX. Mixed paragraphs are
    split, preserving span order; a paragraph whose spans are all promoted
    becomes the equation blocks themselves. Never raises.

    Returns the number of promoted spans.
    """
    promoted = 0
    try:
        for section in report.report.sections:
            out: list[ReportBlock] = []
            for block in section.blocks or []:
                if block.type != "paragraph" or not block.spans:
                    out.append(block)
                    continue
                parts: list[ReportBlock] = []
                pending: list[Span] = []
                changed = False
                for span in block.spans:
                    if _is_bare_equation(span.text):
                        if pending:
                            parts.append(
                                ReportBlock(
                                    type="paragraph",
                                    text=block.text,
                                    spans=pending,
                                    citations=list(block.citations),
                                )
                            )
                            pending = []
                        equation = ReportBlock(
                            type="equation",
                            language="latex",
                            text=span.text,
                            citations=list(span.citations),
                        )
                        equation.spans = []  # text lives in .text, not a span
                        parts.append(equation)
                        changed = True
                        promoted += 1
                    else:
                        pending.append(span)
                if pending:
                    parts.append(
                        ReportBlock(
                            type="paragraph",
                            text=block.text,
                            spans=pending,
                            citations=list(block.citations),
                        )
                    )
                out.extend(parts if changed else [block])
            section.blocks = out
    except Exception:
        pass
    return promoted


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

    Steps: collapse adjacent duplicate blocks, flag unresolvable citations,
    drop bare-numeric citations, map the cited registry entries to deduped
    Source records (plan §8.1), then renumber citation arrays and rewrite
    [D#]/[W#] text markers onto those final records (plan §6.3 — positions
    are 1-based into the deduped array, so they never go out of range),
    drop sub-heading blocks with no content zone, promote undelimited
    display-equation spans to equation blocks, wrap undelimited inline LaTeX
    runs in prose spans with $...$, normalize
    comparison_table row widths to the header, and compute quality metrics.
    `evidence_json` is accepted for interface stability and reserved for
    future provenance fields.
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

    _collapse_adjacent_duplicate_blocks(report)

    unresolvable = find_unresolvable_citations(report, registry)
    dropped = drop_bare_numeric_citations(report, registry)

    not_generated = _apply_empty_section_guards(report)
    orphan_gaps = _drop_orphan_subheadings(report)

    cited_keys = _collect_cited_keys(report, registry)

    source_dicts = registry_to_sources(registry, cited_keys)
    report.report.sources = [Source.model_validate(d) for d in source_dicts]

    _remap_citations_to_final_sources(report, registry)
    _promote_bare_equation_spans(report)
    _wrap_undelimited_latex(report)
    _normalize_comparison_table_widths(report)

    report.quality.citation_density = compute_citation_density(report)
    report.quality.verification = {
        **(verification_status or {}),
        "unresolvable_citations": unresolvable,
        "dropped_bare_citations": dropped,
    }
    if not_generated or orphan_gaps:
        gaps = list(report.quality.verification.get("gaps") or [])
        gaps.extend(not_generated)
        gaps.extend(orphan_gaps)
        report.quality.verification["gaps"] = gaps

    doc_count = sum(1 for s in report.report.sources if s.type == "report")
    web_count = sum(1 for s in report.report.sources if s.type == "webpage")
    report.quality.sources_count = {"documents": doc_count, "web": web_count}

    report.quality.total_words = count_total_words(report)

    return report


def _collapse_adjacent_duplicate_blocks(report: ResearchReport) -> None:
    """Collapse runs of ADJACENT byte-identical blocks to one, in place.

    Final defense against a degenerate LLM loop that re-emits the same block
    (e.g. one note callout written 14x). Conservative by design: only
    consecutive identical blocks (normalized JSON via model_dump_json, so
    type + spans + type-specific fields) are collapsed; identical blocks
    separated by other content are kept, and non-adjacent duplicates across
    sections never interact. Never raises.
    """
    try:
        for section in report.report.sections:
            kept: list[ReportBlock] = []
            for block in section.blocks or []:
                if kept and block.model_dump_json() == kept[-1].model_dump_json():
                    continue
                kept.append(block)
            section.blocks = kept
    except Exception:
        pass


def _normalize_comparison_table_widths(report: ResearchReport) -> int:
    """Make every comparison_table row as wide as its header, in place.

    Ragged tables (a row with fewer or more cells than ``len(columns)``)
    pass the schema — there is no width constraint — but they break
    downstream consumers that index rows up to the header width (paperbot's
    normalizer crashed reading ``row[i]`` on a 3-column table whose rows
    carried 2 cells). Short rows are padded with empty cells, over-long rows
    truncated, and non-Span cells coerced to spans so every cell keeps the
    ``.text``/``.citations`` shape the plain-text and quality helpers read
    (a raw ``""`` cell would read fine in paperbot but AttributeError in
    ``sections_plain_text``). Tables with no columns are left untouched.
    Never raises. Returns the number of rows modified.
    """
    changed = 0
    try:
        for section in report.report.sections:
            for block in section.blocks or []:
                if block.type != BlockType.comparison_table:
                    continue
                width = len(block.columns or [])
                if width == 0 or not isinstance(block.rows, list):
                    continue
                for i, row in enumerate(block.rows):
                    cells = [
                        cell if isinstance(cell, Span) else Span(text=str(cell), citations=[])
                        for cell in row
                    ] if isinstance(row, list) else []
                    if len(cells) < width:
                        cells += [Span(text="", citations=[]) for _ in range(width - len(cells))]
                    elif len(cells) > width:
                        cells = cells[:width]
                    if cells != row:
                        block.rows[i] = cells
                        changed += 1
    except Exception:
        pass
    return changed


# ---------------------------------------------------------------------------
# Undelimited-LaTeX sanitizer (model-agnostic math hygiene)
#
# Some models ignore the $...$ inline-math rule and emit raw LaTeX runs
# embedded in prose (|\psi\rangle, 2^{N}-dimensional, S_A = S_B). KaTeX can
# only typeset delimited math, so those runs print as literal backslash soup.
# This pass wraps each undelimited math RUN with $...$. It is deliberately
# conservative — a run starts ONLY on strong signals and extends only over
# math-shaped characters — so prose can never be captured into math.
# ---------------------------------------------------------------------------

_LATEX_PROTECTED_RE = re.compile(
    r"\$\$[\s\S]+?\$\$|\$[^$\n]+?\$|\\\[[\s\S]+?\\\]|\\\(.*?\\\)"
)

# Space variants models emit (regular, nbsp, thin/hair/narrow nbsp).
_LATEX_SPACES = " \u00a0\u2007\u2009\u202f"

# Ket/bra delimiters (note the different lengths: \langle = 8, \rangle = 7).
_LANGLE = "\\langle"
_RANGLE = "\\rangle"

# Max distance (chars) between a ket/bra opener and its closing delimiter.
# Bounds the search so a lone prose `|` can never swallow the rest of a span.
_KET_UNIT_WINDOW = 64


def _snip(text: str, width: int = 60) -> str:
    """Head+tail snippet for log lines (993abfa pattern)."""
    text = (text or "").replace("\n", "\\n")
    if len(text) <= width * 2:
        return text
    return f"{text[:width]}…{text[-width:]}"


def _is_backslash_seq(text: str, i: int) -> int:
    """If text[i:] starts a backslash sequence, return its length; else 0."""
    if i >= len(text) or text[i] != "\\":
        return 0
    j = i + 1
    if j < len(text) and text[j].isalpha():
        while j < len(text) and text[j].isalpha():
            j += 1
        return j - i
    if j < len(text):
        return 2  # \, \; \\ etc.
    return 1


def _letter_run(text: str, i: int) -> int:
    """Length of the letter run starting at i (case-insensitive)."""
    j = i
    while j < len(text) and text[j].isalpha():
        j += 1
    return j - i


def _find_unit_close(text: str, open_idx: int, close_is_pipe: bool) -> int:
    """Index of the closing delimiter of the ket/bra unit opened at open_idx,
    within _KET_UNIT_WINDOW, or -1.

    The opener is text[open_idx] (``|`` for a ket, ``<`` for a bra). The
    interior must be math-ish: a space followed by a lowercase letter run
    (prose words) disqualifies the unit, as does a missing closer.
    """
    n = len(text)
    i = open_idx + 1
    limit = min(n, open_idx + _KET_UNIT_WINDOW + 1)
    while i < limit:
        c = text[i]
        if (
            c == "\\"
            and text.startswith(_RANGLE, i)
            and (i + len(_RANGLE) >= n or text[i + len(_RANGLE)] != "a")
        ):
            return i
        if close_is_pipe and c == "|":
            return i
        if c in _LATEX_SPACES:
            k = i
            while k < n and text[k] in _LATEX_SPACES:
                k += 1
            if k + 2 <= n and text[k].islower() and text[k + 1].islower():
                return -1  # prose word inside => not a ket/bra unit
        i += 1
    return -1


def _math_run_start(text: str, i: int, protected) -> int:
    """Return the true start of a math run at/just before index i, or -1.

    Strong start signals (and only these):
      S1: a backslash command (\\lambda, \\psi, ...)
      S2: a braced super/subscript (^{ ... _{) — absorbing ONE preceding
          alphanumeric base char (2^{N} -> base 2)
      S3: a ket/bra unit: ``|`` opening a ``|…\\rangle`` ket, or ``\\langle``
          opening a ``\\langle…|`` bra. The opener is accepted only when its
          closing delimiter sits within _KET_UNIT_WINDOW with a math-ish
          interior, so a literal prose pipe (``A | B``) never starts a run.
      S4: base ^/_ arg where the arg is uppercase/digit/braced (S_A, x^2)
          — a lowercase arg (3^rd party) is treated as prose and rejected
    """
    n = len(text)
    c = text[i]
    # S1: backslash command
    if c == "\\" and i + 1 < n and text[i + 1].isalpha():
        s = i
        if s > 0 and text[s - 1] == "{" and not protected[s - 1]:
            s -= 1  # absorb an opening group brace: {\lambda_i}
        return s
    # S2: braced super/subscript with a base
    if c in "^_" and i + 1 < n and text[i + 1] == "{":
        if i > 0 and text[i - 1].isalnum() and not protected[i - 1]:
            return i - 1
        return i
    # S3: ket/bra unit opener (| with a \rangle partner, or \langle with a |
    # partner within the window) — the WHOLE unit becomes one run so the
    # ``|00`` opener is typeset in math, not body font.
    if c == "|" and i + 1 < n and (
        text[i + 1] == "\\" or text[i + 1].isalnum()
    ):
        if _find_unit_close(text, i, False) >= 0:
            return i
    if c == "|" and i + 1 < n and text[i + 1] in _LATEX_SPACES:
        k = i + 1
        while k < n and text[k] in _LATEX_SPACES:
            k += 1
        if k < n and (text[k] == "\\" or text[k].isalnum()):
            if _find_unit_close(text, i, False) >= 0:
                return i
    # S4: base ^/_ arg with a non-lowercase arg
    if i > 0 and text[i - 1].isalnum() and c in "^_" and i + 1 < n:
        a = text[i + 1]
        if a == "{" or a == "\\" or a.isupper() or a.isdigit():
            return i - 1
    return -1


def _math_run_extent(text: str, start: int, protected) -> int:
    """Extend a math run rightward; return the end index (exclusive).

    Continues over math-shaped characters (backslash sequences, braces,
    ^ _, digits, letters, math operators, kets, parens) and across a single
    space when the next token looks mathematical. Stops at a run of >=2
    consecutive lowercase letters (prose words), >=3 uppercase, or a
    non-math character. A single lowercase letter continues (p_n, e^{i...}).
    """
    j = start
    n = len(text)
    # Unit mode: the run began with a ket/bra opener — it then extends only
    # until the unit's closing delimiter (\rangle for kets, | for bras).
    mode = None
    if start < n and text[start] == "|":
        mode = "ket"
    elif text.startswith(_LANGLE, start) and (
        start + len(_LANGLE) >= n or text[start + len(_LANGLE)] != "a"
    ):
        mode = "bra"
    unit_closed = False
    close_at = -1  # position just past a ket/bra closing delimiter
    while j < n and not protected[j]:
        if mode is not None and not unit_closed and (j - start) > _KET_UNIT_WINDOW:
            return start  # closer never found: collapse (opener stays prose)
        c = text[j]
        if c == "\\":
            if (
                mode == "ket"
                and text.startswith(_RANGLE, j)
                and (j + len(_RANGLE) >= n or text[j + len(_RANGLE)] != "a")
            ):
                j += len(_RANGLE)
                unit_closed = True
                # Close of a ket: unless the expression continues DIRECTLY
                # through it (|u_i\rangle\otimes|v_i\rangle stays one run),
                # the unit ended and the run stops — so |00\rangle keeps its
                # opener, and |+.../-dimensional keep their prose tails.
                close_at = j
                if j >= n or text[j] in _LATEX_SPACES or text[j] in ".,;:!()]":
                    break
                if text[j].islower() and j + 1 < n and text[j + 1].islower():
                    break  # a prose word follows
                continue
            ln = _is_backslash_seq(text, j)
            if ln:
                j += ln
                continue
            break
        if mode == "bra" and c == "|":
            j += 1
            unit_closed = True
            close_at = j
            if j >= n or text[j] in ".,;:!()]:" or text[j] in _LATEX_SPACES:
                break
            if text[j].islower() and j + 1 < n and text[j + 1].islower():
                break  # a prose word follows the closing |
            continue
        if c.isalpha():
            lr = _letter_run(text, j)
            if lr and c.islower() and lr >= 2:
                break  # prose word
            if lr and c.isupper() and lr >= 3:
                break  # prose acronym
            j += lr
            continue
        if c in "{}^_0123456789=+-*/.~:;,()|":
            j += 1
            continue
        if c in _LATEX_SPACES:
            # a space directly after the unit's close ends the run (a ket
            # cannot span a space into a new token, e.g. |00\rangle + ...)
            if mode is not None and unit_closed and close_at == j:
                break
            k = j
            while k < n and text[k] in _LATEX_SPACES:
                k += 1
            if k - j == 1 and k < n and not protected[k]:
                nk = text[k]
                base_signal = nk.isalpha() and k + 1 < n and text[k + 1] in "^_{"
                if (
                    _math_run_start(text, k, protected) >= 0
                    or base_signal
                    or nk.isdigit()
                ) and not (mode is not None and unit_closed):
                    j = k  # rescan from the signaled char (keeps backslash/base)
                    continue
            break
        break
    return j


def _trim_math_run(run: str) -> tuple:
    """Drop dangling operators/punctuation/whitespace from the run edges.

    Returns (core, head, tail) where head/tail are the trimmed-overhang
    characters, re-emitted outside the $...$ so nothing is lost (a hyphen in
    ``2^{N}-dimensional`` stays in the prose; a sentence-final ``.`` stays a
    period). Never lets the core become empty of a real math token.
    """
    danglers = set("+-=*/.~:;,() ") | set(_LATEX_SPACES)
    head = ""
    tail = ""
    while run and run[-1] in danglers and _run_has_core(run[:-1]):
        tail = run[-1] + tail
        run = run[:-1]
    # a trailing unbalanced closing brace (kept its opener outside the run)
    while run and run[-1] == "}" and run.count("{") < run.count("}"):
        tail = run[-1] + tail
        run = run[:-1]
    while run and run[0] in "()+ " and _run_has_core(run[1:]):
        head = run[0] + head
        run = run[1:]
    return run, head, tail


def _run_has_core(run: str) -> bool:
    return bool(re.search(r"\\[a-zA-Z]|[A-Za-z0-9]", run))


def _wrap_latex_in_text(text: str) -> tuple:
    r"""Wrap every undelimited math run in $...$; return (new_text, n_runs).

    Text already inside $...$/$$..$$/\(..\)/\[..\] is protected and never
    touched, which also makes the pass idempotent. Never raises.
    """
    if not text:
        return text, 0
    try:
        protected = [False] * len(text)
        for m in _LATEX_PROTECTED_RE.finditer(text):
            for k in range(m.start(), m.end()):
                protected[k] = True
        out = []
        i = 0
        n = len(text)
        wrapped = 0
        while i < n:
            if protected[i]:
                j = i
                while j < n and protected[j]:
                    j += 1
                out.append(text[i:j])
                i = j
                continue
            s = _math_run_start(text, i, protected)
            if s < 0:
                out.append(text[i])
                i += 1
                continue
            e = _math_run_extent(text, max(s, i), protected)
            run, head, tail = _trim_math_run(text[s:e])
            if not run or not _run_has_core(run):
                out.append(text[i])
                i += 1
                continue
            if s < i:
                # absorbed base/brace/ket char was already appended
                out.pop()
            out.append(head + "$" + run + "$" + tail)
            wrapped += 1
            i = max(e, i + 1)
        return ("".join(out), wrapped) if wrapped else (text, 0)
    except Exception:
        return text, 0


def _wrap_undelimited_latex(report: ResearchReport) -> int:
    """Wrap undelimited LaTeX runs in every span/item/cell/block text and in
    the executive summary, in place. Logs a WARNING (before/after snippets)
    for each modified text. Returns the number of modified texts; never
    raises. Model-agnostic: catches whatever math a model leaks into prose.
    """
    changed = 0
    try:
        for idx, para in enumerate(report.report.executive_summary or []):
            if isinstance(para, str):
                new, k = _wrap_latex_in_text(para)
                if k:
                    logger.warning(
                        "wrapped %d undelimited LaTeX run(s) in exec summary (before: %s | after: %s)",
                        k,
                        _snip(para, 50),
                        _snip(new, 50),
                    )
                    report.report.executive_summary[idx] = new
                    changed += 1
        for section in report.report.sections:
            for block in section.blocks or []:
                for target in [
                    *(block.spans or []),
                    *(block.items or []),
                    *(c for row in (block.rows or []) for c in (row or [])),
                ]:
                    new, k = _wrap_latex_in_text(target.text or "")
                    if k:
                        logger.warning(
                            "wrapped %d undelimited LaTeX run(s) in span (before: %s | after: %s)",
                            k,
                            _snip(target.text, 50),
                            _snip(new, 50),
                        )
                        target.text = new
                        changed += 1
                new, k = _wrap_latex_in_text(block.text or "")
                if k:
                    logger.warning(
                        "wrapped %d undelimited LaTeX run(s) in block text (before: %s | after: %s)",
                        k,
                        _snip(block.text, 50),
                        _snip(new, 50),
                    )
                    block.text = new
                    changed += 1
    except Exception:
        pass
    return changed


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
