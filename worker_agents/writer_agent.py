from .model_runner import run_model
from typing import Any, Optional
from utils.config import get_config
from utils.json_extract import extract_json_payload_span
import json
import re
from datetime import datetime, timezone

from models import BlockType, Metadata, Report, ReportBlock, Section

"""
Writer Agent 
=====================================================================================
It used by the Orchestrator to write a report on the retrieved information
from the retriever agent.
Main role is to draft a clear, grounded response from the evidence it receives.
"""

_CITATION_KEY_RE = re.compile(r"[DW]\d+")

_JSON_DECODER = json.JSONDecoder()


# Shared math-writing rules, appended to every writer prompt. Math must be
# LaTeX (the PDF engine renders it with KaTeX); the current failure modes
# are Unicode super/subscripts in prose (x², y₀, ẋ, ⁿ) and display math
# dumped into code_block (renders as a monospace snippet, not a formula).
_MATH_RULES_JSON = """
MATH — all mathematics is written in LaTeX (the PDF engine renders it with
KaTeX). These rules are mandatory:
- Inline math: use $...$ inside a span's text — e.g. "growth of $e^{rt}$
  outpaced the baseline". Put no space after an opening $ or before a
  closing $, and close the $ before any following word or digit ($5 and
  $10 stays plain text). A literal dollar sign is written as \\$
- Display math: emit a dedicated equation block, ONE formula per block:
  {"type":"equation","text":"E = mc^2","language":"latex"}. The text is
  RAW LaTeX — never wrap it in $$ or \\[ \\]. NEVER put math in a
  code_block; that renders as a monospace code snippet, not a formula.
- Use LaTeX commands, never Unicode: \\frac{a}{b}, \\sqrt{}, \\sum, \\int,
  ^{} for superscripts, _{} for subscripts, \\dot{x}, \\le, \\approx,
  \\alpha. NEVER emit Unicode super/subscripts or math symbols in prose
  (x², y₀, ẋ, ⁿ, ≤, ≈).
- Explanatory prose (what each symbol means) stays in normal span text
  outside the math delimiters, with its citations as usual.
"""

# Markdown-mode variant of the same rules: display math is a standalone
# $$…$$ paragraph (never a fenced code block).
_MATH_RULES_MD = r"""
MATH — all mathematics is written in LaTeX (the PDF engine renders it with
KaTeX). These rules are mandatory:
- Inline math: use $...$ inside a sentence — no space right after an
  opening $ or before a closing $; a literal dollar sign is written as \$.
- Display math: a standalone $$...$$ paragraph on its own line. NEVER put
  display math in a fenced code block — that renders as monospace text,
  not a formula.
- Use LaTeX commands, never Unicode: \frac{a}{b}, \sqrt{}, \sum, \int,
  ^{} for superscripts, _{} for subscripts, \dot{x}, \le, \approx,
  \alpha. NEVER emit Unicode super/subscripts or math symbols in prose
  (x², y₀, ẋ, ⁿ, ≤, ≈).
- Explanatory prose (what each symbol means) stays outside the math
  delimiters, with citations as usual.
"""


def _extract_json_object_span(text: str) -> tuple[dict | None, int]:
    """Extract a single JSON object from a model response (best-effort),
    returning (object, end-offset).

    Delegates to the shared utils.json_extract extractor (single source of
    truth for preamble/fence-tolerant JSON recovery) with the writer's
    preferred keys: among top-level objects, one carrying "blocks" or
    "heading" wins (largest decoded span) — a prose decoy like
    `example: {"heading": "wrong"}` may precede the real section —
    otherwise the first top-level object wins, exactly as the original
    object-only implementation did. Returns (None, -1) when no object
    parses (soft-fail). The end offset is the position in the stripped
    (de-fenced) text where the chosen object's decode ended, for
    post-object premature-close recovery.
    """
    value, end = extract_json_payload_span(text, prefer_keys=("blocks", "heading"))
    if value is None or not isinstance(value, dict):
        return None, -1
    return value, end


def _extract_json_object(text: str) -> dict | None:
    """Extract a single JSON object from a model response (best-effort).
    Same selection rules as _extract_json_object_span; returns the parsed
    dict, or None when nothing parses (soft-fail)."""
    return _extract_json_object_span(text)[0]


_BLOCK_VOCAB = frozenset(t.value for t in BlockType)
_MAX_RECOVERED_BLOCKS = 20


def _recover_trailing_blocks(
    text: str, end: int, existing: list[ReportBlock] | None = None
) -> list[ReportBlock]:
    """Premature-close recovery: some models close the section object early
    and keep writing the remaining blocks as trailing JSON. Scan `text[end:]`
    (raw_decode at every '{') and keep complete objects whose "type" is a
    block-vocabulary value and which validate as ReportBlock (the lenient
    string-cell coercion applies; invalids are skipped). Ignored entirely
    when the remainder is insignificant (<200 non-ws chars and no '{').
    Capped at _MAX_RECOVERED_BLOCKS blocks.

    Degenerate-loop guard: a recovered block that is an EXACT duplicate of a
    block already in the section (`existing`, normalized via
    model_dump_json) or of an earlier recovery in the same remainder is
    dropped, so a model that re-emits one block after its premature close
    cannot bloat the section; genuinely novel blocks are still salvaged."""
    if end < 0:
        return []
    remainder = text[end:]
    if len(remainder.strip()) < 200 and "{" not in remainder:
        return []
    existing_keys = {block.model_dump_json() for block in (existing or [])}
    out: list[ReportBlock] = []
    seen: set[str] = set()
    spans: list[tuple[int, int]] = []
    for i, ch in enumerate(remainder):
        if ch != "{" or len(out) >= _MAX_RECOVERED_BLOCKS:
            continue
        try:
            obj, b_end = _JSON_DECODER.raw_decode(remainder, i)
        except ValueError:
            continue
        if not isinstance(obj, dict) or obj.get("type") not in _BLOCK_VOCAB:
            continue
        if any(start <= i < e for start, e in spans):
            continue
        try:
            block = ReportBlock.model_validate(obj)
        except Exception:
            continue
        key = block.model_dump_json()
        if key in existing_keys or key in seen:
            continue
        out.append(block)
        seen.add(key)
        spans.append((i, b_end))
    return out


def _is_truncated_response(response) -> bool:
    """True when the API signals the response was cut short by OUTPUT
    LENGTH — the one case a 2x retry can help.

    When incomplete_details is present it is authoritative: only
    reason == "max_output_tokens" counts (a content_filter termination
    will filter again at 2x — retrying is pointless). The bare
    status == "incomplete" check is used only when details is missing.
    Tolerates missing attributes.
    """
    details = getattr(response, "incomplete_details", None)
    if details is not None:
        reason = (
            details.get("reason")
            if isinstance(details, dict)
            else getattr(details, "reason", None)
        )
        return reason == "max_output_tokens"
    return getattr(response, "status", None) == "incomplete"


def _looks_truncated_json(text: str) -> bool:
    """Heuristic for a cut-off JSON object: after stripping one
    full-response code fence, more "{" than "}" means the object never
    closed."""
    if not text:
        return False
    candidate = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    return candidate.count("{") > candidate.count("}")


def _has_no_json_content(text: Optional[str]) -> bool:
    """True when the response carries no JSON at all — empty or pure prose.

    The brace-imbalance heuristic (_looks_truncated_json) and the API's
    truncation signal (_is_truncated_response) both miss this shape: the
    local Responses shim never sets incomplete_details, and prose has no
    braces to be imbalanced. A thinking-heavy model that spends its fixed
    output budget on prose would otherwise yield a silent empty section, so
    this also triggers the bounded 2x retry.
    """
    if text is None:
        return True
    candidate = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    return "{" not in candidate and "[" not in candidate


# Bounded per-run budget for truncated-draft retries: a truncated
# section/synthesis draft gets ONE retry at 2x max_output_tokens, and the
# budget keeps a chronically truncating model from eating the whole run's
# LLM allowance. Reset at the start of each deep run (deep_research) and in
# tests.
_writer_retry_budget = 4


def reset_writer_retry_budget(n: int = 4) -> None:
    """Reset the per-run truncation-retry budget (call at run start)."""
    global _writer_retry_budget
    _writer_retry_budget = max(0, int(n))


def _take_writer_retry() -> bool:
    """Consume one retry if the budget allows; False when spent."""
    global _writer_retry_budget
    if _writer_retry_budget <= 0:
        return False
    _writer_retry_budget -= 1
    return True


_HEADING_ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"


def _normalize_heading(text: str) -> str:
    """Tolerant heading comparison form: strip zero-width chars, collapse
    whitespace, casefold, and drop trailing punctuation (:;.-—…)."""
    s = "".join(ch for ch in (text or "") if ch not in _HEADING_ZERO_WIDTH)
    s = " ".join(s.split()).casefold()
    return s.rstrip(" \t.:;.-—…")


def _strip_duplicate_heading(section: Section) -> None:
    """Drop blocks[0] when it re-emits the section's own title (tolerant
    match on the normalized text). Only the FIRST block is considered —
    later subsection headings are never touched. Mutates section in place.
    """
    if not section.blocks:
        return
    first = section.blocks[0]
    if first.type != "heading":
        return
    text = "".join(span.text for span in first.spans or []) or first.text
    if _normalize_heading(text) and _normalize_heading(text) == _normalize_heading(section.heading):
        section.blocks = section.blocks[1:]


def _slug(heading: str) -> str:
    """Lowercase-hyphen slug: non-alphanumeric runs collapse to one '-'."""
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def _salvage_blocks(blocks: Any) -> list[ReportBlock]:
    """Keep only block entries that individually validate as ReportBlock
    (salvage-on-validation-failure): a single malformed block (e.g. a
    table whose cells a weak model wrote with the wrong shape) must not
    sink the whole section. Non-list input yields []."""
    if not isinstance(blocks, list):
        return []
    kept: list[ReportBlock] = []
    for block in blocks:
        if isinstance(block, ReportBlock):
            kept.append(block)
            continue
        try:
            kept.append(ReportBlock.model_validate(block))
        except Exception:
            continue
    return kept


WRITE_SECTION_JSON_INSTRUCTIONS = """
You are writing ONE section of a deep-research report. You will be given:
- the overall user query (for context only — do not answer it directly),
- the full report outline (for context: where this section sits and what the
  other sections cover — do not duplicate their content),
- THIS section's heading and its context (the sub-question and angle it must
  answer),
- the citation-keyed evidence assigned to this section: every item carries a
  key like [D1] or [W2],
- one-line summaries of the sections already written (for coherence: do not
  repeat what they established).

CONTRACT — all of these are mandatory:
1. Write at least 300 words of substance: concrete facts, mechanisms,
   comparisons, and examples grounded in the evidence for THIS section —
   not filler. Keep the section between about 300 and 1100 words of
   substance.
2. Cite every factual sentence with a key that is ACTUALLY PROVIDED in the
   evidence for this section: every factual sentence is one containing a
   claim, name, number, date, or specific finding. Density target for this
   section: at least 4 citations per 100 words; reusing the same key in
   multiple sentences is correct and expected. NEVER invent a key that is
   not present in this section's evidence. Each key goes at the end of the
   sentence it supports.
3. Output EXACTLY ONE JSON object — no Markdown, no code fences, no
   commentary. Shape: {"id": "<lowercase-hyphen slug of the section
   heading>", "heading": "<the exact section heading provided in the input>",
   "blocks": [...]}. The exact section heading provided in the input is:
   {SECTION_HEADING} — use it verbatim as "heading" and its lowercase-hyphen
   slug as "id".
4. Block vocabulary (one line each):
   paragraph {"type":"paragraph","spans":[{"text":"...","citations":["D1"]}]}
   heading {"type":"heading","level":3,"text":"..."}
   unordered_list/ordered_list {"type":"...","items":[{"text":"...","citations":[...]}]}
   callout {"type":"callout","callout_type":"note|warning|info","spans":[...]}
   comparison_table {"type":"comparison_table","caption":"...","columns":["A","B"],"rows":[[{"text":"cell","citations":["D1"]},{"text":"cell2","citations":[]}]]}
   code_block {"type":"code_block","language":"...","text":"..."}   (code snippets only — NEVER for math)
   equation {"type":"equation","text":"<raw LaTeX, no outer delimiters>","language":"latex"}
   page_break {"type":"page_break"}
   citation_note {"type":"citation_note","spans":[...]}
   Table cells and list items must be span objects, never bare strings.
5. Per-sentence spans: split each paragraph into spans so the span that ends
   a cited sentence carries that sentence's citation keys; uncited
   transition spans get citations [].
6. Use at least 2 blocks per section (a subsection heading is encouraged).
   Do NOT include the section's own title as a heading block — the title is
   carried by the `heading` field. Include subsection heading blocks only
   for subsections you actually write.

Example of one paragraph block with per-sentence citation spans:
{"type":"paragraph","spans":[{"text":"Grid-scale deployment rose sharply","citations":[]},{"text":"in several major electricity markets","citations":["D1","W2"]},{"text":"through 2024","citations":["D1"]}]}
""" + _MATH_RULES_JSON


SYNTHESIS_JSON_INSTRUCTIONS = """
You are writing the FINAL SYNTHESIS section of a deep-research report. You
will be given the overall user query (context only) and the finished content
sections of the report, each with its heading.

CONTRACT — all of these are mandatory:
1. Output EXACTLY ONE JSON object — no Markdown, no code fences, no
   commentary. Shape: {"id": "synthesis", "heading": "Synthesis", "blocks":
   [...]}. The heading MUST be exactly "Synthesis".
2. Write 300-500 words that connect the report's sections: explicitly
   compare or contrast the findings of at least 3 different sections BY
   NAME (use their headings).
3. If the sections contain tensions or contradictions, identify them
   explicitly; if they do not, say so honestly.
4. State 2-3 field-level implications that follow from the combined
   findings of the report.
5. Citations: you may use ONLY [D#]/[W#] keys that appear in the provided
   section texts, and only at the end of the sentence they support — never
   invent a key. A pure-synthesis sentence may have citations [].
6. Do NOT introduce new factual claims not present in the provided
   sections: you are connecting what is already there, not researching.
7. Block vocabulary (one line each):
   paragraph {"type":"paragraph","spans":[{"text":"...","citations":["D1"]}]}
   heading {"type":"heading","level":3,"text":"..."}
   unordered_list/ordered_list {"type":"...","items":[{"text":"...","citations":[...]}]}
   callout {"type":"callout","callout_type":"note|warning|info","spans":[...]}
   comparison_table {"type":"comparison_table","caption":"...","columns":["A","B"],"rows":[[{"text":"cell","citations":["D1"]},{"text":"cell2","citations":[]}]]}
   code_block {"type":"code_block","language":"...","text":"..."}   (code snippets only — NEVER for math)
   equation {"type":"equation","text":"<raw LaTeX, no outer delimiters>","language":"latex"}
   page_break {"type":"page_break"}
   citation_note {"type":"citation_note","spans":[...]}
   Table cells and list items must be span objects, never bare strings.
8. Per-sentence spans: split each paragraph into spans so the span that ends
   a cited sentence carries that sentence's citation keys; uncited
   transition spans get citations []. Use at least 2 blocks.
""" + _MATH_RULES_JSON


WRITE_REPORT_JSON_INSTRUCTIONS = """
Answer the user using only the evidence, as one structured JSON report.

CONTRACT — all of these are mandatory:
1. Output EXACTLY ONE JSON object — no Markdown, no code fences, no
   commentary. Shape: {"title": "...", "subtitle": "", "executive_summary":
   ["..."], "sections": [... 7-8 section objects ...]}.
2. "title" describes the topic (not a copy of the query); "subtitle" may be
   the empty string; "executive_summary" is a list of 2-4 self-contained
   summary paragraphs.
3. "sections" holds 7-8 section objects. Use this default set, adapting
   section names to the topic when a better fit exists:
   1. Definition & Background
   2. Core Components & Mechanics
   3. Major Variants & Alternative Approaches
   4. Dynamics / Evaluation / Analysis
   5. Applications
   6. Tools & Ecosystem
   7. Limitations & Open Problems
   8. Synthesis
   Each section object: {"id": "<lowercase-hyphen slug of its heading>",
   "heading": "<the section heading>", "blocks": [...]} with at least 300
   words of substance — concrete facts, mechanisms, comparisons, and
   examples grounded in the evidence, not filler.
4. Block vocabulary (one line each):
   paragraph {"type":"paragraph","spans":[{"text":"...","citations":["D1"]}]}
   heading {"type":"heading","level":3,"text":"..."}
   unordered_list/ordered_list {"type":"...","items":[{"text":"...","citations":[...]}]}
   callout {"type":"callout","callout_type":"note|warning|info","spans":[...]}
   comparison_table {"type":"comparison_table","caption":"...","columns":["A","B"],"rows":[[{"text":"cell","citations":["D1"]},{"text":"cell2","citations":[]}]]}
   code_block {"type":"code_block","language":"...","text":"..."}   (code snippets only — NEVER for math)
   equation {"type":"equation","text":"<raw LaTeX, no outer delimiters>","language":"latex"}
   page_break {"type":"page_break"}
   citation_note {"type":"citation_note","spans":[...]}
   Table cells and list items must be span objects, never bare strings.
5. Per-sentence spans: split each paragraph into spans so the span that ends
   a cited sentence carries that sentence's citation keys; uncited
   transition spans get citations [].
6. Cite every factual sentence with a key that is ACTUALLY PROVIDED in the
   evidence. Density target: at least 4 citations per 100 words; reusing the
   same key in multiple sentences is correct and expected. NEVER invent a
   key that is not present in the evidence. Quantitative claims must always
   carry a citation.
7. Fully explain every concept (how it works, why, formulation, practical
   implications); include mathematical formulas, architectural details,
   quantitative comparisons, and specific examples from the evidence.
8. When the evidence does not cover a fact you would need, write "the
   available evidence does not cover X" — an honest gap. Never fill from
   memory. If the evidence is weak or incomplete, say so.
""" + _MATH_RULES_JSON


def writer_agent(
    user_query: str,
    evidence_text: str,
    verbose: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    output_format: str = "json",
) -> str | Report:
    """
    Execute the writer agent.
    
    Args:
        user_query: The user's original query
        evidence_text: The evidence to base the report on
        verbose: Whether to print debug information
        endpoint: Optional custom endpoint URL
        api_key: Optional custom API key
        output_format: "markdown" (default, unchanged) or "json" — JSON mode
            asks for one structured JSON report object and returns a
            validated models.Report (soft-fail: on any parse/validation
            problem a minimal empty Report is returned, never raises).
        
    Returns:
        The written report as a string
        (markdown mode) or a models.Report (json mode)
    """
    output_format = "markdown" if output_format not in ("markdown", "json") else output_format
    if verbose:
        print("[Writer Agent] Writing report...")

    markdown_instructions = (
        """
        Answer the user using only the evidence.
        Start with the answer.
        Provide a thorough, detailed, and comprehensive report.
        Include the key supporting details needed to fully answer the question.
        When the evidence supports it, include the most important comparisons, caveats, or specific facts rather than giving a minimal summary.
        For judgments, comparisons, recommendations, or conclusions state the best-supported conclusion clearly.
        For short follow-ups, answer comprehensively and with detail.
        Do not add unsupported facts.
        Use the exact citation field from Document evidence for PDF citations.
        If web evidence is present, synthesize it into a comprehensive, self-contained answer and include explicit web citations with the exact title and exact URL from Web evidence.
        When citing web sources, use Markdown links in the form [Exact Source Title](Exact URL).
        Do not omit web source URLs when web evidence is used.
        If the evidence is weak or incomplete, say so.
        Do not end with a question, a suggestion for the user to ask a follow-up, or an offer for more help.

        REPORT DEPTH REQUIREMENTS:
        - Every concept should be fully explained, not just named. For example, don't just define "multi-head attention" — explain how it works, why it's better than single-head, the mathematical formulation, and practical implications.
        - Include mathematical formulas, architectural details, and quantitative comparisons where available in the evidence.
        - Include specific examples, analogies, and visual descriptions from the evidence.
        - Organize the report with clear section headings and sub-headings for each major topic.
        - Aim for a comprehensive, graduate-level technical report that could serve as a standalone reference on the topic.

        REQUIRED OUTLINE (follow this two-step process):
        - Step 1: Begin the report with a "## Report Outline" section listing the 7-8 "##" sections you will write. Use this default set, adapting section names to the topic when a better fit exists:
          1. Definition & Background
          2. Core Components & Mechanics
          3. Major Variants & Alternative Approaches
          4. Dynamics / Evaluation / Analysis
          5. Applications
          6. Tools & Ecosystem
          7. Limitations & Open Problems
          8. Synthesis
        - Step 2: After the outline, write each section under its own "##" heading with at least 300 words of substance — concrete facts, mechanisms, comparisons, and examples grounded in the evidence, not filler.
        """ + _MATH_RULES_MD
    )
    instructions = WRITE_REPORT_JSON_INSTRUCTIONS if output_format == "json" else markdown_instructions

    # Pass the user query together with the retrieval context for drafting.
    input_text = (
        f"User query: {user_query}\n\n"
        f"Evidence:\n{evidence_text}"
    )

    config = get_config()
    response = run_model(
        instructions=instructions,
        input_data=input_text,
        reasoning_effort=config.get_reasoning_effort("writer"),
        max_output_tokens=config.get_max_output_tokens("writer"),
        tools=None,
        agent_name="writer",
        endpoint=endpoint,
        api_key=api_key,
    )
    if output_format != "json":
        return response.output_text

    obj = _extract_json_object(response.output_text or "")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if obj is not None:
        try:
            title = str(obj.get("title") or "").strip() or (user_query[:100])
            sections: list[Section] = []
            for s in obj.get("sections", []):
                if not isinstance(s, dict):
                    continue
                try:
                    sections.append(Section.model_validate(s))
                except Exception as exc:
                    if verbose:
                        print(f"[Writer Agent] WARNING: skipping invalid section: {exc}")
            return Report(
                metadata=Metadata(
                    title=title,
                    subtitle=str(obj.get("subtitle") or ""),
                    query=user_query,
                    session_id="",
                    generated_at=generated_at,
                    report_type="standard",
                ),
                executive_summary=[str(p) for p in obj.get("executive_summary", []) if str(p).strip()],
                sections=sections,
                sources=[],
            )
        except Exception as exc:
            if verbose:
                print(f"[Writer Agent] WARNING: report JSON validation failed ({exc}); returning empty report")
    elif verbose:
        print("[Writer Agent] WARNING: no JSON object in writer response; returning empty report")
    return Report(
        metadata=Metadata(
            title=user_query[:100],
            query=user_query,
            session_id="",
            generated_at=generated_at,
            report_type="standard",
        ),
        executive_summary=[],
        sections=[],
        sources=[],
    )


"""
Per-section drafting (P1-3, DEEP.md §3.4 / plan B.3)
=====================================================================================
Writes ONE section of a deep-research report against that section's own
citation-keyed evidence subset, instead of the legacy whole-report monologue.
The contract is enforced by the prompt and by the critic (P1-4), which sends
weak sections back here with expansion_gaps.
"""

WRITE_SECTION_INSTRUCTIONS = """
You are writing ONE section of a deep-research report. You will be given:
- the overall user query (for context only — do not answer it directly),
- the full report outline (for context: where this section sits and what the
  other sections cover — do not duplicate their content),
- THIS section's heading and its context (the sub-question and angle it must
  answer),
- the citation-keyed evidence assigned to this section: every item carries a
  key like [D1] or [W2],
- one-line summaries of the sections already written (for coherence: do not
  repeat what they established).

CONTRACT — all of these are mandatory:
1. Write at least 300 words of substance: concrete facts, mechanisms,
   comparisons, and examples grounded in the evidence — not filler.
2. Cite every factual sentence with a key that is ACTUALLY PROVIDED in the
   evidence for this section (e.g. "[D1]", "[D1, W2]") at the end of the
   sentence it supports: every factual sentence is one containing a claim,
   name, number, date, or specific finding. Density target for this
   section: at least 4 citations per 100 words; reusing the same key in
   multiple sentences is correct and expected. NEVER invent a key that is
   not present in this section's evidence. When a sentence cannot be cited
   (pure synthesis or transition) — keep such sentences to a
   minority of the section.
3. Include at least 2 concrete specifics (named works, named researchers,
   numbers, dates) whenever the evidence supports them. Quantitative claims
   (numbers, dates, named results) must always carry a citation.
4. When the evidence does not cover a fact you would need, write "the
   available evidence does not cover X" — an honest gap. Never fill from
   memory.
5. Output Markdown starting exactly with the line: ## {SECTION_HEADING}
   (the heading is provided in the input). Do not add a title above it.

BANNED — the critic rejects sections that contain any of these:
- Restating the section heading as an opening paragraph (start with the
  strongest cited fact instead).
- Filler that carries no fact: "it is important to note that…", "in
  today's world…", "in conclusion it is worth mentioning…".
- Closing paragraphs of pure hedging or follow-up offers ("more research
  could…", "future work may explore…"). End on a substantive, cited
  statement.
""" + _MATH_RULES_MD


def write_section(
    user_query: str,
    outline: str,
    section_heading: str,
    section_context: str,
    evidence_text: str,
    prior_summaries: str = "",
    expansion_gaps: list[str] | None = None,
    verbose: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    output_format: str = "json",
) -> str | Section:
    """
    Write ONE section of the deep-research report (P1-3).

    Args:
        user_query: The overall user query (context only).
        outline: The full report outline (headings) for positioning.
        section_heading: Heading for THIS section; the output starts with
            "## {section_heading}".
        section_context: This section's sub-question and angle.
        evidence_text: Citation-keyed evidence for this section only
            (global [D#]/[W#] keys from the shared registry).
        prior_summaries: One-line summaries of sections already written
            (coherence; avoid repetition).
        expansion_gaps: When non-empty (revision round), the critic's gap
            list for a previous draft of this section; the section is
            rewritten in full with every gap fixed.
        verbose: Print a one-line progress note.
        endpoint: Optional custom endpoint URL.
        api_key: Optional custom API key.
        output_format: "markdown" (default, unchanged) or "json" — JSON mode
            returns a validated models.Section (soft-fail: on any
            parse/validation problem a minimal empty Section is returned,
            never raises).

    Returns:
        The section markdown starting with "## {section_heading}"
        (markdown mode) or a models.Section (json mode)
    """
    output_format = "markdown" if output_format not in ("markdown", "json") else output_format
    if verbose:
        print(f"[WRITER] Writing section: {section_heading}")

    if output_format == "json":
        instructions = WRITE_SECTION_JSON_INSTRUCTIONS.replace("{SECTION_HEADING}", section_heading)
    else:
        instructions = WRITE_SECTION_INSTRUCTIONS.replace("{SECTION_HEADING}", section_heading)

    parts = [
        f"Overall user query: {user_query}",
        f"\nReport outline (full report — for context only):\n{outline}",
        f"\nTHIS section:\nHeading: {section_heading}\n{section_context}",
        f"\nEvidence for this section (cite ONLY these provided keys):\n{evidence_text}",
    ]
    if prior_summaries:
        parts.append(f"\nSections already written (one line each — do not repeat):\n{prior_summaries}")
    if expansion_gaps:
        gap_list = "; ".join(g for g in expansion_gaps if g and g.strip())
        parts.append(
            "\nREVISION REQUIRED: A previous review found these problems in "
            f"your draft of this section: {gap_list}. Fix every one while keeping "
            "the section coherent; rewrite the section in full."
        )
    input_text = "\n".join(parts)

    config = get_config()
    response = run_model(
        instructions=instructions,
        input_data=input_text,
        reasoning_effort=config.get_reasoning_effort("writer"),
        max_output_tokens=config.get_max_output_tokens("writer"),
        tools=None,
        agent_name="writer",
        endpoint=endpoint,
        api_key=api_key,
    )
    text = (response.output_text or "").strip()

    if output_format == "json":
        obj, obj_end = _extract_json_object_span(text)
        if obj is None:
            # An explicit API termination (incomplete_details, e.g.
            # content_filter) wins over the pure-prose heuristic: filtered
            # content is prose-shaped, but retrying it only re-filters.
            api_details = getattr(response, "incomplete_details", None)
            if _is_truncated_response(response) or _looks_truncated_json(text):
                retry_reason = "looks truncated"
            elif api_details is None and _has_no_json_content(text):
                retry_reason = "contains no JSON object (empty or pure-prose output)"
            else:
                retry_reason = None
            if retry_reason is not None and _take_writer_retry():
                if verbose:
                    print(
                        f"[WRITER] WARNING: '{section_heading}' output {retry_reason}; "
                        "retrying at 2x max_output_tokens"
                    )
                response = run_model(
                    instructions=instructions,
                    input_data=input_text,
                    reasoning_effort=config.get_reasoning_effort("writer"),
                    max_output_tokens=2 * config.get_max_output_tokens("writer"),
                    tools=None,
                    agent_name="writer",
                    endpoint=endpoint,
                    api_key=api_key,
                )
                text = (response.output_text or "").strip()
                obj, obj_end = _extract_json_object_span(text)
        if obj is not None:
            # Repair-before-validate: a single missing/None heading or id
            # must not kill an otherwise complete section.
            if isinstance(obj, dict):
                obj["heading"] = obj.get("heading") or section_heading
                obj["id"] = obj.get("id") or _slug(section_heading)
            try:
                section = Section.model_validate(obj)
            except Exception:
                section = None
            if section is None and isinstance(obj, dict):
                salvaged = _salvage_blocks(obj.get("blocks"))
                if salvaged:
                    if verbose:
                        print(
                            f"[WRITER] WARNING: '{section_heading}' — kept "
                            f"{len(salvaged)} of {len(obj.get('blocks') or [])} blocks "
                            "after dropping invalid content"
                        )
                    try:
                        section = Section(
                            id=obj.get("id") or _slug(section_heading),
                            heading=obj.get("heading") or section_heading,
                            blocks=salvaged,
                        )
                    except Exception:
                        # Wrong-typed heading/id (e.g. 123) would raise out
                        # of the function; never-raise contract wins — fall
                        # through to the soft-fail empty section below.
                        section = None
                        if verbose:
                            print(
                                f"[WRITER] WARNING: '{section_heading}' — "
                                "salvage construction failed; returning empty section"
                            )
            if section is not None:
                if obj_end >= 0:
                    recovered = _recover_trailing_blocks(text, obj_end, section.blocks)
                    if recovered:
                        if verbose:
                            # INFO, not WARNING: live measurement showed the
                            # model closes the object early but then emits
                            # every intended block, so this recovery is
                            # lossless — the section is complete as written.
                            print(
                                f"[WRITER] INFO: '{section_heading}' — "
                                f"recovered {len(recovered)} block(s) written "
                                "past the section object's premature close"
                            )
                        section.blocks.extend(recovered)
                if not section.heading:
                    section.heading = section_heading
                if not section.id:
                    section.id = _slug(section_heading)
                _strip_duplicate_heading(section)
                return section
        if verbose:
            print(f"[WRITER] WARNING: section JSON validation failed for '{section_heading}'; returning empty section")
        return Section(id=_slug(section_heading), heading=section_heading, blocks=[])

    # Defensive: the contract requires the section to start with its heading.
    if text and not text.startswith("## "):
        text = f"## {section_heading}\n\n{text}"
    elif not text:
        text = f"## {section_heading}"
    return text


"""
Cross-section synthesis (P2-4a)
=====================================================================================
Writes the final "## Synthesis" section that connects the finished content
sections. Drafted AFTER the critic (it only connects existing claims, so the
critic does not review it) and appended as the LAST content section before
assembly. One-shot call, no tools — same style as write_section.
"""

SYNTHESIS_INSTRUCTIONS = """
You are writing the FINAL SYNTHESIS section of a deep-research report. You
will be given the overall user query (context only) and the finished content
sections of the report, each with its heading.

CONTRACT — all of these are mandatory:
1. Output Markdown starting exactly with the line: ## Synthesis
   (do not add a title above it).
2. Write 300-500 words that connect the report's sections: explicitly
   compare or contrast the findings of at least 3 different sections BY
   NAME (use their headings).
3. If the sections contain tensions or contradictions, identify them
   explicitly; if they do not, say so honestly.
4. State 2-3 field-level implications that follow from the combined
   findings of the report.
5. Citations: you may use ONLY [D#]/[W#] keys that appear in the provided
   section texts, and only at the end of a sentence they support — never
   invent a key. A pure-synthesis sentence may carry no citation.
6. Do NOT introduce new factual claims not present in the provided
   sections: you are connecting what is already there, not researching.
""" + _MATH_RULES_MD


def write_synthesis(
    user_query: str,
    sections: list[tuple[str, str]],
    verbose: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    output_format: str = "json",
) -> str | Section:
    """
    Write the final cross-section Synthesis section (P2-4a).

    Args:
        user_query: The overall user query (context only).
        sections: (heading, section_text) pairs for every finished content
            section, in report order.
        verbose: Print a one-line progress note.
        endpoint: Optional custom endpoint URL.
        api_key: Optional custom API key.
        output_format: "markdown" (default, unchanged) or "json" — JSON mode
            returns a validated models.Section (soft-fail: on any
            parse/validation problem a minimal empty Section is returned,
            never raises).

    Returns:
        The synthesis markdown starting with "## Synthesis"
        (markdown mode) or a models.Section (json mode)
    """
    output_format = "markdown" if output_format not in ("markdown", "json") else output_format
    if verbose:
        print("[WRITER] Writing synthesis section")

    section_blocks = "\n\n".join(
        f"### {heading}\n{text}" for heading, text in sections
    )
    input_text = (
        f"Overall user query: {user_query}\n\n"
        f"Finished content sections (connect these by name; cite ONLY keys "
        f"already present in them):\n{section_blocks}"
    )

    instructions = SYNTHESIS_JSON_INSTRUCTIONS if output_format == "json" else SYNTHESIS_INSTRUCTIONS

    config = get_config()
    response = run_model(
        instructions=instructions,
        input_data=input_text,
        reasoning_effort=config.get_reasoning_effort("writer"),
        max_output_tokens=config.get_max_output_tokens("writer"),
        tools=None,
        agent_name="writer",
        endpoint=endpoint,
        api_key=api_key,
    )
    text = (response.output_text or "").strip()

    if output_format == "json":
        obj, obj_end = _extract_json_object_span(text)
        if obj is None:
            # An explicit API termination (incomplete_details, e.g.
            # content_filter) wins over the pure-prose heuristic: filtered
            # content is prose-shaped, but retrying it only re-filters.
            api_details = getattr(response, "incomplete_details", None)
            if _is_truncated_response(response) or _looks_truncated_json(text):
                retry_reason = "looks truncated"
            elif api_details is None and _has_no_json_content(text):
                retry_reason = "contains no JSON object (empty or pure-prose output)"
            else:
                retry_reason = None
            if retry_reason is not None and _take_writer_retry():
                if verbose:
                    print(
                        f"[WRITER] WARNING: synthesis output {retry_reason}; "
                        "retrying at 2x max_output_tokens"
                    )
            response = run_model(
                instructions=instructions,
                input_data=input_text,
                reasoning_effort=config.get_reasoning_effort("writer"),
                max_output_tokens=2 * config.get_max_output_tokens("writer"),
                tools=None,
                agent_name="writer",
                endpoint=endpoint,
                api_key=api_key,
            )
            text = (response.output_text or "").strip()
            obj, obj_end = _extract_json_object_span(text)
        if obj is not None:
            # Repair-before-validate: a single missing/None heading or id
            # must not kill an otherwise complete section.
            if isinstance(obj, dict):
                obj["heading"] = obj.get("heading") or "Synthesis"
                obj["id"] = obj.get("id") or "synthesis"
            try:
                section = Section.model_validate(obj)
            except Exception:
                section = None
            if section is None and isinstance(obj, dict):
                salvaged = _salvage_blocks(obj.get("blocks"))
                if salvaged:
                    if verbose:
                        print(
                            f"[WRITER] WARNING: 'Synthesis' — kept "
                            f"{len(salvaged)} of {len(obj.get('blocks') or [])} blocks "
                            "after dropping invalid content"
                        )
                    try:
                        section = Section(
                            id=obj.get("id") or "synthesis",
                            heading=obj.get("heading") or "Synthesis",
                            blocks=salvaged,
                        )
                    except Exception:
                        # Wrong-typed heading/id (e.g. 123) would raise out
                        # of the function; never-raise contract wins — fall
                        # through to the soft-fail empty section below.
                        section = None
                        if verbose:
                            print(
                                "[WRITER] WARNING: 'Synthesis' — salvage "
                                "construction failed; returning empty section"
                            )
            if section is not None:
                if obj_end >= 0:
                    recovered = _recover_trailing_blocks(text, obj_end, section.blocks)
                    if recovered:
                        if verbose:
                            # INFO, not WARNING: the premature-close recovery
                            # is measured lossless (the model emits every
                            # intended block; none are dropped).
                            print(
                                f"[WRITER] INFO: 'Synthesis' — "
                                f"recovered {len(recovered)} block(s) written "
                                "past the section object's premature close"
                            )
                        section.blocks.extend(recovered)
                if not section.heading:
                    section.heading = "Synthesis"
                if not section.id:
                    section.id = "synthesis"
                _strip_duplicate_heading(section)
                return section
        if verbose:
            print("[WRITER] WARNING: synthesis JSON validation failed; returning empty section")
        return Section(id="synthesis", heading="Synthesis", blocks=[])

    # Defensive: the contract requires the section to start with its heading.
    if text and not text.startswith("## "):
        text = f"## Synthesis\n\n{text}"
    elif not text:
        text = "## Synthesis"
    return text
