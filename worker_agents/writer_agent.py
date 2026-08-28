from .model_runner import run_model
from typing import Optional
from utils.config import get_config
import json
import re
from datetime import datetime, timezone

from models import Metadata, Report, Section

"""
Writer Agent 
=====================================================================================
It used by the Orchestrator to write a report on the retrieved information
from the retriever agent.
Main role is to draft a clear, grounded response from the evidence it receives.
"""

_CITATION_KEY_RE = re.compile(r"[DW]\d+")

_JSON_DECODER = json.JSONDecoder()


def _extract_json_object(text: str) -> dict | None:
    """Extract a single JSON object from a model response (best-effort).

    Strips ```json/``` code fences if the whole response is fenced, then
    tries to raw_decode a JSON value at EVERY "{" index and collects the
    dicts that parse. (The old first-"{"→last-"}" slice silently returned
    None when prose around the JSON contained braces, or when two objects
    appeared.) If any parsed dict contains a "blocks" or "heading" key the
    FIRST such dict (in order of appearance) is returned; otherwise the
    first parsed dict. Returns None when nothing parses (soft-fail).
    """
    if not text:
        return None
    candidate = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    parsed = []
    for i, ch in enumerate(candidate):
        if ch != "{":
            continue
        try:
            obj, _end = _JSON_DECODER.raw_decode(candidate, i)
        except ValueError:
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    for obj in parsed:
        if "blocks" in obj or "heading" in obj:
            return obj
    return parsed[0] if parsed else None


def _is_truncated_response(response) -> bool:
    """True when the API signals the response was cut short.

    status == "incomplete", or incomplete_details (object or dict form)
    with reason == "max_output_tokens". Tolerates missing attributes.
    """
    if getattr(response, "status", None) == "incomplete":
        return True
    details = getattr(response, "incomplete_details", None)
    if details is None:
        return False
    reason = (
        details.get("reason")
        if isinstance(details, dict)
        else getattr(details, "reason", None)
    )
    return reason == "max_output_tokens"


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


def _slug(heading: str) -> str:
    """Lowercase-hyphen slug: non-alphanumeric runs collapse to one '-'."""
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


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
   not filler.
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
   comparison_table {"type":"comparison_table","caption":"...","columns":[...],"rows":[[...]]}
   code_block {"type":"code_block","language":"...","text":"..."}
   page_break {"type":"page_break"}
   citation_note {"type":"citation_note","spans":[...]}
5. Per-sentence spans: split each paragraph into spans so the span that ends
   a cited sentence carries that sentence's citation keys; uncited
   transition spans get citations [].
6. Use at least 2 blocks per section (a subsection heading is encouraged).

Example of one paragraph block with per-sentence citation spans:
{"type":"paragraph","spans":[{"text":"Grid-scale deployment rose sharply","citations":[]},{"text":"in several major electricity markets","citations":["D1","W2"]},{"text":"through 2024","citations":["D1"]}]}
"""


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
   comparison_table {"type":"comparison_table","caption":"...","columns":[...],"rows":[[...]]}
   code_block {"type":"code_block","language":"...","text":"..."}
   page_break {"type":"page_break"}
   citation_note {"type":"citation_note","spans":[...]}
8. Per-sentence spans: split each paragraph into spans so the span that ends
   a cited sentence carries that sentence's citation keys; uncited
   transition spans get citations []. Use at least 2 blocks.
"""


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
   comparison_table {"type":"comparison_table","caption":"...","columns":[...],"rows":[[...]]}
   code_block {"type":"code_block","language":"...","text":"..."}
   page_break {"type":"page_break"}
   citation_note {"type":"citation_note","spans":[...]}
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
"""


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
        """
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
"""


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
        obj = _extract_json_object(text)
        if obj is not None:
            try:
                section = Section.model_validate(obj)
                if not section.heading:
                    section.heading = section_heading
                if not section.id:
                    section.id = _slug(section_heading)
                return section
            except Exception:
                pass
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
"""


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
        obj = _extract_json_object(text)
        if obj is not None:
            try:
                section = Section.model_validate(obj)
                if not section.heading:
                    section.heading = "Synthesis"
                if not section.id:
                    section.id = "synthesis"
                return section
            except Exception:
                pass
        if verbose:
            print("[WRITER] WARNING: synthesis JSON validation failed; returning empty section")
        return Section(id="synthesis", heading="Synthesis", blocks=[])

    # Defensive: the contract requires the section to start with its heading.
    if text and not text.startswith("## "):
        text = f"## Synthesis\n\n{text}"
    elif not text:
        text = "## Synthesis"
    return text
