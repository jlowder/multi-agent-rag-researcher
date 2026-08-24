from .model_runner import run_model
from typing import Optional
from utils.config import get_config

"""
Writer Agent 
=====================================================================================
It used by the Orchestrator to write a report on the retrieved information
from the retriever agent.
Main role is to draft a clear, grounded response from the evidence it receives.
"""
def writer_agent(
    user_query: str,
    evidence_text: str,
    verbose: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Execute the writer agent.
    
    Args:
        user_query: The user's original query
        evidence_text: The evidence to base the report on
        verbose: Whether to print debug information
        endpoint: Optional custom endpoint URL
        api_key: Optional custom API key
        
    Returns:
        The written report as a string
    """
    if verbose:
        print("[Writer Agent] Writing report...")

    instructions = (
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
    return response.output_text


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
2. Cite every factual claim with a key that is ACTUALLY PROVIDED in the
   evidence for this section (e.g. "[D1]", "[D1, W2]") at the end of the
   sentence it supports. Cite at least once per 2-3 factual sentences.
   NEVER invent or reuse a key that is not present in this section's
   evidence.
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
) -> str:
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

    Returns:
        The section markdown starting with "## {section_heading}".
    """
    if verbose:
        print(f"[WRITER] Writing section: {section_heading}")

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

    # Defensive: the contract requires the section to start with its heading.
    if text and not text.startswith("## "):
        text = f"## {section_heading}\n\n{text}"
    elif not text:
        text = f"## {section_heading}"
    return text