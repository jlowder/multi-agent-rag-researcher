import json
import logging
from .model_runner import run_model
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field
from memory import debug, info
from utils.config import get_config

logger = logging.getLogger(__name__)


def parse_evidence_status(text: str) -> Optional[dict]:
    """Parse the EVIDENCE_STATUS block from verifier output using delimiter-based extraction."""
    if not text:
        return None

    try:
        # Find the content between the first and second ~~~ delimiters
        first_tilde = text.find('~~~')
        debug(f" first_tilde found at: {first_tilde}")
        if first_tilde == -1:
            return None

        # Search for EVIDENCE_STATUS: after the first ~~~
        status_marker = text.find('EVIDENCE_STATUS:', first_tilde)
        debug(f" EVIDENCE_STATUS: found at: {status_marker}")
        if status_marker == -1:
            return None

        # Find the closing ~~~ (after the marker)
        close_tilde = text.find('~~~', status_marker)
        debug(f" closing ~~~ found at: {close_tilde}")
        if close_tilde == -1:
            # No closing delimiter; take everything after marker
            block_text = text[status_marker:]
        else:
            block_text = text[status_marker:close_tilde]

        debug(f" block_text ({len(block_text)} chars): {repr(block_text[:300])}")

        status: dict[str, Any] = {}
        lines = block_text.strip().split('\n')
        debug(f" lines count: {len(lines)}")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                debug(f" line {i}: SKIPPED (empty)")
                continue

            # Strip leading bullet point if present
            content = stripped.lstrip('- ').strip()

            debug(f" line {i}: '{stripped}' -> content: '{content}'")

            if content.startswith('confidence:'):
                status['confidence'] = content.split(':', 1)[1].strip()
                debug(f"   -> parsed confidence: {status['confidence']}")
            elif content.startswith('coverage:'):
                status['coverage'] = content.split(':', 1)[1].strip()
                debug(f"   -> parsed coverage: {status['coverage']}")
            elif content.startswith('gaps:'):
                gap_text = content.split(':', 1)[1].strip()
                # Handle various formats:
                if gap_text.startswith('[') and gap_text.endswith(']'):
                    inner = gap_text[1:-1].strip()
                    if not inner:
                        status['gaps'] = []
                    else:
                        status['gaps'] = [
                            g.strip().strip('"').strip("'").strip('-')
                            for g in inner.split(',')
                            if g.strip()
                        ]
                    debug(f"   -> parsed gaps (list): {status['gaps']}")
                elif not gap_text or gap_text.lower() in ('none', 'n/a', 'no gaps', 'none significant', 'none identified'):
                    status['gaps'] = []
                    debug(f"   -> parsed gaps (empty keyword): []")
                else:
                    status['gaps'] = [gap_text]
                    debug(f"   -> parsed gaps (freeform): {status['gaps']}")
            elif content.startswith('re_retrieve:'):
                val = content.split(':', 1)[1].strip().lower()
                status['re_retrieve'] = val in ('true', 'yes', '1', 'yes please')
                debug(f"   -> parsed re_retrieve: {status['re_retrieve']} (from '{val}')")
            elif content.startswith('suggested_queries:'):
                query_text = content.split(':', 1)[1].strip()
                if not query_text or query_text.lower() in ('none', 'n/a', 'no', 'none', 'none at this time'):
                    status['suggested_queries'] = []
                elif query_text.startswith('[') and query_text.endswith(']'):
                    inner = query_text[1:-1].strip()
                    if not inner:
                        status['suggested_queries'] = []
                    else:
                        status['suggested_queries'] = [
                            q.strip().strip('"').strip("'")
                            for q in inner.split(',')
                            if q.strip()
                        ]
                else:
                    status['suggested_queries'] = [query_text.strip('"').strip("'")]
                debug(f"   -> parsed suggested_queries: {status['suggested_queries']}")
            else:
                debug(f"   -> UNMATCHED")

        debug(f" Final status dict: {status}")
        return status if status else None
    except Exception as e:
        debug(f" Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def _strip_evidence_status_block(text: str) -> str:
    """Remove a trailing EVIDENCE_STATUS block (and its ~~~ delimiters).

    P0-3: the EVIDENCE_STATUS block is machine-side only and must never
    reach user-facing text.
    """
    marker = text.find('EVIDENCE_STATUS:')
    if marker == -1:
        return text.rstrip()
    opening = text.rfind('~~~', 0, marker)
    cut = opening if opening != -1 else marker
    return text[:cut].rstrip()


"""
Verifier Agent
================================================
Verifier agent used by the orchestrator to verify the report written
by the writer agent.

It checks the content and claims in the draft against the evidence
before returning the final verified report.
"""
def verifier_agent(
    user_query: str,
    written_draft: str,
    evidence_text: str,
    verbose: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    has_doc_evidence: bool = False,
    has_web_evidence: bool = False,
    short_draft: bool = False,
) -> tuple[str, dict]:
    """
    Execute the verifier agent.

    Args:
        user_query: The user's original query
        written_draft: The draft report to verify
        evidence_text: The evidence to verify against
        verbose: Whether to print debug information
        endpoint: Optional custom endpoint URL
        api_key: Optional custom API key
        has_doc_evidence: Whether the active evidence includes document chunks
            (route signal computed by the orchestrator from state, P0-2)
        has_web_evidence: Whether the active evidence includes web results
            (route signal computed by the orchestrator from state, P0-2)
        short_draft: Whether the draft is under 1500 characters (P0-8)

    Returns:
        Tuple of (clean_text, status_dict): the verified report with the
        EVIDENCE_STATUS block stripped, and the machine-side status dict
        (confidence, coverage, gap_queries, specific_queries, re_retrieve,
        route, short_draft). The status dict is never appended to the text.
    """
    if verbose:
        info("Verifier Agent: Verifying report...")

    instructions = (
        """
        Verify the draft against the evidence and the user query, then return only the final answer.
        Start with the answer and make sure it directly answers the user's question.
        Remove anything that is supported by the evidence but does not answer the query.
        No preamble, review notes, or meta lead-ins such as "Below is", "Here is", or "Based on the cited sources".
        Keep the writing concise, natural, and complete enough to fully answer the question.
        Preserve useful supporting detail when it helps answer the question more completely.
        When the evidence supports it, keep the most important comparisons, caveats, and specific facts instead of collapsing the answer into a minimal summary.
        For judgments, comparisons, recommendations, or conclusions, state the best-supported conclusion clearly if supported.
        For short follow-ups, answer briefly and directly.
        Keep only supported statements.
        Add citations at the end of supported sentences.

        DEPTH & COVERAGE ASSESSMENT (MUST ADDRESS):
        1. COVERAGE: Given the user's query scope, is the available evidence THIN (< 10 useful chunks), MODERATE (10-20 chunks), or COMPREHENSIVE (20+ chunks)? Be honest.
        2. SCOPE GAPS: What specific sub-topics of the query are NOT addressed by the evidence? List each one explicitly.
        3. DEPTH GAPS: Are there advanced topics (e.g., attention variants, optimization techniques, theoretical properties, empirical results) that would be expected in a thorough answer but are missing from the evidence?
        4. SINGLE-SOURCE RISK: If all evidence comes from one document, note that alternative perspectives or additional sources are missing.

        If any of the above reveals gaps, thin coverage, or scope limitations, set re_retrieve to true and provide specific follow-up queries.

        Use the exact citation field from Document evidence for PDF citations.
        Use the exact title and exact URL from Web evidence for web citations.
        When citing web evidence, use Markdown links in the form [Exact Source Title](Exact URL).
        If the final answer uses web evidence, do not omit the URL citations.
        If the evidence is weak, incomplete, or not enough for a confident conclusion, say so.
        End the answer immediately after the last cited sentence.

        End your response with a structured EVIDENCE_STATUS block in this EXACT format. Use '~~~' as block delimiters:

        ~~~
        EVIDENCE_STATUS:
        - confidence: high|medium|low
        - coverage: thin|moderate|comprehensive
        - gaps: [list of specific topic gaps]
        - re_retrieve: true|false
        - suggested_queries: [list of suggested follow-up search queries if re_retrieve is true]
        ~~~
        """
    )

    # P0-8: note a short draft so the verifier judges depth accordingly.
    if short_draft:
        instructions += (
            "\n\nDRAFT LENGTH NOTE: The draft is under 1500 characters. Judge depth "
            "accordingly: a short answer may be appropriate for a narrow query, but "
            "for a broad query scope treat the short length as a depth gap and "
            "reflect it in the EVIDENCE_STATUS block."
        )

    # Pass the query, report draft, and evidence together so the verifier can
    # check the claims and return the final verified report.
    input_text = (
        f"User query: {user_query}\n\n"
        f"Report draft:\n{written_draft}\n\n"
        f"Evidence:\n{evidence_text}"
    )

    config = get_config()
    response = run_model(
        instructions=instructions,
        input_data=input_text,
        reasoning_effort=config.get_reasoning_effort("verifier"),
        max_output_tokens=config.get_max_output_tokens("verifier"),
        tools=None,
        agent_name="verifier",
        endpoint=endpoint,
        api_key=api_key,
    )
    raw_output = response.output_text or ""

    # Parse the model's EVIDENCE_STATUS block (machine-side only, P0-2/P0-3)
    parsed_status = parse_evidence_status(raw_output)

    # Parse-failure default (P0-2): do NOT force re-retrieval; just log.
    if not parsed_status:
        debug("verifier_agent: EVIDENCE_STATUS block missing or unparseable; defaulting re_retrieve=False")
        parsed_status = {
            'confidence': 'medium',
            'coverage': 'moderate',
            'gaps': [],
            're_retrieve': False,
            'suggested_queries': [],
        }

    confidence = parsed_status.get('confidence', 'medium')
    coverage = parsed_status.get('coverage', 'moderate')
    gap_queries = [g for g in parsed_status.get('gaps', []) if g]
    specific_queries = [q for q in parsed_status.get('suggested_queries', []) if q]
    re_retrieve = bool(parsed_status.get('re_retrieve', False))

    # Force-override (P0-2): ONLY when coverage is thin AND the critic listed
    # concrete gaps. No single-source / no-web / moderate-coverage forcing, and
    # no boilerplate query injection (P0-4).
    if not re_retrieve and coverage == "thin" and gap_queries:
        debug(f"verifier_agent: forcing re_retrieve=True (coverage=thin, {len(gap_queries)} gaps)")
        re_retrieve = True

    # Strip the EVIDENCE_STATUS block so it never reaches user-facing text (P0-3)
    clean_text = _strip_evidence_status_block(raw_output)

    status_dict = {
        'confidence': confidence,
        'coverage': coverage,
        'gap_queries': gap_queries,
        'gaps': gap_queries,  # alias kept for existing readers (save_report, prompt context)
        'specific_queries': specific_queries,
        're_retrieve': re_retrieve,
        'route': {'doc': bool(has_doc_evidence), 'web': bool(has_web_evidence)},
        'short_draft': bool(short_draft),
    }

    return clean_text, status_dict


"""
Structured critic (P1-4, DEEP.md §3.6 / plan B.4)
=====================================================================================
The deep pipeline's critic: evaluates the per-section draft against the
citation-keyed evidence and returns a machine-readable VerificationReport.
Unlike verifier_agent (which stays the legacy final-editor with the EVIDENCE
STATUS contract), the critic NEVER rewrites or summarizes the report — the
writer owns the final text. Only sections failing grounded/depth_ok go back
to write_section with their gap list.
"""


class PerSectionReport(BaseModel):
    """Critic verdict for one section."""

    section_id: str
    grounded: bool = Field(
        default=True,
        description="True only if every factual claim is supported by a cited [D#]/[W#] evidence item",
    )
    depth_ok: bool = Field(
        default=True,
        description="True if the section has >=300 words of substance with concrete specifics and no filler",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Concrete, fixable problems (missing citation, thin claim, filler, no specifics)",
    )
    expand_queries: List[str] = Field(
        default_factory=list,
        description="At most 2 targeted retrieval queries that would close the gaps",
    )


class VerificationReport(BaseModel):
    """Structured output of verification_critic."""

    is_supported: bool = Field(
        default=True,
        description="True if the report as a whole is supported by the evidence",
    )
    hallucinated_claims: List[str] = Field(
        default_factory=list,
        description="Claims asserted as fact but contradicted or absent from the evidence",
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Claims with no supporting evidence item (citation missing or unresolvable)",
    )
    per_section: List[PerSectionReport] = Field(default_factory=list)
    confidence_level: Literal["high", "medium", "low"] = "medium"
    re_retrieve_suggested: bool = Field(
        default=False,
        description="True only if additional retrieval would materially improve the report",
    )
    specific_queries: List[str] = Field(
        default_factory=list,
        description="Targeted queries for the re-retrieval (empty when re_retrieve_suggested is false)",
    )


CRITIC_INSTRUCTIONS = """
You are a CRITIC for a deep-research report. Do NOT rewrite, edit, or
summarize the report — evaluate it against the evidence and return only the
structured report.

You receive:
- the user query,
- the report draft (one section per ## heading; each section is labeled with
  its id in the input),
- the citation-keyed evidence: every item carries a global key like [D1] or
  [W2]. A claim is grounded only when it is supported by an item whose key
  the section actually cites.

Per section (ids provided in the input), judge:
- grounded: is every factual claim supported by a [D#]/[W#]-keyed item that
  the section cites? Any claim not supported by a cited item is a problem —
  list it in gaps.
- depth_ok: at least 300 words of substance, concrete specifics (named works,
  numbers, dates), no filler ("it is important to note that…"), no heading
  restatement, no hedging-only closer.
- gaps: concrete, fixable problems (one short phrase each). Empty when the
  section passes.
- expand_queries: at most 2 targeted retrieval queries that would close the
  gaps (empty when grounded and depth_ok).

Overall:
- is_supported: true only if the report as a whole stands on the evidence.
- hallucinated_claims: claims asserted as fact that the evidence contradicts
  or contains no trace of (quote the claim, keep it short).
- unsupported_claims: claims with no supporting evidence (quote, keep short).
- confidence_level: high (thorough, well-cited, no material gaps), medium
  (solid with minor gaps), low (material gaps or ungrounded claims).
- re_retrieve_suggested: true only if more evidence would materially improve
  the report; specific_queries: the targeted queries (empty otherwise).

Return only the structured report; no prose.
"""


def _neutral_critic_report(section_ids: List[str]) -> dict:
    """Final fallback: a neutral pass for every section (P1-4 §7.2)."""
    return {
        "is_supported": True,
        "hallucinated_claims": [],
        "unsupported_claims": [],
        "per_section": [
            {
                "section_id": sid,
                "grounded": True,
                "depth_ok": True,
                "gaps": [],
                "expand_queries": [],
            }
            for sid in section_ids
        ],
        "confidence_level": "medium",
        "re_retrieve_suggested": False,
        "specific_queries": [],
        "source": "fallback",
    }


def verification_critic(
    user_query: str,
    report_text: str,
    evidence_text: str,
    section_ids: List[str],
    verbose: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run the structured critic over the per-section draft (P1-4).

    ONE run_model call with text_format=VerificationReport (agent "verifier"
    config). Parse chain: structured output -> JSON-in-text extraction ->
    neutral pass (warning logged). ALWAYS returns a valid report dict shaped
    like VerificationReport.model_dump() plus a "source" tag; never raises.

    per_section is normalized to cover exactly the provided section_ids in
    order (missing sections get a neutral pass; unknown ids are dropped; each
    section keeps at most 2 expand_queries).
    """
    section_ids = [sid for sid in (section_ids or []) if sid and sid.strip()]
    if verbose:
        info(f"Critic: evaluating {len(section_ids)} section(s)...")

    id_lines = "\n".join(f"- section id: {sid}" for sid in section_ids)
    input_text = (
        f"User query: {user_query}\n\n"
        f"Report sections to evaluate (ids):\n{id_lines}\n\n"
        f"Report draft:\n{report_text}\n\n"
        f"Citation-keyed evidence (a claim is grounded only when supported by an item whose key the section cites):\n{evidence_text}"
    )

    report: Optional[VerificationReport] = None
    response: Any = None
    source = "structured"
    try:
        config = get_config()
        response = run_model(
            instructions=CRITIC_INSTRUCTIONS,
            input_data=input_text,
            text_format=VerificationReport,
            reasoning_effort=config.get_reasoning_effort("verifier"),
            max_output_tokens=config.get_max_output_tokens("verifier"),
            tools=None,
            agent_name="verifier",
            endpoint=endpoint,
            api_key=api_key,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            report = VerificationReport.model_validate(parsed)
        else:
            raise ValueError("no structured output on response")
    except Exception as exc:
        source = "json-fallback"
        try:
            raw_text = getattr(response, "output_text", None) or ""
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end > start:
                report = VerificationReport.model_validate(
                    json.loads(raw_text[start : end + 1])
                )
            else:
                raise ValueError("no JSON object in text")
        except Exception as exc2:
            logger.warning(
                f"verification_critic: report could not be parsed "
                f"(structured: {exc}; text: {exc2}); using neutral pass"
            )
            return _neutral_critic_report(section_ids)

    data = report.model_dump()

    # Normalize per_section to exactly the provided ids, in order.
    by_id = {}
    for entry in data.get("per_section") or []:
        if isinstance(entry, dict) and entry.get("section_id"):
            by_id[str(entry["section_id"]).strip()] = entry
    normalized: List[dict] = []
    for sid in section_ids:
        entry = by_id.get(sid)
        if entry is None:
            entry = {
                "section_id": sid,
                "grounded": True,
                "depth_ok": True,
                "gaps": [],
                "expand_queries": [],
            }
        normalized.append(
            {
                "section_id": sid,
                "grounded": bool(entry.get("grounded", True)),
                "depth_ok": bool(entry.get("depth_ok", True)),
                "gaps": [g for g in (entry.get("gaps") or []) if g and g.strip()],
                "expand_queries": [q for q in (entry.get("expand_queries") or []) if q and q.strip()][
                    :2
                ],
            }
        )
    data["per_section"] = normalized
    data["source"] = source
    return data