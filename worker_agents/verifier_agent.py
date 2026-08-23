from .model_runner import run_model
from typing import Any, Optional
from memory import debug, info
from utils.config import get_config


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