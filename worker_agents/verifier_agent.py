from .model_runner import run_model
from typing import Any, Optional


def parse_evidence_status(text: str) -> Optional[dict]:
    """Parse the EVIDENCE_STATUS block from verifier output using delimiter-based extraction."""
    if not text:
        return None

    try:
        # Find the content between the first and second ~~~ delimiters
        first_tilde = text.find('~~~')
        print(f"[PARSER DEBUG] first_tilde found at: {first_tilde}")
        if first_tilde == -1:
            return None

        # Search for EVIDENCE_STATUS: after the first ~~~
        status_marker = text.find('EVIDENCE_STATUS:', first_tilde)
        print(f"[PARSER DEBUG] EVIDENCE_STATUS: found at: {status_marker}")
        if status_marker == -1:
            return None

        # Find the closing ~~~ (after the marker)
        close_tilde = text.find('~~~', status_marker)
        print(f"[PARSER DEBUG] closing ~~~ found at: {close_tilde}")
        if close_tilde == -1:
            # No closing delimiter; take everything after marker
            block_text = text[status_marker:]
        else:
            block_text = text[status_marker:close_tilde]

        print(f"[PARSER DEBUG] block_text ({len(block_text)} chars): {repr(block_text[:300])}")

        status: dict[str, Any] = {}
        lines = block_text.strip().split('\n')
        print(f"[PARSER DEBUG] lines count: {len(lines)}")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                print(f"[PARSER DEBUG] line {i}: SKIPPED (empty)")
                continue

            # Strip leading bullet point if present
            content = stripped.lstrip('- ').strip()

            print(f"[PARSER DEBUG] line {i}: '{stripped}' -> content: '{content}'")

            if content.startswith('confidence:'):
                status['confidence'] = content.split(':', 1)[1].strip()
                print(f"[PARSER DEBUG]   -> parsed confidence: {status['confidence']}")
            elif content.startswith('coverage:'):
                status['coverage'] = content.split(':', 1)[1].strip()
                print(f"[PARSER DEBUG]   -> parsed coverage: {status['coverage']}")
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
                    print(f"[PARSER DEBUG]   -> parsed gaps (list): {status['gaps']}")
                elif not gap_text or gap_text.lower() in ('none', 'n/a', 'no gaps', 'none significant', 'none identified'):
                    status['gaps'] = []
                    print(f"[PARSER DEBUG]   -> parsed gaps (empty keyword): []")
                else:
                    status['gaps'] = [gap_text]
                    print(f"[PARSER DEBUG]   -> parsed gaps (freeform): {status['gaps']}")
            elif content.startswith('re_retrieve:'):
                val = content.split(':', 1)[1].strip().lower()
                status['re_retrieve'] = val in ('true', 'yes', '1', 'yes please')
                print(f"[PARSER DEBUG]   -> parsed re_retrieve: {status['re_retrieve']} (from '{val}')")
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
                print(f"[PARSER DEBUG]   -> parsed suggested_queries: {status['suggested_queries']}")
            else:
                print(f"[PARSER DEBUG]   -> UNMATCHED")

        print(f"[PARSER DEBUG] Final status dict: {status}")
        return status if status else None
    except Exception as e:
        print(f"[PARSER DEBUG] Exception: {e}")
        import traceback
        traceback.print_exc()
        return None

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
) -> str:
    """
    Execute the verifier agent.
    
    Args:
        user_query: The user's original query
        written_draft: The draft report to verify
        evidence_text: The evidence to verify against
        verbose: Whether to print debug information
        endpoint: Optional custom endpoint URL
        api_key: Optional custom API key
        
    Returns:
        The verified report as a string
    """
    if verbose:
        print("[Verifier Agent] Verifying report...")

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

    # Pass the query, report draft, and evidence together so the verifier can
    # check the claims and return the final verified report.
    input_text = (
        f"User query: {user_query}\n\n"
        f"Report draft:\n{written_draft}\n\n"
        f"Evidence:\n{evidence_text}"
    )

    response = run_model(
        instructions=instructions,
        input_data=input_text,
        reasoning_effort="low",
        tools=None,
        agent_name="verifier",
        endpoint=endpoint,
        api_key=api_key,
    )
    output = response.output_text or ""

    # Strip any existing EVIDENCE_STATUS block from the raw output
    existing_block_start = output.find('EVIDENCE_STATUS:')
    if existing_block_start >= 0:
        output = output[:existing_block_start].strip()

    # Parse status from the existing block (if any)
    parsed_status = parse_evidence_status(response.output_text or "")

    # Synthesize defaults if parser failed
    if not parsed_status:
        parsed_status = {
            'confidence': 'medium',
            'coverage': 'moderate',
            'gaps': ['Coverage assessment failed due to parsing issue; recommend deeper analysis'],
            're_retrieve': True,
            'suggested_queries': ['Check if the retrieved evidence covers the full scope of the query'],
        }

    confidence = parsed_status.get('confidence', 'medium')
    coverage = parsed_status.get('coverage', 'moderate')
    gaps = parsed_status.get('gaps', [])
    re_retrieve = str(parsed_status.get('re_retrieve', False)).lower()
    queries = parsed_status.get('suggested_queries', [])

    # Conservative override: if coverage is thin/moderate but LLM didn't re-retrieve,
    # the verifier may have been overconfident
    if coverage in ('thin', 'moderate') and not re_retrieve:
        re_retrieve = True
        if not queries:
            queries = ['Search for deeper coverage: attention variants, optimization techniques, and real-world applications']

    # FORCE depth-awareness: the LLM is biased to say "comprehensive" on thin evidence.
    # Parse evidence depth from evidence_text directly.
    import re as _re

    # Count chunks by looking for unique chunk_id patterns
    chunk_ids = set()
    for match in _re.finditer(r'chunk_id[":\s]*(\S+)', evidence_text):
        chunk_ids.add(match.group(1).strip('":\n'))
    chunk_count = len(chunk_ids)

    # Count document sources
    doc_matches = _re.findall(r'\[(\S+\.pdf)', evidence_text)
    unique_sources = set(doc_matches)
    has_web = 'web_search' in evidence_text or 'web_evidence' in evidence_text

    is_single_source = len(unique_sources) <= 1
    draft_length = len(written_draft) if written_draft else 0
    is_thin_evidence = chunk_count < 15 or (chunk_count < 25 and draft_length < 2000)
    is_no_web = not has_web

    depth_assessment = []
    if is_single_source:
        depth_assessment.append("SINGLE-SOURCE: evidence from one document only")
    if is_thin_evidence:
        depth_assessment.append(f"THIN EVIDENCE: only {chunk_count} chunks retrieved (expected 15+ for comprehensive coverage)")
    if is_no_web:
        depth_assessment.append("NO_WEB_SEARCH: did not supplement with web search")

    depth_override = ""
    if depth_assessment:
        depth_override = f"\n\n[DEPTH OVERRIDE: {'; '.join(depth_assessment)}. Consider whether coverage should be rated 'thin' or 'moderate' rather than 'comprehensive'.]"

    # Conservative override: force re-retrieval when:
    # 1. Coverage is thin/moderate, OR
    # 2. The depth override detected problematic conditions (single source, thin evidence, no web)
    if (coverage in ('thin', 'moderate') or depth_override) and not re_retrieve:
        re_retrieve = True
        suggested = []
        if not queries:
            suggested.append('Search for deeper coverage: attention variants (multi-query, grouped-query, flash), optimization techniques, and real-world applications')
        if is_thin_evidence:
            suggested.append('Use web_search to supplement document gaps')
        if is_single_source:
            suggested.append('Search for additional documents or perspectives on the topic')
        queries = suggested if not queries else queries

    # Always ensure the block is appended exactly once
    block = f"""

~~~
EVIDENCE_STATUS:
- confidence: {confidence}
- coverage: {coverage}
- gaps: {gaps}
- re_retrieve: {re_retrieve}
- suggested_queries: {queries}
{depth_override}
~~~
"""
    output = output.rstrip() + block

    return output