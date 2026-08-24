import json
import os
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from memory import infer_route_used
from qdrant_vector_database import get_indexed_document_catalog, similarity_search
from .model_runner import run_model
from tavily import TavilyClient
from utils.config import get_config

"""
Retriever Agent 
=====================================================================================
It used by the Orchestrator for information retrieval. 
It uses two tools: document retrieval and web search.

Given a query, it decides whether to retrieve evidence from the indexed PDFs,
search the web for up-to-date information or use both tools.

If local document evidence is weak or missing, it can fall back to web search
to gather broader context.
"""

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) 

# Structured output returned by the retriever agent.
# extra="allow" so goal mode (P1-2) can attach the last SufficiencyReport
# under "sufficiency" without changing the standard-mode serialized shape
# (the extra field is only present when goal mode actually sets it).
class ResearchEvidencePack(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str
    route_used: Literal["documents", "web", "both", "none"]
    summary: str
    document_evidence: Optional[Dict[str, Any]] = None
    web_evidence: Optional[Dict[str, Any]] = None


# Retrieve relevant document evidence for a query.
# Maximum characters to retain per retrieved document chunk.
# PDF page content can easily be 2000-8000+ chars; we cap it to keep
# the LLM context window manageable when accumulated across iterations.
_CHUNK_CONTENT_MAX_CHARS = 800


def retrieve_document(
    query: str,
    per_doc_topk: int = 8,
    score_threshold: Optional[float] = 0.2,
) -> Dict[str, Any]:
    try:
        results = similarity_search(
            query=query,
            per_doc_topk=per_doc_topk,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        return {
            "query": query,
            "summary": f"Document retrieval failed: {type(exc).__name__}",
            "chunks": [],
        }

    def _truncate(text: str) -> str:
        if not text:
            return text
        return text[:_CHUNK_CONTENT_MAX_CHARS]

    def _as_page_number(value: Any) -> Optional[int]:
        # Defensive parse: legacy/missing payloads can lack a usable page.
        # None means "unknown" so downstream can omit "p. N" instead of crashing.
        try:
            page = int(value)
        except (TypeError, ValueError):
            return None
        return page if page > 0 else None

    chunks = []
    for item in results:
        document_name = item.get("document_name") or ""
        page_number = _as_page_number(item.get("page_number"))
        citation = item.get("citation") or ""
        if not citation and document_name and page_number is not None:
            # Derive in the same shape the chunker writes at ingest time.
            citation = f"[{document_name} p.{page_number}]"
        chunks.append(
            {
                "document_name": document_name,
                # Derived: fall back to the file name when the extracted
                # title is empty so the writer always has a title to cite.
                "document_title": item.get("document_title") or document_name,
                "page_number": page_number,
                "chunk_id": item.get("chunk_id") or "",
                "citation": citation,
                "content": _truncate(item.get("content") or ""),
                "score": float(item.get("score") or 0.0),
            }
        )

    return {
        "query": query,
        "summary": (
            "Retrieved relevant evidence from the uploaded PDFs."
            if results else
            "No sufficiently relevant evidence was found in the uploaded PDFs."
        ),
        "chunks": chunks,
    }

# Search the web for supporting context.
# Maximum characters to retain per web search result content.
# Tavily can return 3-15KB of raw HTML/Markdown per result; we cap it
# so accumulated results across iterations stay within the context window.
_WEB_RESULT_CONTENT_MAX_CHARS = 600


def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    if tavily is None:
        return {"query": query, "results": []}

    try:
        result = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=num_results,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )
        results = result.get("results", [])

        def _whitelist_result(r: Dict[str, Any]) -> Dict[str, Any]:
            """Keep only the citation-required fields; drop all Tavily extras."""
            title = r.get("title")
            url = r.get("url")
            content = r.get("content")
            try:
                score = float(r.get("score"))
            except (TypeError, ValueError):
                score = 0.0
            kept: Dict[str, Any] = {
                "title": title if isinstance(title, str) else "",
                "url": url if isinstance(url, str) else "",
                "content": (
                    content[:_WEB_RESULT_CONTENT_MAX_CHARS]
                    if isinstance(content, str)
                    else ""
                ),
                "score": score,
            }
            published = r.get("published_date")
            if isinstance(published, str) and published:
                kept["published_date"] = published
            return kept

        return {"query": query, "results": [_whitelist_result(r) for r in results]}
    except Exception:
        return {"query": query, "results": []}


# ---------------------------------------------------------------------------
# Goal-driven per-sub-question investigator (P1-2)
# ---------------------------------------------------------------------------

# Keep at most this many follow-up queries per sufficiency round so a single
# sub-question cannot fan out into unbounded retrieval.
_MAX_FOLLOW_UP_QUERIES = 2


def _apply_budget(
    pack: Dict[str, Any], budget_doc: int, budget_web: int
) -> Dict[str, Any]:
    """
    Pure helper: return a copy of the evidence pack with at most budget_doc
    doc chunks and budget_web web results, kept by score descending.
    Never mutates the input; tolerates missing/empty evidence sections.
    """
    out = dict(pack or {})
    doc_budget = max(int(budget_doc), 0)
    web_budget = max(int(budget_web), 0)

    doc_evidence = out.get("document_evidence")
    if isinstance(doc_evidence, dict) and doc_evidence.get("chunks"):
        doc_evidence = dict(doc_evidence)
        doc_evidence["chunks"] = sorted(
            doc_evidence["chunks"],
            key=lambda c: float(c.get("score") or 0.0),
            reverse=True,
        )[:doc_budget]
        out["document_evidence"] = doc_evidence

    web_evidence = out.get("web_evidence")
    if isinstance(web_evidence, dict) and web_evidence.get("results"):
        web_evidence = dict(web_evidence)
        web_evidence["results"] = sorted(
            web_evidence["results"],
            key=lambda r: float(r.get("score") or 0.0),
            reverse=True,
        )[:web_budget]
        out["web_evidence"] = web_evidence

    return out


class SufficiencyReport(BaseModel):
    """Structured verdict on whether collected evidence covers the goal."""

    is_sufficient: bool = False
    missing_aspects: List[str] = Field(default_factory=list)
    follow_up_queries: List[str] = Field(default_factory=list)


SUFFICIENCY_INSTRUCTIONS = """
You are the sufficiency evaluator for a single research sub-goal.

You receive a research goal and a compact digest of the evidence collected so
far (document chunks and web results, truncated to their most informative
prefix).

Decide whether the collected evidence is sufficient to write a well-grounded
answer to the goal.

Rules:
- is_sufficient: true only if the evidence directly covers the goal's core
  aspects; do not reward quantity over relevance.
- missing_aspects: the specific aspects of the goal still missing or
  under-covered (empty list when sufficient).
- follow_up_queries: at most 2 self-contained search queries that target the
  missing aspects (empty list when sufficient). Each must be a concrete
  retrieval query, not a meta-instruction.
- Return only the structured report; no prose.
"""


def _build_sufficiency_input(
    research_goal: str,
    document_evidence: Optional[Dict[str, Any]],
    web_evidence: Optional[Dict[str, Any]],
) -> str:
    """Compact digest of what is collected so far, for the sufficiency call."""
    doc_chunks = (document_evidence or {}).get("chunks") or []
    web_results = (web_evidence or {}).get("results") or []

    doc_lines = []
    for index, chunk in enumerate(doc_chunks, start=1):
        title = (chunk.get("document_title") or "").strip()
        name = (chunk.get("document_name") or "").strip()
        page = chunk.get("page_number")
        location = name or "unknown.pdf"
        if page is not None:
            location = f"{location} p. {page}"
        header = f"{title} — {location}" if title else location
        doc_lines.append(f"[D{index}] {header}: {(chunk.get('content') or '')[:300]}")

    web_lines = []
    for index, result in enumerate(web_results, start=1):
        title = (result.get("title") or "").strip()
        url = (result.get("url") or "").strip()
        header = f"{title} ({url})" if title else url
        web_lines.append(f"[W{index}] {header}: {(result.get('content') or '')[:200]}")

    return (
        f"Research goal: {research_goal}\n\n"
        f"Evidence collected so far — document chunks ({len(doc_chunks)}):\n"
        f"{chr(10).join(doc_lines) if doc_lines else 'none yet'}\n\n"
        f"Web results ({len(web_results)}):\n"
        f"{chr(10).join(web_lines) if web_lines else 'none yet'}"
    )


def _evaluate_sufficiency(
    research_goal: str,
    document_evidence: Optional[Dict[str, Any]],
    web_evidence: Optional[Dict[str, Any]],
    *,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One structured run_model call judging evidence sufficiency for the goal.
    Parse failure is treated as insufficient with no follow-ups (the caller
    then stops: there is no target for a targeted re-query).
    """
    config = get_config()
    input_text = _build_sufficiency_input(research_goal, document_evidence, web_evidence)
    try:
        response = run_model(
            instructions=SUFFICIENCY_INSTRUCTIONS,
            input_data=input_text,
            text_format=SufficiencyReport,
            reasoning_effort=config.get_reasoning_effort("sufficiency"),
            max_output_tokens=config.get_max_output_tokens("sufficiency"),
            agent_name="sufficiency",
            endpoint=endpoint,
            api_key=api_key,
        )
    except Exception as exc:
        print(f"[RETRIEVER] Sufficiency evaluation failed: {type(exc).__name__}: {exc}")
        return {
            "is_sufficient": False,
            "missing_aspects": [],
            "follow_up_queries": [],
            "source": "fallback",
        }

    report: Optional[SufficiencyReport] = None
    source = "structured"
    try:
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            report = SufficiencyReport.model_validate(parsed)
        else:
            raise ValueError("no structured output on response")
    except Exception as exc:
        source = "json-fallback"
        try:
            raw_text = getattr(response, "output_text", None) or ""
            report = SufficiencyReport.model_validate(_extract_plan_json(raw_text))
        except Exception as exc2:
            print(
                f"[RETRIEVER] Sufficiency report could not be parsed "
                f"(structured: {exc}; text: {exc2}); "
                "treating as insufficient and stopping."
            )
            return {
                "is_sufficient": False,
                "missing_aspects": [],
                "follow_up_queries": [],
                "source": "fallback",
            }

    data = report.model_dump()
    data["source"] = source
    # Never let a malformed report fan out beyond the cap.
    data["follow_up_queries"] = [q for q in data["follow_up_queries"] if q and q.strip()][
        :_MAX_FOLLOW_UP_QUERIES
    ]
    return data


def _extract_plan_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object embedded in raw text."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return json.loads(text[start : end + 1])


def _merge_doc_evidence(
    document_evidence: Optional[Dict[str, Any]],
    function_response: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], int]:
    """Merge retrieved doc chunks using the legacy chunk_id dedup."""
    chunks = function_response.get("chunks")
    if not chunks:
        return document_evidence, 0
    if document_evidence is None:
        return dict(function_response), len(chunks)
    existing_ids = {
        c.get("chunk_id", "") for c in document_evidence.get("chunks", [])
    }
    new_chunks = [c for c in chunks if c.get("chunk_id", "") not in existing_ids]
    if new_chunks:
        document_evidence = dict(document_evidence)
        document_evidence["chunks"] = (
            list(document_evidence.get("chunks", [])) + new_chunks
        )
        document_evidence["summary"] = (
            f"Accumulated {len(document_evidence['chunks'])} total chunks "
            f"across goal-driven retrieval rounds."
        )
    return document_evidence, len(new_chunks)


def _merge_web_evidence(
    web_evidence: Optional[Dict[str, Any]],
    function_response: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], int]:
    """Merge web results using the legacy URL dedup."""
    results = function_response.get("results")
    if not results:
        return web_evidence, 0
    if web_evidence is None:
        return dict(function_response), len(results)
    existing_urls = {r.get("url", "") for r in web_evidence.get("results", [])}
    new_results = [r for r in results if r.get("url", "") not in existing_urls]
    if new_results:
        web_evidence = dict(web_evidence)
        web_evidence["results"] = list(web_evidence.get("results", [])) + new_results
        web_evidence["summary"] = (
            f"Accumulated {len(web_evidence['results'])} total web results "
            f"across goal-driven retrieval rounds."
        )
    return web_evidence, len(new_results)


def _normalize_routes(routes: Optional[List[str]]) -> set:
    """Normalize goal-mode route selection. None = both routes (current
    behavior); "doc"/"web" (case-insensitive, legacy alias "documents")
    restrict which retrieval routes run."""
    if routes is None:
        return {"doc", "web"}
    normalized = set()
    for route in routes:
        route = (route or "").strip().lower()
        if route in ("doc", "docs", "documents", "document"):
            normalized.add("doc")
        elif route in ("web",):
            normalized.add("web")
    return normalized


def _goal_driven_retrieval(
    *,
    user_query: str,
    research_goal: str,
    max_rounds: int,
    budget_doc: int,
    budget_web: int,
    routes: Optional[List[str]] = None,
    verbose: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ResearchEvidencePack:
    """
    Goal-driven investigator loop (P1-2). Each round retrieves for the current
    query set (round 1: the goal itself; later rounds: the previous round's
    sufficiency follow-up queries, max 2), merges with the legacy dedup, then
    (unless this is the final round) asks ONE structured sufficiency call.
    Stops on: sufficiency, round cap, unparseable sufficiency, or 0 new items
    after dedup (diminishing returns). The final pack is truncated to the
    per-sub-question budget by _apply_budget.

    routes: None = both routes (default); ["doc"] / ["web"] restrict which
    retrieval routes run (e.g. web-only mode, P1-6, or a sub-question the
    decomposer marked as doc-only).
    """
    config = get_config()
    goal = research_goal.strip()
    allowed_routes = _normalize_routes(routes)
    document_evidence: Optional[Dict[str, Any]] = None
    web_evidence: Optional[Dict[str, Any]] = None
    sufficiency: Optional[Dict[str, Any]] = None
    rounds_total = max(int(max_rounds), 1)
    rounds_used = 0
    round_queries = [goal]

    for round_num in range(1, rounds_total + 1):
        rounds_used = round_num
        new_items = 0
        for query in round_queries:
            query = query.strip()
            if not query:
                continue
            if verbose:
                print(f"[INVESTIGATOR] Round {round_num}: retrieving '{query}'")
            if "doc" in allowed_routes:
                doc_response = retrieve_document(query, per_doc_topk=8)
                document_evidence, new_doc = _merge_doc_evidence(document_evidence, doc_response)
            else:
                new_doc = 0
            if "web" in allowed_routes:
                web_response = web_search(query)
                web_evidence, new_web = _merge_web_evidence(web_evidence, web_response)
            else:
                new_web = 0
            new_items += new_doc + new_web

        if round_num == rounds_total:
            # Final round: stop after merge, no sufficiency call.
            break

        report = _evaluate_sufficiency(
            goal,
            document_evidence,
            web_evidence,
            endpoint=endpoint,
            api_key=api_key,
        )
        sufficiency = report
        if verbose:
            print(
                f"[INVESTIGATOR] Round {round_num} sufficiency: "
                f"is_sufficient={report['is_sufficient']} "
                f"missing={report['missing_aspects']} "
                f"follow_ups={report['follow_up_queries']}"
            )
        if report["is_sufficient"]:
            break
        if new_items == 0:
            # Diminishing returns: nothing new to build a next round on.
            if verbose:
                print(f"[INVESTIGATOR] Round {round_num}: 0 new items after dedup, stopping.")
            break
        if not report["follow_up_queries"]:
            # No target for a targeted re-query.
            break
        round_queries = report["follow_up_queries"][:_MAX_FOLLOW_UP_QUERIES]

    budgeted = _apply_budget(
        {
            "document_evidence": document_evidence,
            "web_evidence": web_evidence,
        },
        budget_doc,
        budget_web,
    )
    document_evidence = budgeted["document_evidence"]
    web_evidence = budgeted["web_evidence"]

    document_chunks = (document_evidence or {}).get("chunks", [])
    web_results = (web_evidence or {}).get("results", [])
    route_used = infer_route_used(document_chunks or [], web_results or [])

    verdict = "not evaluated" if sufficiency is None else (
        "sufficient" if sufficiency["is_sufficient"] else "insufficient"
    )
    summary = (
        f"Goal-driven retrieval for '{goal}': {rounds_used} round(s), "
        f"{len(document_chunks)} doc chunk(s), {len(web_results)} web result(s); "
        f"last sufficiency verdict: {verdict}."
    )

    pack = ResearchEvidencePack(
        query=user_query,
        route_used=route_used,
        summary=summary,
        document_evidence=document_evidence if document_evidence and document_chunks else {},
        web_evidence=web_evidence if web_evidence and web_results else {},
    )
    if sufficiency is not None:
        pack.sufficiency = sufficiency
    return pack


# Guides the retriever agent on how to interact with the available tools
# (document retrieval and web search).
RETRIEVER_TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "retrieve_document",
        "description": (
            """Search the indexed PDF corpus and return the most relevant chunks with
            document names, titles, page numbers, exact citation strings, and
            scores. Preserve the returned document names, titles, page numbers, and
            citation strings because they are needed downstream for accurate PDF
            citations in the final answer. Prefer this when the query is plausibly
            covered by the uploaded PDFs, is closely related to the indexed document
            titles or topics, when the user explicitly asks about the uploaded PDFs,
            or when document-grounded evidence is needed."""
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Rewrite the user's request into a self-contained search query "
                        "for the indexed PDFs. Include omitted subject details from "
                        "follow-up context when needed."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "web_search",
        "description": (
            """Search the web for recent, changing, external, or clearly non-PDF
            information and return concise results with exact source titles, exact
            URLs, and source metadata. Preserve the returned titles and URLs because
            they are needed downstream for accurate web citations in the final
            answer. Prefer this when the query does not match the indexed document
            titles or topics, when external or current evidence is needed, or after
            document retrieval is empty or insufficient."""
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Rewrite the user's request into a self-contained web search "
                        "query. Include omitted subject details from follow-up context "
                        "when needed."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

# Instructions guiding the LLM's behavior in retriever agent
RETRIEVER_INSTRUCTIONS = """
You are the retriever worker for a research assistant.

Indexed document titles and topic hints are provided in the input.

Rules:
- This worker is only called for evidence requests, so use at least one retrieval tool unless the query is empty or malformed.
- Use the indexed document titles/topics to decide whether the active PDFs are likely relevant.
- If the query does not appear related to the indexed document topics and the user is not explicitly asking about the PDFs, go straight to web_search.
- If retrieve_document returns no relevant chunks and the user is not explicitly asking about the PDFs, call web_search before finishing.
- If retrieve_document returns relevant chunks, do not call web_search unless newer or external evidence is still needed.
- Use the last user query only when the current query is a follow-up that depends on it.
- Evidence accumulates across rounds. Do NOT repeat queries that returned results in a prior round.
- Vary your queries across rounds to gather complementary, non-redundant evidence. Search for related angles, specific sub-topics, and different aspects of the question.
- After tool use, return a short retrieval summary, not a user-facing answer.
"""

def retriever_agent(
    user_query: str,
    *,
    last_user_query: str = "",
    verbose: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    research_goal: str = "",
    max_rounds: int = 3,
    budget_doc: int = 10,
    budget_web: int = 5,
    routes: Optional[List[str]] = None,
) -> ResearchEvidencePack:
    """
    Execute the retriever agent.
    
    Args:
        user_query: The user's query to retrieve evidence for
        last_user_query: The previous user query for context
        verbose: Whether to print debug information
        endpoint: Optional custom endpoint URL
        api_key: Optional custom API key
        research_goal: P1-2 goal-driven mode. When non-empty, the legacy
            LLM tool-call loop is replaced by a per-goal investigator loop
            (retrieve -> structured sufficiency -> targeted re-query) capped
            at max_rounds rounds and the per-sub-question evidence budget.
            When empty (default), behavior is EXACTLY the legacy standard
            mode.
        max_rounds: Max investigator rounds in goal mode (default 3).
        budget_doc: Max doc chunks kept per goal (default 10, top-scored).
        budget_web: Max web results kept per goal (default 5, top-scored).
        routes: Goal mode only. None = both routes (default, current
            behavior); ["doc"] / ["web"] restrict which retrieval routes run.
            Ignored in standard mode (legacy path untouched).
        
    Returns:
        ResearchEvidencePack with the retrieved evidence (plus a "sufficiency"
        extra field with the last SufficiencyReport dict in goal mode).
    """
    # P1-2 guard: goal-driven mode. An empty research_goal falls through to
    # the unchanged legacy path below (standard mode behavior is identical).
    if research_goal:
        return _goal_driven_retrieval(
            user_query=user_query,
            research_goal=research_goal,
            max_rounds=max_rounds,
            budget_doc=budget_doc,
            budget_web=budget_web,
            routes=routes,
            verbose=verbose,
            endpoint=endpoint,
            api_key=api_key,
        )

    previous_response_id: Optional[str] = None

    # Expose indexed document titles so the model can judge whether the PDFs
    # are relevant before choosing a retrieval tool.
    document_catalog = get_indexed_document_catalog()
    indexed_documents_text = "\n".join(
        f"- {item['title']} (file: {item['file_name']})"
        for item in document_catalog
    ) or "- None"
    pending_input: Any = (
        f"Current user query: {user_query.strip()}\n"
        f"Last user query: {last_user_query.strip() or 'None'}\n"
        f"Indexed document titles and topic hints:\n{indexed_documents_text}"
    )
    document_evidence: Optional[Dict[str, Any]] = None
    web_evidence: Optional[Dict[str, Any]] = None
    summary = ""

    config = get_config()
    # Allow a few tool-call rounds before finalizing the retrieval summary.
    for _ in range(4):
        response = run_model(
            instructions=RETRIEVER_INSTRUCTIONS,
            input_data=pending_input,
            tools=RETRIEVER_TOOL_SCHEMAS,
            previous_response_id=previous_response_id,
            reasoning_effort=config.get_reasoning_effort("retriever"),
            max_output_tokens=config.get_max_output_tokens("retriever"),
            agent_name="retriever",
            endpoint=endpoint,
            api_key=api_key,
        )
        previous_response_id = response.id

        tool_results = []
        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            # If the model stops calling tools, treat its final text as the
            # retriever-facing summary for this turn.
            summary = (response.output_text or "").strip()
            break

        for call in function_calls:
            if call.name == "retrieve_document":
                try:
                    args = json.loads(call.arguments)
                    query = args.get("query", "").strip()
                except (json.JSONDecodeError, TypeError, AttributeError):
                    query = ""

                if not query:
                    query = last_user_query.strip()
                if not query:
                    query = user_query.strip()

                if not query:
                    print("[RETRIEVER] Skipping retrieve_document: no query found in arguments")
                    continue

                if verbose:
                    print("[Retriever Agent] Retrieving document evidence...")
                function_response = retrieve_document(query, per_doc_topk=8)
                if function_response.get("chunks"):
                    if document_evidence is None:
                        document_evidence = function_response
                    else:
                        # Accumulate: merge chunks, deduplicating by chunk_id
                        existing_chunks = {
                            c["chunk_id"] for c in document_evidence.get("chunks", [])
                        }
                        new_chunks = [
                            c
                            for c in function_response["chunks"]
                            if c["chunk_id"] not in existing_chunks
                        ]
                        document_evidence["chunks"] = (
                            document_evidence.get("chunks", []) + new_chunks
                        )
                        document_evidence["summary"] = (
                            f"Accumulated {len(document_evidence['chunks'])} total chunks "
                            f"across retrieval rounds."
                        )
            elif call.name == "web_search":
                try:
                    args = json.loads(call.arguments)
                    query = args.get("query", "").strip()
                except (json.JSONDecodeError, TypeError, AttributeError):
                    query = ""

                if not query:
                    query = last_user_query.strip()
                if not query:
                    query = user_query.strip()

                if not query:
                    print("[RETRIEVER] Skipping web_search: no query found in arguments")
                    continue

                if verbose:
                    print("[Retriever Agent] Searching the web...")
                function_response = web_search(query)
                if function_response.get("results"):
                    if web_evidence is None:
                        web_evidence = function_response
                    else:
                        # Accumulate: merge results, deduplicating by URL
                        existing_urls = {
                            r.get("url", "")
                            for r in web_evidence.get("results", [])
                        }
                        new_results = [
                            r
                            for r in function_response["results"]
                            if r.get("url", "") not in existing_urls
                        ]
                        web_evidence["results"] = (
                            web_evidence.get("results", []) + new_results
                        )
                        web_evidence["summary"] = (
                            f"Accumulated {len(web_evidence['results'])} total web "
                            f"results across retrieval rounds."
                        )

            tool_results.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(function_response),
                }
            )

        pending_input = tool_results

    document_chunks = document_evidence.get("chunks") if document_evidence else []
    web_results = web_evidence.get("results") if web_evidence else []
    route_used = infer_route_used(document_chunks or [], web_results or [])

    # Return the route used, a short summary and any collected evidence.
    return ResearchEvidencePack(
        query=user_query,
        route_used=route_used,
        summary=summary,
        document_evidence=document_evidence if document_evidence and document_evidence.get("chunks") else {},
        web_evidence=web_evidence if web_evidence and web_evidence.get("results") else {},
    )