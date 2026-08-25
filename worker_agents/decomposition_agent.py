import json
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .model_runner import run_model
from utils.config import get_config

logger = logging.getLogger(__name__)

"""
Decomposition Agent (P1-1)
=====================================================================================
One-shot planning step for the deep research pipeline (DEEP.md §3.2).

The orchestrator LLM is a hostile environment for planning (tool_choice=
"required" with full state re-injected every iteration), so decomposition is a
dedicated single structured call: the user query in, a ResearchPlan of 5-10
MECE sub-questions out. Simple queries short-circuit to a single sub-question.
The plan never raises: every failure path lands on a valid single-sub-question
fallback plan, tagged with its "source" for observability.
"""


class SubQuestion(BaseModel):
    """One investigable slice of the overall research query."""

    id: str = Field(
        default="",
        description='Short identifier, e.g. "sq1"',
    )
    question: str = Field(description="Self-contained, answerable question")
    angle: str = Field(description="Which aspect of the report this covers")
    expected_sources: Literal["doc", "web", "both"] = Field(
        description=(
            "doc: indexed PDFs plausibly cover it; web: they clearly do not; "
            "both: otherwise"
        )
    )
    priority: int = Field(ge=1, le=5, description="1 = most central, 5 = least")
    heading: str = Field(
        default="",
        description=(
            "Concise section heading for the report section that will answer "
            "this sub-question (max 8 words). Optional: left empty when "
            "uncertain."
        ),
    )


class ResearchPlan(BaseModel):
    """Structured output of the decomposer."""

    is_simple: bool = Field(
        default=False,
        description="True only for a single narrow factual question",
    )
    sub_questions: List[SubQuestion] = Field(default_factory=list)


DECOMPOSER_INSTRUCTIONS = """
You are the decomposition worker for a deep research pipeline.

Decompose the user's query into the sub-questions that a comprehensive research
report on the query must answer.

Rules:
- Produce 5 to 10 sub-questions that are MECE: each is self-contained and
  answerable on its own, no two overlap substantially, and together they fully
  cover a comprehensive research report on the query.
- Every sub-question must be investigable with the available retrieval tools:
  local document retrieval over the indexed PDFs (the catalog is listed in the
  input) and/or web search.
- expected_sources: "doc" ONLY when the indexed document catalog plausibly
  covers the sub-question; "web" when the local documents clearly do not cover
  it; "both" otherwise.
- priority: 1 for the sub-questions most central to the report, 5 for the
  least central.
- heading: a concise section heading (max 8 words, no trailing punctuation)
  for the report section that will answer this sub-question.
- Order sub_questions from highest to lowest priority and number them sq1,
  sq2, ... in that order.
- If the query is a single narrow factual question (not a request for a
  comprehensive report or broad coverage), set is_simple=true and return
  exactly ONE sub-question that restates the query.
- Return only the structured plan; no prose.
"""


def _fallback_plan(user_query: str, source: str = "fallback") -> Dict[str, Any]:
    """Valid single-sub-question plan used when decomposition cannot be parsed."""
    return {
        "is_simple": True,
        "sub_questions": [
            {
                "id": "sq1",
                "question": user_query,
                "angle": "original query",
                "expected_sources": "both",
                "priority": 3,
            }
        ],
        "source": source,
    }


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object embedded in raw text.

    Uses raw_decode starting at the first '{', so preamble prose and any
    trailing data after the plan object (models sometimes append more text
    or a second blob) do not break extraction. Returns None on any failure;
    never raises.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _ensure_ids(data: Dict[str, Any]) -> Dict[str, Any]:
    """Assign sq1, sq2, ... to sub-questions that arrived without an id
    (structured/JSON outputs are not guaranteed to include one)."""
    for idx, sq in enumerate(data.get("sub_questions") or []):
        if isinstance(sq, dict) and not (sq.get("id") or "").strip():
            sq["id"] = f"sq{idx + 1}"
    return data


def _parse_plan(response: Any, user_query: str) -> Dict[str, Any]:
    """Robust parse (DEEP.md §7.2 style): structured -> text JSON -> fallback."""
    # 1) Structured output from responses.parse().
    try:
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            data = _ensure_ids(ResearchPlan.model_validate(parsed).model_dump())
            data["source"] = "structured"
            return data
    except Exception as exc:
        logger.warning(f"Decomposer structured output failed to validate: {exc}")

    # 2) JSON object embedded in the raw text.
    try:
        raw_text = getattr(response, "output_text", None) or ""
        candidate = _extract_json_object(raw_text)
        if candidate is not None:
            data = _validated_plan_from_candidate(candidate)
            return data
    except Exception as exc:
        logger.warning(f"Decomposer text JSON fallback failed: {exc}")

    # 3) Final fallback: always a valid plan.
    logger.warning("Decomposer produced no usable plan; using single-sub-question fallback")
    return _fallback_plan(user_query)


def _normalize_plan_candidate(data: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort normalization of observed model shape drift BEFORE schema
    validation (lenient text-JSON path only): `sub_questions` as a dict
    keyed by id ({"sq1": {...}}) becomes a list (id taken from the key),
    `sub_question` is mapped to `question`, and missing optional-in-practice
    fields get neutral defaults (angle="n/a", priority=3,
    expected_sources="both"). Unknown shapes pass through for rejection.
    """
    if not isinstance(data, dict):
        return data
    sqs = data.get("sub_questions")
    if isinstance(sqs, dict):
        items: List[Dict[str, Any]] = []
        for idx, (key, val) in enumerate(sqs.items()):
            if not isinstance(val, dict):
                continue
            item = dict(val)
            if not (item.get("id") or "").strip():
                item["id"] = str(key) or f"sq{idx + 1}"
            items.append(item)
    elif isinstance(sqs, list):
        items = [dict(item) for item in sqs if isinstance(item, dict)]
    else:
        return data
    for item in items:
        if "sub_question" in item and not (item.get("question") or "").strip():
            item["question"] = item["sub_question"]
        if not (item.get("angle") or "").strip():
            item["angle"] = "n/a"
        try:
            prio = int(item.get("priority"))
        except (TypeError, ValueError):
            prio = 3
        item["priority"] = max(1, min(5, prio))
        if (item.get("expected_sources") or "") not in ("doc", "web", "both"):
            item["expected_sources"] = "both"
    data = dict(data)
    data["sub_questions"] = items
    return data


def _validated_plan_from_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize, validate against ResearchPlan, tag source=json-fallback.
    Raises when the candidate cannot be made valid."""
    data = _normalize_plan_candidate(candidate)
    data = _ensure_ids(ResearchPlan.model_validate(data).model_dump())
    data["source"] = "json-fallback"
    return data


def _plain_text_plan_json(
    query: str,
    doc_lines: str,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Plain (non-structured) re-ask used when the structured call failed:
    returns the raw text so the caller can extract an embedded JSON plan.
    Ornith-style models sometimes refuse `responses.parse()` (prepending
    preamble prose), but reliably emit a JSON plan on a plain call. Returns
    None when the call itself fails or the server yields no text."""
    input_data = (
        f"User query: {query}\n\n"
        f"Indexed document catalog (the only local documents available):\n{doc_lines}\n\n"
        "Respond with ONLY a single raw JSON object — no markdown, no code "
        "fences, no commentary before or after — matching exactly this shape "
        "(fill in the values, repeat the sub_questions entry per question):\n"
        '{\n'
        '  "is_simple": false,\n'
        '  "sub_questions": [\n'
        '    {"id": "sq1", "question": "...", "angle": "...", '
        '"expected_sources": "doc", "priority": 1, "heading": "..."}\n'
        '  ]\n'
        '}'
    )
    config = get_config()
    try:
        response = run_model(
            instructions=DECOMPOSER_INSTRUCTIONS,
            input_data=input_data,
            reasoning_effort=config.get_reasoning_effort("decomposer"),
            max_output_tokens=config.get_max_output_tokens("decomposer"),
            agent_name="decomposer",
            endpoint=endpoint,
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning(f"Decomposer plain-text re-ask failed: {type(exc).__name__}: {exc}")
        return None
    return getattr(response, "output_text", None) or ""


def decompose_query(
    user_query: str,
    doc_catalog: List[Dict[str, str]],
    verbose: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decompose a user query into a research plan with ONE structured LLM call.

    Args:
        user_query: The user's query to decompose.
        doc_catalog: Indexed document catalog entries
            ({"document_name": str, "document_title": str}); may be empty.
        verbose: Print a one-line summary of the result.
        endpoint: Optional custom endpoint URL.
        api_key: Optional custom API key.

    Returns:
        Plan dict: {"is_simple": bool, "sub_questions": [
            {"id", "question", "angle", "expected_sources", "priority"}],
            "source": "structured" | "json-fallback" | "fallback"}.
        Never raises: every failure path returns a valid fallback plan.
    """
    query = (user_query or "").strip()
    if not query:
        logger.warning("decompose_query called with an empty query; using fallback plan")
        return _fallback_plan(user_query or "")

    doc_lines = "\n".join(
        f"- {item.get('document_title') or item.get('document_name') or 'Untitled'} "
        f"(file: {item.get('document_name') or 'unknown.pdf'})"
        for item in (doc_catalog or [])
    ) or "- None (no indexed documents)"

    input_data = (
        f"User query: {query}\n\n"
        f"Indexed document catalog (the only local documents available):\n{doc_lines}"
    )

    config = get_config()
    plan: Optional[Dict[str, Any]] = None
    try:
        response = run_model(
            instructions=DECOMPOSER_INSTRUCTIONS,
            input_data=input_data,
            text_format=ResearchPlan,
            reasoning_effort=config.get_reasoning_effort("decomposer"),
            max_output_tokens=config.get_max_output_tokens("decomposer"),
            agent_name="decomposer",
            endpoint=endpoint,
            api_key=api_key,
        )
        plan = _parse_plan(response, query)
    except Exception as exc:
        logger.warning(
            f"Decomposer LLM call failed ({type(exc).__name__}: {exc})"
        )
        # The structured call raised (e.g. a model that prepends preamble
        # prose to its JSON breaks responses.parse()), so the text-JSON
        # fallback inside _parse_plan never got the raw output. Re-ask once
        # with a plain call, then try the same embedded-JSON extraction on
        # that text.
        raw_text = _plain_text_plan_json(query, doc_lines, endpoint, api_key)
        if raw_text:
            candidate = _extract_json_object(raw_text)
            if candidate is not None:
                try:
                    plan = _validated_plan_from_candidate(candidate)
                except Exception as exc2:
                    logger.warning(
                        f"Decomposer plain-text JSON failed to validate: {exc2}"
                    )
        if plan is None:
            plan = _fallback_plan(query)

    if verbose:
        print(
            f"[DECOMPOSER] source={plan.get('source')} "
            f"is_simple={plan.get('is_simple')} "
            f"sub_questions={len(plan.get('sub_questions', []))}"
        )
    return plan
