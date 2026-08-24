"""
Deep Research Orchestrator (P1-3/P1-4, plan Section B)
=====================================================================================
The 5-stage deep-research pipeline (DEEP.md), parallel to the standard
orchestrator (which stays untouched for cheap queries):

    Query → [1] DECOMPOSER (1 structured call)
          → [2] INVESTIGATOR loop per sub-question (retrieve → sufficiency →
                re-query; ≤3 rounds, per-sub-question evidence budget)
          → [3] PER-SECTION DRAFT (1 writer call per section, own evidence
                subset only, global citation keys)
          → [4] CRITIC (structured VerificationReport) → budgeted EXPANSION
                back to [3] (≤2 revisions/section, ≤8 expansion calls, ONE
                optional re-retrieval)
          → [5] ASSEMBLY (exec summary LAST + resolved references +
                machine-side state for save_report)

A global LLM call budget (MAX_LLM_CALLS) counts every run_model call made in
the pipeline (decomposer, sufficiency, writers, critic, exec summary). When
the budget is exhausted the pipeline stops issuing LLM calls and assembles
the report from what exists (warning logged).

Standard mode (orchestrator_agent) is NOT modified.
"""

import importlib
import json
import re
import time
from typing import Any, Dict, List, Optional

from memory.helpers import (
    assign_citation_keys,
    format_references,
    infer_route_used,
    render_evidence_text,
)
from qdrant_vector_database import get_qdrant_client
from qdrant_vector_database.vector_store import COLLECTION_NAME

# importlib (not `import ... as`): worker_agents/__init__.py re-exports the
# retriever_agent/verifier_agent/writer_agent functions, which would shadow
# the same-named module attributes in the package namespace.
_decomposition_mod = importlib.import_module("worker_agents.decomposition_agent")
_retriever_mod = importlib.import_module("worker_agents.retriever_agent")
_writer_mod = importlib.import_module("worker_agents.writer_agent")
_verifier_mod = importlib.import_module("worker_agents.verifier_agent")
from worker_agents.decomposition_agent import decompose_query
from worker_agents.model_runner import run_model
from worker_agents.retriever_agent import retriever_agent
from worker_agents.verifier_agent import verification_critic
from worker_agents.writer_agent import write_section
from utils.config import get_config

# Global LLM call budget for one deep-research run (plan B / P2-3 ≈ 40).
MAX_LLM_CALLS = 40

# Revision caps (P1-4): per section and global expansion calls.
_MAX_REVISIONS_PER_SECTION = 2
_MAX_EXPANSION_CALLS = 8


class _BudgetExhausted(Exception):
    """Raised by the tracked run_model wrapper when the budget is spent."""


class _LLMBudget:
    """Counts every run_model call made inside the deep pipeline."""

    def __init__(self, limit: int):
        self.limit = max(int(limit), 1)
        self.count = 0

    @property
    def exhausted(self) -> bool:
        return self.count >= self.limit

    def can_afford(self, n: int = 1) -> bool:
        return self.count + n <= self.limit

    def charge(self, label: str, verbose: bool = False) -> None:
        if self.exhausted:
            raise _BudgetExhausted(
                f"LLM budget exhausted ({self.count}/{self.limit} calls) "
                f"before {label}"
            )
        self.count += 1
        if verbose:
            print(f"[DEEP] LLM call {self.count}/{self.limit} ({label})")


def _install_tracked_run_models(budget: _LLMBudget, verbose: bool) -> list:
    """Swap each agent module's run_model with a budget-charging wrapper.

    The agents call `run_model` via their own module globals, so the swap
    must happen per module. Returns (module, original) pairs for restore.
    """
    originals = []
    for module, label in (
        (_decomposition_mod, "decomposer"),
        (_retriever_mod, "sufficiency/retriever"),
        (_writer_mod, "writer"),
        (_verifier_mod, "critic"),
    ):
        original = module.run_model

        def make_wrapper(original, label):
            def wrapper(*args, **kwargs):
                budget.charge(label, verbose=verbose)
                return original(*args, **kwargs)

            return wrapper

        module.run_model = make_wrapper(original, label)
        originals.append((module, original))
    return originals


def _restore_tracked_run_models(originals: list) -> None:
    for module, original in originals:
        module.run_model = original


def _read_doc_catalog() -> List[Dict[str, str]]:
    """Unique {document_name, document_title} pairs from the Qdrant
    `document_reports` payloads (same scroll pattern as
    get_indexed_document_catalog). Empty list on error or empty collection
    → the pipeline runs in web-only mode (P1-6)."""
    try:
        client = get_qdrant_client()
        if not client.collection_exists(COLLECTION_NAME):
            return []
        catalog: Dict[str, str] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=["document_name", "document_title"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                name = payload.get("document_name")
                if name:
                    catalog.setdefault(name, payload.get("document_title") or "")
            if offset is None:
                break
        return [
            {"document_name": name, "document_title": title}
            for name, title in sorted(catalog.items())
        ]
    except Exception as exc:
        print(
            f"[DEEP] WARNING: could not read the document catalog "
            f"({type(exc).__name__}: {exc}); continuing in web-only mode"
        )
        return []


def _merge_evidence(packs: List[dict]) -> tuple:
    """Merge per-sub-question evidence packs into global deduped lists:
    docs by chunk_id and web by url, keeping the highest-score occurrence
    of each. Deterministic (input order stable for equal scores)."""
    from memory.helpers import _dedupe_doc_chunks, _dedupe_web_results

    all_doc: List[dict] = []
    all_web: List[dict] = []
    for pack in packs or []:
        doc_evidence = pack.get("document_evidence") or {}
        web_evidence = pack.get("web_evidence") or {}
        all_doc.extend(doc_evidence.get("chunks") or [])
        all_web.extend(web_evidence.get("results") or [])
    return _dedupe_doc_chunks(all_doc), _dedupe_web_results(all_web)


def _merge_new_evidence(
    registry: dict,
    doc_key_map: dict,
    web_key_map: dict,
    all_doc: List[dict],
    all_web: List[dict],
) -> tuple:
    """Assign CONTINUATION keys to evidence items not yet keyed (after a
    re-retrieval merge). Existing registry entries and keys are untouched.

    Returns (registry, doc_key_map, web_key_map, added_docs, added_webs)
    where added_* are the chunks/results that received new keys (in key
    order) — they are what revision calls should include.
    """
    from memory.helpers import _citation_page, _citation_score

    registry = dict(registry or {})
    doc_key_map = dict(doc_key_map or {})
    web_key_map = dict(web_key_map or {})

    next_doc = max((int(k[1:]) for k in registry if k.startswith("D")), default=0) + 1
    added_docs: List[dict] = []
    for chunk in sorted(
        (
            c
            for c in all_doc
            if (c.get("chunk_id") or "").strip()
            and (c.get("chunk_id") or "").strip() not in doc_key_map
            and (c.get("content") or "").strip()
            and ((c.get("document_name") or "").strip() or (c.get("document_title") or "").strip())
        ),
        key=lambda c: (-_citation_score(c.get("score")), (c.get("document_name") or "")),
    ):
        key = f"D{next_doc}"
        next_doc += 1
        doc_key_map[(chunk.get("chunk_id") or "").strip()] = key
        registry[key] = {
            "kind": "doc",
            "title": (chunk.get("document_title") or "").strip(),
            "document_name": (chunk.get("document_name") or "").strip(),
            "page_number": _citation_page(chunk.get("page_number")),
            "citation": (chunk.get("citation") or "").strip(),
            "score": _citation_score(chunk.get("score")),
        }
        added_docs.append(chunk)

    next_web = max((int(k[1:]) for k in registry if k.startswith("W")), default=0) + 1
    added_webs: List[dict] = []
    for result in sorted(
        (
            r
            for r in all_web
            if (r.get("url") or "").strip()
            and (r.get("url") or "").strip() not in web_key_map
            and (r.get("content") or "").strip()
        ),
        key=lambda r: (-_citation_score(r.get("score")), (r.get("title") or "")),
    ):
        key = f"W{next_web}"
        next_web += 1
        web_key_map[(result.get("url") or "").strip()] = key
        registry[key] = {
            "kind": "web",
            "title": (result.get("title") or "").strip(),
            "url": (result.get("url") or "").strip(),
            "published_date": (result.get("published_date") or "").strip() or None,
            "score": _citation_score(result.get("score")),
        }
        added_webs.append(result)

    return registry, doc_key_map, web_key_map, added_docs, added_webs


def _strip_invented_keys(text: str, registry: dict) -> tuple:
    """Deterministically drop citation keys that do not resolve against the
    registry (invented citations — plan Section D, validation 2, no LLM).

    Handles single-key ("[D22]"), multi-key ("[D1, W2]"), and mixed
    ("[41, D13]") brackets: bare-number items are not registry keys and are
    always kept. "[D22, D103]" -> "[D22]"; "[D103]" -> removed (bracket
    dropped). Returns (cleaned_text, invented_keys_in_appearance_order).
    """
    invented: List[str] = []

    def _repl(match: re.Match) -> str:
        items = re.findall(r"[DW]\d+|\d+", match.group(1))
        kept: List[str] = []
        for item in items:
            if item[0] in "DW":
                if item in registry:
                    kept.append(item)
                elif item not in invented:
                    invented.append(item)
            else:
                kept.append(item)
        if not kept:
            return ""
        if len(kept) == len(items):
            return match.group(0)
        return "[" + ", ".join(kept) + "]"

    cleaned = _CITATION_BRACKET_RE.sub(_repl, text or "")
    cleaned = re.sub(r"  +", " ", cleaned)  # spaces left behind by removals
    cleaned = re.sub(r" \.", ".", cleaned)
    cleaned = re.sub(r" \)", ")", cleaned)
    return cleaned, invented


# A citation bracket is a comma-separated list of items, each an internal
# evidence key ("[D1]", "[W2]") or a bare number. Greedy \d+ consumes the
# full number, so "[D10]" matches as D10 (never as D1 plus a stray digit).
# Writers occasionally emit mixed brackets ("[41, D13]"), so both item kinds
# are handled in one pattern.
_CITATION_BRACKET_RE = re.compile(
    r"\[((?:[DW]\d+|\d+)(?:\s*,\s*(?:[DW]\d+|\d+))*)\]"
)


def _collect_cited_keys(text: str, registry: dict) -> List[str]:
    """Citation keys found in the body (single-, multi-, and mixed-key
    brackets), in first-appearance order (walks once; duplicates collapse).
    Keys absent from the registry are still recorded — format_references
    drops them when building the list."""
    seen: List[str] = []
    for match in _CITATION_BRACKET_RE.finditer(text or ""):
        for key in re.findall(r"[DW]\d+", match.group(1)):
            if key not in seen:
                seen.append(key)
    return seen


def _key_number_map(cited_keys: List[str], registry: dict) -> Dict[str, int]:
    """key -> reference number, mirroring format_references' numbering
    exactly: first-appearance order, keys missing from the registry
    skipped, duplicates collapsed, contiguous 1..N."""
    mapping: Dict[str, int] = {}
    for key in cited_keys or []:
        if key in mapping or not (registry or {}).get(key):
            continue
        mapping[key] = len(mapping) + 1
    return mapping


def _renumber_inline_keys(text: str, key_to_number: Dict[str, int]) -> tuple:
    """Replace every [D#]/[W#] occurrence in `text` (single-, multi-, and
    mixed brackets) with its [n] reference number; bare-number items in
    mixed brackets pass through untouched (validated later by
    _drop_unresolved_numbers). Keys missing from the map are removed
    (whole bracket dropped, or the entry stripped) so no internal key
    survives the final body.

    Returns (renumbered_text, removed_keys_in_appearance_order).
    """
    removed: List[str] = []

    def _repl(match: re.Match) -> str:
        group = match.group(1)
        items = re.findall(r"[DW]\d+|\d+", group)
        if not re.search(r"[DW]\d+", group):
            return match.group(0)  # pure-numeric bracket: leave as-is
        out_items: List[str] = []
        for item in items:
            if item[0] in "DW":
                number = key_to_number.get(item)
                if number is None:
                    if item not in removed:
                        removed.append(item)
                    continue
                num = str(number)
                if num not in out_items:
                    out_items.append(num)
            elif item not in out_items:
                out_items.append(item)
        if not out_items:
            return ""
        return "[" + ", ".join(out_items) + "]"

    out = _CITATION_BRACKET_RE.sub(_repl, text or "")
    out = re.sub(r"  +", " ", out)  # spaces left behind by removals
    out = re.sub(r" \.", ".", out)
    out = re.sub(r" \)", ")", out)
    return out, removed


def _drop_unresolved_numbers(text: str, valid: set) -> tuple:
    """Drop bracketed numbers that do not resolve to a rendered reference
    entry (model-hallucinated numeric citations, which the writer contract
    forbids) so every inline [n] in the final body resolves. Pure prose
    brackets whose numbers all fall in `valid` are left untouched.

    Returns (text, dropped_numbers_in_appearance_order).
    """
    dropped: List[str] = []
    numeric_bracket = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

    def _repl(match: re.Match) -> str:
        items = [i.strip() for i in match.group(1).split(",")]
        kept = [i for i in items if int(i) in valid]
        for i in items:
            if int(i) not in valid and i not in dropped:
                dropped.append(i)
        if len(kept) == len(items):
            return match.group(0)
        if not kept:
            return ""
        return "[" + ", ".join(kept) + "]"

    out = numeric_bracket.sub(_repl, text or "")
    out = re.sub(r"  +", " ", out)
    out = re.sub(r" \.", ".", out)
    out = re.sub(r" \)", ")", out)
    return out, dropped


def _one_line_summary(section_text: str, max_words: int = 50) -> str:
    """One-line (~first 50 words) content summary of a written section, for
    prior_summaries coherence context and the exec-summary input."""
    lines = [
        line.strip()
        for line in (section_text or "").strip().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    words = " ".join(" ".join(lines).split())
    if len(words.split()) > max_words:
        words = " ".join(words.split()[:max_words]) + "…"
    return words


def _derive_heading(question: str) -> str:
    """Fallback heading when the decomposer did not provide one: the
    question truncated to ≤60 chars, title-cased."""
    q = " ".join((question or "").split()).rstrip("?.!")
    if len(q) > 60:
        q = q[:60].rsplit(" ", 1)[0]
    return q.title()


def _heading_for(sub_question: dict) -> str:
    heading = (sub_question.get("heading") or "").strip().rstrip(".: ")
    if not heading:
        heading = _derive_heading(sub_question.get("question") or "")
    return heading


def _routes_for(expected_sources: str, web_only: bool) -> List[str]:
    """Map the decomposer's expected_sources to retrieval routes; an empty
    doc catalog forces web-only mode (P1-6)."""
    if web_only:
        return ["web"]
    return {
        "doc": ["doc"],
        "web": ["web"],
        "both": ["doc", "web"],
    }.get((expected_sources or "both").strip().lower(), ["doc", "web"])


EXEC_SUMMARY_INSTRUCTIONS = """
You are writing the executive summary of a finished deep-research report.

Write a 150-250 word executive summary that SYNTHESIZES the findings across
the section summaries below. It must be a connected synthesis of what the
report establishes overall — its key numbers, named works, and strongest
claims — NOT a list or enumeration of the sections. Do not introduce facts
that are not present in the summaries, and do not use citations.

Output only the summary prose: no heading, no preamble.
"""


def deep_research(
    user_query: str,
    verbose: bool = True,
    max_rounds: int = 3,
    budget_doc: int = 10,
    budget_web: int = 5,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run the 5-stage deep-research pipeline (plan Section B).

    Args:
        user_query: The user's research query.
        verbose: Print stage/retrieval/writer progress.
        max_rounds: Max investigator rounds per sub-question (default 3).
        budget_doc: Max doc chunks kept per sub-question (default 10).
        budget_web: Max web results kept per sub-question (default 5).
        endpoint: Optional custom endpoint URL for all agents.
        api_key: Optional custom API key for all agents.

    Returns:
        {"final_answer": str, "state": dict, "stats": dict} where stats =
        {llm_calls, wall_s, sections, revisions, re_retrieves,
        section_failures, exec_summary_failed}. state carries
        the fields save_report consumes (verification, draft, evidence_json,
        verification_status) plus the investigation state (plan, per-sub-
        question packs, citation registry/maps, sections, critic report).
    """
    started = time.time()
    budget = _LLMBudget(MAX_LLM_CALLS)
    stats: Dict[str, Any] = {
        "llm_calls": 0,
        "wall_s": 0.0,
        "sections": 0,
        "revisions": 0,
        "re_retrieves": 0,
        "section_failures": 0,
        "exec_summary_failed": False,
    }
    state: Dict[str, Any] = {
        "plan": None,
        "sub_question_evidence": {},
        "citation_registry": {},
        "doc_key_map": {},
        "web_key_map": {},
        "sections": [],
        "critic": None,
    }
    sections: List[tuple] = []  # (sq_id, heading, section_text)
    packs: Dict[str, dict] = {}
    all_doc: List[dict] = []
    all_web: List[dict] = []
    critic: Optional[dict] = None

    if verbose:
        print(f"[DEEP] deep_research: '{user_query}' (budget {budget.limit} LLM calls)")

    def _log_stage(name: str, extra: str = "") -> None:
        stats["llm_calls"] = budget.count
        wall = time.time() - started
        print(
            f"[DEEP] stage {name} done: wall={wall:.1f}s "
            f"llm_calls={budget.count}/{budget.limit} {extra}".rstrip()
        )

    def _finish(final_answer: str) -> dict:
        stats["llm_calls"] = budget.count
        stats["wall_s"] = round(time.time() - started, 1)
        stats["sections"] = len(sections)
        return {"final_answer": final_answer, "state": state, "stats": stats}

    originals = _install_tracked_run_models(budget, verbose)
    try:
        # ------------------------------------------------------------------
        # Doc catalog (empty → web-only mode, P1-6)
        # ------------------------------------------------------------------
        catalog = _read_doc_catalog()
        web_only = not catalog
        if verbose:
            print(
                f"[DEEP] document catalog: {len(catalog)} document(s)"
                + (" (web-only mode)" if web_only else "")
            )

        # ------------------------------------------------------------------
        # STAGE 1 — DECOMPOSE
        # ------------------------------------------------------------------
        if not budget.can_afford(1):
            print("[DEEP] WARNING: LLM budget exhausted before decomposition; nothing to assemble.")
            return _finish(
                "Deep research failed: the LLM call budget was exhausted before "
                "the query could be decomposed."
            )
        plan = decompose_query(
            user_query, catalog, verbose=verbose, endpoint=endpoint, api_key=api_key
        )
        if plan.get("source") == "fallback":
            # One retry: a garbage structured call lands on the
            # single-sub-question fallback plan. Keep whichever plan has MORE
            # sub-questions (on a tie the first); no further retries.
            if budget.can_afford(1):
                if verbose:
                    print("[DEEP] decomposition source=fallback; retrying once")
                retry_plan = decompose_query(
                    user_query, catalog, verbose=verbose,
                    endpoint=endpoint, api_key=api_key,
                )
                if len(retry_plan.get("sub_questions") or []) > len(
                    plan.get("sub_questions") or []
                ):
                    plan = retry_plan
            else:
                print(
                    "[DEEP] WARNING: LLM budget exhausted before a decomposition "
                    "retry; keeping the fallback plan."
                )
        state["plan"] = plan
        sub_questions = list(plan.get("sub_questions") or [])
        if not sub_questions:
            _log_stage("1 DECOMPOSE", "FAILED (no sub-questions)")
            return _finish(
                "Deep research failed: the query could not be decomposed into "
                "sub-questions (the decomposer returned an empty plan). Please "
                "rephrase the request and try again."
            )
        # Priority order (1 first), stable within a priority.
        sub_questions.sort(key=lambda sq: (int(sq.get("priority") or 3),))
        _log_stage(
            "1 DECOMPOSE",
            f"sub_questions={len(sub_questions)} source={plan.get('source')}",
        )

        # ------------------------------------------------------------------
        # STAGE 2 — INVESTIGATE (per sub-question, priority order)
        # ------------------------------------------------------------------
        for sub_question in sub_questions:
            sq_id = sub_question.get("id") or f"sq{len(packs) + 1}"
            routes = _routes_for(sub_question.get("expected_sources"), web_only)
            goal = " ".join(
                part for part in [
                    (sub_question.get("question") or "").strip(),
                    (sub_question.get("angle") or "").strip(),
                ] if part
            )
            if verbose:
                print(f"[DEEP] investigating {sq_id} (routes={routes}): {goal[:90]}")
            pack = retriever_agent(
                user_query,
                research_goal=goal,
                max_rounds=max_rounds,
                budget_doc=budget_doc,
                budget_web=budget_web,
                routes=routes,
                verbose=verbose,
                endpoint=endpoint,
                api_key=api_key,
            )
            packs[sq_id] = pack.model_dump()
            state["sub_question_evidence"][sq_id] = packs[sq_id]
        _log_stage("2 INVESTIGATE", f"packs={len(packs)}")

        all_doc, all_web = _merge_evidence(list(packs.values()))
        registry, doc_key_map, web_key_map = assign_citation_keys(all_doc, all_web)
        state["citation_registry"] = registry
        state["doc_key_map"] = doc_key_map
        state["web_key_map"] = web_key_map
        if verbose:
            print(
                f"[DEEP] global citation keys: {len(doc_key_map)} doc, "
                f"{len(web_key_map)} web"
            )

        # ------------------------------------------------------------------
        # STAGE 3 — PER-SECTION DRAFT
        # ------------------------------------------------------------------
        outline_text = "\n".join(
            f"{i}. {_heading_for(sq)}" for i, sq in enumerate(sub_questions, start=1)
        )
        for sub_question in sub_questions:
            if not budget.can_afford(1):
                print(
                    "[DEEP] WARNING: LLM budget exhausted during drafting; "
                    "assembling the report from the sections written so far."
                )
                break
            sq_id = sub_question.get("id")
            heading = _heading_for(sub_question)
            pack = packs.get(sq_id, {})
            section_docs = (pack.get("document_evidence") or {}).get("chunks") or []
            section_webs = (pack.get("web_evidence") or {}).get("results") or []
            evidence_text = render_evidence_text(
                section_docs, section_webs, doc_key_map, web_key_map
            )
            prior_summaries = "\n".join(
                f"- {sec_heading}: {_one_line_summary(sec_text)}"
                for _sid, sec_heading, sec_text in sections
            )
            section_context = (
                f"Sub-question: {sub_question.get('question') or ''}. "
                f"Angle: {sub_question.get('angle') or 'n/a'}."
            )
            try:
                section_text = write_section(
                    user_query,
                    outline_text,
                    heading,
                    section_context,
                    evidence_text,
                    prior_summaries=prior_summaries,
                    verbose=verbose,
                    endpoint=endpoint,
                    api_key=api_key,
                )
            except _BudgetExhausted:
                print(
                    "[DEEP] WARNING: LLM budget exhausted in write_section; "
                    "assembling from what exists."
                )
                break
            except Exception as exc:
                # Transient LLM failure (500/timeout): skip this section and
                # keep drafting the rest — a gap in the report, never a crash.
                stats["section_failures"] += 1
                print(
                    f"[DEEP] WARNING: write_section failed for '{heading}' "
                    f"({type(exc).__name__}: {exc}); skipping this section."
                )
                continue
            sections.append((sq_id, heading, section_text))
        draft_extra = f"sections={len(sections)}"
        if stats["section_failures"]:
            draft_extra += f" section_failures={stats['section_failures']}"
        _log_stage("3 PER-SECTION DRAFT", draft_extra)

        # ------------------------------------------------------------------
        # STAGE 4 — CRITIC (+ budgeted expansion / one re-retrieval)
        # ------------------------------------------------------------------
        if sections:
            draft_text = "\n\n".join(text for _sid, _h, text in sections)
            section_ids = [sq_id for sq_id, _h, _t in sections]
            if budget.can_afford(1):
                critic = verification_critic(
                    user_query,
                    draft_text,
                    render_evidence_text(all_doc, all_web, doc_key_map, web_key_map),
                    section_ids,
                    verbose=verbose,
                    endpoint=endpoint,
                    api_key=api_key,
                )
            else:
                print(
                    "[DEEP] WARNING: LLM budget exhausted before the critic; "
                    "skipping verification and revisions."
                )

            added_docs: List[dict] = []
            added_webs: List[dict] = []

            # ONE optional re-retrieval (before revisions, so its evidence can
            # be included in the revision calls for the affected sections).
            if (
                critic
                and critic.get("re_retrieve_suggested")
                and critic.get("specific_queries")
                and stats["re_retrieves"] == 0
                and budget.can_afford(1)
            ):
                combined_goal = " ".join(
                    q.strip() for q in critic["specific_queries"] if q and q.strip()
                )
                if verbose:
                    print(f"[DEEP] re-retrieval: {combined_goal[:120]}")
                rr_pack = retriever_agent(
                    user_query,
                    research_goal=combined_goal,
                    max_rounds=1,
                    budget_doc=5,
                    budget_web=3,
                    verbose=verbose,
                    endpoint=endpoint,
                    api_key=api_key,
                )
                rr_dump = rr_pack.model_dump()
                # Merge the re-retrieval pack into the global (deduped) lists.
                from memory.helpers import _dedupe_doc_chunks, _dedupe_web_results

                rr_docs = (rr_dump.get("document_evidence") or {}).get("chunks") or []
                rr_webs = (rr_dump.get("web_evidence") or {}).get("results") or []
                all_doc = _dedupe_doc_chunks(list(all_doc) + rr_docs)
                all_web = _dedupe_web_results(list(all_web) + rr_webs)
                (
                    registry,
                    doc_key_map,
                    web_key_map,
                    added_docs,
                    added_webs,
                ) = _merge_new_evidence(
                    registry, doc_key_map, web_key_map, all_doc, all_web
                )
                state["citation_registry"] = registry
                state["doc_key_map"] = doc_key_map
                state["web_key_map"] = web_key_map
                stats["re_retrieves"] += 1
                if verbose:
                    print(
                        f"[DEEP] re-retrieval merged: +{len(added_docs)} doc, "
                        f"+{len(added_webs)} web keyed items"
                    )

            # Revisions: weak sections go back to write_section with gaps.
            if critic:
                rev_counts: Dict[str, int] = {}
                for verdict in critic.get("per_section") or []:
                    if not isinstance(verdict, dict):
                        continue
                    if verdict.get("grounded", True) and verdict.get("depth_ok", True):
                        continue
                    gaps = [g for g in (verdict.get("gaps") or []) if g and g.strip()]
                    if not gaps:
                        continue
                    sid = str(verdict.get("section_id") or "").strip()
                    index = next(
                        (i for i, (s, _h, _t) in enumerate(sections) if s == sid),
                        None,
                    )
                    if index is None:
                        continue
                    if (
                        stats["revisions"] >= _MAX_EXPANSION_CALLS
                        or rev_counts.get(sid, 0) >= _MAX_REVISIONS_PER_SECTION
                    ):
                        if verbose:
                            print(
                                f"[DEEP] revision cap reached for {sid}; skipping."
                            )
                        continue
                    if not budget.can_afford(1):
                        print(
                            "[DEEP] WARNING: LLM budget exhausted during revisions; "
                            "keeping the current draft."
                        )
                        break
                    sub_question = next(
                        (sq for sq in sub_questions if sq.get("id") == sid), {}
                    )
                    pack = packs.get(sid, {})
                    section_docs = (pack.get("document_evidence") or {}).get("chunks") or []
                    section_webs = (pack.get("web_evidence") or {}).get("results") or []
                    # Affected sections see their own pool + any re-retrieved
                    # evidence (continuation keys), all under the global maps.
                    evidence_text = render_evidence_text(
                        section_docs + added_docs,
                        section_webs + added_webs,
                        doc_key_map,
                        web_key_map,
                    )
                    _sid, heading, _old_text = sections[index]
                    prior_summaries = "\n".join(
                        f"- {sec_heading}: {_one_line_summary(sec_text)}"
                        for i, (_s, sec_heading, sec_text) in enumerate(sections)
                        if i != index
                    )
                    section_context = (
                        f"Sub-question: {sub_question.get('question') or ''}. "
                        f"Angle: {sub_question.get('angle') or 'n/a'}."
                    )
                    try:
                        new_text = write_section(
                            user_query,
                            outline_text,
                            heading,
                            section_context,
                            evidence_text,
                            prior_summaries=prior_summaries,
                            expansion_gaps=gaps,
                            verbose=verbose,
                            endpoint=endpoint,
                            api_key=api_key,
                        )
                    except _BudgetExhausted:
                        print(
                            "[DEEP] WARNING: LLM budget exhausted in a revision; "
                            "keeping the current draft."
                        )
                        break
                    except Exception as exc:
                        # Transient LLM failure: keep the current draft for
                        # this section and continue with remaining verdicts —
                        # a gap, never a crash.
                        stats["section_failures"] += 1
                        print(
                            f"[DEEP] WARNING: revision failed for '{heading}' "
                            f"({type(exc).__name__}: {exc}); keeping the "
                            "current draft for this section."
                        )
                        continue
                    sections[index] = (sid, heading, new_text)
                    stats["revisions"] += 1
                    rev_counts[sid] = rev_counts.get(sid, 0) + 1
        critic_extra = (
            f"revisions={stats['revisions']} re_retrieves={stats['re_retrieves']} "
            f"source={critic.get('source') if critic else 'skipped'}"
        )
        if stats["section_failures"]:
            critic_extra += f" section_failures={stats['section_failures']}"
        _log_stage("4 CRITIC", critic_extra)

        # ------------------------------------------------------------------
        # STAGE 5 — ASSEMBLY
        # ------------------------------------------------------------------
        # (a) Executive summary, written LAST.
        exec_summary = ""
        if sections:
            if budget.can_afford(1):
                summaries = "\n".join(
                    f"- {heading}: {_one_line_summary(text)}"
                    for _sid, heading, text in sections
                )
                try:
                    budget.charge("exec-summary", verbose=verbose)
                    config = get_config()
                    response = run_model(
                        instructions=EXEC_SUMMARY_INSTRUCTIONS,
                        input_data=(
                            f"User query: {user_query}\n\n"
                            f"Section summaries of the finished report:\n{summaries}"
                        ),
                        reasoning_effort=config.get_reasoning_effort("writer"),
                        max_output_tokens=config.get_max_output_tokens("writer"),
                        tools=None,
                        agent_name="writer",
                        endpoint=endpoint,
                        api_key=api_key,
                    )
                    summary_prose = (response.output_text or "").strip()
                    # Strip a heading the model may have added anyway.
                    summary_lines = [
                        line
                        for line in summary_prose.splitlines()
                        if not line.strip().startswith("## ")
                    ]
                    summary_prose = "\n".join(summary_lines).strip()
                    if summary_prose:
                        # Prose only: save_report's build_enriched_markdown wraps
                        # the body under its own "## Executive Summary" heading
                        # (same as standard mode), so the pipeline must not add a
                        # second one.
                        exec_summary = summary_prose
                except Exception as exc:
                    # Transient LLM failure: assemble WITHOUT an executive
                    # summary (sections + references only). Never crash.
                    stats["exec_summary_failed"] = True
                    print(
                        f"[DEEP] WARNING: executive summary call failed "
                        f"({type(exc).__name__}: {exc}); assembling without one."
                    )
            else:
                print(
                    "[DEEP] WARNING: LLM budget exhausted before the executive "
                    "summary; assembling without one."
                )

        # (b) References resolved deterministically from the cited keys.
        # Invented citation keys (cited but absent from the registry) are
        # dropped deterministically first (plan Section D, validation 2) so
        # every key left in the body resolves to a reference entry.
        body_parts = [part for part in [exec_summary] + [t for _s, _h, t in sections] if part]
        cleaned_parts: List[str] = []
        invented_keys: List[str] = []
        for part in body_parts:
            cleaned, invented = _strip_invented_keys(part, registry)
            cleaned_parts.append(cleaned)
            for k in invented:
                if k not in invented_keys:
                    invented_keys.append(k)
        body = "\n\n".join(cleaned_parts)
        cited_keys = _collect_cited_keys(body, registry)
        # Renumber the internal [D#]/[W#] keys across the WHOLE assembled
        # body (exec summary + sections) to the [1..N] numbering the
        # References section uses, so every inline marker resolves to a
        # reference entry.
        key_to_number = _key_number_map(cited_keys, registry)
        body, unresolvable = _renumber_inline_keys(body, key_to_number)
        # Hallucinated bare-number citations (the writer contract forbids
        # them) that do not resolve to a rendered reference are dropped so
        # every inline [n] in the final body resolves.
        body, dead_numbers = _drop_unresolved_numbers(
            body, set(key_to_number.values())
        )
        if unresolvable:
            shown = ", ".join(unresolvable[:10])
            print(
                f"[DEEP] WARNING: dropped {len(unresolvable)} unresolvable "
                f"key marker(s) during renumbering: {shown}"
            )
        if dead_numbers:
            shown = ", ".join(dead_numbers[:10])
            print(
                f"[DEEP] WARNING: dropped {len(dead_numbers)} unresolvable "
                f"numeric citation(s): {shown}"
            )
        references = format_references(registry, cited_keys)
        final_parts = [body] if body else []
        if references:
            final_parts.append(f"## References\n\n{references}")
        if invented_keys:
            shown = ", ".join(invented_keys[:10])
            if len(invented_keys) > 10:
                shown += "\u2026"
            final_parts.append(
                f"*Note: {len(invented_keys)} citation key(s) ({shown}) were cited "
                "without a matching evidence item and were removed."
            )

        final_answer = "\n\n".join(final_parts)
        if not final_answer:
            final_answer = (
                "Deep research produced no sections: the LLM call budget was "
                "exhausted before any section could be drafted."
            )

        # (c) Machine-side state for save_report / observability.
        merged_pack = {
            "query": user_query,
            "route_used": infer_route_used(all_doc, all_web),
            "summary": (
                f"Deep research evidence: {len(all_doc)} doc chunk(s) and "
                f"{len(all_web)} web result(s) across {len(packs)} sub-question(s)."
            ),
            "document_evidence": {
                "query": user_query,
                "summary": "Merged per-sub-question document evidence (deduped).",
                "chunks": all_doc,
            },
            "web_evidence": {
                "query": user_query,
                "summary": "Merged per-sub-question web evidence (deduped).",
                "results": all_web,
            },
        }
        state["sections"] = [
            {"id": sq_id, "heading": heading, "text": text}
            for sq_id, heading, text in sections
        ]
        state["critic"] = critic
        state["verification"] = final_answer
        state["draft"] = final_answer
        state["evidence_json"] = json.dumps(merged_pack)
        if critic:
            coverage = {
                "high": "comprehensive",
                "medium": "moderate",
                "low": "thin",
            }.get(critic.get("confidence_level"), "moderate")
            state["verification_status"] = {
                "confidence": critic.get("confidence_level", "medium"),
                "coverage": coverage,
                "gaps": [
                    g
                    for verdict in critic.get("per_section") or []
                    if isinstance(verdict, dict)
                    for g in (verdict.get("gaps") or [])
                    if g and g.strip()
                ],
            }
        else:
            state["verification_status"] = {
                "confidence": "medium",
                "coverage": "moderate",
                "gaps": [],
            }
        assembly_extra = (
            f"chars={len(final_answer)} refs={len(references.splitlines()) if references else 0} "
            f"invented_keys_dropped={len(invented_keys)} "
            f"unresolved_dropped={len(unresolvable) + len(dead_numbers)}"
        )
        if stats["exec_summary_failed"]:
            assembly_extra += " exec_summary_failed=1"
        _log_stage("5 ASSEMBLY", assembly_extra)

        return _finish(final_answer)
    finally:
        _restore_tracked_run_models(originals)
