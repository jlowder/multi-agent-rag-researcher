import json
import re
from typing import Any, Optional

# infer route used from available document and web evidence
def infer_route_used(document_chunks: list[Any], web_results: list[Any]) -> str:
    if document_chunks and web_results:
        return "both"
    if document_chunks:
        return "documents"
    if web_results:
        return "web"
    return "none"


# build evidence context from saved evidence json
def build_evidence_context(
    evidence_json: str,
    *,
    include_formatted_evidence: bool = False,
) -> dict[str, Any]:
    if not evidence_json:
        return {"has_evidence": False, "summary": "None", "formatted_evidence": ""}

    try:
        payload = json.loads(evidence_json)
    except json.JSONDecodeError:
        return {
            "has_evidence": False,
            "summary": "Active evidence is available, but it could not be summarized.",
            "formatted_evidence": evidence_json,
        }

    document_evidence = payload.get("document_evidence") or {}
    web_evidence = payload.get("web_evidence") or {}
    document_chunks = document_evidence.get("chunks") or []
    web_results = web_evidence.get("results") or []
    route_used = payload.get("route_used") or infer_route_used(document_chunks, web_results)
    retrieval_summary = payload.get("summary") or "None"
    has_evidence = bool(document_chunks or web_results)
    summary = "\n".join(
        [
            f"Route used: {route_used}",
            f"Summary: {retrieval_summary}",
            f"Document chunk count: {len(document_chunks)}",
            f"Web result count: {len(web_results)}",
        ]
    )

    if not include_formatted_evidence or not has_evidence:
        return {
            "has_evidence": has_evidence,
            "summary": summary,
            "formatted_evidence": "",
        }

    formatted_evidence = "\n\n".join(
        section for section in [
            f"Retrieval summary:\nRoute used: {route_used}\nSummary: {retrieval_summary}",
            f"Document evidence:\n{json.dumps(document_evidence, indent=2)}" if document_chunks else "",
            f"Web evidence:\n{json.dumps(web_evidence, indent=2)}" if web_results else "",
        ]
        if section
    )

    return {
        "has_evidence": has_evidence,
        "summary": summary,
        "formatted_evidence": formatted_evidence,
    }


# ---------------------------------------------------------------------------
# Citation-keyed evidence (P1-5): deterministic D*/W* keys + metadata registry
# ---------------------------------------------------------------------------


def _citation_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _citation_page(value: Any) -> Optional[int]:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _dedupe_doc_chunks(doc_chunks: list) -> list:
    """Dedupe doc chunks by chunk_id, keeping the highest-score occurrence.

    Chunks without a chunk_id are kept as-is (they cannot collide and will
    simply be uncitable). Input order is stable for equal scores.
    """
    by_id: dict[str, dict] = {}
    order: list[str] = []
    no_id: list[dict] = []
    for chunk in doc_chunks or []:
        chunk_id = (chunk.get("chunk_id") or "").strip()
        if not chunk_id:
            no_id.append(chunk)
            continue
        existing = by_id.get(chunk_id)
        if existing is None:
            by_id[chunk_id] = chunk
            order.append(chunk_id)
        elif _citation_score(chunk.get("score")) > _citation_score(existing.get("score")):
            by_id[chunk_id] = chunk
    return [by_id[cid] for cid in order] + no_id


def _dedupe_web_results(web_results: list) -> list:
    """Dedupe web results by url, keeping the highest-score occurrence."""
    by_url: dict[str, dict] = {}
    order: list[str] = []
    no_url: list[dict] = []
    for result in web_results or []:
        url = (result.get("url") or "").strip()
        if not url:
            no_url.append(result)
            continue
        existing = by_url.get(url)
        if existing is None:
            by_url[url] = result
            order.append(url)
        elif _citation_score(result.get("score")) > _citation_score(existing.get("score")):
            by_url[url] = result
    return [by_url[u] for u in order] + no_url


def assign_citation_keys(
    doc_chunks: list[dict],
    web_results: list[dict],
) -> tuple[dict, dict, dict]:
    """
    Assign GLOBAL citation keys across all evidence from a whole
    investigation (merged across sub-questions, P1-3 deep pipeline).

    Args:
        doc_chunks: doc chunk dicts (see build_citation_context). Deduped by
            chunk_id internally (highest score wins) if not already.
        web_results: web result dicts. Deduped by url internally (highest
            score wins) if not already.

    Returns:
        (registry, doc_key_map, web_key_map) where:
        - registry: same shape as build_citation_context's registry —
              doc: {kind, title, document_name, page_number, citation, score}
              web: {kind, title, url, published_date, score}
        - doc_key_map: {chunk_id: "D3"} for every keyed chunk.
        - web_key_map: {url: "W2"} for every keyed result.

    Key assignment is deterministic: docs sorted by score descending with a
    stable tiebreak on document_name -> D1..Dn; web by score descending with a
    stable tiebreak on title -> W1..Wn. Chunks missing required identity
    fields (doc: both document_name and document_title; web: url) get NO key
    and are absent from both the maps and the registry (Section D: drop, don't
    degrade). No LLM is involved.
    """
    registry: dict[str, dict[str, Any]] = {}
    doc_key_map: dict[str, str] = {}
    web_key_map: dict[str, str] = {}

    doc_index = 0
    for chunk in sorted(
        _dedupe_doc_chunks(doc_chunks),
        key=lambda c: (-_citation_score(c.get("score")), (c.get("document_name") or "")),
    ):
        if not (chunk.get("content") or "").strip():
            continue
        name = (chunk.get("document_name") or "").strip()
        title = (chunk.get("document_title") or "").strip()
        if not name and not title:
            continue  # unidentifiable: no key
        doc_index += 1
        key = f"D{doc_index}"
        chunk_id = (chunk.get("chunk_id") or "").strip()
        if chunk_id:
            doc_key_map[chunk_id] = key
        registry[key] = {
            "kind": "doc",
            "title": title,
            "document_name": name,
            "page_number": _citation_page(chunk.get("page_number")),
            "citation": (chunk.get("citation") or "").strip(),
            "score": _citation_score(chunk.get("score")),
        }

    web_index = 0
    for result in sorted(
        _dedupe_web_results(web_results),
        key=lambda r: (-_citation_score(r.get("score")), (r.get("title") or "")),
    ):
        if not (result.get("content") or "").strip():
            continue
        url = (result.get("url") or "").strip()
        if not url:
            continue  # unidentifiable: no key
        web_index += 1
        key = f"W{web_index}"
        web_key_map[url] = key
        registry[key] = {
            "kind": "web",
            "title": (result.get("title") or "").strip(),
            "url": url,
            "published_date": (result.get("published_date") or "").strip() or None,
            "score": _citation_score(result.get("score")),
        }

    return registry, doc_key_map, web_key_map


def render_evidence_text(
    doc_chunks: list[dict],
    web_results: list[dict],
    doc_key_map: dict[str, str],
    web_key_map: dict[str, str],
) -> str:
    """
    Render citation-keyed evidence blocks using GLOBAL keys from
    assign_citation_keys (per-section subsets use the same global keys so a
    citation resolves against the single shared registry).

    Block format is identical to build_citation_context:
        [D1] {title} — {document_name} p. {page_number}\n{content}\n
    (" p. N" omitted when the page is unknown)
        [W1] {title} ({url}, {date})\n{content}\n
    (date omitted when absent). Chunks without a key are rendered under an
    "(uncited)" marker as plain context. Ordering: docs before web; within
    each kind, score descending (stable tiebreak document_name / title), so
    keyed chunks render in D1..Dn / W1..Wn order.
    """
    doc_blocks: list[str] = []
    for chunk in sorted(
        _dedupe_doc_chunks(doc_chunks),
        key=lambda c: (-_citation_score(c.get("score")), (c.get("document_name") or "")),
    ):
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        chunk_id = (chunk.get("chunk_id") or "").strip()
        key = doc_key_map.get(chunk_id) if chunk_id else None
        if key is None:
            doc_blocks.append(f"(uncited)\n{content}\n")
            continue
        name = (chunk.get("document_name") or "").strip()
        title = (chunk.get("document_title") or "").strip()
        page = _citation_page(chunk.get("page_number"))
        header = " — ".join(part for part in (title, name) if part)
        if page is not None:
            header += f" p. {page}"
        doc_blocks.append(f"[{key}] {header}\n{content}\n")

    web_blocks: list[str] = []
    for result in sorted(
        _dedupe_web_results(web_results),
        key=lambda r: (-_citation_score(r.get("score")), (r.get("title") or "")),
    ):
        content = (result.get("content") or "").strip()
        if not content:
            continue
        url = (result.get("url") or "").strip()
        key = web_key_map.get(url) if url else None
        if key is None:
            title = (result.get("title") or "").strip()
            header = f"(uncited) {title}" if title else "(uncited)"
            web_blocks.append(f"{header}\n{content}\n")
            continue
        title = (result.get("title") or "").strip()
        date = (result.get("published_date") or "").strip()
        source = url + (f", {date}" if date else "")
        header = f"{title} ({source})" if title else f"({source})"
        web_blocks.append(f"[{key}] {header}\n{content}\n")

    return "".join(doc_blocks) + "".join(web_blocks)


def build_citation_context(evidence_pack: dict) -> tuple[str, dict]:
    """
    Build citation-keyed evidence text and a key -> metadata registry.

    Args:
        evidence_pack: parsed evidence pack dict of the shape
            {"document_evidence": {"chunks": [...]},
             "web_evidence": {"results": [...]}}
            (extra keys such as query/summary/route_used are ignored).

    Returns:
        (evidence_text, registry) where:
        - evidence_text holds blocks in the form
              [D1] {title} — {document_name} p. {page_number}\n{content}\n
          (" p. N" is omitted when the page is unknown) and
              [W1] {title} ({url}, {date})\n{content}\n
          (date omitted when absent). Chunks missing required identity fields
          (doc: both document_name and document_title; web: url) get NO key;
          they may still appear under an "(uncited)" marker as plain context.
        - registry maps each key to
              doc: {kind, title, document_name, page_number, citation, score}
              web: {kind, title, url, published_date, score}

    Key assignment is deterministic: within each kind, score descending with a
    stable tiebreak by name (docs: document_name then title; web: url then
    title). No LLM is involved.

    Implemented as assign_citation_keys + render_evidence_text so the deep
    pipeline (P1-3) can share the exact same keying/formatting logic at
    investigation-wide scale.
    """
    evidence_pack = evidence_pack or {}
    doc_chunks = (evidence_pack.get("document_evidence") or {}).get("chunks") or []
    web_results = (evidence_pack.get("web_evidence") or {}).get("results") or []
    registry, doc_key_map, web_key_map = assign_citation_keys(doc_chunks, web_results)
    evidence_text = render_evidence_text(
        doc_chunks, web_results, doc_key_map, web_key_map
    )
    return evidence_text, registry


_FILENAME_EXT_RE = re.compile(r"\.\w{1,5}$", re.IGNORECASE)


def _title_is_filename_like(title: str, document_name: str) -> bool:
    """True when a doc `title` is a bare filename rather than a real title:
    no whitespace plus a short trailing file extension (e.g. "gpem.dvi"), or
    the title equals the document_name's basename or stem (case-insensitive).
    """
    t = (title or "").strip()
    if not t:
        return False
    if " " not in t and _FILENAME_EXT_RE.search(t):
        return True
    base = (document_name or "").strip().rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else ""
    return t.lower() in {v.lower() for v in (base, stem) if v}


def format_references(registry: dict, cited_keys: list) -> str:
    """
    Render a References section for exactly the cited keys.

    Numbering follows first appearance: the order given in `cited_keys` is
    taken as the first-appearance order (duplicates collapse to one entry,
    unknown keys are skipped), numbered contiguously 1..N. Lines render as:
        doc: [1] {title}. *{document_name}* p. {page}.
        web: [1] {title}. {url} ({date}).
    Missing parts (title, page, date) are omitted gracefully.
    Returns "" when nothing is cited. Deterministic; no LLM.
    """
    lines: list[str] = []
    seen: set[str] = set()
    number = 0
    for key in cited_keys or []:
        if key in seen:
            continue
        seen.add(key)
        entry = (registry or {}).get(key)
        if not entry:
            continue
        number += 1
        title = (entry.get("title") or "").strip()
        if entry.get("kind") == "doc":
            name = (entry.get("document_name") or "").strip()
            # A bare filename used as the title ("gpem.dvi") adds no
            # information next to the document_name; render without it.
            if _title_is_filename_like(title, name):
                title = ""
            parts: list[str] = []
            if title:
                parts.append(f"{title}.")
            if name:
                parts.append(f"*{name}*")
            page = entry.get("page_number")
            if page:
                parts.append(f"p. {page}.")
            line = " ".join(parts)
            if line and not line.endswith("."):
                line += "."
            lines.append(f"[{number}] " + line)
        else:
            url = (entry.get("url") or "").strip()
            date = (entry.get("published_date") or "").strip()
            url_part = f"{url} ({date})." if date else f"{url}."
            prefix = f"{title}. " if title else ""
            lines.append(f"[{number}] {prefix}{url_part}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry -> Sources mapping (plan §8.1): CSL-JSON-compatible source records
# ---------------------------------------------------------------------------


def registry_to_sources(registry: dict, cited_keys: list, include_uncited: bool = False) -> list[dict]:
    """Convert the citation registry into CSL-JSON compatible source records.

    Only keys present in `cited_keys` are included unless `include_uncited`
    is True. Multiple registry entries from the same document are deduplicated
    into one source record (doc: same document_name or title; web: same url);
    the record keeps the primary (lowest-numbered) citation_key. Missing
    bibliographic fields come out empty ("" / [] / {}), never null.

    Returns list of dicts suitable for Source.model_validate (plan §8.1).
    """
    registry = registry or {}
    if include_uncited:
        keys: list[str] = list(registry.keys())
    else:
        seen: set[str] = set()
        keys = []
        for key in cited_keys or []:
            if key in seen or key not in registry:
                continue
            seen.add(key)
            keys.append(key)

    records: list[dict] = []
    dedupe_index: dict[tuple, int] = {}
    for key in keys:
        entry = registry.get(key) or {}
        if entry.get("kind") == "doc":
            dedupe_key = (
                "doc",
                (entry.get("document_name") or entry.get("title") or "").strip().lower(),
            )
            url = ""
            source_type = "report"
            issued: dict = {}
        else:
            dedupe_key = (
                "web",
                (entry.get("url") or "").strip().lower()
                or (entry.get("title") or "").strip().lower(),
            )
            url = (entry.get("url") or "").strip()
            source_type = "webpage"
            issued = {}
            published = entry.get("published_date")
            if isinstance(published, str) and published.strip():
                year = re.search(r"(\d{4})", published)
                if year:
                    issued = {"date-parts": [[int(year.group(1))]]}

        if dedupe_key in dedupe_index:
            continue  # same source already recorded under an earlier (primary) key
        dedupe_index[dedupe_key] = len(records)
        records.append(
            {
                "id": f"source-{key.lower()}",
                "type": source_type,
                "title": (entry.get("title") or "").strip(),
                "author": [],
                "issued": issued,
                "URL": url,
                "publisher": "",
                "DOI": "",
                "citation_key": key,
                "accessed": "",
            }
        )
    return records