from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

"""
Evidence cache
=====================================================================================
Per-sub-question evidence memory for deep mode (P2-2).

Deep runs store each retrieved evidence pack (keyed by session + normalized
sub-topic) so a later run asking a near-identical sub-question can reuse the
pack instead of re-retrieving. Lookup is cross-session: any recent,
sufficient pack whose question has high token overlap counts.

Follows memory.py's connection pattern (SQLite, utils/ directory) in its own
file so memory.py's single-row session schema stays untouched for standard
mode.
"""

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
EVIDENCE_CACHE_DB_PATH = UTILS_DIR / "evidence_cache.db"

# A stored pack is reusable only if its token overlap with the incoming
# question meets this Jaccard threshold (see lookup_evidence_detail).
JACCARD_THRESHOLD = 0.8

# TTL used by the once-per-process lazy purge when a save is the process's
# first cache use and no explicit ttl_days is passed (lookups always pass
# their own ttl_days; the pipeline threads its configured value through).
_DEFAULT_TTL_DAYS = 30

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evidence_cache (
    session_id TEXT NOT NULL,
    sub_topic TEXT NOT NULL,
    question TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    sufficiency_ok INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, sub_topic)
)
"""

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Lazy purge bookkeeping: stale rows are deleted once per process (first
# cache use), not on every call.
_purged_this_process = False


# get sqlite connection for evidence-cache operations (creates the table
# idempotently on first use)
def get_evidence_cache_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(EVIDENCE_CACHE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE_SQL)
    return conn


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_retrieved_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# normalize a question for sub-topic keys and similarity matching:
# lowercase, strip punctuation, collapse whitespace
def normalize_question(text: str) -> str:
    lowered = str(text or "").lower()
    no_punct = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", no_punct).strip()


# token-set Jaccard similarity on normalized text
def jaccard(a: str, b: str) -> float:
    tokens_a = set(normalize_question(a).split())
    tokens_b = set(normalize_question(b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _delete_stale(conn: sqlite3.Connection, ttl_days: int) -> None:
    cutoff = (_utc_now() - timedelta(days=ttl_days)).isoformat()
    conn.execute("DELETE FROM evidence_cache WHERE retrieved_at < ?", (cutoff,))


def _lazy_purge_once(conn: sqlite3.Connection, ttl_days: int) -> None:
    """Purge stale rows once per process (first cache use), not per call."""
    global _purged_this_process
    if _purged_this_process:
        return
    _purged_this_process = True
    _delete_stale(conn, ttl_days)


# delete rows older than ttl_days (UTC)
def purge_stale(ttl_days: int) -> None:
    with get_evidence_cache_connection() as conn:
        _delete_stale(conn, ttl_days)


# drop every cached evidence pack (whole table) and return the rowcount.
# Called at ingest entry points when the corpus was reconciled away, so
# stale-document packs can't be reused for a new corpus.
def clear_evidence_cache() -> int:
    with get_evidence_cache_connection() as conn:
        cursor = conn.execute("DELETE FROM evidence_cache")
        return cursor.rowcount


# store (or replace) the evidence pack for this session + sub-topic
def save_evidence(
    session_id: str,
    question: str,
    evidence_pack: dict,
    sufficiency_ok: bool,
    ttl_days: int = _DEFAULT_TTL_DAYS,
) -> None:
    sub_topic = normalize_question(question)
    if not session_id or not sub_topic:
        return
    with get_evidence_cache_connection() as conn:
        _lazy_purge_once(conn, ttl_days)
        conn.execute(
            """
            INSERT OR REPLACE INTO evidence_cache
                (session_id, sub_topic, question, retrieved_at, evidence_json,
                 sufficiency_ok)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                sub_topic,
                str(question or "").strip(),
                _utc_now().isoformat(),
                json.dumps(evidence_pack),
                1 if sufficiency_ok else 0,
            ),
        )


# find the best reusable pack for `question` across ALL sessions
def lookup_evidence_detail(question: str, ttl_days: int) -> dict | None:
    """
    Cross-session lookup of a reusable evidence pack.

    A candidate row must be within ttl_days, marked sufficient
    (sufficiency_ok = 1), and have jaccard(stored question, question)
    >= JACCARD_THRESHOLD (0.8). The highest score wins; ties go to the most
    recent retrieved_at. Returns {"evidence", "jaccard", "question",
    "retrieved_at", "age_days"} or None.
    """
    q_norm = normalize_question(question)
    if not q_norm:
        return None
    conn = get_evidence_cache_connection()
    try:
        _lazy_purge_once(conn, ttl_days)
        cutoff = (_utc_now() - timedelta(days=ttl_days)).isoformat()
        rows = conn.execute(
            """
            SELECT question, evidence_json, retrieved_at
            FROM evidence_cache
            WHERE retrieved_at >= ? AND sufficiency_ok = 1
            """,
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    best = None
    best_score = 0.0
    for row in rows:
        score = jaccard(row["question"], q_norm)
        if score < JACCARD_THRESHOLD:
            continue
        if best is None or score > best_score or (
            score == best_score and row["retrieved_at"] > best["retrieved_at"]
        ):
            best = row
            best_score = score
    if best is None:
        return None
    return {
        "evidence": json.loads(best["evidence_json"]),
        "jaccard": best_score,
        "question": best["question"],
        "retrieved_at": best["retrieved_at"],
        "age_days": max(
            0, (_utc_now() - _parse_retrieved_at(best["retrieved_at"])).days
        ),
    }


# reusable evidence pack for `question` (or None); see lookup_evidence_detail
def lookup_evidence(question: str, ttl_days: int) -> dict | None:
    detail = lookup_evidence_detail(question, ttl_days)
    return detail["evidence"] if detail is not None else None
