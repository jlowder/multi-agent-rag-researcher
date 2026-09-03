"""Shared, preamble/fence-tolerant JSON extraction for LLM output.

Local models (e.g. Ornith on an OpenAI-compatible MLX server) often wrap
the requested JSON payload in conversational preamble/postamble or markdown
code fences and do not honor the strict ``json_schema`` ``text.format`` the
SDK sends. The openai SDK's ``responses.parse`` parses client-side with
``model_validate_json(full_text)`` and *raises* on any preamble, before any
agent fallback can see the raw text.

``extract_json_payload`` recovers the JSON payload from such text using only
the standard library. It generalizes
``worker_agents.writer_agent._extract_json_object_span`` (the one
preamble-proof path, which now delegates here) so every caller shares a
single implementation. Standard library only: no third-party dependencies.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence, Tuple

__all__ = ["extract_json_payload", "extract_json_payload_span"]

# At most one leading fence line ("```json" / "```" with an optional language
# tag) and at most one trailing fence line.
_FENCE_OPEN_RE = re.compile(r"^\s*```[A-Za-z0-9_+-]*[ \t]*\r?\n?")
_FENCE_CLOSE_RE = re.compile(r"\r?\n?```\s*$")

_JSON_DECODER = json.JSONDecoder()


def _strip_code_fence(text: str) -> str:
    """Strip at most one leading and one trailing markdown fence line."""
    candidate = text
    open_match = _FENCE_OPEN_RE.match(candidate)
    if open_match and open_match.end() > 0:
        candidate = candidate[open_match.end():]
    close_match = _FENCE_CLOSE_RE.search(candidate)
    if close_match:
        candidate = candidate[: close_match.start()]
    return candidate.strip()


def _collect_top_level(candidate: str) -> list[Tuple[Any, int, int]]:
    """Decode a JSON value at every "{" / "[" index; keep TOP-LEVEL ones.

    A value whose start index falls inside a previously-accepted span is a
    part of that value (nested dict/list), not a competing payload, so it is
    skipped. Returns a list of (value, start, end) tuples.
    """
    accepted: list[Tuple[Any, int, int]] = []
    accepted_spans: list[Tuple[int, int]] = []
    for idx, ch in enumerate(candidate):
        if ch not in "{[":
            continue
        try:
            value, end = _JSON_DECODER.raw_decode(candidate, idx)
        except ValueError:
            continue
        if any(start <= idx < end for start, end in accepted_spans):
            continue
        accepted.append((value, idx, end))
        accepted_spans.append((idx, end))
    return accepted


def extract_json_payload_span(
    text: Optional[str],
    prefer_keys: Optional[Sequence[str]] = None,
) -> Tuple[Any, int]:
    """Extract the best JSON object/array from raw LLM text.

    Returns ``(value, end)`` where ``end`` is the index in the de-fenced,
    stripped text just past the chosen value, or ``(None, -1)`` when nothing
    usable was found. Handles: bare JSON, leading prose, trailing prose, one
    markdown code fence (language tag optional), and decoy/partial objects
    inside prose. Never raises.

    Selection rule: the LARGEST valid top-level value by decoded span wins
    (a tiny decoy object in the preamble cannot steal the response). When
    ``prefer_keys`` is given, a top-level object carrying any of those keys
    is preferred over other objects (a single-element object array is
    unwrapped for this check) — this reproduces the writer's original
    "blocks"/"heading" rule; if no such object exists, the first top-level
    object wins (writer parity), else the largest value overall.
    """
    if not text:
        return None, -1
    candidate = _strip_code_fence(text)
    if not candidate:
        return None, -1

    # Fast path: the whole (de-fenced) text is one JSON value.
    try:
        value = json.loads(candidate)
    except ValueError:
        value = None
    if isinstance(value, (dict, list)):
        return value, len(candidate)

    accepted = _collect_top_level(candidate)
    if not accepted:
        return None, -1

    if prefer_keys is not None:
        # Writer parity: unwrap a single-element object array so a model that
        # wraps the one object in [ ... ] still surfaces the object.
        expanded = []
        for value, start, end in accepted:
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
                expanded.append((value[0], start, end))
            else:
                expanded.append((value, start, end))
        keyed = [
            item
            for item in expanded
            if isinstance(item[0], dict) and any(key in item[0] for key in prefer_keys)
        ]
        if keyed:
            pool = keyed
        elif any(isinstance(item[0], dict) for item in expanded):
            for value, _start, end in expanded:
                if isinstance(value, dict):
                    return value, end
            pool = []
        else:
            pool = accepted
    else:
        pool = accepted
    if not pool:
        return None, -1
    best = max(pool, key=lambda item: item[2] - item[1])
    return best[0], best[2]


def extract_json_payload(text: Optional[str]) -> Optional[Any]:
    """Extract a JSON object or array from raw LLM text, or ``None``.

    See :func:`extract_json_payload_span` for the handled shapes and the
    largest-span selection rule. Never raises.
    """
    value, _end = extract_json_payload_span(text)
    return value
