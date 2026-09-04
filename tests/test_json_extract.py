"""Unit tests for utils.json_extract — preamble/fence-tolerant JSON recovery.

Covers the shapes observed from chatty local models (Ornith on the MLX
server): conversational preamble, postamble, markdown fences, decoy objects
in prose, truncation, and the exact failure shapes reported for the
verifier critic and sufficiency evaluation.
"""

import json

from utils.json_extract import extract_json_payload


def test_bare_object():
    assert extract_json_payload('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_bare_array():
    assert extract_json_payload('[{"a": 1}, {"a": 2}]') == [{"a": 1}, {"a": 2}]


def test_bare_primitive_array():
    assert extract_json_payload("[1, 2, 3]") == [1, 2, 3]


def test_leading_prose_plus_json():
    text = "Let me evaluate this carefully before answering.\n" + '{"ok": true}'
    assert extract_json_payload(text) == {"ok": True}


def test_json_plus_trailing_prose():
    text = '{"ok": true}\nI hope that helps — let me know if you need more.'
    assert extract_json_payload(text) == {"ok": True}


def test_json_code_fence():
    assert extract_json_payload('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_payload('```\n{"a": 1}\n```') == {"a": 1}


def test_fence_with_prose_around():
    text = "Here you go:\n```json\n{\"a\": 1}\n```\nLet me know!"
    assert extract_json_payload(text) == {"a": 1}


def test_decoy_smaller_first_larger_wins():
    small = '{"n": 1}'
    big = json.dumps({"n": 2, "pad": "x" * 100})
    assert extract_json_payload(small + "\n" + big) == json.loads(big)
    # Order must not matter: the larger decoded span wins either way.
    assert extract_json_payload(big + "\n" + small) == json.loads(big)


def test_user_shape_critic_preamble_and_postamble():
    # The reported VerificationReport failure shape: "Let me carefully
    # evaluat..." preamble and "... n31. [W45] (GNS derived" postamble.
    payload = {
        "is_supported": True,
        "hallucinated_claims": [],
        "unsupported_claims": [],
        "per_section": [
            {
                "section_id": "s1",
                "grounded": True,
                "depth_ok": True,
                "citation_density_ok": True,
                "gaps": [],
                "expand_queries": [],
            }
        ],
        "confidence_level": "medium",
        "re_retrieve_suggested": False,
        "specific_queries": [],
    }
    text = (
        "Let me carefully evaluate the report section by section.\n"
        + json.dumps(payload)
        + "\nOverall the draft is solid; see section 31. [W45] (GNS derived features"
    )
    assert extract_json_payload(text) == payload


def test_user_shape_sufficiency_preamble_and_postamble():
    # The reported SufficiencyReport failure shape: "The user wants me to
    # eva..." preamble and "... type III factors, etc.)" postamble.
    payload = {
        "is_sufficient": False,
        "missing_aspects": ["calibration under distribution drift"],
        "follow_up_queries": ["calibration drift evaluation study"],
    }
    text = (
        "The user wants me to evaluate whether the evidence is sufficient.\n"
        + json.dumps(payload)
        + "\nMissing aspects include calibration, drift coverage, and type III factors, etc.)"
    )
    assert extract_json_payload(text) == payload


def test_truncated_json_returns_none():
    assert extract_json_payload('{"a": 1, "b": ') is None
    assert extract_json_payload('preamble prose {"a": 1, "b": ') is None


def test_pure_prose_returns_none():
    assert extract_json_payload("No JSON here at all, just words.") is None
    assert extract_json_payload("") is None
    assert extract_json_payload(None) is None


def test_brackets_in_prose_do_not_break_extraction():
    text = "See [W45] and [12] for details.\n{\"ok\": true}\nDone (see [99])."
    assert extract_json_payload(text) == {"ok": True}


def test_span_offset_points_past_chosen_value():
    from utils.json_extract import extract_json_payload_span

    text = "intro\n" + '{"a": 1}' + "\noutro"
    value, end = extract_json_payload_span(text)
    assert value == {"a": 1}
    assert text.strip()[end - 1] == "}"
