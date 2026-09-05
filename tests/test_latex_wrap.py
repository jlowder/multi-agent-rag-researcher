"""Undelimited-LaTeX sanitization (model-agnostic math hygiene).

Weak models emit raw LaTeX runs inside prose spans (|\\psi\\rangle,
2^{N}-dimensional, S_A = S_B) which KaTeX cannot typeset; the assembly pass
wraps each run in $...$. Conservative by design: prose must never be
captured into math.
"""

from deep_research_structured import (
    _heal_malformed_math_regions,
    _wrap_latex_in_text,
    _wrap_undelimited_latex,
)
from models.report_schema import (
    BlockType,
    ResearchReport,
    ReportBlock,
    Span,
)

METADATA = {
    "title": "T",
    "query": "q",
    "session_id": "s1",
    "generated_at": "2026-01-01T00:00:00Z",
}


def _para(text):
    return ReportBlock(
        type=BlockType.paragraph,
        spans=[Span(text=text, citations=[])],
    )


def _report(*blocks):
    return ResearchReport.model_validate(
        {
            "schema_version": "1.0",
            "report": {
                "metadata": METADATA,
                "sections": [
                    {
                        "id": "s1",
                        "heading": "S",
                        "blocks": [b.model_dump() for b in blocks],
                    }
                ],
                "sources": [],
            },
            "quality": {},
        }
    )


# ---------------------------------------------------------------------------
# text-level scanner
# ---------------------------------------------------------------------------


def test_ket_sum_run_wrapped_as_one_run():
    # real shape from the hilbert report
    t = (
        "a canonical form "
        + "|\\psi\\rangle=\\sum_{i=1}^{k} \\lambda_i |u_i\\rangle\\otimes|v_i\\rangle where "
        + "{\\lambda_i} are coefficients"
    )
    new, k = _wrap_latex_in_text(t)
    # unit semantics: the pre-ket part of the sum, the lambda+ket/tensor
    # unit, and the lone {λ_i} are three runs — all fully wrapped, prose
    # intact, and every |…\rangle unit keeps its opener inside the math
    assert k == 3
    assert "$|\\psi\\rangle=\\sum_{i=1}^{k}$" in new
    assert "$\\lambda_i |u_i\\rangle\\otimes|v_i\\rangle$" in new
    assert new.endswith("${\\lambda_i}$ are coefficients")
    # no prose captured: the words around the runs are intact
    assert new.startswith("a canonical form $")


def test_superscript_with_hyphenated_word_keeps_hyphen_outside():
    new, k = _wrap_latex_in_text("the state lives in a 2^{N}-dimensional space")
    assert k == 1
    assert "$2^{N}$-dimensional" in new


def test_subscript_equation_split_across_operators():
    new, k = _wrap_latex_in_text("Same spectrum, S_A = S_B, and the entropy vanishes.")
    assert k == 2
    assert "$S_A$ = $S_B$, and" in new


def test_braced_dagger_product_wrapped():
    new, k = _wrap_latex_in_text(
        "a unitary U satisfies U^{\\dagger}U=I, where U is invertible"
    )
    assert "$U^{\\dagger}U=I$," in new
    assert k == 1


def test_sentence_final_period_stays_outside_math():
    new, _ = _wrap_latex_in_text("their Hilbert spaces, H^{AB}=H^{A}\\otimes H^{B}.")
    assert "$H^{AB}=H^{A}\\otimes H^{B}$." in new


def test_log_sum_run_merged_across_spaces():
    new, _ = _wrap_latex_in_text(
        "entropy S_A = -\\sum_n p_n \\log p_n, where p_n are coefficients."
    )
    assert "$\\sum_n p_n \\log p_n$," in new
    assert " -" in new  # the minus stays in the prose (not a run starter)


def test_bare_caret_ordinal_and_hyphenated_prose_untouched():
    for t in ("a 3^rd party review", "state-of-the-art method", "x-ray imaging"):
        new, k = _wrap_latex_in_text(t)
        assert k == 0 and new == t, t


def test_already_delimited_math_is_protected():
    for t in (
        "grows as $e^{rt}$ over time",
        "uses \\(\\alpha\\) here",
        "with $$w = A$$ below",
    ):
        new, k = _wrap_latex_in_text(t)
        assert k == 0 and new == t, t


def test_plain_prose_untouched():
    for t in (
        "The area is 400\u20132300\u202fnm and the range is wide.",
        "Control qubits Ai and Si encode outcomes m1,i and m2,i.",
        "Version 1.2.3 shipped in March 2025.",
    ):
        new, k = _wrap_latex_in_text(t)
        assert k == 0 and new == t, t


def test_single_lowercase_base_arg_stays_math():
    new, k = _wrap_latex_in_text("the ratio x^2/|x| is bounded")
    assert k == 1
    assert "$x^2/|x|$ is bounded" in new


def test_idempotent_second_pass_is_noop():
    t = "a 2^{N}-dimensional space with S_A = S_B entropy"
    once, k1 = _wrap_latex_in_text(t)
    twice, k2 = _wrap_latex_in_text(once)
    assert k1 == 3 and k2 == 0 and twice == once


def test_never_raises_on_weird_input():
    assert _wrap_latex_in_text("") == ("", 0)
    new, _ = _wrap_latex_in_text("lone backslash \\ at end")
    assert isinstance(new, str)


# ---------------------------------------------------------------------------
# report-level pass
# ---------------------------------------------------------------------------


def test_report_level_wraps_spans_cells_and_exec_summary():
    rep = _report(_para("the joint state lives in a 2^{N}-dimensional space"))
    rep.report.executive_summary = [
        "Entanglement S_A = S_B for symmetric pairs."
    ]
    rep.report.sections[0].blocks.append(
        ReportBlock(
            type=BlockType.comparison_table,
            columns=["a", "b"],
            rows=[
                [
                    Span(text="correction X^{m1,i} Z^{m2,i}", citations=[]),
                    Span(text="ok", citations=[]),
                ],
                [Span(text="ok2", citations=[]), Span(text="ok3", citations=[])],
            ],
        )
    )
    changed = _wrap_undelimited_latex(rep)
    assert changed == 3  # span + exec paragraph + cell
    assert any(
        "$2^{N}$-dimensional"
        in (s.text or "")
        for s in rep.report.sections[0].blocks[0].spans
    )
    assert rep.report.executive_summary[0] == (
        "Entanglement $S_A$ = $S_B$ for symmetric pairs."
    )
    cell_text = rep.report.sections[0].blocks[1].rows[0][0].text
    assert cell_text == "correction $X^{m1,i} Z^{m2,i}$"
    # idempotent
    assert _wrap_undelimited_latex(rep) == 0


def test_report_level_is_noop_when_no_math():
    rep = _report(_para("pure prose with no math at all"))
    assert _wrap_undelimited_latex(rep) == 0

# ---------------------------------------------------------------------------
# ket/bra units: |…\rangle and \langle…| must become single math runs
# ---------------------------------------------------------------------------


def test_ket_units_fully_wrapped():
    """Bell span: every |…\\rangle is one run INCLUDING the | opener."""
    text = (
        "The maximally entangled Bell state |\\Phi^+\\rangle = "
        "(|00\\rangle + |11\\rangle)/\\sqrt{2} and each term has equal weight."
    )
    new, _ = _wrap_latex_in_text(text)
    assert "$|\\Phi^+\\rangle$" in new
    assert "$|00\\rangle$" in new
    assert "$|11\\rangle$" in new
    assert "$\\sqrt{2}$" in new
    assert " and each term has equal weight." in new
    new2, _ = _wrap_latex_in_text(new)
    assert new2 == new


def test_prose_pipe_without_partner_untouched():
    text = "Filter A | B gives C when the input matches A | B or none at all."
    new, n = _wrap_latex_in_text(text)
    assert new == text
    assert n == 0


def test_bra_unit_fully_wrapped():
    text = (
        "For any state |\\psi\\rangle the dual |\\psi\\rangle is "
        "\\langle\\psi| and that is all."
    )
    new, _ = _wrap_latex_in_text(text)
    assert "$|\\psi\\rangle$" in new
    assert "$\\langle\\psi|$" in new
    assert " and that is all." in new


def test_sandwich_bra_ket_runs_valid():
    text = "the inner product \\langle\\psi|\\hat{U}|\\phi\\rangle is unitary here"
    new, _ = _wrap_latex_in_text(text)
    # the closing | is followed by a math token, so the whole sandwich is one run
    assert "$\\langle\\psi|\\hat{U}|\\phi\\rangle$" in new
    assert " is unitary here" in new
    new2, _ = _wrap_latex_in_text(new)
    assert new2 == new


def test_ket_with_interior_space():
    text = "the state | 00\\rangle plus | 11\\rangle here"
    new, _ = _wrap_latex_in_text(text)
    assert "$| 00\\rangle$" in new
    assert "$| 11\\rangle$" in new


def test_pipe_far_from_partner_not_captured():
    text = "Column A | B | C" + " " * 70 + "ends with \\rangle somewhere"
    new, _ = _wrap_latex_in_text(text)
    assert "A | B | C" in new
    assert "$A" not in new


def test_ket_idempotent_and_mixed_prose():
    text = "Prepare |\\phi\\rangle, apply U, then measure in the computational basis."
    a, _ = _wrap_latex_in_text(text)
    b, _ = _wrap_latex_in_text(a)
    assert a == b
    assert "$|\\phi\\rangle$" in a
    assert " apply U, then measure in the computational basis." in a


# ---------------------------------------------------------------------------
# post-wrap validity gate (_heal_malformed_math_regions)
# ---------------------------------------------------------------------------

BELL = (
    "A concrete illustration is the Bell state $\\ket{\\Phi^+}="
    "\\frac{1}{\\sqrt{2}}\\bigl(|00\\rangle+|11\\rangle\\bigr$) is maximal."
)
BASIS = (
    "Specifying a basis \\{$|\\psi_k\\rangle\\}_{k=1}^{d_1d_2$} and assigning "
    "each basis vector to a subsystem."
)
KRAUS = (
    "The maps must satisfy K'_i = $\\sum_{j} V_{$ij} K_j, yielding the "
    "same quadratic form."
)


def test_gate_bell_dangling_bigr_heals_to_raw():
    out, n = _heal_malformed_math_regions(BELL)
    assert n == 1
    assert "$" not in out
    assert "\\bigl(|00\\rangle+|11\\rangle\\bigr)" in out  # raw, parens kept


def test_gate_basis_escaped_brace_unbalanced_heals_to_raw():
    out, n = _heal_malformed_math_regions(BASIS)
    assert n == 1
    assert "$" not in out
    assert "^{d_1d_2} and assigning" in out  # raw text, delimiters stripped


def test_gate_kraus_unclosed_brace_heals_to_raw():
    out, n = _heal_malformed_math_regions(KRAUS)
    assert n == 1
    assert "$" not in out
    assert "\\sum_{j} V_{ij} K_j" in out


def test_gate_nested_dollar_resolves_no_double():
    out, n = _heal_malformed_math_regions("val $$a$ b$ end")
    assert n >= 1
    assert "$$" not in out
    assert "$" not in out
    assert "val a b end" in out


def test_gate_valid_left_right_equation_kept():
    text = (
        "The bound $\\left(\\sum_{j=0}^{t} 3^{j}\\binom{n}{j}\\right)2^{k}"
        "\\leq 2^{n}$ is tight."
    )
    out, n = _heal_malformed_math_regions(text)
    assert n == 0
    assert out == text  # byte-identical — a good equation must survive


def test_gate_valid_ket_and_simple_regions_kept():
    for text in (
        "Prepare $|\\psi\\rangle$ now.",
        "A $2^{N}$-dimensional space.",
        "Here $S_A=S_B$ exactly.",
    ):
        out, n = _heal_malformed_math_regions(text)
        assert n == 0
        assert out == text


def test_gate_idempotent():
    once, n1 = _heal_malformed_math_regions(BELL)
    twice, n2 = _heal_malformed_math_regions(once)
    assert twice == once
    assert n2 == 0


def test_full_pass_heals_model_emitted_malformed_regions():
    """Model-emitted malformed $..$ (wrap is a no-op on them) must still be
    healed to clean raw text by the report-level pass."""
    report = _report(_para(BELL), _para(BASIS), _para(KRAUS))
    _wrap_undelimited_latex(report)
    texts = [s.text for b in report.report.sections[0].blocks for s in b.spans]
    assert all("$" not in t for t in texts)


def test_full_pass_keeps_good_wraps_and_idempotent():
    report = _report(
        _para("Prepare |\\psi\\rangle in a $2^{N}$-dimensional space."),
        _para("Here $S_A=S_B$ exactly and $\\alpha$ decays."),
    )
    _wrap_undelimited_latex(report)
    texts = [s.text for b in report.report.sections[0].blocks for s in b.spans]
    assert "$|\\psi\\rangle$" in texts[0]        # still wrapped, not healed
    assert "$2^{N}$" in texts[0]
    assert "$S_A=S_B$" in texts[1]               # model's valid region kept
    _wrap_undelimited_latex(report)              # second pass: no churn
    texts2 = [s.text for b in report.report.sections[0].blocks for s in b.spans]
    assert texts2 == texts
