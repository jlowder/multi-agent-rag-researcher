"""Undelimited-LaTeX sanitization (model-agnostic math hygiene).

Weak models emit raw LaTeX runs inside prose spans (|\\psi\\rangle,
2^{N}-dimensional, S_A = S_B) which KaTeX cannot typeset; the assembly pass
wraps each run in $...$. Conservative by design: prose must never be
captured into math.
"""

from deep_research_structured import (
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
    assert k == 2
    assert (
        "$|\\psi\\rangle=\\sum_{i=1}^{k} \\lambda_i |u_i\\rangle\\otimes|v_i\\rangle$"
        in new
    )
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
