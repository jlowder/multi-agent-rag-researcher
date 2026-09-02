"""Tests for promoting undelimited display-equation spans to equation blocks.

The writer sometimes emits a display equation as a bare paragraph span (no
$/$$, no \\(\\[ delimiters); paperbot only typesets delimited math or
equation blocks, so such a span printed raw TeX. The structured assembly
now promotes those spans (deep_research_structured._promote_bare_equation_spans).
"""

from deep_research_structured import (
    _is_bare_equation,
    _promote_bare_equation_spans,
    assemble_structured_report,
)
from models.report_schema import (
    BlockType,
    Metadata,
    QualityMetrics,
    Report,
    ReportBlock,
    ResearchReport,
    Section,
    Span,
)

SFT_EQ = r"L^{SFT} = \mathbb{E}_{x\sim X,\,y\sim p_T(y|x)}\!\big[-\log p_S(y|x)\big]"


def _report(*blocks) -> ResearchReport:
    section = Section(id="s1", heading="H", blocks=list(blocks))
    return ResearchReport(
        report=Report(
            metadata=Metadata(
                title="T",
                query="q",
                session_id="s",
                generated_at="2025-01-01T00:00:00Z",
            ),
            executive_summary=[],
            sections=[section],
            sources=[],
        ),
        quality=QualityMetrics(),
    )


def _para(*spans) -> ReportBlock:
    return ReportBlock(type=BlockType.paragraph, spans=list(spans))


class TestIsBareEquation:
    def test_promotes_undelimited_latex(self):
        assert _is_bare_equation(SFT_EQ)

    def test_rejects_delimited(self):
        assert not _is_bare_equation("$x^2$")
        assert not _is_bare_equation("$$" + SFT_EQ + "$$")
        assert not _is_bare_equation(r"\(x = y\)")
        assert not _is_bare_equation(r"\[x = y\]")

    def test_rejects_plain_prose(self):
        assert not _is_bare_equation("Loss functions trade off accuracy.")

    def test_rejects_prose_with_embedded_command(self):
        assert not _is_bare_equation("we set \\alpha = 0.5 for the model")


class TestPromote:
    def test_three_span_paragraph_splits(self):
        # The exact shape from the distillation report: prose, bare equation,
        # prose with delimited inline math.
        rep = _report(
            _para(
                Span(
                    text="The simplest objective is supervised fine-tuning (SFT), "
                    "which trains the student to reproduce the sequences the "
                    "teacher emits rather than matching an entire distribution.",
                    citations=[],
                ),
                Span(text=SFT_EQ, citations=[]),
                Span(
                    text="Here $y$ is the teacher's output sequence and $p_S$ the "
                    "student's conditional distribution [D30].",
                    citations=["34"],
                ),
            )
        )
        assert _promote_bare_equation_spans(rep) == 1
        blocks = rep.report.sections[0].blocks
        assert [b.type for b in blocks] == [
            BlockType.paragraph,
            BlockType.equation,
            BlockType.paragraph,
        ]
        assert blocks[0].spans[0].text.startswith("The simplest objective")
        assert blocks[1].text == SFT_EQ
        assert blocks[1].language == "latex"
        assert blocks[2].spans[0].text.startswith("Here $y$")
        assert blocks[2].spans[0].citations == ["34"]

    def test_single_span_paragraph_becomes_equation(self):
        rep = _report(_para(Span(text=r"\alpha = \beta / T", citations=[])))
        assert _promote_bare_equation_spans(rep) == 1
        blocks = rep.report.sections[0].blocks
        assert len(blocks) == 1
        assert blocks[0].type == BlockType.equation
        assert blocks[0].text == r"\alpha = \beta / T"

    def test_prose_span_unchanged(self):
        rep = _report(_para(Span(text="we set \\alpha = 0.5 for the model", citations=[])))
        assert _promote_bare_equation_spans(rep) == 0
        block = rep.report.sections[0].blocks[0]
        assert block.type == BlockType.paragraph
        assert block.spans[0].text == "we set \\alpha = 0.5 for the model"

    def test_delimited_span_unchanged(self):
        rep = _report(_para(Span(text="$x^2$", citations=[])))
        assert _promote_bare_equation_spans(rep) == 0
        block = rep.report.sections[0].blocks[0]
        assert block.type == BlockType.paragraph
        assert block.spans[0].text == "$x^2$"

    def test_assemble_promotes(self):
        s1 = Section(
            id="s1",
            heading="H",
            blocks=[
                _para(
                    Span(text="SFT fits the teacher's emissions. " + "word " * 30, citations=[]),
                    Span(text=SFT_EQ, citations=[]),
                )
            ],
        )
        rep = assemble_structured_report(
            sections=[s1],
            registry={},
            user_query="q",
            session_id="s",
            exec_paragraphs=["Ex."],
            verification_status={},
            title="T",
        )
        types = [b.type for b in rep.report.sections[0].blocks]
        assert BlockType.equation in types
