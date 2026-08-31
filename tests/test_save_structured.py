"""Phase 4 tests: render_markdown + save_structured_report (plan section 7)."""

import importlib
import json
import re
from pathlib import Path

import pytest

import deep_research_orchestrator as dpo  # noqa: F401  (import-surface check)
from deep_research_structured import parse_exec_summary

wmod = importlib.import_module("worker_agents.writer_agent")
save_mod = importlib.import_module("memory.save_report")
from memory.save_report import (
    render_markdown,
    save_structured_report,
    unescape_double_escaped_quotes,
)
from models.report_schema import (
    BlockType,
    Metadata,
    QualityMetrics,
    Report,
    ReportBlock,
    ResearchReport,
    Section,
    Source,
    Span,
)

save_report = save_mod.save_report


def _report(sources=None, exec_summary=None, sections=None):
    report = Report(
        metadata=Metadata(
            title="Diffusion Model Fundamentals",
            query="diffusion models",
            session_id="s1",
            generated_at="2025-01-01T00:00:00Z",
        ),
        executive_summary=list(exec_summary or []),
        sections=list(sections or []),
        sources=list(sources or []),
    )
    return ResearchReport(report=report, quality=QualityMetrics())


def _src(key, title, url="", type_="webpage"):
    return Source(
        id=f"source-{key.lower()}",
        type=type_,
        title=title,
        URL=url,
        citation_key=key,
    )


class TestRenderMarkdown:
    def test_title_exec_summary_paragraph(self):
        rep = _report(
            exec_summary=["First para.", "Second para."],
            sections=[
                Section(
                    id="s1",
                    heading="Forward Process",
                    blocks=[
                        ReportBlock(
                            type=BlockType.paragraph,
                            spans=[
                                Span(
                                    text="Diffusion models are generative models",
                                    citations=["1"],
                                ),
                                Span(text=" that learn noise.", citations=["1", "2"]),
                            ],
                        )
                    ],
                )
            ],
            sources=[
                _src("1", "A Math Intro"),
                _src("2", "TESS 2", url="https://t.co/x"),
            ],
        )
        md = render_markdown(rep)
        assert md.startswith("# Diffusion Model Fundamentals\n")
        assert "## Executive Summary" in md
        assert "First para." in md and "Second para." in md
        assert "### Forward Process" in md
        # Footnote markers: per-span, adjacent for multi-cite spans.
        assert "generative models[^1]" in md
        assert "learn noise.[^1][^2]" in md
        # Reference definitions in source order.
        assert "[^1]: A Math Intro" in md
        assert "[^2]: TESS 2" in md
        assert md.index("[^1]: A Math Intro") < md.index("[^2]: TESS 2")

    def test_all_block_types(self):
        section = Section(
            id="s1",
            heading="H",
            blocks=[
                ReportBlock(type=BlockType.heading, level=4, text="Sub", spans=[Span(text="Sub")]),
                ReportBlock(
                    type=BlockType.unordered_list,
                    items=[Span(text="one", citations=["1"]), Span(text="two")],
                ),
                ReportBlock(
                    type=BlockType.ordered_list,
                    items=[Span(text="first"), Span(text="second")],
                ),
                ReportBlock(
                    type=BlockType.callout,
                    callout_type="warning",
                    callout_title="Careful",
                    text="Do not do this.",
                ),
                ReportBlock(
                    type=BlockType.comparison_table,
                    columns=["A", "B"],
                    rows=[[Span(text="a1"), Span(text="b1")], [Span(text="a2"), Span(text="b2")]],
                ),
                ReportBlock(type=BlockType.code_block, language="python", text="x = 1"),
                ReportBlock(
                    type=BlockType.equation,
                    text="L = \\sum_i l_i",
                    language="latex",
                ),
                ReportBlock(type=BlockType.page_break),
                ReportBlock(
                    type=BlockType.citation_note,
                    text="Evidence does not cover X.",
                ),
            ],
        )
        md = render_markdown(_report(sections=[section], sources=[_src("1", "S")]))
        assert "#### Sub" in md
        assert "- one[^1]" in md
        assert "- two" in md
        assert "1. first" in md
        assert "2. second" in md
        assert "> **Careful:** Do not do this." in md
        assert "| A | B |" in md
        assert "| --- | --- |" in md
        assert "| a1 | b1 |" in md
        assert "```python\nx = 1\n```" in md
        assert "$$ L = \\sum_i l_i $$" in md
        assert "---" in md
        assert "> Evidence does not cover X." in md

    def test_equation_not_double_wrapped(self):
        # A model that wraps the equation text in $$ anyway must not get
        # $$…$$ inside $$…$$.
        section = Section(
            id="s1",
            heading="H",
            blocks=[
                ReportBlock(
                    type=BlockType.equation,
                    text="$$E = mc^2$$",
                    language="latex",
                )
            ],
        )
        md = render_markdown(_report(sections=[section]))
        assert "$$E = mc^2$$" in md
        assert "$$ $$" not in md

    def test_accepts_inner_report_and_rejects_garbage(self):
        rep = _report()
        inner = rep.report
        assert render_markdown(inner).startswith("# Diffusion Model Fundamentals")
        with pytest.raises(TypeError):
            render_markdown("not a report")

    def test_empty_report(self):
        md = render_markdown(_report())
        assert md == "# Diffusion Model Fundamentals\n"


class TestSaveStructuredReport:
    def test_writes_json_sources_markdown(self, tmp_path):
        rep = _report(
            exec_summary=["Para."],
            sections=[
                Section(
                    id="s1",
                    heading="S1",
                    blocks=[ReportBlock(
                        type=BlockType.paragraph,
                        spans=[Span(text="Fact.", citations=["1"])],
                    )],
                )
            ],
            sources=[_src("1", "Src One", url="https://x.test/1")],
        )
        returned = save_structured_report(rep, output_dir=tmp_path)
        returned_path = Path(returned)
        assert returned_path.suffix == ".json"
        assert returned_path.exists()
        stem = returned_path.stem
        assert (tmp_path / f"{stem}.sources.json").exists()
        assert (tmp_path / f"{stem}.markdown.md").exists()

        parsed = ResearchReport.model_validate_json(returned_path.read_text())
        assert parsed.report.metadata.title == "Diffusion Model Fundamentals"

        sources = json.loads((tmp_path / f"{stem}.sources.json").read_text())
        assert isinstance(sources, list)
        assert sources[0]["citation_key"] == "1"
        assert sources[0]["URL"] == "https://x.test/1"

        md = (tmp_path / f"{stem}.markdown.md").read_text()
        assert md.startswith("# Diffusion Model Fundamentals")
        assert "Fact.[^1]" in md

    def test_accepts_inner_report(self, tmp_path):
        rep = _report()
        returned = save_structured_report(rep.report, output_dir=tmp_path)
        assert ResearchReport.model_validate_json(Path(returned).read_text())

    def test_evidence_side_file_when_state_has_evidence(self, tmp_path):
        rep = _report()
        evidence = json.dumps(
            {
                "document_evidence": {"chunks": [
                    {
                        "document_name": "a.pdf",
                        "document_title": "A",
                        "chunk_id": "c1",
                        "content": "text",
                        "score": 0.9,
                    }
                ]},
                "web_evidence": {"results": [
                    {"title": "W", "url": "https://w.test", "content": "c", "score": 0.8},
                ]},
            }
        )
        from memory.save_report import ReportConfig

        cfg = ReportConfig.research()
        cfg.include_evidence_dump = True
        returned = save_structured_report(
            rep,
            output_dir=tmp_path,
            state={"evidence_json": evidence, "verification_status": {}},
            config=cfg,
        )
        stem = Path(returned).stem
        assert (tmp_path / f"{stem}.evidence.md").exists()

    def test_rejects_garbage(self, tmp_path):
        with pytest.raises(TypeError):
            save_structured_report("nope", output_dir=tmp_path)


class TestQuoteSanitizer:
    """Guard against writer-LLM double-escaped quotes (see save_structured_report)."""

    def test_unescapes_prose_and_skips_math(self):
        assert unescape_double_escaped_quotes('To \\"solve\\" an ODE') == 'To "solve" an ODE'
        assert unescape_double_escaped_quotes("x\\'s") == "x's"
        # composite: prose quotes fixed, TeX backslash inside math untouched
        assert (
            unescape_double_escaped_quotes('a \\"q\\" and $x \\ne y$ end')
            == 'a "q" and $x \\ne y$ end'
        )
        # no artifact present: byte-identical, TeX commands left alone
        unchanged = 'check $\\frac{a}{b}$ now'
        assert unescape_double_escaped_quotes(unchanged) == unchanged

    def test_save_sanitizes_spans_not_code(self, tmp_path):
        rep = _report(
            sections=[
                Section(
                    id="s1",
                    heading="S1",
                    blocks=[
                        ReportBlock(
                            type=BlockType.paragraph,
                            spans=[Span(text='To \\"solve\\" an ODE')],
                        ),
                        ReportBlock(
                            type=BlockType.code_block,
                            language="js",
                            text='alert(\\"hi\\")',  # backslashes must stay
                        ),
                    ],
                )
            ],
        )
        returned = save_structured_report(rep, output_dir=tmp_path)
        saved = json.loads(Path(returned).read_text())
        blocks = saved["report"]["sections"][0]["blocks"]
        assert blocks[0]["spans"][0]["text"] == 'To "solve" an ODE'
        assert blocks[1]["text"] == 'alert(\\"hi\\")', "code_block content must be untouched"
        md = Path(returned).with_suffix(".markdown.md").read_text()
        assert 'To "solve" an ODE' in md


def test_markdown_save_report_still_works(tmp_path):
    """Backward compatibility (plan 7.2): the old entry point is untouched."""
    path = save_report("Some content.", query="q", output_dir=tmp_path)
    p = Path(path)
    assert p.exists() and p.suffix == ".md"
    assert "Some content." in p.read_text()


class TestHeadingDedup:
    """Leading heading block that duplicates a section title must not render
    the title twice; genuine subsections always render."""

    def test_own_heading_variant_deduped(self):
        section = Section(
            id="s1",
            heading="Forward Process",
            blocks=[
                ReportBlock(type=BlockType.heading, level=3, text="  forward process:  "),
                ReportBlock(
                    type=BlockType.paragraph,
                    spans=[Span(text="Body.")],
                ),
            ],
        )
        md = render_markdown(_report(sections=[section]))
        assert md.count("### Forward Process") == 1
        assert "forward process" not in md  # the duplicate variant is gone

    def test_sibling_heading_deduped(self):
        s1 = Section(
            id="s1",
            heading="Forward Process",
            blocks=[ReportBlock(type=BlockType.paragraph, spans=[Span(text="One body.")])],
        )
        s2 = Section(
            id="s2",
            heading="Sampling",
            blocks=[
                ReportBlock(type=BlockType.heading, level=3, text="Forward Process"),
                ReportBlock(type=BlockType.paragraph, spans=[Span(text="Two body.")]),
            ],
        )
        md = render_markdown(_report(sections=[s1, s2]))
        assert md.count("### Forward Process") == 1  # only s1's own boundary
        assert md.count("### Sampling") == 1
        assert "Two body." in md

    def test_genuine_subsection_kept(self):
        section = Section(
            id="s1",
            heading="H",
            blocks=[
                ReportBlock(type=BlockType.heading, level=3, text="Reverse Steps"),
                ReportBlock(type=BlockType.paragraph, spans=[Span(text="Body.")]),
            ],
        )
        md = render_markdown(_report(sections=[section]))
        assert "### H" in md
        assert "### Reverse Steps" in md

    def test_injected_headings_neutralized(self):
        section = Section(
            id="s1",
            heading="S",
            blocks=[
                ReportBlock(
                    type=BlockType.paragraph,
                    spans=[Span(text="ok\n### Injected\nmore", citations=["1"])],
                ),
                ReportBlock(
                    type=BlockType.code_block,
                    language="python",
                    text="a = 1\n### Injected\nb = 2",
                ),
            ],
        )
        rep = _report(
            sections=[section],
            sources=[_src("1", "Evil\n## Injected\n\ntext")],
        )
        md = render_markdown(rep)
        # No line OUTSIDE code fences opens a heading from payload text
        # (inside a fence, newlines are literal content, not structure).
        outside = re.sub(r"```.*?```", "", md, flags=re.DOTALL)
        heading_lines = [l for l in outside.splitlines() if l.lstrip().startswith("#")]
        assert all("Injected" not in l for l in heading_lines)
        # ...inline instead: the span collapses to one line and keeps its marker.
        assert "ok ### Injected more[^1]" in md
        # Source labels collapse too (no heading line in references).
        assert "[^1]: Evil ## Injected text" in md
        # Code-block newlines are literal content and stay verbatim.
        assert "a = 1\n### Injected\nb = 2" in md


class TestParseExecSummary:
    """deep_research_structured.parse_exec_summary residue semantics."""

    def test_apology_with_json_array(self):
        text = 'I am sorry, [I failed] here is the summary: ["Para one.", "Para two."]'
        assert parse_exec_summary(text) == ["Para one.", "Para two."]

    def test_jsonish_garbage_discarded(self):
        assert parse_exec_summary("{not json") == []


def test_citation_token_newline_collapse():
    # F8: a newline inside a (legacy) citation token must not split the
    # [^n]: footnote definition line.
    section = Section(
        id="s1",
        heading="S",
        blocks=[
            ReportBlock(
                type=BlockType.paragraph,
                spans=[Span(text="Fact.", citations=["1\n"])],
            )
        ],
    )
    rep = _report(sections=[section], sources=[_src("1", "Source One")])
    md = render_markdown(rep)
    assert "Fact.[^1]" in md
    assert "[^1]: Source One" in md
    assert "[^1\n" not in md


def _retitled(rep, title: str, query: str):
    meta = rep.report.metadata.model_copy(update={"title": title, "query": query})
    rep.report = rep.report.model_copy(update={"metadata": meta})
    return rep


class TestSaveStructuredNaming:
    def test_title_names_all_artifacts(self, tmp_path):
        rep = _retitled(
            _report(),
            title="Comprehensive Report on Genetic Programming",
            query="Produce a comprehensive report on Genetic Programming and its successes",
        )
        returned = Path(save_structured_report(rep, output_dir=tmp_path))
        base = "comprehensive_report_on_genetic_programming"
        stem = returned.stem
        assert stem.startswith(base + "_")
        for p in tmp_path.iterdir():
            assert p.name.startswith(base + "_"), p.name
        assert {p.suffix for p in tmp_path.iterdir()} <= {".json", ".md"}

    def test_missing_title_falls_back_to_query(self, tmp_path):
        rep = _retitled(_report(), title="", query="my fallback query")
        returned = Path(save_structured_report(rep, output_dir=tmp_path))
        assert returned.stem.startswith("my fallback query_")

    def test_all_punctuation_title_falls_back_to_query(self, tmp_path):
        rep = _retitled(_report(), title="!!!", query="punctuated title query")
        returned = Path(save_structured_report(rep, output_dir=tmp_path))
        assert returned.stem.startswith("punctuated title query_")
