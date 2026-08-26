"""Phase 4 tests: render_markdown + save_structured_report (plan section 7)."""

import importlib
import json
from pathlib import Path

import pytest

import deep_research_orchestrator as dpo  # noqa: F401  (import-surface check)

wmod = importlib.import_module("worker_agents.writer_agent")
save_mod = importlib.import_module("memory.save_report")
from memory.save_report import render_markdown, save_structured_report
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
        assert "---" in md
        assert "> Evidence does not cover X." in md

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


def test_markdown_save_report_still_works(tmp_path):
    """Backward compatibility (plan 7.2): the old entry point is untouched."""
    path = save_report("Some content.", query="q", output_dir=tmp_path)
    p = Path(path)
    assert p.exists() and p.suffix == ".md"
    assert "Some content." in p.read_text()
