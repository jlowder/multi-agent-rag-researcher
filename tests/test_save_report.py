"""Unit tests for save_report module."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.save_report import (
    save_report,
    _sanitize_text,
    get_reports_directory,
    get_latest_report,
    ReportConfig,
)


class TestSanitization(unittest.TestCase):
    """Test sanitization functions."""
    
    def test_sanitize_text_valid_input(self):
        """Test _sanitize_text with valid input."""
        result = _sanitize_text("Hello, World!")
        self.assertEqual(result, "Hello, World!")
    
    def test_sanitize_text_null_bytes(self):
        """Test _sanitize_text removes null bytes."""
        result = _sanitize_text("Hello\x00World")
        self.assertNotIn("\x00", result)
    
    def test_sanitize_text_control_chars(self):
        """Test _sanitize_text removes control characters."""
        result = _sanitize_text("Hello\x1fWorld")
        self.assertNotIn("\x1f", result)
    
    def test_sanitize_text_truncation(self):
        """Test _sanitize_text truncates to max_length."""
        long_text = "A" * 10000
        result = _sanitize_text(long_text, max_length=100)
        self.assertLessEqual(len(result), 100)
    
    def test_sanitize_text_empty_string(self):
        """Test _sanitize_text with empty string."""
        result = _sanitize_text("")
        self.assertEqual(result, "")
    
    def test_sanitize_text_non_string(self):
        """Test _sanitize_text converts non-string input to str (baseline
        behavior: warns and converts, does not drop the value)."""
        result = _sanitize_text(12345)
        self.assertEqual(result, "12345")
    
    def test_sanitize_text_preserves_printable(self):
        """Test _sanitize_text preserves printable characters."""
        text = "Hello\nWorld\t!"
        result = _sanitize_text(text)
        self.assertIn("\n", result)
        self.assertIn("\t", result)


class TestSaveReport(unittest.TestCase):
    """Test save_report function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_content = "Test report content with meaningful information."
        self.test_query = "Test query about climate change impacts"
        self.test_session = "test_session_123"
    
    def test_save_report_creates_file(self):
        """Test that save_report creates a markdown file."""
        result = save_report(self.test_content, self.test_query, self.test_session)
        self.assertTrue(result.endswith('.md'))
        self.assertTrue(os.path.exists(result))
        # Clean up
        if os.path.exists(result):
            os.unlink(result)
    
    def test_save_report_invalid_content(self):
        """Test save_report handles non-string content."""
        result = save_report(12345, self.test_query, self.test_session)
        self.assertTrue(result.endswith('.md'))
        if os.path.exists(result):
            os.unlink(result)
    
    def test_save_report_empty_query(self):
        """Test save_report with empty query."""
        result = save_report(self.test_content, "", self.test_session)
        self.assertTrue("untitled" in result)
        if os.path.exists(result):
            os.unlink(result)
    
    def test_save_report_with_state_includes_evidence(self):
        """P0-5: with default config the body stays clean (no inline dump,
        no side file) even when evidence is present in state."""
        test_state = {
            "user_query": self.test_query,
            "final_answer": self.test_content,
            "evidence_json": json.dumps({
                "document_evidence": {
                    "chunks": [
                        {
                            "document_name": "nature_paper.pdf",
                            "document_title": "Biodiversity Loss",
                            "page_number": 3,
                            "chunk_id": "doc1",
                            "citation": "Smith et al., 2023, p. 3",
                            "content": "Climate change significantly impacts biodiversity.",
                            "score": 0.92
                        }
                    ]
                },
                "web_evidence": {
                    "results": [
                        {
                            "title": "Climate research",
                            "url": "https://example.com",
                            "content": "Research shows climate change effects.",
                            "score": 0.85
                        }
                    ]
                }
            }),
            "verification_status": {
                "confidence": 0.85,
                "coverage": "comprehensive",
                "gaps": []
            }
        }
        result = save_report(self.test_content, self.test_query, self.test_session, state=test_state)
        self.assertTrue(result.endswith('.md'))
        try:
            with open(result, 'r') as f:
                content = f.read()
            # Dump is disabled by default: no inline section, no pointer
            self.assertNotIn("## Supporting Evidence", content)
            self.assertNotIn("Full evidence dump", content)
            # No side file written
            stem = Path(result).stem
            self.assertFalse((Path(result).parent / f"{stem}.evidence.md").exists())
        finally:
            if os.path.exists(result):
                os.unlink(result)
    
    def test_save_report_evidence_side_file_when_enabled(self):
        """P0-5: include_evidence_dump=True writes {stem}.evidence.md next to
        the report (correctly attributed from real write-side fields) and a
        one-line pointer in the body; the dump is never inline."""
        test_state = {
            "evidence_json": json.dumps({
                "document_evidence": {
                    "chunks": [
                        {
                            "document_name": "nature_paper.pdf",
                            "document_title": "Biodiversity Loss",
                            "page_number": 3,
                            "chunk_id": "doc1",
                            "citation": "Smith et al., 2023, p. 3",
                            "content": "Climate change significantly impacts biodiversity.",
                            "score": 0.92
                        }
                    ]
                },
                "web_evidence": {
                    "results": [
                        {
                            "title": "Climate research",
                            "url": "https://example.com",
                            "content": "Research shows climate change effects.",
                            "score": 0.85
                        }
                    ]
                }
            })
        }
        result = save_report(
            self.test_content, self.test_query, self.test_session,
            state=test_state, config=ReportConfig(include_evidence_dump=True)
        )
        stem = Path(result).stem
        side = Path(result).parent / f"{stem}.evidence.md"
        try:
            with open(result, 'r') as f:
                content = f.read()
            # Body: one-line pointer, no inline dump
            self.assertIn(f"> Full evidence dump: `{stem}.evidence.md`", content)
            self.assertNotIn("## Supporting Evidence", content)
            self.assertNotIn("Climate change significantly impacts biodiversity.", content)
            # Side file exists with correctly attributed evidence
            self.assertTrue(side.exists())
            with open(side, 'r') as f:
                dump = f.read()
            self.assertIn("## Supporting Evidence", dump)
            self.assertIn("### Document Evidence", dump)
            self.assertIn("Biodiversity Loss", dump)
            self.assertIn("nature_paper.pdf", dump)
            self.assertIn("p. 3", dump)
            self.assertIn("Smith et al., 2023, p. 3", dump)
            self.assertIn("Climate change significantly impacts biodiversity.", dump)
            self.assertIn("### Web Evidence", dump)
            self.assertIn("Climate research", dump)
            self.assertIn("https://example.com", dump)
            self.assertIn("Research shows climate change effects.", dump)
            self.assertNotIn("Unknown", dump)
        finally:
            if side.exists():
                side.unlink()
            if os.path.exists(result):
                os.unlink(result)


class TestReportStructure(unittest.TestCase):
    """Test markdown report structure."""
    
    def test_report_has_header(self):
        """Test report has proper header."""
        result = save_report("Test content", "Test query", "test")
        with open(result, 'r') as f:
            content = f.read()
            self.assertIn("# Research Report", content)
            self.assertIn("**Query:** Test query", content)
            self.assertIn("**Date:**", content)
        if os.path.exists(result):
            os.unlink(result)
    
    def test_report_has_executive_summary(self):
        """Test report has executive summary section."""
        result = save_report("Test content", "Test query", "test")
        with open(result, 'r') as f:
            content = f.read()
            self.assertIn("## Executive Summary", content)
            self.assertIn("Test content", content)
        if os.path.exists(result):
            os.unlink(result)
    
    def test_report_has_evidence_section(self):
        """P0-5: with default config there is no inline evidence section in
        the body, even when evidence is present in state."""
        test_state = {
            "evidence_json": json.dumps({
                "document_evidence": {
                    "chunks": [{
                        "document_name": "test_source.pdf",
                        "document_title": "Test Source",
                        "page_number": 1,
                        "citation": "Test Citation",
                        "content": "Evidence content",
                        "score": 0.8
                    }]
                },
                "web_evidence": {
                    "results": [{
                        "title": "Web result",
                        "url": "https://test.com",
                        "content": "Web result content",
                        "score": 0.7
                    }]
                }
            })
        }
        result = save_report("Test content", "Test query", "test", state=test_state)
        try:
            with open(result, 'r') as f:
                content = f.read()
            self.assertNotIn("## Supporting Evidence", content)
            self.assertNotIn("Full evidence dump", content)
        finally:
            if os.path.exists(result):
                os.unlink(result)


class TestReportsDirectory(unittest.TestCase):
    """Test reports directory functions."""
    
    def test_get_reports_directory_returns_path(self):
        """Test get_reports_directory returns Path object."""
        result = get_reports_directory()
        self.assertIsInstance(result, Path)
        self.assertTrue(result.is_absolute())
    
    def test_get_latest_report(self):
        """Test get_latest_report returns most recent file."""
        # Create a test report first
        result = save_report("Test content", "Test query", "test")
        
        # Get latest report
        latest = get_latest_report()
        
        # Verify it exists
        self.assertIsNotNone(latest)
        self.assertTrue(latest.exists())
        
        # Clean up
        if os.path.exists(result):
            os.unlink(result)


class TestResearchConfig(unittest.TestCase):
    """P0-1: research() un-caps report bodies/verification; strict() still
    caps (regression guard)."""
    
    def test_research_body_uncapped(self):
        """P0-1a: research() leaves a >5000-char body intact."""
        body = "Alpha section one.\n" + ("A" * 6000) + "\nOmega closing paragraph."
        result = save_report(body, "uncap test", "test", config=ReportConfig.research())
        try:
            with open(result, 'r') as f:
                content = f.read()
            # Head and tail of the long body both survive (no truncation)
            self.assertIn("Alpha section one.", content)
            self.assertIn("Omega closing paragraph.", content)
            self.assertIn("A" * 6000, content)
        finally:
            if os.path.exists(result):
                os.unlink(result)
    
    def test_research_verification_uncapped(self):
        """P0-1b: research() leaves a >2000-char verification output intact."""
        state = {
            "verification": "V" * 3000,
            "evidence_json": "",
            "verification_status": {},
        }
        result = save_report(
            "Body.", "verif test", "test", state=state,
            config=ReportConfig.research()
        )
        try:
            with open(result, 'r') as f:
                content = f.read()
            self.assertIn("## Verification Output", content)
            self.assertIn("V" * 3000, content)
        finally:
            if os.path.exists(result):
                os.unlink(result)
    
    def test_research_strips_control_chars(self):
        """P0-1c: research() still strips control characters from the body."""
        body = "Start\x00End\nFinal line.\x01Tail"
        result = save_report(body, "ctrl test", "test", config=ReportConfig.research())
        try:
            with open(result, 'r') as f:
                content = f.read()
            self.assertNotIn("\x00", content)
            self.assertNotIn("\x01", content)
            self.assertIn("StartEnd\nFinal line.Tail", content)
        finally:
            if os.path.exists(result):
                os.unlink(result)
    
    def test_strict_still_caps_body(self):
        """P0-1d regression guard: strict() still truncates the body at 5000."""
        body = "Head marker.\n" + ("B" * 6000) + "\nTail marker."
        result = save_report(body, "strict cap test", "test", config=ReportConfig.strict())
        try:
            with open(result, 'r') as f:
                content = f.read()
            # Body truncated to exactly 5000 chars: 13 head + 4987 B's
            self.assertIn("B" * 4987, content)
            self.assertNotIn("B" * 4988, content)
            self.assertNotIn("Tail marker.", content)
        finally:
            if os.path.exists(result):
                os.unlink(result)
    
    def test_research_cli_identical_verification_not_duplicated(self):
        """CLI path: content == state["verification"] (final answer is the
        clean verification text) → the body is written exactly once and the
        duplicate "## Verification Output" section is skipped; control
        characters are still stripped."""
        first_line = "## Lead finding: scaling behavior in large models"
        body = first_line + "\n\n"
        body += "Scaling laws remain the dominant framework for capacity planning. " * 180
        body += "\n\nClosing paragraph with the final recommendation."
        self.assertGreater(len(body), 5000)
        raw = body + "\x00\x1f"
        state = {
            "verification": raw,
            "verification_status": {
                "confidence": 0.9,
                "coverage": "comprehensive",
                "gaps": []
            },
        }
        result = save_report(
            raw, "dup check", "test", state=state,
            config=ReportConfig.research()
        )
        try:
            with open(result, 'r') as f:
                content = f.read()
            # Body appears exactly once: the full text's first line occurs once
            self.assertEqual(content.count(first_line), 1)
            # No duplicate full-text section
            self.assertNotIn("## Verification Output", content)
            # Control char stripping still applies
            self.assertNotIn("\x00", content)
            self.assertNotIn("\x1f", content)
        finally:
            if os.path.exists(result):
                os.unlink(result)

    def test_strict_still_caps_verification(self):
        """Regression guard: strict() still caps verification output at 2000."""
        state = {"verification": "V" * 3000}
        result = save_report(
            "Body.", "strict verif", "test", state=state,
            config=ReportConfig.strict()
        )
        try:
            with open(result, 'r') as f:
                content = f.read()
            self.assertIn("V" * 2000, content)
            self.assertNotIn("V" * 2001, content)
        finally:
            if os.path.exists(result):
                os.unlink(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)