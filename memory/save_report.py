"""Save research reports to markdown files with rich formatting and structure.

This module provides functionality for generating high-quality, well-structured
markdown reports from research data, including support for evidence tracking,
verification status, and metadata enrichment.

## Features

- Structured markdown report generation with consistent formatting
- Evidence tracking from document chunks and web results
- Verification status reporting with confidence scores
- Automatic sanitization and validation of inputs
- File path management with unique timestamp-based naming
- Support for session-based reporting with metadata

## Examples

Basic usage:

```python
from memory.save_report import save_report

# Simple report
report_path = save_report(
    content="Research findings about LLM architectures...",
    query="What are modern LLM architectures?"
)

# Enhanced report with evidence (real retriever write-side fields)
state = {
    "evidence_json": json.dumps({
        "document_evidence": {
            "chunks": [{
                "document_name": "paper.pdf",
                "document_title": "Transformer Architecture",
                "page_number": 3,
                "chunk_id": "doc1",
                "citation": "Vaswani et al., 2017, p. 3",
                "content": "Transformer architecture...",
                "score": 0.95
            }]
        },
        "web_evidence": {
            "results": [{
                "title": "Recent advances",
                "url": "https://example.com",
                "content": "Recent advances...",
                "score": 0.88
            }]
        }
    }),
    "verification": "Cross-referenced with 3 sources",
    "verification_status": {
        "confidence": 0.92,
        "coverage": "comprehensive",
        "gaps": ["No data on 2025 models"]
    }
}

report_path = save_report(
    content="Summary of findings...",
    query="LLM architectures",
    session_id="session_123",
    state=state
)
```
"""

import json
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Tuple, Callable
)
from dataclasses import dataclass, field

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)

# =============================================================================
# Type Aliases
# =============================================================================

JsonDict = Dict[str, Any]
"""Type alias for JSON-compatible dictionaries."""

EvidenceChunk = Dict[str, Any]
"""Represents a single evidence chunk from documents."""

WebResult = Dict[str, Any]
"""Represents a single web search result."""

VerificationStatus = Dict[str, Any]
"""Represents verification status information."""

ReportMetadata = Dict[str, Any]
"""Metadata associated with a research report."""

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvidenceItem:
    """Represents a single piece of evidence with source attribution.
    
    Attributes:
        source: Source identifier (file path, URL, etc.)
        content: The actual evidence content
        score: Relevance/confidence score (0.0 to 1.0)
        source_type: Type of source ('document' or 'web')
        snippet_length: Maximum length of content to display
        title: Optional display title (document_title / web result title)
        page_number: Optional document page number
        citation: Optional exact citation string
    """
    source: str
    content: str
    score: float
    source_type: str = "document"
    snippet_length: int = 800
    title: Optional[str] = None
    page_number: Optional[int] = None
    citation: Optional[str] = None

    def to_markdown(self) -> str:
        """Convert evidence item to markdown format.

        Returns:
            Formatted markdown string for this evidence item
        """
        score_percent = self.score * 100
        badge = f"**{self.source_type.title()} Score:** {score_percent:.1f}%"

        parts: List[str] = []
        if self.title and self.title != self.source:
            parts.append(f"**{self.title}**")
        parts.append(self.source)
        if self.page_number is not None:
            parts.append(f"p. {self.page_number}")
        if self.citation:
            parts.append(f"Citation: {self.citation}")
        attribution = " | ".join(parts)

        return f"""### {badge}

{attribution}

```text
{self.content[:self.snippet_length]}
```"""


@dataclass 
class ReportConfig:
    """Configuration for report generation.

    Attributes:
        max_content_length: Maximum length for content fields
            (None disables body truncation)
        max_snippet_length: Maximum length for snippets
        max_gaps_display: Maximum number of gaps to display
        sanitize_control_chars: Whether to remove control characters
        preserve_whitespace: Whether to preserve whitespace formatting
        include_evidence_dump: Whether to write the evidence dump to a side
            file ({report_stem}.evidence.md) next to the report. The dump is
            never written inline in the report body.
    """
    max_content_length: Optional[int] = 10000
    max_snippet_length: int = 1000
    max_gaps_display: int = 5
    sanitize_control_chars: bool = True
    preserve_whitespace: bool = False
    include_evidence_dump: bool = False

    @classmethod
    def default(cls) -> "ReportConfig":
        """Get default configuration."""
        return cls()

    @classmethod
    def strict(cls) -> "ReportConfig":
        """Get strict configuration for security-sensitive contexts."""
        return cls(
            max_content_length=5000,
            max_snippet_length=500,
            sanitize_control_chars=True,
            preserve_whitespace=False
        )

    @classmethod
    def research(cls) -> "ReportConfig":
        """Get research configuration for long-form research reports.

        Body and verification output are uncapped (no mid-sentence
        truncation); evidence snippets are capped; control characters are
        still sanitized. The evidence dump stays off unless
        include_evidence_dump is set (side file next to the report).
        """
        return cls(
            max_content_length=None,
            max_snippet_length=800,
            sanitize_control_chars=True,
            preserve_whitespace=False
        )


# =============================================================================
# Utility Functions
# =============================================================================


def _sanitize_text(
    text: str,
    max_length: Optional[int] = 5000,
    sanitize_control_chars: bool = True
) -> str:
    """Sanitize text input to prevent injection attacks and excessive length.
    
    This function performs multiple sanitization steps to ensure safe text
    handling, including truncation, control character removal, and whitespace
    normalization.
    
    Args:
        text: The text to sanitize
        max_length: Maximum allowed length after truncation (None
            disables truncation)
        sanitize_control_chars: Whether to remove non-printable characters
        
    Returns:
        Sanitized and truncated text string
    """
    # Input validation
    if text is None:
        return ""
    
    if not isinstance(text, str):
        logger.warning(f"_sanitize_text: input is not a string, converting. Type: {type(text)}")
        text = str(text)
    
    # Truncate to max length (skipped when max_length is None)
    if max_length is not None and len(text) > max_length:
        logger.debug(f"Truncating text from {len(text)} to {max_length} characters")
        text = text[:max_length]
    
    # Remove null bytes and other control characters (preserve newlines and tabs)
    if sanitize_control_chars:
        text = ''.join(
            c for c in text 
            if c.isprintable() or c in '\n\t\r'
        )
    
    # Limit consecutive newlines to prevent excessive spacing
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def _whitespace_normalized(text: str) -> str:
    """Collapse all whitespace runs to single spaces (strip included).
    
    Used to compare the verification text against the main content so that
    trailing-newline / blank-line differences do not fool the duplicate check.
    
    Args:
        text: The text to normalize
        
    Returns:
        Whitespace-normalized text string
    """
    return " ".join(text.split())


def _validate_query(query: str) -> Tuple[bool, str]:
    """Validate query string for report generation.
    
    Args:
        query: The query string to validate
        
    Returns:
        Tuple of (is_valid, sanitized_query)
    """
    if not query:
        return True, "untitled"
    
    # Sanitize for filename safety
    safe = "".join(c for c in query if c.isalnum() or c in " -_").lower()
    
    if not safe:
        return True, "untitled"
    
    # Truncate to reasonable length for filename
    safe = safe[:60]
    return True, safe


def _parse_evidence_json(evidence_json: str) -> Tuple[List[EvidenceChunk], List[WebResult]]:
    """Parse evidence JSON string into structured data.
    
    Args:
        evidence_json: JSON string containing evidence data
        
    Returns:
        Tuple of (document_chunks, web_results) lists
    """
    document_chunks: List[EvidenceChunk] = []
    web_results: List[WebResult] = []
    
    if not evidence_json or not isinstance(evidence_json, str):
        logger.debug("No evidence JSON to parse")
        return document_chunks, web_results
    
    try:
        evidence_data = json.loads(evidence_json)
        
        if not isinstance(evidence_data, dict):
            logger.warning("Evidence JSON is not a dictionary")
            return document_chunks, web_results
        
        # Parse document evidence (real write-side fields from the
        # retriever: document_name/document_title/page_number/citation/
        # content/score — see REPORT_QUALITY_IMPROVEMENT_PLAN.md P0-5)
        doc_evidence = evidence_data.get("document_evidence", {})
        if isinstance(doc_evidence, dict):
            chunks = doc_evidence.get("chunks", [])
            if isinstance(chunks, list):
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        document_title = chunk.get("document_title") or chunk.get("document_name")
                        document_chunks.append({
                            "source": chunk.get("document_name") or document_title or "Unknown",
                            "title": document_title or "Unknown",
                            "page_number": chunk.get("page_number"),
                            "citation": chunk.get("citation", ""),
                            "content": chunk.get("content", "")[:1000],
                            "score": float(chunk.get("score", 0.5)) if chunk.get("score") else 0.5
                        })
        
        # Parse web evidence (real Tavily fields: title/url/content/score)
        web_evidence = evidence_data.get("web_evidence", {})
        if isinstance(web_evidence, dict):
            results = web_evidence.get("results", [])
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        web_results.append({
                            "url": result.get("url", "Unknown"),
                            "title": result.get("title", ""),
                            "content": result.get("content", "")[:500],
                            "score": float(result.get("score", 0.5)) if result.get("score") else 0.5
                        })
        
        logger.info(f"Parsed {len(document_chunks)} document chunks and {len(web_results)} web results")
        
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"Could not parse evidence JSON: {e}")
    
    return document_chunks, web_results


def _build_evidence_markdown(
    document_chunks: List[EvidenceChunk],
    web_results: List[WebResult],
    verification_status: VerificationStatus,
    config: ReportConfig
) -> str:
    """Render the evidence dump block for the side file.

    This is the content that used to be inlined in the report body as the
    "Supporting Evidence" section. When ReportConfig.include_evidence_dump
    is True it is written to {report_stem}.evidence.md next to the report;
    it is never inlined in the report body.

    Args:
        document_chunks: Parsed document evidence chunks
        web_results: Parsed web search results
        verification_status: Verification status dictionary (may be empty)
        config: Report configuration (snippet length, gaps display)
        
    Returns:
        Markdown string for the evidence side file
    """
    lines: List[str] = []
    
    lines.append("## Supporting Evidence")
    lines.append("")
    
    # Document Evidence
    if document_chunks:
        lines.append("### Document Evidence")
        lines.append("")
        for chunk in document_chunks:
            evidence_item = EvidenceItem(
                source=chunk["source"],
                content=chunk["content"],
                score=chunk["score"],
                source_type="document",
                title=chunk.get("title"),
                page_number=chunk.get("page_number"),
                citation=chunk.get("citation")
            )
            lines.append(evidence_item.to_markdown())
            lines.append("")
    
    # Web Evidence
    if web_results:
        lines.append("### Web Evidence")
        lines.append("")
        for result in web_results:
            evidence_item = EvidenceItem(
                source=result["url"],
                content=result["content"],
                score=result["score"],
                source_type="web",
                title=result.get("title"),
                snippet_length=config.max_snippet_length
            )
            lines.append(evidence_item.to_markdown())
            lines.append("")
    
    # Verification Summary
    if isinstance(verification_status, dict) and verification_status:
        lines.append("### Verification Summary")
        lines.append("")
        
        confidence_raw = verification_status.get("confidence", 0)
        
        # Parse confidence - handle both string ("high", "medium", "low") and float (0.0-1.0)
        confidence: float = 0.0
        if isinstance(confidence_raw, str):
            # Convert qualitative string to float
            confidence_map = {
                "high": 0.9,
                "medium": 0.6,
                "low": 0.3,
                "none": 0.0
            }
            confidence = confidence_map.get(confidence_raw.lower(), 0.5)
        elif isinstance(confidence_raw, (int, float)):
            # Convert numeric to float, ensuring it's in 0.0-1.0 range
            confidence = float(max(0.0, min(1.0, confidence_raw)))
        
        coverage = verification_status.get("coverage", "unknown")
        gaps = verification_status.get("gaps", [])
        
        lines.append(f"- **Confidence Level:** {confidence * 100:.1f}%")
        lines.append(f"- **Coverage:** {coverage}")
        
        if gaps:
            lines.append(f"- **Identified Gaps:** {len(gaps)}")
            lines.append("")
            
            gaps_to_show = gaps[:config.max_gaps_display]
            for gap in gaps_to_show:
                lines.append(f"  - {gap}")
            
            if len(gaps) > config.max_gaps_display:
                lines.append(f"  - ... and {len(gaps) - config.max_gaps_display} more gaps")
        
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Main Report Functions
# =============================================================================


def build_enriched_markdown(
    content: str,
    query: str,
    session_id: str = "default",
    state: Optional[JsonDict] = None,
    config: Optional[ReportConfig] = None,
    evidence_pointer: Optional[str] = None
) -> str:
    """Build a comprehensive markdown report with rich structure.
    
    This function creates a well-formatted markdown report that includes
    the main content, evidence from various sources, and verification status.
    
    Args:
        content: The main report content to include
        query: The original research query
        session_id: Session identifier for tracking
        state: Optional state dictionary containing evidence data
        config: Optional configuration for report generation
        evidence_pointer: Optional filename of the evidence side file
            (relative to the report), used for the one-line in-body
            pointer when the evidence dump is enabled
        
    Returns:
        Formatted markdown string ready for saving
    """
    if config is None:
        config = ReportConfig.default()
    
    # Sanitize inputs
    sanitized_content = _sanitize_text(
        content,
        max_length=config.max_content_length,
        sanitize_control_chars=config.sanitize_control_chars
    )
    
    sanitized_query = _sanitize_text(query, max_length=1000)
    
    sanitized_session_id = _sanitize_text(
        session_id,
        max_length=100,
        sanitize_control_chars=config.sanitize_control_chars
    )
    
    lines: List[str] = []
    timestamp = datetime.now(timezone.utc)
    
    # =============================================================================
    # Header Section
    # =============================================================================
    
    lines.append("# Research Report")
    lines.append("")
    lines.append(f"**Query:** {sanitized_query}")
    lines.append(f"**Date:** {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append(f"**Session ID:** {sanitized_session_id}")
    lines.append(f"**Generated:** {timestamp.isoformat()}")
    lines.append("")
    
    # =============================================================================
    # Executive Summary Section
    # =============================================================================
    
    lines.append("---")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(sanitized_content)
    lines.append("")
    
    # =============================================================================
    # Evidence Section (if state is provided)
    # =============================================================================
    
    if state and isinstance(state, dict):
        evidence_json = state.get("evidence_json", "")
        verification = state.get("verification", "")
        verification_status = state.get("verification_status", {})
        
        document_chunks, web_results = _parse_evidence_json(evidence_json)
        
        # Evidence dump (P0-5): the raw chunk/result dump is never written
        # inline in the body. When the config enables it and evidence
        # exists, save_report() writes the dump to a side file next to the
        # report and passes its filename here; the body gets a one-line
        # pointer instead.
        if config.include_evidence_dump and (document_chunks or web_results) and evidence_pointer:
            lines.append(f"> Full evidence dump: `{evidence_pointer}`")
            lines.append("")
        
        # Verification Output (raw)
        # Capped at 2000 chars for capped configs (default/strict); uncapped
        # when the body is uncapped (research config).
        if verification:
            verification_max_length = 2000 if config.max_content_length is not None else None
            sanitized_verification = _sanitize_text(
                verification,
                max_length=verification_max_length,
                sanitize_control_chars=config.sanitize_control_chars
            )
            
            # The CLI path passes content=final_answer, which is the same
            # clean text as state["verification"]; re-writing it under
            # "## Verification Output" would double the body length. Skip
            # the full-text section when the two are identical after
            # strip/whitespace normalization. (The report body renders no
            # other verification metadata; the evidence side file, when
            # enabled, is written independently and keeps its summary.)
            if (
                sanitized_verification
                and _whitespace_normalized(sanitized_verification)
                != _whitespace_normalized(sanitized_content)
            ):
                lines.append("## Verification Output")
                lines.append("")
                lines.append("```text")
                lines.append(sanitized_verification)
                lines.append("```")
                lines.append("")
    
    # =============================================================================
    # Footer Section
    # =============================================================================
    
    lines.append("---")
    lines.append("")
    lines.append("*This report was automatically generated by the AI Research Orchestrator.*")
    
    return "\n".join(lines)


def save_report(
    content: str,
    query: str = "",
    session_id: str = "default",
    state: Optional[JsonDict] = None,
    output_dir: Optional[Path] = None,
    config: Optional[ReportConfig] = None
) -> str:
    """Save a research report to a markdown file with comprehensive structure.

    Standard-mode path (Markdown). Deep mode saves via save_structured_report.    
    This function generates a richly formatted markdown report from the provided
    content and optional evidence state, then saves it to a timestamped file.
    
    Args:
        content: The report text to save. Will be sanitized and truncated.
        query: The original user query. Used for filename generation.
        session_id: The session identifier for tracking purposes.
        state: Optional state dictionary from orchestrator_agent containing
               evidence and other metadata for enriched reporting.
        output_dir: Optional custom output directory. Defaults to reports/ directory.
        config: Optional configuration for report generation.
        
    Returns:
        The absolute file path where the report was saved.
        
    Raises:
        OSError: If there's an error creating directories or writing the file.
        ValueError: If content is None or empty.
    """
    # Input validation
    if content is None:
        raise ValueError("Content cannot be None")
    
    if not isinstance(content, str):
        logger.warning("save_report: content is not a string, converting to string")
        content = str(content)
    
    if not content.strip():
        logger.warning("save_report: content is empty or whitespace-only")
    
    # Determine output directory
    if output_dir is None:
        reports_dir = Path(__file__).parent.parent / "reports"
    else:
        reports_dir = output_dir
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Reports directory: {reports_dir.absolute()}")
    
    # Sanitize and truncate query for filename
    is_valid, safe_query = _validate_query(query)
    
    if not is_valid:
        safe_query = "untitled"
    
    if not safe_query:
        safe_query = "untitled"
    
    # Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_query}_{timestamp}.md"
    filepath = reports_dir / filename
    
    cfg = config or ReportConfig.default()
    
    # Evidence side file (P0-5): when enabled and evidence is present, the
    # dump is written to {report_stem}.evidence.md next to the report and
    # the body gets a one-line pointer instead of the inline dump.
    evidence_path: Optional[Path] = None
    document_chunks: List[EvidenceChunk] = []
    web_results: List[WebResult] = []
    if cfg.include_evidence_dump and state and isinstance(state, dict):
        document_chunks, web_results = _parse_evidence_json(state.get("evidence_json", ""))
        if document_chunks or web_results:
            evidence_path = filepath.with_name(f"{filepath.stem}.evidence.md")
    
    # Build markdown content
    markdown_content = build_enriched_markdown(
        content=content,
        query=query,
        session_id=session_id,
        state=state,
        config=cfg,
        evidence_pointer=evidence_path.name if evidence_path is not None else None
    )
    
    # Write to file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        if evidence_path is not None:
            evidence_markdown = _build_evidence_markdown(
                document_chunks,
                web_results,
                state.get("verification_status", {}),
                cfg
            )
            with open(evidence_path, "w", encoding="utf-8") as f:
                f.write(evidence_markdown + "\n")
            logger.info(f"Evidence dump saved to: {evidence_path.absolute()}")
        
        logger.info(f"Markdown report saved to: {filepath.absolute()}")
        
    except OSError as e:
        logger.error(f"Failed to save markdown report to {filepath}: {e}")
        raise
    
    return str(filepath.absolute())


# =============================================================================
# Report Discovery Functions
# =============================================================================


def get_reports_directory(base_dir: Optional[Path] = None) -> Path:
    """Get the reports directory path.
    
    Args:
        base_dir: Base directory to use. Defaults to parent of this file's directory.
        
    Returns:
        Path object pointing to the reports directory
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    return reports_dir


def get_latest_report(base_dir: Optional[Path] = None) -> Optional[Path]:
    """Get the most recently created report.
    
    Args:
        base_dir: Base directory to search. Defaults to parent of this file's directory.
        
    Returns:
        Path to the latest report, or None if no reports exist.
    """
    reports_dir = get_reports_directory(base_dir)
    
    if not reports_dir.exists():
        logger.debug(f"Reports directory does not exist: {reports_dir}")
        return None
    
    md_files = list(reports_dir.glob("*.md"))
    
    if not md_files:
        logger.debug("No markdown reports found")
        return None
    
    latest = max(md_files, key=lambda p: p.stat().st_mtime)
    logger.debug(f"Latest report: {latest}")
    
    return latest


def get_reports_by_session(
    session_id: str,
    base_dir: Optional[Path] = None
) -> List[Path]:
    """Get all reports for a specific session.
    
    Args:
        session_id: Session identifier to search for
        base_dir: Base directory to search
        
    Returns:
        List of paths to matching report files, sorted by modification time (newest first)
    """
    reports_dir = get_reports_directory(base_dir)
    
    if not reports_dir.exists():
        return []
    
    # Find all markdown files (will filter by session_id later)
    md_files = list(reports_dir.glob("*.md"))
    
    # Filter and parse session IDs
    matching_reports: List[Tuple[Path, datetime]] = []
    
    for filepath in md_files:
        try:
            # Read file to check for session ID
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
                if f"**Session ID:** {session_id}" in content:
                    matching_reports.append((filepath, filepath.stat().st_mtime))
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Could not read {filepath}: {e}")
            continue
    
    # Sort by modification time (newest first)
    matching_reports.sort(key=lambda x: x[1], reverse=True)
    
    return [report[0] for report in matching_reports]


def get_reports_by_date_range(
    start_date: datetime,
    end_date: Optional[datetime] = None,
    base_dir: Optional[Path] = None
) -> List[Path]:
    """Get reports created within a date range.
    
    Args:
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive). Defaults to now.
        base_dir: Base directory to search
        
    Returns:
        List of paths to matching report files, sorted by modification time
    """
    if end_date is None:
        end_date = datetime.now()
    
    reports_dir = get_reports_directory(base_dir)
    
    if not reports_dir.exists():
        return []
    
    md_files = list(reports_dir.glob("*.md"))
    
    matching_reports: List[Path] = []
    
    for filepath in md_files:
        try:
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            
            if start_date <= mtime <= end_date:
                matching_reports.append(filepath)
        except OSError as e:
            logger.warning(f"Could not get mtime for {filepath}: {e}")
            continue
    
    # Sort by modification time (newest first)
    matching_reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return matching_reports


# =============================================================================
# Utility: Report Statistics
# =============================================================================


def get_report_statistics(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Get statistics about saved reports.
    
    Args:
        base_dir: Base directory to analyze
        
    Returns:
        Dictionary containing report statistics
    """
    reports_dir = get_reports_directory(base_dir)
    
    stats: Dict[str, Any] = {
        "total_reports": 0,
        "total_size_bytes": 0,
        "oldest_report": None,
        "newest_report": None,
        "reports_by_day": {},
        "error_count": 0
    }
    
    if not reports_dir.exists():
        return stats
    
    md_files = list(reports_dir.glob("*.md"))
    stats["total_reports"] = len(md_files)
    
    timestamps: List[Tuple[Path, datetime]] = []
    
    for filepath in md_files:
        try:
            size = filepath.stat().st_size
            mtime = filepath.stat().st_mtime
            
            stats["total_size_bytes"] += size
            
            # Track date
            day = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            stats["reports_by_day"][day] = stats["reports_by_day"].get(day, 0) + 1
            
            timestamps.append((filepath, datetime.fromtimestamp(mtime)))
            
        except OSError as e:
            stats["error_count"] += 1
            logger.warning(f"Could not process {filepath}: {e}")
            continue
    
    if timestamps:
        timestamps.sort(key=lambda x: x[1])
        
        oldest = timestamps[0]
        newest = timestamps[-1]
        
        stats["oldest_report"] = {
            "path": str(oldest[0]),
            "date": oldest[1].isoformat()
        }
        
        stats["newest_report"] = {
            "path": str(newest[0]),
            "date": newest[1].isoformat()
        }
    
    return stats


# =============================================================================
# Type Guards
# =============================================================================


def is_valid_evidence_state(state: Any) -> bool:
    """Check if state is a valid evidence dictionary.
    
    Args:
        state: The state to validate
        
    Returns:
        True if state appears to be a valid evidence dictionary
    """
    if not isinstance(state, dict):
        return False
    
    # Check for expected keys
    expected_keys = {"evidence_json", "verification", "verification_status"}
    actual_keys = set(state.keys())
    
    # At least one expected key should be present
    return bool(actual_keys & expected_keys)


def is_valid_report_path(filepath: Union[str, Path]) -> bool:
    """Check if filepath appears to be a valid report path.
    
    Args:
        filepath: Path to check
        
    Returns:
        True if filepath appears to be a valid markdown report path
    """
    path = Path(filepath)
    
    # Check extension
    if path.suffix.lower() != ".md":
        return False
    
    # Check name pattern
    name = path.name
    if not re.match(r'.*_\d{8}_\d{6}\.md$', name):
        return False
    
    # Check it's in reports directory
    try:
        reports_dir = get_reports_directory()
        path.absolute().relative_to(reports_dir.absolute())
        return True
    except ValueError:
        # Path is not relative to reports directory
        return False


# =============================================================================
# Structured reports (Phase 4, plan section 7)
# =============================================================================


_HEADING_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff]")


def _normalize_heading_text(value: str) -> str:
    """Tolerant heading comparison form (mirror of the writer's, kept local
    to avoid a dependency edge into worker_agents): remove zero-width
    chars, casefold, collapse whitespace to single spaces, strip trailing
    punctuation (\u2026 and :;.-—)."""
    s = _HEADING_ZERO_WIDTH_RE.sub("", value or "")
    s = " ".join(s.split()).casefold()
    return s.rstrip(" \t.:;.-—…")


def _collapse_ws(value: str) -> str:
    """Collapse whitespace runs to single spaces. Span/cell text and
    source labels can come from web content (attacker-controllable); an
    embedded newline there would inject Markdown structure into the saved
    report. Code-block text is intentionally NOT collapsed (literal)."""
    return re.sub(r"\s+", " ", value or "").strip()


def _render_span(text: str, citations) -> str:
    """Render one span: its text (whitespace collapsed, see _collapse_ws)
    followed by [^n] footnote markers. Citation tokens are appended
    verbatim — the collapse applies to the prose text only."""
    out = _collapse_ws(text)
    for c in citations or []:
        out += f"[^{c}]"
    return out


def _render_block(block) -> List[str]:
    """Render one ReportBlock to Markdown lines (no trailing blank line)."""
    lines: List[str] = []
    btype = block.type
    if btype == "heading":
        level = max(1, min(6, int(block.level or 3)))
        text = (
            "".join(_render_span(s.text, s.citations) for s in block.spans)
            or _collapse_ws(block.text)
        )
        lines.append("#" * level + " " + text.strip())
    elif btype == "paragraph":
        if block.spans:
            text = "".join(_render_span(s.text, s.citations) for s in block.spans)
        else:
            text = _collapse_ws(block.text)
        lines.append(text.strip())
    elif btype in ("ordered_list", "unordered_list"):
        for i, item in enumerate(block.items or []):
            marker = f"{i + 1}." if btype == "ordered_list" else "-"
            lines.append(f"{marker} {_render_span(item.text, item.citations)}")
    elif btype == "callout":
        body = (
            "".join(_render_span(s.text, s.citations) for s in block.spans)
            or _collapse_ws(block.text)
        )
        label = block.callout_title or block.callout_type.title()
        lines.append(f"> **{label}:** {body.strip()}")
    elif btype == "comparison_table":
        cols = list(block.columns or [])
        rows = list(block.rows or [])
        if not cols and rows:
            cols = [f"col{i + 1}" for i in range(len(rows[0]))]

        def _cell(c) -> str:
            cell_text = getattr(c, "text", None)
            if isinstance(cell_text, str):
                return _render_span(cell_text, getattr(c, "citations", [])).strip()
            if isinstance(c, list):
                return " ".join(_render_span(s.text, s.citations) for s in c).strip()
            return str(c).strip()

        if cols:
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join("---" for _ in cols) + " |")
            for r in rows:
                lines.append("| " + " | ".join(_cell(c) for c in r) + " |")
    elif btype == "code_block":
        lang = (block.language or "").strip()
        lines.append(f"```{lang}")
        lines.append((block.text or "").rstrip("\n"))
        lines.append("```")
    elif btype == "page_break":
        lines.append("---")
    elif btype == "citation_note":
        body = (
            "".join(_render_span(s.text, s.citations) for s in block.spans)
            or _collapse_ws(block.text)
        )
        lines.append(f"> {body.strip()}")
    else:
        if block.text:
            lines.append(_collapse_ws(block.text))
    return [l for l in lines if l != ""]


def render_markdown(report) -> str:
    """Deterministic Markdown rendering of a structured report (plan 7.3).

    Debug / copy-paste export only — the canonical output is JSON. Accepts a
    ResearchReport or the inner Report. Citations render as [^n] footnote
    markers; references render as [^n]: definition lines in source order
    (sources are stored in first-cited-key order, which matches the marker
    numbers). Deterministic; raises TypeError on other input types.
    """
    from models.report_schema import Report as _Report
    from models.report_schema import ResearchReport as _ResearchReport

    if isinstance(report, _ResearchReport):
        report = report.report
    if not isinstance(report, _Report):
        raise TypeError(
            f"render_markdown expects a Report/ResearchReport, "
            f"got {type(report).__name__}"
        )

    lines: List[str] = []
    meta = report.metadata
    lines.append(f"# {meta.title}")
    lines.append("")
    if meta.subtitle:
        lines.append(f"*{meta.subtitle}*")
        lines.append("")

    if report.executive_summary:
        lines.append("## Executive Summary")
        lines.append("")
        for para in report.executive_summary:
            lines.append(str(para).strip())
            lines.append("")

    section_headings = [_normalize_heading_text(s.heading) for s in report.sections]
    for section in report.sections:
        lines.append(f"### {section.heading}")
        lines.append("")
        blocks = list(section.blocks or [])
        # Skip a LEADING heading block whose text duplicates this section's
        # own heading (or any sibling section's heading — a phantom
        # boundary): the heading line above already prints it, so a block
        # restating it would render the title twice. Only blocks[0] is
        # considered; later subsection headings always render.
        if blocks and blocks[0].type == "heading":
            first_text = (
                "".join(_render_span(s.text, s.citations) for s in blocks[0].spans)
                or blocks[0].text
            )
            first_norm = _normalize_heading_text(first_text)
            if first_norm and first_norm in section_headings:
                blocks = blocks[1:]
        for block in blocks:
            block_lines = _render_block(block)
            if block_lines:
                lines.extend(block_lines)
                lines.append("")

    sources = list(report.sources or [])
    if sources:
        if lines and lines[-1] != "":
            lines.append("")
        for i, s in enumerate(sources, start=1):
            label = _collapse_ws(s.title or s.URL or s.id)
            lines.append(f"[^{i}]: {label}")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def save_structured_report(
    report,
    output_dir: Optional[Path] = None,
    state: Optional[JsonDict] = None,
    config: Optional[ReportConfig] = None,
) -> str:
    """Save a validated structured report to reports/ (plan section 7.1).

    Writes, all sharing the stem {safe_query}_{timestamp}:
    - {stem}.json         — the canonical structured document
    - {stem}.sources.json — standalone sources array (for the doc-gen project)
    - {stem}.markdown.md  — deterministic Markdown export (render_markdown)
    - {stem}.evidence.md  — evidence side file when state carries evidence
      (unchanged behavior from save_report, plan 7.4)

    Returns the path (str) of the canonical .json file.
    """
    import json

    from models.report_schema import (
        QualityMetrics as _QualityMetrics,
        Report as _Report,
        ResearchReport as _ResearchReport,
    )

    if isinstance(report, _Report):
        report = _ResearchReport(report=report, quality=_QualityMetrics())
    if not isinstance(report, _ResearchReport):
        raise TypeError(
            f"save_structured_report expects a ResearchReport, "
            f"got {type(report).__name__}"
        )

    if output_dir is None:
        reports_dir = Path(__file__).parent.parent / "reports"
    else:
        reports_dir = output_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or ReportConfig.default()
    _valid, safe_query = _validate_query(report.report.metadata.query or "")
    if not safe_query:
        safe_query = "untitled"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{safe_query}_{timestamp}"

    json_path = reports_dir / f"{stem}.json"
    sources_path = reports_dir / f"{stem}.sources.json"
    markdown_path = reports_dir / f"{stem}.markdown.md"

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    with open(sources_path, "w", encoding="utf-8") as f:
        f.write(json.dumps([s.model_dump() for s in report.report.sources], indent=2))
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    if cfg.include_evidence_dump and state and isinstance(state, dict):
        document_chunks, web_results = _parse_evidence_json(
            state.get("evidence_json", "")
        )
        if document_chunks or web_results:
            evidence_path = json_path.with_name(f"{json_path.stem}.evidence.md")
            evidence_markdown = _build_evidence_markdown(
                document_chunks,
                web_results,
                state.get("verification_status", {}),
                cfg,
            )
            with open(evidence_path, "w", encoding="utf-8") as f:
                f.write(evidence_markdown)

    logger.debug(f"save_structured_report: wrote {json_path}")
    return str(json_path)
