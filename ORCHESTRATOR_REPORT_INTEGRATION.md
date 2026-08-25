# Orchestrator Report Integration

This document explains how the orchestrator integrates with the improved markdown report generation system.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                               │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 Orchestrator Agent                          │
│  - Coordinates retriever, writer, verifier agents           │
│  - Manages evidence collection and verification             │
│  - Returns comprehensive state dictionary                   │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  │ result (state dict)
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                save_report()                                │
│  - Receives full orchestration state                        │
│  - Parses evidence_json (documents + web)                   │
│  - Includes verification_status with confidence scores      │
│  - Generates richly formatted markdown report               │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  │ filepath
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  reports/                                   │
│  - Timestamped markdown files                               │
│  - Structured with evidence and verification                │
└─────────────────────────────────────────────────────────────┘
```

## State Dictionary Structure

The orchestrator returns a comprehensive state dictionary that includes:

```python
{
    # Core information
    "user_query": str,              # Original user query
    "current_date": str,            # Date in ISO format
    "final_answer": str,            # Final verified answer
    
    # Evidence data
    "evidence_json": str,           # JSON string with evidence
    "document_evidence": list,      # Document chunks from PDFs
    "web_evidence": list,           # Web search results
    
    # Writer and Verifier outputs
    "written_draft": str,           # Writer agent's draft
    "verification": str,            # Verifier agent's output
    
    # Verification status
    "verification_status": {
        "confidence": float,        # 0.0 to 1.0
        "coverage": str,           # thin/moderate/comprehensive
        "gaps": list,              # Identified information gaps
        "re_retrieve": bool,       # Whether to re-retrieve
        "suggested_queries": list  # Queries to address gaps
    },
    
    # Orchestration state
    "retrieval_attempted": bool,
    "written_draft": str,
    "re_retrieve_rounds": int,
    "needs_more_evidence": bool,
    "gap_queries": list,
    "cached_query": str,            # Previous query for reuse
    "cached_evidence_summary": str,
}
```

## Report Generation Flow

1. **User Query**: User enters a research question
2. **Orchestration**: Orchestrator coordinates agents to:
   - Retrieve evidence (documents + web)
   - Write a draft
   - Verify the draft against evidence
3. **State Capture**: Full orchestration state is captured
4. **Report Generation**: `save_report()` processes the state:
   - Parses `evidence_json` into structured evidence items
   - Extracts verification status and confidence scores
   - Generates formatted markdown with evidence
5. **File Output**: Report saved to `reports/` directory

## Enhanced Features

### 1. Evidence Parsing

The improved `save_report` automatically parses evidence from the orchestrator state:

```python
# Document evidence from PDFs
{
    "source": "paper.pdf",
    "content": "Transformer architecture enables...",
    "score": 0.95
}

# Web search results
{
    "url": "https://example.com",
    "snippet": "Recent advances in transformer models...",
    "relevance_score": 0.88
}
```

### 2. Verification Status

Verification status is included with confidence metrics:

```markdown
### Verification Summary

- **Confidence Level:** 92.0%
- **Coverage:** comprehensive
- **Identified Gaps:** 1

  - No data on 2025 models
```

### 3. Security Configuration

The orchestrator uses `ReportConfig.strict()` for enhanced security:

```python
config = ReportConfig.strict()
# max_content_length: 5000
# max_snippet_length: 500
# sanitize_control_chars: True
# preserve_whitespace: False
```

### 4. Debug Output

When `--debug` is enabled, additional metadata is logged:

```python
[DEBUG] Evidence length: 4521 chars
[DEBUG] Verification: present
[DEBUG] Status: {'confidence': 0.92, 'coverage': 'comprehensive', ...}
```

## Usage Examples

### Basic Usage

```bash
python run_orchestrator.py
```

### With Debug Output

```bash
python run_orchestrator.py --debug
```

### Custom Session ID

```python
from run_orchestrator import chat_with_supervisor

chat_with_supervisor(session_id="my_custom_session")
```

## Report Output Format

Generated reports include:

```markdown
# Research Report

**Query:** What are transformer models?
**Date:** 2024-01-15 14:30:22 UTC
**Session ID:** session_123
**Generated:** 2024-01-15T14:30:22+00:00

---

## Executive Summary

Transformer models use attention mechanisms to process sequences...

---

## Supporting Evidence

### Document Evidence

#### Document Score: 95.0%

paper.pdf

```text
Transformer architecture enables parallel processing of...
```

### Web Evidence

#### Web Score: 88.0%

https://example.com/transformers

```text
Recent advances in transformer models for NLP...
```

### Verification Summary

- **Confidence Level:** 92.0%
- **Coverage:** comprehensive
- **Identified Gaps:** 1

  - No data on 2025 models

---

## Verification Output

```text
EVIDENCE_STATUS:
  confidence: 0.92
  coverage: comprehensive
  gaps: []
```

---

*This report was automatically generated by the AI Research Orchestrator.*
```

## Integration Points

### 1. Orchestrator Agent

The orchestrator is responsible for:
- Coordinating worker agents
- Managing evidence collection
- Running verification
- Returning comprehensive state

### 2. Save Report Module

The save_report module handles:
- Parsing evidence JSON
- Building markdown structure
- Sanitizing inputs
- Saving to files

### 3. Memory System

The memory system provides:
- Session context for follow-up questions
- Cached evidence for reuse
- Query tracking

## Best Practices

1. **Always pass state**: The full orchestration state contains valuable metadata
2. **Use strict config**: Security-sensitive applications should use `ReportConfig.strict()`
3. **Enable debug**: Use `--debug` during development to see evidence details
4. **Check verification status**: Verify confidence scores before using reports
5. **Monitor evidence gaps**: Review identified gaps for incomplete research

## Troubleshooting

### No Evidence Found

```python
# Check if evidence was retrieved
if not result.get("evidence_json"):
    print("No evidence found for query")
```

### Low Confidence Score

```python
status = result.get("verification_status", {})
confidence = status.get("confidence", 0)
if confidence < 0.7:
    print("Low confidence - consider re-retrieval")
```

### Large Reports

Use `ReportConfig.strict()` for security:

```python
config = ReportConfig.strict()
saved_path = save_report(..., config=config)
```

## Performance Considerations

- Evidence JSON is parsed once during report generation
- Sanitization prevents injection attacks
- Length limits prevent excessive file sizes
- Strict mode reduces memory usage

## Future Enhancements

Potential improvements:
- Async report generation for high-volume use
- Report aggregation and consolidation
- Export to multiple formats (PDF, HTML)
- Report versioning and history
- Automated report summary generation