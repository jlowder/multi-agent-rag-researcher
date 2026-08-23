# Orchestrator Report Integration Summary

## Overview

The `run_orchestrator.py` has been updated to properly pass orchestration state to the improved markdown report generation system. This ensures rich, structured reports with evidence tracking and verification status.

## Changes Made

### 1. Updated Imports in `run_orchestrator.py`

```python
from memory.save_report import save_report, ReportConfig, build_enriched_markdown
```

**Added:** `ReportConfig` and `build_enriched_markdown` imports for enhanced report generation.

### 2. Report Configuration

```python
# Research configuration for long-form research reports (P0-1)
report_config = ReportConfig.research()
```

**Benefit:** Research reports are no longer truncated mid-sentence:
- Body content is uncapped (no length truncation)
- Verification output is uncapped (no length truncation)
- Evidence snippet length limited to 800 characters
- Control characters sanitized
- Whitespace not preserved (prevents formatting attacks)

The UI (`ui/gradio_handlers.py::handle_save_report`) also passes
`ReportConfig.research()` to `save_report()`.

### 3. Enhanced Orchestrator Integration

```python
# Pass the full orchestration state to save_report for enriched reporting
saved_path = save_report(
    content=answer,
    query=user_query,
    session_id=session_id,
    state=result,  # Pass orchestration state with evidence, verification, etc.
    config=report_config  # Use research configuration (uncapped body)
)
```

**Benefit:** Full orchestration state is now passed to `save_report()`, including:
- Evidence from document chunks and web results
- Verification status with confidence scores
- Gap identification
- All metadata from orchestration

### 4. Debug Output

```python
if debug:
    evidence_json = result.get("evidence_json", "")
    verification = result.get("verification", "")
    status = result.get("verification_status", {})
    print(f"[DEBUG] Evidence length: {len(evidence_json)} chars")
    print(f"[DEBUG] Verification: {'present' if verification else 'missing'}")
    print(f"[DEBUG] Status: {status}")
```

**Benefit:** Debug mode now provides visibility into the evidence and verification process.

## State Structure

The orchestrator returns a comprehensive state dictionary:

```python
{
    "user_query": str,
    "current_date": str,
    "final_answer": str,
    "evidence_json": str,  # JSON with document_evidence and web_evidence
    "written_draft": str,
    "verification": str,
    "verification_status": {
        "confidence": float,  # 0.0 to 1.0
        "coverage": str,      # thin/moderate/comprehensive
        "gaps": list,         # identified information gaps
        "re_retrieve": bool,
        "suggested_queries": list
    },
    "retrieval_attempted": bool,
    "re_retrieve_rounds": int,
    "needs_more_evidence": bool,
    "gap_queries": list,
    # ... additional metadata
}
```

## Report Generation Flow

1. **User Query** → Orchestrator Agent
2. **Orchestration** → Evidence retrieval, drafting, verification
3. **State Capture** → Full orchestration state
4. **Report Generation** → `save_report()` parses state and builds markdown
5. **Output** → Structured markdown file in `reports/`

## Key Features

### 1. Evidence Parsing

Automatically parses evidence from `evidence_json`, reading the real
write-side fields emitted by the retriever
(`worker_agents/retriever_agent.py`):

```python
# Document evidence (Qdrant chunks)
{
    "document_name": "paper.pdf",   # fallback for document_title
    "document_title": "Paper Title",
    "page_number": 3,
    "chunk_id": "doc1",
    "citation": "Vaswani et al., 2017, p. 3",
    "content": "...",               # ≤ 800 chars when written
    "score": 0.95
}

# Web evidence (raw Tavily results)
{
    "title": "Recent advances",
    "url": "...",
    "content": "...",               # ≤ 600 chars when written
    "score": 0.88
}
```

The legacy read-side fields (`source`, `snippet`, `relevance_score`)
are no longer read — that mismatch is why Qdrant chunks previously
rendered as "Unknown" in reports (see
`REPORT_QUALITY_IMPROVEMENT_PLAN.md` P0-5).

**Evidence dump (`include_evidence_dump`)** — a `ReportConfig` field,
defaulting to `False` for all configs:
- `False` (default): no "Supporting Evidence" section is written in
  the report body at all.
- `True`: the dump is written to a side file
  `{report_stem}.evidence.md` in the same output directory as the
  report, with each chunk/result correctly attributed (title, source,
  page, citation, score). The report body receives a single-line
  pointer ("> Full evidence dump: `...`") instead.
  The dump is never written inline in the report body.

### 2. Verification Status

Includes verification metrics in reports:

```markdown
### Verification Summary

- **Confidence Level:** 92.0%
- **Coverage:** comprehensive
- **Identified Gaps:** 1
```

### 3. Security Configuration

The CLI (`run_orchestrator.py`) and the UI
(`ui/gradio_handlers.py::handle_save_report`) use
`ReportConfig.research()`:
- Input sanitization
- Control character removal
- Body/verification uncapped; evidence snippets capped

`ReportConfig.default()` and `ReportConfig.strict()` are unchanged for
other consumers. The evidence dump is off by default in all configs.

### 4. Report Structure

Generated reports include:
- Header with metadata
- Executive summary (full body, uncapped under `research()`)
- One-line pointer to the evidence side file (only when
  `include_evidence_dump=True` and evidence is present)
- Verification output (uncapped under `research()`)

The "Supporting Evidence" and "Verification Summary" sections live in
the `{report_stem}.evidence.md` side file, not in the report body.

## Usage

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

chat_with_supervisor(session_id="custom_session")
```

## Testing

A test script is provided: `test_orchestrator_report.py`

```bash
python test_orchestrator_report.py
```

This tests:
- Basic report generation
- Strict security configuration
- Minimal state handling
- Empty state handling
- Long content truncation

## Benefits

### 1. Enhanced Reporting

Reports now include rich evidence tracking:
- Multiple source types (documents + web)
- Confidence scores for each piece of evidence
- Verification status

### 2. Better Debugging

Debug output shows:
- Evidence length
- Verification presence
- Verification status

### 3. Security

Strict configuration prevents:
- Injection attacks
- Excessive memory usage
- Malicious content

### 4. Flexibility

- Configurable report generation
- Multiple output options
- Custom configurations

## Troubleshooting

### No Evidence Found

```python
if not result.get("evidence_json"):
    print("No evidence found for query")
```

### Low Confidence Score

```python
status = result.get("verification_status", {})
if status.get("confidence", 0) < 0.7:
    print("Low confidence - review gaps")
```

### Large Reports

Use research configuration:
```python
config = ReportConfig.research()
```

## Files Modified

1. `run_orchestrator.py` - Updated to pass state and use research config
2. `memory/save_report.py` - Real evidence fields, `include_evidence_dump` + side file, `research()` config
3. `ui/gradio_handlers.py` - `handle_save_report` passes `ReportConfig.research()`

## Files Added

1. `test_orchestrator_report.py` - Integration tests
2. `ORCHESTRATOR_REPORT_INTEGRATION.md` - Detailed documentation
3. `ORCHESTRATOR_REPORT_SUMMARY.md` - This file

## Verification

To verify the integration:

1. Run the orchestrator with `--debug` flag
2. Check the evidence length and verification status
3. Review the generated markdown report
4. Verify evidence is properly parsed and formatted
5. Check verification status is included

## Next Steps

Potential improvements:
- Async report generation
- Report aggregation
- Export to multiple formats
- Report versioning
- Automated summary generation