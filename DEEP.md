# Deep Research Mode: Analysis & Implementation Plan

## Executive Summary

The current `multi-agent-rag-researcher` implements a linear R → W → V pipeline that is functional for simple factual queries but has **fundamental depth limitations** for complex, multi-faceted research tasks. The system cannot iteratively deepen its evidence collection, decompose complex questions, assess whether it has *enough* evidence, or synthesize findings across sources.

This document proposes a **Deep Research mode** — a fundamentally different orchestration paradigm that replaces the linear pipeline with an iterative, recursive, multi-agent research loop. The proposed architecture introduces recursive sub-question decomposition, iterative evidence depth control, multi-pass memory accumulation, synthesis/analysis layers, divergent thinking, and iterative verification feedback loops.

**Key insight**: The current system asks "can I find evidence?" The Deep Research system asks "have I found *enough* evidence to confidently answer this question?"

---

## 1. Current Architecture Analysis

### 1.1 Pipeline Overview

```
User Query → Orchestrator → [Retriever → Writer → Verifier]
                   ↑                                │
                   └────────────────────────────────┘
                         (no feedback loop)
```

**Key files:**
- `orchestrator_agent.py` — Coordinates 4 tools, max 4 rounds
- `worker_agents/retriever_agent.py` — PDF + web retrieval
- `worker_agents/writer_agent.py` — Draft generation
- `worker_agents/verifier_agent.py` — Hallucination checking
- `worker_agents/model_runner.py` — LLM API wrapper
- `memory/memory.py` — SQLite session memory
- `run_orchestrator.py` — CLI entry point

### 1.2 Detailed Limitation Analysis

#### Limitation 1: Destructive Retrieval (`worker_agents/retriever_agent.py`, lines 117-141)

```python
# Inside the 4-round loop, each round OVERRIDES previous results:
if call.name == "retrieve_document":
    function_response = retrieve_document(query)
    if function_response.get("chunks"):
        document_evidence = function_response  # ← Overwrites!
```

The retriever runs 4 internal rounds, but each round's results **replace** the previous round's results. After round 4, only that single round's 16 chunks (4 per doc × 4 docs) survive. Three rounds of potentially different, complementary evidence are silently destroyed. The same pattern applies to `web_evidence`.

#### Limitation 2: No Recursive Follow-up Generation (`worker_agents/retriever_agent.py`)

The retriever rewrites queries to be self-contained but never generates follow-up questions. There is no mechanism for the retriever to say "I found X about topic A, but I need to investigate Y to fully answer the user's question."

```python
# The retriever only rewrites for self-containment:
"Rewrite the user's request into a self-contained search query "
"for the indexed PDFs. Include omitted subject details from "
"follow-up context when needed."
```

No instruction exists for generating investigative follow-up queries.

#### Limitation 3: Single-shot Memory (`memory/memory.py`)

```sql
CREATE TABLE IF NOT EXISTS evidence_memory (
    session_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    evidence_json TEXT NOT NULL,  -- Only ONE row per session (PRIMARY KEY)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

The `evidence_memory` table uses `session_id` as PRIMARY KEY, meaning only the **latest** evidence per session is stored. Earlier evidence is overwritten. There is no concept of accumulating evidence across turns.

#### Limitation 4: Verifier Checks Hallucination, Not Sufficiency (`worker_agents/verifier_agent.py`)

```python
# Verifier instructions only mention:
"Keep only supported statements."
"If the evidence is weak, incomplete, or not enough for a confident conclusion, say so."
```

The verifier can flag weak evidence as a textual caveat but produces **no structured gap report**. The orchestrator has no machine-parseable signal to decide whether to re-retrieve. There is no confidence score, no gap list, no "re-retrieve this specific topic" directive.

#### Limitation 5: No Cross-Document Synthesis

The writer receives a flat concatenation of evidence chunks and writes a draft. There is no agent that explicitly:
- Compares findings across different documents
- Identifies agreements and contradictions
- Weighs evidence quality across sources

#### Limitation 6: No Sub-question Decomposition

Complex queries like "How do Transformer architecture differences between GPT-4 and Claude 3 affect their reasoning capabilities in multi-step tasks?" are treated identically to simple queries like "What is GPT-4?" — a single retrieval pass followed by writing.

#### Limitation 7: No Divergent Thinking

The system only gathers supporting evidence. It never deliberately seeks counter-evidence, alternative interpretations, or edge cases. The final answer is biased toward the first evidence found.

#### Limitation 8: Fixed Iteration Budget (`orchestrator_agent.py`, line ~137)

```python
# Allow up to 4 orchestration rounds before stopping.
for _ in range(4):
```

The 4-round limit is arbitrary, not evidence-driven. A complex question might need 6 rounds; a simple one might only need 1. The system cannot decide based on evidence quality whether to continue.

---

## 2. Proposed Deep Research Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  DEEP RESEARCH ORCHESTRATOR              │
│                                                         │
│  1. Query Decomposition Agent                            │
│     Breaks complex query into 3-7 sub-questions          │
│                                                         │
│  2. Sub-Question Investigators (parallel/sequential)     │
│     Each runs its own R→W→V cycle with iterative depth  │
│                                                         │
│  3. Synthesis Agent                                      │
│     Compares, contrasts, and merges findings             │
│     Identifies agreements, contradictions, gaps           │
│                                                         │
│  4. Divergent Thinking Agent                             │
│     Challenges conclusions, seeks counter-evidence       │
│                                                         │
│  5. Evidence Quality Assessor                            │
│     Confidence scores per claim (high/medium/low)        │
│     Evidence gap detection with re-retrieval triggers    │
│                                                         │
│  6. Final Report Generator                               │
│     Integrates all layers into a nuanced final answer    │
│                                                         │
└─────────────────────────────────────────────────────────┘
         ↑ Evidence gaps feed back to Sub-Question Investigators
```

### 2.2 Data Flow

```
Phase 1: DECOMPOSITION
  Input: user_query
  Output: structured sub_question_plan = [
    {id, question, priority, parent_topic, status}
  ]

Phase 2: INVESTIGATION (per sub-question)
  Input: sub_question
  Output: sub_answer + evidence_pack + confidence_score

Phase 3: SYNTHESIS
  Input: [sub_answer_1, sub_answer_2, ...]
  Output: merged_findings + contradictions + gaps

Phase 4: GAP CLOSING (iterative)
  Input: gaps from synthesis
  Output: additional evidence packs

Phase 5: DIVERGENT THINKING
  Input: merged_findings + gaps
  Output: alternative_interpretations + counter_evidence

Phase 6: FINAL REPORT
  Input: all findings + confidence scores + alternatives
  Output: final_answer with confidence levels
```

---

## 3. Component Design

### 3.1 `deep_research_orchestrator.py` — New File

**Purpose**: Replace `orchestrator_agent.py` for deep research mode. Manages the full lifecycle.

**Key responsibilities:**
- Decide when to use deep research vs. standard mode
- Run query decomposition
- Manage sub-question lifecycle (create, assign, track completion)
- Orchestrate synthesis and gap-closing loops
- Collect and aggregate confidence scores
- Terminate based on evidence sufficiency, not just iteration count

**Data structures:**

```python
class ResearchQuestion(BaseModel):
    id: str
    question: str
    priority: int  # 1-7
    status: Literal["pending", "investigating", "completed", "failed"]
    sub_answer: str = ""
    confidence: float = 0.0
    evidence_pack: str = ""  # JSON
    gaps: list[str] = []

class ResearchReport(BaseModel):
    original_query: str
    sub_questions: list[ResearchQuestion]
    synthesis_findings: str
    contradictions: list[str]
    alternative_interpretations: list[str]
    final_answer: str
    overall_confidence: float  # 0.0 - 1.0
    confidence_breakdown: dict[str, float]
    evidence_summary: str
    total_retrieval_rounds: int
    total_llm_calls: int
```

### 3.2 `worker_agents/decomposition_agent.py` — New File

**Purpose**: Break complex queries into structured sub-questions.

**Input**: user_query + optional context
**Output**: list of ResearchQuestion objects

**Prompt strategy**:
1. Assess query complexity (simple vs. multi-faceted)
2. Identify key dimensions that need investigation
3. Generate 3-7 sub-questions covering all dimensions
4. Each sub-question must be investigable with the available tools (retrieve_document or web_search)

**Decision logic**: If the query is simple (one factual question), short-circuit to standard mode.

### 3.3 `worker_agents/investigator_agent.py` — New File

**Purpose**: Investigate a single sub-question with iterative deepening.

**Architecture**: A self-contained mini-pipeline that:
1. Runs retriever with iterative evidence gathering (accumulates across rounds)
2. Drafts a focused answer
3. Verifies with confidence scoring
4. Identifies gaps and re-retrieves if needed

**Internal loop** (replaces the current retriever's 4-round loop):
```
Round 1: Initial retrieval → draft → verify → check sufficiency
If gaps > 0 and round < MAX_ITERATIONS:
    Round 2+: Targeted retrieval on gaps → update draft → re-verify
Final: Return sub_answer + confidence + evidence + gaps
```

### 3.4 `worker_agents/synthesis_agent.py` — New File

**Purpose**: Compare and merge findings from multiple sub-questions.

**Input**: list of sub_answers with evidence packs
**Output**: merged findings, contradictions, agreement levels, gaps

**Behaviors**:
- Identify where sub-answers agree (reinforcing evidence)
- Flag contradictions and report both sides
- Identify topics that NO sub-question addressed
- Rate overall evidence coverage
- Output a gap list for the gap-closing phase

### 3.5 `worker_agents/divergent_thinking_agent.py` — New File

**Purpose**: Challenge the synthesized findings.

**Input**: merged findings + gap list
**Output**: alternative interpretations, counter-evidence search queries, edge cases

**Prompt strategy**:
1. "What is the strongest counter-argument to the synthesized findings?"
2. "What evidence would contradict these conclusions?"
3. "What edge cases or exceptions should be noted?"
4. "Are there alternative frameworks for understanding this?"

### 3.6 `worker_agents/verifier_agent.py` — Extended

**Current state**: Only checks hallucination, produces flat text.

**New behavior**: Produce structured gap report:
```python
class VerificationReport(BaseModel):
    is_supported: bool           # Overall: is the draft grounded?
    hallucinated_claims: list[str]
    unsupported_claims: list[str]
    missing_topics: list[str]    # What the query asks about but evidence doesn't cover
    confidence_level: Literal["high", "medium", "low"]
    gap_details: list[str]       # Human-readable gap descriptions
    re_retrieve_suggested: bool  # Should orchestrator send more queries?
    specific_queries: list[str]  # Suggested follow-up retrieval queries
```

**Changes needed**:
- Add `VerificationReport` pydantic model
- Use structured output (`text_format=VerificationReport`) in `run_model`
- Update prompt instructions to require gap detection

### 3.7 `worker_agents/retriever_agent.py` — Extended

**Current state**: 4 rounds, each round overwrites previous results.

**New behavior**: Accumulate evidence across rounds. Add "goal-directed retrieval."

**Key changes**:
1. **Accumulation**: Change `document_evidence = function_response` to append-mode:
   ```python
   if document_evidence is None:
       document_evidence = function_response
   else:
       existing_chunks = set(
           (c["document_name"], c["page_number"], c["chunk_id"])
           for c in document_evidence.get("chunks", [])
       )
       new_chunks = [
           c for c in function_response.get("chunks", [])
           if (c["document_name"], c["page_number"], c["chunk_id"]) not in existing_chunks
       ]
       document_evidence["chunks"].extend(new_chunks)
   ```

2. **Goal-directed queries**: Accept an optional `research_goal` parameter that tells the retriever what specifically to look for:
   ```python
   def retriever_agent(
       user_query: str,
       *,
       research_goal: str = "",  # NEW: targeted retrieval directive
       follow_up_queries: list[str] = [],  # NEW: specific things to investigate
       ...
   )
   ```

3. **Internal iteration with memory**: Track which queries have been attempted to avoid redundancy.

### 3.8 `memory/memory.py` — Extended

**Current state**: Single row per session in `evidence_memory`.

**New structure**:
```sql
CREATE TABLE IF NOT EXISTS evidence_memory (
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    research_topic TEXT,          -- Which sub-question this addresses
    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, research_topic, retrieved_at)
)
```

**New functions**:
```python
def save_evidence_accumulated(...):  # Appends, doesn't overwrite
def get_all_evidence(session_id):     # Returns full evidence history
def get_evidence_by_topic(session_id, topic):  # Returns evidence for a sub-question
def clear_topic_evidence(session_id, topic):  # For fresh investigations
```

### 3.9 `orchestrator_agent.py` — Refactored or Mode-Toggled

**Option A (cleaner)**: Create `deep_orchestrator_agent.py` as a parallel file. Standard mode continues using the current orchestrator.

**Option B (integrated)**: Add a `mode="standard" | "deep"` parameter to the existing orchestrator.

**Recommendation**: Option A. The deep orchestrator is complex enough to warrant its own file, and keeping the proven standard path untouched reduces risk.

### 3.10 `run_orchestrator.py` — Extended

**Changes**:
- Add `--mode deep|standard` CLI flag
- Pass mode through to orchestrator
- Deep mode logs more verbose progress (sub-questions, confidence scores, synthesis steps)

---

## 4. Specific Code Changes by File

### 4.1 New Files (Create)

| File | Purpose | Dependencies |
|------|---------|--------------|
| `worker_agents/decomposition_agent.py` | Sub-question decomposition | `model_runner` |
| `worker_agents/investigator_agent.py` | Iterative per-sub-question investigation | `retriever_agent`, `writer_agent`, `verifier_agent`, `model_runner` |
| `worker_agents/synthesis_agent.py` | Cross-sub-question synthesis | `model_runner` |
| `worker_agents/divergent_thinking_agent.py` | Counter-evidence exploration | `model_runner` |
| `deep_research_orchestrator.py` | Full deep research pipeline | All of the above |
| `models/research_models.py` | Shared Pydantic models | None |

### 4.2 Modified Files

| File | Changes |
|------|---------|
| `worker_agents/worker_agents/verifier_agent.py` | Add `VerificationReport` model, structured output, gap detection logic |
| `worker_agents/retriever_agent.py` | Accumulate evidence across rounds, add `research_goal` parameter, dedup chunks |
| `worker_agents/__init__.py` | Export new agents |
| `memory/memory.py` | Schema migration for accumulation, add new functions |
| `memory/__init__.py` | Export new functions |
| `run_orchestrator.py` | Add `--mode` CLI flag |
| `ui/gradio_app.py` | Display deep research metadata (confidence, sub-questions) |
| `ui/gradio_handlers.py` | Pass mode flag, show progress for deep mode |

### 4.3 Exact Code Patterns to Change

#### Change 1: Accumulate, don't overwrite, in retriever

**File**: `worker_agents/retriefer_agent.py`
**Lines**: 117-141 (the 4-round loop)

Current:
```python
if call.name == "retrieve_document":
    function_response = retrieve_document(query)
    if function_response.get("chunks"):
        document_evidence = function_response  # ← DESTROY previous
```

New:
```python
if call.name == "retrieve_document":
    function_response = retrieve_document(query)
    if function_response.get("chunks"):
        document_evidence = document_evidence or {"query": query, "chunks": []}
        existing = {(c["document_name"], c["page_number"], c["chunk_id"])
                    for c in document_evidence["chunks"]}
        new_chunks = [
            c for c in function_response["chunks"]
            if (c["document_name"], c["page_number"], c["chunk_id"]) not in existing
        ]
        document_evidence["chunks"].extend(new_chunks)
```

#### Change 2: Structured verification output

**File**: `worker_agents/verifier_agent.py`

Current:
```python
response = run_model(
    instructions=instructions,
    input_data=input_text,
    reasoning_effort="low",
    tools=None,
    agent_name="verifier",
    endpoint=endpoint,
    api_key=api_key,
)
return response.output_text
```

New:
```python
from pydantic import BaseModel, Literal
from typing import Optional

class VerificationReport(BaseModel):
    is_supported: bool
    hallucinated_claims: list[str] = []
    unsupported_claims: list[str] = []
    missing_topics: list[str] = []
    confidence_level: Literal["high", "medium", "low"]
    gap_details: list[str] = []
    re_retrieve_suggested: bool = False
    specific_queries: list[str] = []

response = run_model(
    instructions=instructions,
    input_data=input_text,
    reasoning_effort="low",
    tools=None,
    text_format=VerificationReport,  # ← STRUCTURED OUTPUT
    agent_name="verifier",
    endpoint=endpoint,
    api_key=api_key,
)
return response.parsed  # Pydantic model, not raw text
```

#### Change 3: Memory accumulation

**File**: `memory/memory.py`

Add new table:
```sql
CREATE TABLE IF NOT EXISTS evidence_accumulation (
    session_id TEXT NOT NULL,
    sub_topic TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    retrieval_query TEXT NOT NULL,
    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, sub_topic, retrieved_at)
)
```

Add function:
```python
def append_evidence(
    session_id: str,
    sub_topic: str,
    query: str,
    evidence_json: str
) -> None:
    """Append evidence for a sub-topic without overwriting other topics."""
    with get_memory_connection() as conn:
        conn.execute(
            """
            INSERT INTO evidence_accumulation (session_id, sub_topic, evidence_json, retrieval_query)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, sub_topic, evidence_json, query),
        )

def get_topic_evidence(session_id: str, sub_topic: str) -> list[dict]:
    """Get all evidence accumulated for a sub-topic."""
    with get_memory_connection() as conn:
        rows = conn.execute(
            """
            SELECT evidence_json FROM evidence_accumulation
            WHERE session_id = ? AND sub_topic = ?
            ORDER BY retrieved_at
            """,
            (session_id, sub_topic),
        ).fetchall()
    return [json.loads(r["evidence_json"]) for r in rows]
```

---

## 5. Implementation Roadmap

### P0: Foundation (Week 1-2) — Core depth loop

These changes are **necessary** for any form of deep research. Without them, higher-level agents have nothing to work with.

**1. Evidence accumulation in retriever** (`worker_agents/retriever_agent.py`)
- Replace overwrite-with-last with append-mode accumulation
- Add chunk deduplication by (document, page, chunk_id)
- Add optional `research_goal` parameter
- Add internal query tracking to avoid redundant searches
- **Risk**: Low. Purely additive change to retrieval logic.
- **Test**: Verify that running retriever with 4 rounds on a multi-topic query returns all unique chunks, not just the last round's.

**2. Structured verifier output** (`worker_agents/verifier_agent.py`)
- Add `VerificationReport` Pydantic model
- Switch to `text_format=` structured output in `run_model`
- Update prompt to require gap detection
- Add `re_retrieve_suggested` and `specific_queries` fields
- **Risk**: Medium. Structured output from LLMs can be brittle. Need fallback for unparseable output.
- **Test**: Run verifier on known hallucinated and fully-supported drafts. Verify gap detection accuracy.

**3. Memory schema migration** (`memory/memory.py`)
- Add `evidence_accumulation` table
- Add `append_evidence()` and `get_topic_evidence()` functions
- Backward compatibility: ensure existing single-row schema still works
- **Risk**: Low. Additive DB schema change.
- **Test**: Create session, accumulate 3 rounds of evidence, verify all 3 are retrievable.

### P1: Orchestration (Week 3-4) — The deep research loop

**4. Decomposition agent** (`worker_agents/decomposition_agent.py`)
- Query complexity classifier
- Sub-question generator (3-7 questions)
- Each sub-question must be independently investigable
- **Risk**: Medium. Quality of decomposition determines everything downstream.
- **Test**: Feed 10 diverse complex queries. Verify sub-questions are mutually exclusive and collectively exhaustive.

**5. Investigation agent** (`worker_agents/investigator_agent.py`)
- Self-contained mini-pipeline using retriever + writer + verifier
- Iterative: investigate → verify → gap → re-investigate
- Terminate on evidence sufficiency or max iterations
- **Risk**: Medium-High. This is the most complex single component.
- **Test**: Investigate sub-questions with known gaps. Verify it self-corrects.

**6. Deep research orchestrator** (`deep_research_orchestrator.py`)
- Connect decomposition → investigation → synthesis → gap-closing → divergent → final
- Manage sub-question lifecycle
- Confidence aggregation
- Termination conditions
- **Risk**: High. Integration complexity. Many edge cases.
- **Test**: End-to-end test with 5 multi-faceted queries of increasing complexity.

### P2: Polish & Integration (Week 5-6) — UX and robustness

**7. Synthesis agent** (`worker_agents/synthesis_agent.py`)
- Cross-source comparison
- Contradiction detection
- Coverage assessment
- **Risk**: Medium.
- **Test**: Provide contradictory evidence sets. Verify contradictions are flagged.

**8. Divergent thinking agent** (`worker_agents/divergent_thinking_agent.py`)
- Counter-argument generation
- Alternative hypothesis exploration
- **Risk**: Low.
- **Test**: Investigate controversial topics. Verify counter-evidence is surfaced.

**9. UI integration** (`ui/gradio_app.py`, `ui/gradio_handlers.py`)
- Display sub-question progress
- Show confidence scores
- Toggle deep/standard mode
- Show evidence gap status
- **Risk**: Low.
- **Test**: Manual UI testing with deep mode queries.

**10. CLI integration** (`run_orchestrator.py`)
- `--mode deep|standard` flag
- Verbose logging for deep mode (sub-questions, rounds, confidence)
- **Risk**: Low.

---

## 6. Termination Conditions

The current system terminates on **iteration count** (4 rounds). Deep research should terminate on **evidence sufficiency**:

```python
def should_terminate(state: DeepResearchState) -> tuple[bool, str]:
    """Return (should_stop, reason)."""
    
    # Always stop if max budget exhausted
    if state.total_llm_calls >= MAX_LLM_CALLS:
        return True, "LLM call budget exhausted"
    if state.total_rounds >= MAX_RESEARCH_ROUNDS:
        return True, "Max research rounds reached"
    
    # Stop if all sub-questions have high confidence
    low_confidence_subs = [
        sq for sq in state.sub_questions
        if sq.confidence < MIN_CONFIDENCE_THRESHOLD
    ]
    if not low_confidence_subs and state.synthesis_confidence >= 0.7:
        return True, "All sub-questions resolved with sufficient confidence"
    
    # Stop if re-retrieval yields diminishing returns
    if state.last_round_new_evidence_count == 0 and state.retrieval_rounds > 1:
        return True, "Diminishing returns: no new evidence in last round"
    
    # Stop if gap-closing didn't find anything
    if state.gap_closed_count >= MAX_GAP_CLOSING_ATTEMPTS:
        return True, "Max gap-closing attempts reached"
    
    return False, "Continue researching"
```

**Suggested constants** (configurable):
```python
MAX_LLM_CALLS = 80          # ~2h of API time at typical rates
MAX_RESEARCH_ROUNDS = 6     # Decomposition + 2 investigation cycles + synthesis
MIN_CONFIDENCE_THRESHOLD = 0.7
MAX_GAP_CLOSING_ATTEMPTS = 2
EVIDENCE_SUFFICIENCY_SCORE = 0.7  # min avg confidence to stop
```

---

## 7. Risk Considerations

### 7.1 Cost & Latency

**Risk**: Deep research mode uses significantly more LLM calls (10-30× standard mode).

**Mitigation**:
- Start with `reasoning_effort="low"` across all agents
- Use the decomposition agent to short-circuit simple queries to standard mode
- Configurable budget limits (see termination conditions)
- Cache aggressively — reuse cached evidence across sub-questions when relevant
- Default budget: ~40 LLM calls max per deep research query

### 7.2 Structured Output Reliability

**Risk**: LLMs don't always produce perfect Pydantic-validated output.

**Mitigation**:
- Wrap all `run_model` structured output calls in try/except
- Fall back to text-based parsing if structured output fails
- Use `text_format` only for critical agents (verifier, decomposition); keep text output for exploratory agents (divergent thinking)

### 7.3 Over-Engineering Simple Queries

**Risk**: A query like "What is Python?" gets broken into 5 sub-questions unnecessarily.

**Mitigation**:
- The decomposition agent classifies query complexity first
- Queries with ≤ 1 clear dimension → standard mode
- Budget cap prevents runaway deep research on simple queries

### 7.4 Contradiction Handling

**Risk**: When sources contradict, the synthesis agent might produce a confused or wishy-washy answer.

**Mitigation**:
- Synthesis agent explicitly reports which side has more/better evidence
- Confidence scores differentiate well-supported vs. weakly-supported claims
- Divergent thinking agent frames contradictions as "here's an alternative view" not "the answer is uncertain"

### 7.5 Memory Explosion

**Risk**: Accumulating evidence across many rounds and sub-questions could exceed context windows downstream.

**Mitigation**:
- Truncate evidence chunks at retrieval (already done: 800 chars per chunk)
- Synthesis agent receives summaries, not raw chunks
- Citation-level summarization: "Document A (p.3, p.7, p.12) supports X" not full text
- Track total evidence size and warn if approaching context limits

### 7.6 Backward Compatibility

**Risk**: Changes to verifier output format could break the standard orchestrator.

**Mitigation**:
- Deep orchestrator is a separate file — standard orchestrator unchanged
- Verifier's structured output is opt-in (controlled by a `mode` parameter)
- Standard mode verifier returns the same text it currently returns

---

## 8. Expected Outcomes

### 8.1 Quality Improvements

| Metric | Current | After Deep Research |
|--------|---------|-------------------|
| Depth on multi-faceted queries | Surface-level, single-pass | Multi-dimensional, 3-5 rounds of investigation |
| Evidence coverage | 16 chunks per doc (discarded rounds) | All unique chunks across rounds, targeted follow-up |
| Confidence communication | "Evidence is weak" (text) | "High/Medium/Low" per claim with reasoning |
| Contradiction handling | Ignored | Explicitly surfaced with source attribution |
| Counter-evidence | Never sought | Divergent agent actively searches for it |
| Follow-up quality | Same breadth | Deeper, targeted follow-up on gaps |

### 8.2 Quantitative Improvements (Target)

- **Evidence unique chunks**: +50-200% (from destructive to accumulative retrieval)
- **Claim coverage**: +30-50% (gap-driven re-retrieval catches missed topics)
- **User-perceived depth**: Measurable via A/B testing on complex queries
- **Self-correction rate**: Will correctly identify and fix gaps in 70%+ of cases (vs. 0% currently)

### 8.3 Trade-offs

- **Latency**: Deep research queries will take 2-5× longer (acceptable for research-grade answers)
- **Cost**: 10-30× more LLM calls per query (mitigated by budget caps and standard mode default)
- **Complexity**: More agents = more failure modes = more debugging

---

## 9. Quick-Start: Minimal Viable Deep Research

If full implementation feels overwhelming, here's the **minimum viable change** that captures 80% of the value:

### Only 3 Changes Needed:

1. **Fix the retriever** — Accumulate evidence across its 4 rounds (change 4 lines in `retriever_agent.py`)
2. **Extend the verifier** — Add structured gap detection (add `VerificationReport` model, change 6 lines)
3. **Add one new orchestrator** — `deep_orchestrator.py` that:
   - Calls retriever
   - Calls writer
   - Calls verifier → gets gap report
   - If gaps exist and rounds < 3: calls retriever again with gap-specific queries
   - Repeats until sufficient or budget exhausted

This minimal version gives you:
- ✅ Evidence accumulation (no more destructive overwrites)
- ✅ Gap-driven re-retrieval (asks "what am I missing?" and goes get it)
- ✅ Confidence signaling ( verifier tells orchestrator if evidence is sufficient)
- ✅ Iterative deepening (2-3 rounds of targeted follow-up)

Everything else (decomposition, synthesis, divergent thinking) is progressive enhancement.

---

## 10. Appendix: Current Code Reference Points

### Retriever's Destructive Pattern
- **File**: `worker_agents/retriever_agent.py`
- **Lines**: 117-141
- **Issue**: `document_evidence = function_response` overwrites
- **Fix**: Append-mode accumulation with dedup

### Memory's Single-Row Limitation
- **File**: `memory/memory.py`
- **Line**: 49
- **Issue**: `PRIMARY KEY (session_id)` on `evidence_memory`
- **Fix**: Add `evidence_accumulation` table with composite key

### Verifier's Text-Only Output
- **File**: `worker_agents/verifier_agent.py`
- **Lines**: 52-58
- **Issue**: `return response.output_text` — no structured signal
- **Fix**: `text_format=VerificationReport`, return Pydantic model

### Orchestrator's Fixed Budget
- **File**: `orchestrator_agent.py`
- **Line**: ~137
- **Issue**: `for _ in range(4):` — no evidence-based termination
- **Fix**: Evidence-sufficiency check in deep orchestrator

### Writer's Flat Input
- **File**: `worker_agents/writer_agent.py`
- **Lines**: 34-54
- **Issue**: No synthesis, no cross-source comparison
- **Fix**: New synthesis agent handles this layer

### Model Runner's Reasoning Budget
- **File**: `worker_agents/model_runner.py`
- **Line**: ~108
- **Note**: `reasoning_effort="low"` across all agents — deep mode may benefit from `"medium"` on synthesis/divergent agents
