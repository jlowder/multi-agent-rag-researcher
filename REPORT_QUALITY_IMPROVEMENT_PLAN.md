# Report Quality Improvement Plan — Deep-Research Output

**Repo:** `multi-agent-rag-researcher` · **Date:** 2026-08-23 · **Companion doc:** `DEEP.md` (prior design; section refs below point to it)
**Baseline artifacts:** `reports/create a comprehensive report covering genetic programming_20260822_103759.md` (Report 1), `reports/i want a comprehensive report covering genetic programming_20260822_113611.md` (Report 2), `reports/gemini-research-genetic-programming.pdf` (Gemini benchmark) · **Local corpus:** 5 GP PDFs in `docs/`

## 0. Verified Root Causes (all confirmed against code)

| Claim | Verified at |
|---|---|
| strict caps 5000/2000 | `memory/save_report.py` L159–165 (`strict()`), L333–337 (body sanitize), L461–467 (verification cap); `run_orchestrator.py` L32 + L55–63 |
| confidence "90.0%" | `save_report.py` L429–435: `{"high": 0.9, "medium": 0.6, "low": 0.3}` mapping |
| EVIDENCE_STATUS always appended | `worker_agents/verifier_agent.py` L276–291 (strip model's block, re-synthesize, `output = output.rstrip() + block`) |
| DEPTH OVERRIDE substring trap | `orchestrator_agent.py` L622 (`'[DEPTH OVERRIDE' in output`), L618–632 (coverage `'thin'/'moderate'` also force), L646 (acts on `re_retrieve`), L457–458 (discards draft+verification), L297 (`range(10)`), L720–724 (cap fallback) |
| `tool_choice="required"` | `worker_agents/model_runner.py` L195 → orchestrator's no-tool-call early return (L311) is unreachable; only exit is `finish_research` (L657–671), which sets `final_answer = state["verification"]` (L663) — **the verifier's text is the final report** |
| Boilerplate transformer query | TWO copies: `orchestrator_agent.py` L636–643 and `verifier_agent.py` L266–273 ("attention variants (multi-query, grouped-query, flash)...") |
| Dead length gate | `orchestrator_agent.py` L557–560 (prints, does nothing) |
| Retriever | `retriever_agent.py` L242 (`range(4)`), L283 (`per_doc_topk=8`), L38 (800-char cap), L89 (600-char cap), L99 (`search_depth="basic"`); dedup by `chunk_id`/`url` exists (L286–330); **no global chunk cap** |
| Writer | `writer_agent.py` L70 (`reasoning_effort="low"`), L57 ("at least 2000 words"), single one-shot call; raw `json.dumps(indent=2)` injection via `memory/helpers.py::build_evidence_context` L51–60 |
| Verifier as final editor | `verifier_agent.py` L144 ("return only the final answer"), L148 ("Keep the writing concise"); coverage bucketed by chunk count L151 (thin <10 / moderate 10–20 / comprehensive 20+) |
| Memory single-row | `memory/memory.py` L41–47 (`session_id TEXT PRIMARY KEY`), `save_evidence` upsert overwrites; reuse via `reuse_cached_evidence` handler `orchestrator_agent.py` L339–355 |
| Config | `utils/config.py` L152 `get_reasoning_effort` — zero callers (grep-verified); gpt-5.4 default; `vector_store.py` L98–99 (1000/150), L172–173 (oversample 8 / min 20) |
| DEEP.md | Read fully (801 lines). Implemented so far: evidence accumulation+dedup (§4.3 Change 1), minimal gap-driven re-fetch, fixed 10-iteration budget + `finish_research`. **Not implemented:** Decomposer (§3.2), Investigator (§3.3), Synthesis (§3.4), Divergent (§3.5), structured `VerificationReport` (§3.6/Change 2), per-topic memory table (§3.8/Change 3), sufficiency termination (§6), reasoning-effort tuning (§10) |

**Additional findings that sharpen the plan:**

1. **The NO_WEB_SEARCH flag is a case-sensitivity bug, not a heuristic.** `verifier_agent.py` L239: `has_web = 'web_search' in evidence_text or 'web_evidence' in evidence_text`. The formatted evidence header is `"Web evidence:"` (capital W, from `helpers.py` L55) and the JSON dump never contains the literal `web_search`/`web_evidence`. So `is_no_web` is **always True** → `depth_override` is always non-empty → `re_retrieve` is force-True on *every* verify call, even when web evidence was used. This single line is the engine of the 10-iteration script.
2. **`get_reasoning_effort` is dead twice.** No callers anywhere, *and* `get_config()` never loads `*_REASONING_EFFORT` env vars (the dataclass fields at `config.py` L98–101 default to `"low"`), so the `.env.example` knobs documented in `get_env_example()` are silently ignored.
3. **`run_model` already supports structured output** — `model_runner.py` L217–219: `client.responses.parse(**request, text_format=text_format)`. DEEP.md Change 2 (structured `VerificationReport`) is therefore a verifier-prompt + model-class change, not new infrastructure.
4. **The UI save path is a second, different truncation.** `ui/gradio_handlers.py::handle_save_report` calls `save_report(report, query=..., session_id=...)` with **no state and default config** → body capped at `ReportConfig.default().max_content_length=10000` and no evidence dump. The CLI path (`run_orchestrator.py`) is the one with the 5000 cap + broken dump. Both need fixing.

**Evidence-dump field mismatch (RC7), exact read vs write (verified):**

| Field | Read side (`save_report.py::_parse_evidence_json`, L245–296) | Write side (retriever / Tavily) | Result in reports |
|---|---|---|---|
| doc title | `chunk.get("source", "Unknown")` | `document_name`, `document_title` (Qdrant payload `source` key is `""` — `vector_store.py` `create_document_embeddings`) | 71× "Unknown" |
| doc text | `chunk.get("content", "")` | `content` | renders (matches) |
| doc score | `chunk.get("score", 0.5)` | `score` | renders |
| web url | `result.get("url")` | `url` | renders |
| web text | `result.get("snippet", "")` | Tavily `content` (and `raw_content`) | empty |
| web score | `result.get("relevance_score", 0.5)` | Tavily `score` | uniform "Web Score: 50.0%" |

`ORCHESTRATOR_REPORT_SUMMARY.md` documents the *assumed* schema (`source`/`snippet`/`relevance_score`) — the doc was written to the wrong assumption and must be updated.

## A. Quality Gap Statement

On the same topic (genetic programming) and same 5 local PDFs, the Gemini PDF delivers ~2,824 words of body across 5 topical sections with subsections, 117 inline citations (~1 per 24 words), 32 resolved references, named papers and operator-level mechanics, and an explicit synthesis section — in 11 pages. This tool's Report 1 is a 1,873-line file where only ~670 words are actual prose (~5%), the body is **amputated mid-sentence at 5,000 chars by `ReportConfig.strict()`** (L159–165) before §2.2 finishes, has 8 inline citations, zero resolved references, and is padded by a 71-chunk "Supporting Evidence" dump whose titles are all "Unknown" and whose web chunks have empty text and a uniform fake "50.0%" score — all caused by the `_parse_evidence_json` field mismatch above. Report 2 is worse in kind: the verifier's "return only the final answer / keep it concise" prompt (L144/L148) turns the verifier into the final editor, collapsing the output to a 535-word executive summary. **The gap is depth, specificity, synthesis, and citation density — not length.**

## B. Target Architecture

```
Query → [1] DECOMPOSER (1 structured call)
        → [2] INVESTIGATOR loop per sub-question (retrieve → sufficiency → re-query; ≤3 rounds)
        → [3] PER-SECTION DRAFT (1 writer call per section, own evidence subset only)
        → [4] CRITIC (structured VerificationReport) → budgeted EXPANSION back to [3]
        → [5] ASSEMBLY (exec summary LAST + resolved references + evidence side-file)
```

**1. Query → Research Plan.** Add a **new Decomposer step** (`worker_agents/decomposition_agent.py`, DEEP.md §3.2), not an orchestrator-prompt redesign — one line of reasoning: the orchestrator LLM runs under `tool_choice="required"` (model_runner.py L195) with full state re-injected every iteration, a hostile environment for planning; a one-shot `text_format` call is cheap, independently testable, and can short-circuit simple queries to a single sub-question (DEEP.md §3.9 Option A: parallel `deep_research_orchestrator.py`, standard path untouched). Output: 5–10 sub-questions, each `{id, question, angle, expected_sources(doc|web|both), priority}` — this checklist later drives the eval rubric.

**2. Per-sub-question Investigator loop** (DEEP.md §3.3/§3.7, termination §6). Extend `retriever_agent.py::retriever_agent()` with `research_goal: str` and `max_rounds: int = 3` (the 4-round loop at L242 becomes per-goal); each round: (a) RAG + web retrieval aimed at the sub-question (dedup/accumulation already exists L286-330), (b) **sufficiency eval** — one structured LLM call `SufficiencyReport{is_sufficient, missing_aspects[], follow_up_queries[]}` comparing chunks to the sub-question, (c) if insufficient and rounds remain, targeted re-query on the gaps. Stop on sufficiency, round cap, or diminishing returns (0 new chunks after merge — DEEP.md §6). **Budget per sub-question: ≤10 doc chunks + ≤5 web results, top-scored.**

**3. Per-section DRAFTING.** New call shape `write_section(user_query, outline, section, evidence_text, prior_summaries)` in `writer_agent.py` — one call per section with **only that section's evidence subset** (replaces the ~70-90KB raw-JSON monologue, RC5). Contract: ≥300 words of substance; ≥1 citation per 2-3 factual sentences using provided citation keys; ≥2 concrete specifics (named paper/method/number) when evidence supports; **banned patterns:** (a) restating the section heading as an opening paragraph, (b) "it is important to note that…" / "in today's world…" filler carrying no fact, (c) closing paragraphs of pure hedging or follow-up offers.

**4. CRITIC.** `verifier_agent.py` stops being the final editor: delete L144 "return only the final answer" and L148 "Keep the writing concise"; the **writer owns final text**. Critic outputs a structured `VerificationReport` via the **existing** `text_format` support (model_runner.py L217-219, DEEP.md §3.6/Change 2): `{is_supported, hallucinated_claims[], unsupported_claims[], per_section[{section_id, grounded, depth_ok, gaps[], expand_queries[]}], confidence_level(low|med|high), re_retrieve_suggested, specific_queries[]}`. **Expansion loop:** sections failing `grounded`/`depth_ok` go back to `write_section` with the gap list — **≤2 revisions per section, global budget ≤8 expansion calls**. The EVIDENCE_STATUS block survives only as this machine-side struct; it never enters user-facing text (RC2 leak).

**5. ASSEMBLY.** (a) Executive summary written **last** — one writer call over the finished sections' one-paragraph summaries (DEEP.md §7.5). (b) **References section resolved, not generated**: a citation-key registry built at investigation time maps `D1…/W1… → metadata`; references = cited keys numbered by first appearance. Metadata that **must** ride end-to-end: doc chunks `document_title, document_name, page_number, citation, score` (already in Qdrant payload and `retrieve_document` output); web `title, url, content, score, published_date` (whitelist Tavily fields, drop `raw_content`). (c) Raw evidence dump **out of the body** → optional side file `{report}.evidence.md`, correctly attributed (fixes RC7's "Unknown" dump).

## C. Prioritized Steps

### P0 — QUICK WINS (1-2 days)

| ID | Change (file:line) | RC | Impact | Effort | Deps |
|---|---|---|---|---|---|
| P0-1 | **Un-cap report bodies, cap the appendix.** `save_report.py` L159-165: new `ReportConfig.research()` — body uncapped, `max_snippet_length≈800`; L333-337 body sanitize keeps control-char strip only (no length cut); L461-467 delete the 2000-char Verification cap. `run_orchestrator.py:32` → `research()` config. `gradio_handlers.py::handle_save_report`: same config (kills the second 10000-cap path). Update `tests/test_save_report.py`. | 1 | Mid-sentence amputation gone on both CLI and UI saves | S | — |
| P0-2 | **Kill the always-true depth trap.** `verifier_agent.py` L239: replace the case-buggy substring `has_web` with a real signal — pass `route_used` from the evidence payload as a parameter. L266-273: force-override only when `coverage=='thin' AND gaps non-empty`. Parse-failure default (≈L208): `re_retrieve: True → False` + log. `orchestrator_agent.py` L618-632: delete the `'[DEPTH OVERRIDE'` substring detection and the `moderate` override; act only on the parsed block's actual `re_retrieve` bool (L646). | 2 | Queries stop running the deterministic 10-iter discard-refetch script; loops become evidence-driven | S-M | — |
| P0-3 | **Strip ~~~ block from user text.** `verifier_agent.py` L276-291: return `(clean_text, status_dict)` instead of concatenating the block; `orchestrator_agent.py` L540-616 consumes the pair; `state["verification"]`/`final_answer` (L663, L721) and UI `state["last_report"]` hold clean text. | 2, 6 | Report-2-style leaked `~~~ EVIDENCE_STATUS ~~~` blocks disappear | S | P0-2 |
| P0-4 | **Delete both boilerplate gap-query copies** (orchestrator L636-643, verifier L266-273 "attention variants…"); gap queries come **only** from the critic's structured `specific_queries` (P0-2's parse, P1-4's model). | 3 | No transformer-era text injected into any topic | S | — |
| P0-5 | **Fix/defeature the evidence dump.** `save_report.py::_parse_evidence_json` L245-296: read real write-side fields — doc `document_title`/`document_name`/`page_number`/`citation`/`score`; web `title`/`url`/`content`/`score`. New `ReportConfig.include_evidence_dump=False` default; when on, dump goes to side file `{report}.evidence.md`. Update `ORCHESTRATOR_REPORT_SUMMARY.md` (currently documents the wrong schema). | 7 | No more 71×"Unknown" / empty web text / uniform 50%; body ≈ prose | S | — |
| P0-6 | **Add `max_output_tokens`** to `model_runner.py::run_model` (Responses API param — client is `client.responses.create`, L219); include in request dict L188-201 when set. Writer/verifier 16000; retriever/orchestrator 2000. | 5 | Guarantees full-length writer output (provider default applies today) | S | — |
| P0-7 | **Wire the dead reasoning-effort knob.** `utils/config.py::get_config()`: load `*_REASONING_EFFORT` env vars (documented in `.env.example`, never loaded); defaults writer/verifier `"medium"`, retriever/orchestrator `"low"` (DEEP.md §10); replace all hardcoded `reasoning_effort="low"` (writer L70, verifier, orchestrator, retriever) with `get_config().get_reasoning_effort(agent_name)` (L152, currently zero callers). | 5 | Better drafting/verification; config actually works | S | — |
| P0-8 | Replace "at least 2000 words" (`writer_agent.py` prompt L57) with a **REQUIRED OUTLINE**: step 1 emit `## Report Outline` — 7-8 `##` sections, parameterizable by topic: 1 Definition & Background · 2 Core Components & Mechanics · 3 Major Variants & Alternative Approaches · 4 Dynamics/Evaluation/Analysis · 5 Applications · 6 Tools & Ecosystem · 7 Limitations & Open Problems · 8 Synthesis; step 2 write each section ≥300 words of substance. Make the dead length gate act (`orchestrator_agent.py` L557-560): on `draft < 1500 chars`, pass a `short_draft=True` flag into the critic input (or delete the print). | 5 | Outline-based writer gives immediate structural gain toward Gemini's sectioned shape; length gate finally feeds the depth check instead of printing | S | — |

### P1 — RESEARCH-LOOP REWORK (1–2 weeks)

| ID | Change | RC | Impact | Effort | Deps |
|---|---|---|---|---|---|
| P1-1 | **Decomposer** (DEEP.md §3.2): new `worker_agents/decomposition_agent.py`; one-shot `run_model(text_format=ResearchPlan)`; 5-10 sub-questions `{id, question, angle, expected_sources(doc\|web\|both), priority}`; prompt requires MECE + each investigable via `retrieve_document`/`web_search` (fed the indexed doc catalog); simple query → short-circuit to 1 sub-question (DEEP.md §7.3). | 5 | Per-query depth plan; source of the eval coverage checklist | M | P0-7 |
| P1-2 | **Per-sub-question investigator** (DEEP.md §3.3/§6): extend `retriever_agent.py::retriever_agent()` with `research_goal=""`, `max_rounds=3` (replaces fixed `range(4)` L242 per goal); loop: retrieve → structured `SufficiencyReport{is_sufficient, missing_aspects[], follow_up_queries[]}` (text_format) → targeted re-query on gaps; stop on sufficiency / round cap / 0 new chunks after dedup (diminishing returns, §6); budget **≤10 doc + ≤5 web chunks per sub-question**, top-scored; reuse existing dedup/accumulation (L286-330). | 2, 4 | Adequate, relevant evidence per sub-question; rounds evidence-driven, not fixed | L | P1-1 |
| P1-3 | **Per-section drafting**: `writer_agent.py::write_section(user_query, outline, section, evidence_text, prior_summaries)` — one call per section with only that section's top-scored subset, replacing the whole-pack raw `json.dumps(indent=2)` injection in `memory/helpers.py::build_evidence_context` L51-60; contract: ≥300 words, ≥1 citation per 2-3 factual sentences (keys from P1-5), ≥2 named specifics, banned: heading restatement, no-fact filler ("it is important to note…"), hedging-only closers. | 5 | Core depth gain: focused, cited, specific sections; writer context ~4-5× smaller | M | P1-2, P1-5 |
| P1-4 | **Critic → writer expansion loop** (DEEP.md Change 2): `verifier_agent.py` becomes pure critic (delete L144/L148 final-editor instructions); `VerificationReport` via existing `text_format` (`model_runner.py` L217-219): `{is_supported, hallucinated_claims[], unsupported_claims[], per_section[{section_id, grounded, depth_ok, gaps[], expand_queries[]}], confidence_level, re_retrieve_suggested, specific_queries[]}`; **≤2 revisions/section, ≤8 global expansion calls**; try/except + text-parse fallback (§7.2); writer owns final text; status stays machine-side (P0-3). | 6 | Thin/ungrounded sections fixed in place — no whole-draft discard (L457-458), no compression to summary (Report 2) | L | P0-3, P1-3 |
| P1-5 | **End-to-end attribution**: Qdrant payload `document_title/document_name/page_number/citation/score` must survive `similarity_search` → `retrieve_document` chunk dict → writer context → references; Tavily whitelist `title/url/content/score/published_date` (drop `raw_content`), `search_depth="basic"→"advanced"` (L99), fix `save_report.py::_parse_evidence_json` L245-296 read map (`snippet→content`, `relevance_score→score`, `source→document_title`); `build_evidence_context` L51-60 emits citation-keyed format (D1/W1) + key→metadata registry in orchestrator state. | 7 (5) | Citations mechanically resolvable; references assemblable; no "Unknown" dumps | M | — |
| P1-6 | *(optional)* **Web-only mode**: when the indexed-document catalog is empty/unrelated, skip RAG entirely per sub-question — pure Tavily investigation; exercises P1-5 web-field path. | 4 | Robust no-PDF path (eval topic in E) | S | P1-2, P1-5 |

### P2 — COMPLETION + EVALUATION (ongoing)

| ID | Change | RC | Impact | Effort | Deps |
|---|---|---|---|---|---|
| P2-1 | **Synthesizer** (DEEP.md §3.4): add **only after** P1 assembly proves per-section coverage — the last-written exec summary already stitches sections; the synthesizer's unique value is the explicit "Synthesis & Strategic Trajectories" section + cross-document contradiction handling. Defer if post-P1 eval shows synthesis gap is small. | 5 | Closes the remaining Gemini synthesis gap | M | P1-4 |
| P2-2 | **Per-topic evidence memory** (DEEP.md §3.8/Change 3): new `evidence_accumulation` table in `memory/memory.py` (composite PK `session_id, sub_topic, retrieved_at`) alongside the single-row session schema (L41-47, kept for standard mode); `append_evidence`/`get_topic_evidence`; rework `reuse_cached_evidence` (orchestrator L339-355) to match follow-ups to sub-topics, not the stale latest row. | 8 | Follow-ups reuse relevant evidence without full re-research; history preserved | M | P1-2 |
| P2-3 | **`MAX_LLM_CALLS` global budget** + `should_terminate` (DEEP.md §6): deep orchestrator cap ≈40 calls; log calls/latency/cost per run to report footer + UI trace. | 2 | No runaway cost; per-run accountability | S | P1-1…P1-4 |
| P2-4 | **Reasoning-effort tuning** (DEEP.md §10): per-agent defaults (writer/critic `medium`; retriever/sufficiency/decomposer `low`); A/B via the eval harness. | 5 | Quality-per-dollar optimized | S | P0-7, E |
| P2-5 | **Eval harness (E) + UI**: section-by-section streaming in the `ui/gradio_handlers.py` `chat()` generator — yield after each assembled section (append to the assistant entry; existing 0.1s-poll generator protocol unchanged); **standard-vs-deep mode toggle** (UI checkbox + `--mode` CLI flag + config env var). | — | Visible progress on 3-4× longer runs; cheap queries stay cheap | S-M | P1-4, P2-3 |


## D. Citation & Specificity Strategy

**Chunk metadata requirement (drop, don't degrade):** every doc chunk dict MUST carry `{document_name, document_title, page_number, chunk_id, citation, content, score}`; every web result MUST carry `{title, url, content, score}` (+ `published_date` if present). Any chunk missing a required field is **excluded from the citation registry** (may appear as context, gets no key).

**Writer prompt contract:** cite every factual sentence with a **provided** key (`[D1]`, `[D1, W2]`); never invent keys (validator (2) catches it); quantitative claims (numbers, dates, named results) always cited; when the subset doesn't cover a needed fact, write "the available evidence does not cover X" — honest gap, no memory-fill.

**Post-hoc deterministic validation** (no LLM; run in assembler/`save_report`; results to report footer + eval):
1. **Uncited-claim heuristic** — factual sentences (contain a number, named work, or assertion) lacking a key: section ratio > threshold → expansion-loop input.
2. **Unresolvable citation keys** — extracted `[D#/W#]` absent from registry → flag as invented citation.
3. **Reference list ≡ cited keys** — exactly the cited set: orphans (registry keys never cited) dropped; any key without a reference entry is an error.
4. **Section word counts** — each ≥300 (contract check; catches RC1-style truncation at section granularity).

**Where the references list is built:** in the **assembler**, not by the LLM — the citation-key registry is populated at investigation time (P1-5) from chunk-dict/Tavily metadata; entries numbered by **first appearance** in the body, rendered `[n] Title. *file.pdf* p. X.` / `[n] Title. URL (date).` Hallucinated bibliographies are structurally impossible.

## E. Evaluation Plan

**Rubric:**

| Dimension | How measured | Target (GP topic) |
|---|---|---|
| Coverage | % of decomposed sub-questions addressed with ≥2 specifics each (LLM-judge vs the sub-question checklist) | ≥80% |
| Specificity | Count of named works / numbers / dates (deterministic regex + judge) | ≥ Gemini baseline density (named entities per 100 words) |
| Citation density | Citations per 100 words (deterministic) | ≥4 (Gemini baseline: 117/2824 ≈ 4.1) |
| Structure | All planned sections present; body ends at a section boundary (deterministic truncation check) | 100% |
| Synthesis | Comparisons, limitations, explicit conclusion present (LLM judge) | Present |
| Sourcing | Reference count; % primary sources (arXiv/PMC/university repos/journals); zero "Unknown" titles; zero empty web snippets | ≥20 refs, ≥50% primary, 0 "Unknown" |

**Benchmark procedure:**
1. **Baseline:** the two existing reports (kept in `reports/`) + the Gemini PDF.
2. **Post-P0:** regenerate the GP topic ("create a comprehensive report covering genetic programming") via CLI; run deterministic checks (no mid-sentence truncation, no `~~~` block, no "Unknown", references list present); compare section-by-section vs the Gemini PDF.
3. **Post-P1:** regenerate; full rubric with the LLM-judge pass; target ≥80% rubric score with zero deterministic failures.
4. **Web-only topic:** one topic with NO local PDFs (outside the GP corpus) to exercise the Tavily path end-to-end.

**Automation:** a `scripts/eval_report.py`-style harness: deterministic checks (truncation regex, citation-key count, "Unknown" occurrences, section word counts) + one LLM-judge call with the rubric as the system prompt; save scored results per run for regression tracking.

## F. Risks & Do-Not-Change

- **Cost/latency:** deep mode ≈ decomposer 1 + per-sub-question (3 rounds × retrieve + sufficiency ≈ 6) × up to 8 sub-questions + 8 section drafts + 1 exec summary + 1 critic + ≤8 expansions ≈ **30–45 LLM calls vs ~10 today → ~3–4× cost** per deep report; smaller per-section writer contexts partially offset token cost. Mitigations: hard budgets (P2-3 `MAX_LLM_CALLS≈40`, per-sub-question and revision caps above) and a **standard mode** that keeps today's single-write path for short queries.
- **Gradio streaming:** the generator protocol in `gradio_handlers.py` yields on a 0.1s poll — keep it, and yield after each assembled section (P2-5) so 3–4× longer runs show visible progress; verify the `chat()` generator stays compatible with the multi-section flow.
- **Do NOT change now:** Qdrant ingestion chunking 1000/150 (`vector_store.py` L98–99) — re-evaluate only if post-P1 eval shows retrieval, not writing, as the bottleneck; model choice (gpt-5.4); the orchestrator loop skeleton (budget + `finish_research`) — it is reused by the deep orchestrator rather than replaced.
- **Regression risk:** P0 touches `save_report` (covered by `tests/test_save_report.py` — update it), the orchestrator loop, and the verifier output shape — run the existing test suite plus the `test_orchestrator_report.py` end-to-end script after P0 lands.
- **Sequence:** P0 in one branch → re-run the GP benchmark → P1 in order with **attribution (P1-5) first** (per-section drafting, the critic, and references all depend on clean chunk metadata) → P2 items as needed.
