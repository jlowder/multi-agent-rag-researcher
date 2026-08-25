# Multi Agent RAG Researcher
![cover](rag-researcher.png)

Multi Agent RAG Researcher is based on an orchestrator that coordinates three worker agents to work together on a topic and generate grounded content. It works with two evidence sources: local PDF documents and the web.

The three worker agents are:

- Retriever Agent: retrieves information from local documents (PDFs), the web, or both.
- Writer Agent: writes the response based on the retrieved evidence.
- Verifier Agent: checks the draft written by the Writer Agent and returns the final verified response.

## Key Components

1. A configurable OpenAI-compatible LLM — project default `Ornith-1.0-35B-MLX-oQ8` via a local omlx server — used by all agents, with per-agent model, reasoning-effort, and output-token overrides
2. Step-by-step function calling that allows the agents to interact with one another
3. Qdrant vector database for local PDF retrieval
4. Tavily for web search
5. SQLite for short-term memory
6. Gradio UI for browser-based interaction
7. An optional 5-stage deep-research pipeline (`--mode deep`) for long-form, heavily cited reports

## Multi-Agent Architecture

### Data Sources

1. Qdrant Vector Database

Information retrieval from PDFs is handled in the following stages:

- Multiple PDFs can be loaded from the `docs/` folder or uploaded through the UI.
- Documents are split into chunks, converted into embeddings, and stored in a local Qdrant collection.
- Similarity search is then used to retrieve the most relevant chunks across the indexed documents.
- The retrieved chunks include citation metadata such as document name and page number.

2. Tavily Web Search

Tavily is used to retrieve up-to-date or external information from the web. The retriever can use it when:

- the indexed PDFs do not cover the query
- document evidence is weak or incomplete
- newer information is needed

### Worker Agents

1. Retriever Agent

The role is:

- It uses two tools: PDF document retrieval and web search.
- Given a query, it decides whether to use local documents, web search, or both.
- If local document evidence is missing or weak, it can fall back to web search to gather broader or more up-to-date context.

2. Writer Agent

The role is:

- It receives the retrieved information from the Retriever Agent.
- It writes a grounded draft based on the available evidence.
- It includes supporting citations from PDFs or web sources when they are available.

3. Verifier Agent

The role is:

- It receives the draft from the Writer Agent together with the evidence.
- It checks whether the claims in the draft are supported by the retrieved evidence.
- It returns the final verified response.

### Memory

SQLite is used to provide short-term memory for the multi-agent workflow. For a given session ID, the system stores:

- the latest user query
- the latest retrieved evidence for that session

This allows the orchestrator to reuse relevant evidence for follow-up questions instead of retrieving the same information again every time.

### Orchestrator

The orchestrator coordinates the three worker agents: Retriever, Writer, and Verifier.

#### Working Mechanism

- It receives the user query and, depending on the query, may respond directly or begin the evidence-based workflow.
- For a research query, it first checks whether relevant cached evidence from the current session can be reused.
- If cached evidence is not enough, it calls the Retriever Agent to gather evidence from PDFs, the web, or both.
- If there is document evidence but the evidence is weak, the Retriever Agent can also fetch up-to-date information from the web to supplement the local document information.
- The orchestrator then passes the active evidence and the user query to the Writer Agent so it can generate a grounded draft.
- Next, it sends the draft and evidence to the Verifier Agent, which checks the claims and returns the final verified report.
- During the session, the latest query and retrieved evidence are stored in memory for follow-up questions.
- In follow-up questions, the orchestrator may reuse cached evidence instead of calling the Retriever Agent again, then continue with the Writer Agent and Verifier Agent to generate the final response.

Note: The orchestrator has a guardrail that keeps the system focused on research and factual questions. It refuses unrelated general tasks such as coding help or simple math because the goal of the system is to function as a research assistant.

### Deep Research Mode (5-Stage Pipeline)

Deep mode (`--mode deep`) is a separate pipeline in `deep_research_orchestrator.py` (repo root; entry point `deep_research(user_query)`, returning `{"final_answer", "state", "stats"}`). It reuses the retriever and writer and adds a decomposer:

1. **Decompose** — `worker_agents/decomposition_agent.py` turns the query into a structured ResearchPlan of 5-10 sub-questions (each with `sub_question`, `angle`, `heading`, `expected_sources`, `priority`, and a code-generated `id`). Parsing has four fallback paths — structured output, structured retry, plain-text JSON, then a deterministic fallback — so a plan always succeeds.
2. **Investigate** — one research pack per sub-question: retrieve, then an LLM sufficiency check; up to 3 rounds, up to 10 doc and 5 web chunks per round.
3. **Per-section draft** — one writer call per section, given only that section's evidence plus a citation contract.
4. **Critic** — a structured per-section verdict pass (grounding, depth, and citation density ≥4 per 100 words) with budgeted revisions (max 2 per section, 8 total); under-cited sections are sent back for revision.
5. **Assembly** — the executive summary is written last; inline citation keys are renumbered to `[1..N]` and the References section is rendered deterministically from the citation registry.

With ≥3 sections, a final **Synthesis** section is drafted after the critic pass, connecting the content sections by name (skipped when <3 sections or the budget is tight).

A per-sub-question evidence cache reuses recent retrievals across runs: stage 2 reuses a pack for a near-identical sub-question (≥0.8 token overlap) retrieved within `EVIDENCE_CACHE_TTL_DAYS` (default on, 30 days) — disabled with `EVIDENCE_CACHE_ENABLED=false`.

A global budget caps a run at 40 LLM calls. If the budget runs out or an individual call fails, the pipeline logs a warning and assembles from what exists — it never crashes mid-report.

## Project Structure

```text
.
├── docs/                         # Default PDF files
├── memory/                       # SQLite-backed session memory helpers
├── qdrant_vector_database/       # PDF ingestion and similarity search
├── ui/                           # Gradio app and UI handlers
├── utils/
│   ├── requirements.txt          # Python dependencies
│   ├── var.env                   # Local API keys (gitignored; seeded from .env.example)
│   ├── memory.db                 # Created at runtime
│   └── qdrant_storage/           # Created at runtime
├── worker_agents/                # Retriever, writer, verifier, and decomposer
├── orchestrator_agent.py         # Main coordinator (standard mode)
├── deep_research_orchestrator.py # 5-stage deep-research pipeline (deep mode)
├── tests/                        # pytest suite
└── run_orchestrator.py           # CLI entry point
```

## Setup Project

### Prerequisites

- Python 3.10 or newer
- Tavily API key
- For local LLMs: Ollama, LocalAI, or LM Studio installed and running

### Installation

1. Clone the repository:

```bash
git clone https://github.com/ayoolaolafenwa/multi-agent-rag-researcher.git
cd multi-agent-rag-researcher
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the dependencies:

```bash
pip3 install -r utils/requirements.txt
pip3 install langchain-openai
```

4. Create a `utils/var.env` file with your API keys:

```env
# For OpenAI (default)
LLM_ENDPOINT=https://api.openai.com/v1
LLM_API_KEY=your_openai_api_key
LLM_MODEL=gpt-5.4

# For local LLMs (Ollama example)
# LLM_ENDPOINT=http://localhost:11434/v1
# LLM_MODEL=llama3.1:8b
# LLM_API_KEY=n/a

# Optional: deep-mode decomposer on a different (schema-compliant) model
# DECOMPOSER_MODEL=Qwen3-Coder-Next-MLX-6bit

TAVILY_API_KEY=your_tavily_api_key
```

See the [Configuration](#configuration) section for the full list of supported variables.

**Note:** See [LOCAL_LLM_SETUP.md](LOCAL_LLM_SETUP.md) for comprehensive configuration examples for Ollama, LocalAI, LM Studio, and other OpenAI-compatible services.

5. Place the PDFs you want to index in the `docs/` folder, or upload PDFs later through the UI. The project ships with five genetic-programming papers in `docs/` (`01_koza_genetic_programming_1992.pdf` through `05_statistical_ml_elements_gp.pdf`), so you can index those directly or replace them with your own documents.

## Run Project

When the CLI starts, it ingests the PDFs in `docs/` into the local Qdrant store, then runs an interactive stdin loop: it prompts `User: ` for a query, and the session ends when you type `q` (the prompt also accepts `exit`/`exist`).

### Standard mode (default)

```bash
python3 run_orchestrator.py
```

The original orchestrator loop: the orchestrator drives the retriever, writer, and verifier as LLM tools. A run takes about 10 LLM calls and a few minutes — best for quick questions over the indexed documents.

### Deep mode

```bash
python3 run_orchestrator.py --mode deep
```

The 5-stage deep-research pipeline (decompose → investigate → per-section draft → critic → assembly, described above). A run typically takes 25-35 LLM calls and 15-30 minutes, under a global budget of 40 calls. If the budget runs out or a call fails, the pipeline logs a warning and assembles from what exists — it never crashes mid-report.

Add `--debug` to either mode for verbose logging and saved-report metadata.

### With Local LLMs

Set the environment variables before running:

```bash
# Ollama example
export LLM_ENDPOINT=http://localhost:11434/v1
export LLM_MODEL=llama3.1:8b
export LLM_API_KEY=n/a

python3 run_orchestrator.py
```

See [LOCAL_LLM_SETUP.md](LOCAL_LLM_SETUP.md) for configuration examples for Ollama, LocalAI, LM Studio, and other OpenAI-compatible services.

### Saved Reports

Both modes save each answer to a timestamped `reports/<query-slug>_<timestamp>.md` file through the same flow: uncapped body and verification text, capped evidence snippets, and verification metadata (confidence level, coverage) in the report header.

In deep mode the report is an executive summary, one section per sub-question, and a `## References` section numbered `[1..N]` by first appearance in the body. Every inline citation resolves to a reference entry: local documents show title, source file, and page; web results show title, URL, and date.

An optional raw-evidence side-file (`<report-stem>.evidence.md`, written next to the report) is enabled by passing `include_evidence_dump=True` in `save_report`'s `ReportConfig`; it is off by default in both modes.

## Run UI for Multi-Agent Chat

Start the Gradio UI:

```bash
python3 ui/gradio_app.py
```

The UI automatically loads the default PDFs from `docs/` on startup. If you upload new PDFs, they replace the active indexed document set for that UI session.

The UI has a **Mode** selector (standard, default / deep). Deep mode runs the 5-stage pipeline and streams the report section-by-section as it is drafted, with a live stage/elapsed-time status line.

## Configuration

The live env file is `utils/var.env` (gitignored). It is auto-seeded from `.env.example` on first run, so changes to `.env.example` only affect fresh setups — edit `utils/var.env` to change this project's configuration.

- `LLM_MODEL` — default model for all agents. The project default is `Ornith-1.0-35B-MLX-oQ8` served by a local OpenAI-compatible omlx server (`LLM_ENDPOINT=http://localhost:8080/v1`), which serves several models concurrently.
- `DECOMPOSER_MODEL` — per-agent model override for deep mode's decomposer only. Its structured schema is produced more reliably by the larger model (`Qwen3-Coder-Next-MLX-6bit` in the project config); every other agent stays on `LLM_MODEL`.
- Per-agent `*_REASONING_EFFORT` (`low`/`medium`/`high`) and `*_MAX_OUTPUT_TOKENS` for all six agents (`retriever`, `writer`, `verifier`, `orchestrator`, `decomposer`, `sufficiency`). Defaults: writer and verifier run `medium` effort with 16,000 output tokens; retriever, orchestrator, and decomposer run `low` with 2,000; sufficiency runs `low` with 1,000.
- The four original agents also accept full `*_ENDPOINT`/`*_API_KEY`/`*_MODEL` overrides (the decomposer accepts a model override only).
- `TAVILY_API_KEY` powers web search; `EMBEDDING_ENDPOINT`/`EMBEDDING_MODEL`/`EMBEDDING_API_KEY` configure chunk embeddings.

## Testing

```bash
venv/bin/python -m pytest tests/ -q
```

The pytest suite (84 tests) covers the save-report flow, citation context and key renumbering, decomposition (structured and fallback parsing), per-sub-question investigation, the deep pipeline end-to-end, and config overrides. Use the project virtualenv — system Python lacks the project dependencies.

## Notes

- Session memory is stored in `utils/memory.db`.
- Local Qdrant data is stored in `utils/qdrant_storage/`.
- The system is designed for research and factual question answering, not for unrelated general-purpose tasks.
