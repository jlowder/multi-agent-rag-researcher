# API Server

A FastAPI service exposing the 5-stage deep-research pipeline over HTTP. POST a topic to get a task id, poll status and current step, then GET the finished report as the structured JSON envelope (`{schema_version, report, quality}`) that paperbot consumes for PDF/HTML rendering.

## Overview

- Same pipeline as the Gradio UI's deep mode (`deep_research` in `deep_research_orchestrator.py`) — just the HTTP layer, no UI.
- In-memory task store: single process; tasks do not survive a restart.
- One run at a time: `deep_research` holds a process lock, and the service returns 409 while a run is in progress.
- A per-task watchdog (45 min by default) marks a run that exceeds the deadline as failed.
- The finished report is the canonical `ResearchReport` JSON (the same structured document deep mode saves under `reports/`), which is exactly what paperbot's `POST /render` accepts.

## Quick start

Dependencies are pinned in `utils/requirements.txt` (`fastapi`, `uvicorn`):

```bash
pip3 install -r utils/requirements.txt
venv/bin/python api_server.py   # PORT env (default 8000), HOST env (default 0.0.0.0)
```

The service needs the LLM configuration from `utils/var.env` (`LLM_ENDPOINT`, `LLM_API_KEY`, `LLM_MODEL`, with `OPENAI_*` fallbacks) to actually run research. Check before posting topics:

```bash
curl -s localhost:8000/health
# {"service":"multi-agent-rag-researcher","running":false,"deep_configured":true}
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/research` | Start a run → 202 with task id (409 while one is running, 422 on invalid body) |
| GET | `/research` | List all tasks (summaries) |
| GET | `/research/{id}` | Full task record: status, current_step, step timeline, stats |
| GET | `/research/{id}/report` | Raw structured report JSON once completed |
| GET | `/health` | Service status, incl. whether the deep pipeline is configured |

### POST /research

Body:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `topic` | string | — | Research query. Required, must be non-empty. |
| `max_rounds` | int | 3 | Max investigation rounds per sub-question. |
| `budget_doc` | int | 10 | Max doc chunks kept per sub-question. |
| `budget_web` | int | 5 | Max web results kept per sub-question. |

202 Accepted:

```json
{
  "task_id": "4bd2c028651a4d8091088b48aa14186d",
  "status": "running",
  "current_step": "queued",
  "links": {
    "status": "/research/4bd2c028651a4d8091088b48aa14186d",
    "report": "/research/4bd2c028651a4d8091088b48aa14186d/report"
  }
}
```

Other responses:

- 409, while another run is in progress: `{"error": "a research run is already in progress", "running_task_id": "<id>"}`
- 422, invalid body (e.g. empty `topic`)

### GET /research

```json
{
  "tasks": [
    {
      "id": "4bd2c0…",
      "topic": "What is a vector field",
      "status": "running",
      "current_step": "draft: drafting 1 section(s)",
      "step_count": 5,
      "started_at": 1788272018.03,
      "finished_at": null,
      "error": null
    }
  ]
}
```

### GET /research/{id}

200 — the full record (`error` present only on failure; `stats` present only after the run produced a result):

```json
{
  "id": "4bd2c0…",
  "topic": "What is a vector field",
  "status": "completed",
  "current_step": "assemble: complete (structured): 1 section(s), 3 source(s)",
  "steps": [
    {"stage": "decompose", "detail": "decomposing query: What is a vector field", "ts": 1788272018.1}
  ],
  "started_at": 1788272018.03,
  "finished_at": 1788272243.33,
  "stats": {"llm_calls": 5, "wall_s": 225.3, "sections": 1, "revisions": 1}
}
```

404, unknown id: `{"error": "unknown task: <id>"}`

### GET /research/{id}/report

- 200, when completed — the raw report JSON (`Content-Type: application/json`):

```json
{
  "schema_version": "1.0",
  "report": {
    "metadata": {"title": "Vector Field Overview"},
    "executive_summary": ["…"],
    "sections": [{"id": "sq1", "heading": "Definition of vector field", "blocks": ["…"]}],
    "sources": ["…"]
  },
  "quality": {"citation_density": {"…": "…"}, "verification": {"…": "…"}, "sources_count": {"…": "…"}, "total_words": 265}
}
```

- 409, while running or failed: `{"status": "running"}` (or `"failed"`)
- 404, unknown id: `{"error": "unknown task: <id>"}`

### GET /health

```json
{"service": "multi-agent-rag-researcher", "running": false, "deep_configured": true}
```

`running` is true while any task is in progress; `deep_configured` is true when the config has both an endpoint and an API key.

## Step tracking

`current_step` (and every `steps[]` entry) is populated by the pipeline's `on_stage` / `on_section` progress callbacks. Stage numbers map to `decompose` (1), `investigate` (2), `draft` (3), `critique` (4), `assemble` (5); each drafted section adds a `section i/n` step. A real run of "What is a vector field" (1 round, 1 section) recorded:

```
queued
decompose: decomposing query: What is a vector field
investigate: investigating 1 sub-question(s)
investigate: investigating sub-question 1/1: What is a vector field Definition
draft: drafting 1 section(s)
section 1/1: Definition of vector field
critique: critic pass: checking every drafted section
assemble: assembling final report
assemble: complete (structured): 1 section(s), 3 source(s)
```

Poll `GET /research/{id}` (every ~10 s) and watch `current_step` advance through that sequence.

## Status lifecycle

```
running ──▶ completed   pipeline returns; stats + report stored
        └──▶ failed     exception (error = "<ExceptionType>: <message>")
                            or watchdog timeout (error = "timed out after 2700s")
```

Known limitation: Python threads cannot be killed. A run the watchdog has marked failed keeps executing in the background until the pipeline finishes on its own, and the service keeps rejecting new runs with 409 until it does.

## Integration with paperbot

The report endpoint returns exactly what paperbot consumes — the structured `{schema_version, report, quality}` envelope that paperbot's `POST /render` takes, so a finished run pipes straight into PDF/HTML rendering. Verified end-to-end: topic "What is a vector field" (`max_rounds=1, budget_doc=2, budget_web=1`) → 225 s run → 1 section / 3 sources → 211 KB, 2-page PDF rendered in 0.25 s.

```bash
# 1. Start a run
curl -s -X POST localhost:8000/research -H 'content-type: application/json' \
  -d '{"topic":"What is a vector field","max_rounds":1,"budget_doc":2,"budget_web":1}'
# → {"task_id":"4bd2c0…","status":"running","current_step":"queued","links":{…}}

# 2. Poll until status becomes "completed"
curl -s localhost:8000/research/4bd2c0…

# 3. Fetch the structured report
curl -s localhost:8000/research/4bd2c0…/report -o report.json

# 4. Render with paperbot
curl -s -X POST -H 'content-type: application/json' -d @report.json \
  'http://paperbot-host:3000/render?format=pdf' -o report.pdf
```

`?format=html` renders the same envelope to HTML.

## Notes

- No authentication: bind to localhost or put the service behind an authenticating proxy.
- CORS is not configured — the service is intended for server-to-server use.
- Tests: `venv/bin/python -m pytest tests/ -q`. `tests/test_api_server.py` covers the full task lifecycle with a fake run function (fully offline) plus one end-to-end test that runs the real pipeline with every LLM/retrieval surface stubbed.
