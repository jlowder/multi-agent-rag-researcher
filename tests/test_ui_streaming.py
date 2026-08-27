"""
Tests for P2-5: deep-mode progress callbacks (deep_research on_stage/
on_section) and the Gradio UI streaming deep handler.

No network, Qdrant, Tavily, or LLM calls: agent surfaces are monkeypatched
(the same stub machinery as tests/test_deep_pipeline.py) and the UI handler
is driven with a fake deep_research.
"""

import importlib
import threading
import time

import deep_research_orchestrator as dpo
from test_deep_pipeline import _basic_env, _install_stubs, _run


# ---------------------------------------------------------------------------
# T1 — pipeline progress callbacks
# ---------------------------------------------------------------------------

def test_pipeline_progress_callbacks_record_stages_and_sections(monkeypatch):
    stages: list[tuple] = []
    sections: list[tuple] = []
    _install_stubs(monkeypatch, _basic_env())
    result = dpo.deep_research(
        "test research query",
        verbose=False,
        output_format="markdown",
        on_stage=lambda s, d: stages.append((s, d)),
        on_section=lambda i, t, h, txt, p: sections.append((i, t, h, txt, p)),
    )

    # Stage events: every stage (1-5) starts, in non-decreasing order, and
    # the last event is the post-stage-5 completion detail.
    assert [s for s, _ in stages] == sorted(s for s, _ in stages)
    assert {s for s, _ in stages} == {1, 2, 3, 4, 5}
    assert stages[0][0] == 1
    assert stages[-1][0] == 5
    assert "complete" in stages[-1][1]
    # Stage 2 reports per-sub-question progress (2 sub-questions in _basic_env).
    assert any("sub-question 1/2" in d for s, d in stages if s == 2)
    assert any("sub-question 2/2" in d for s, d in stages if s == 2)

    # Section events: one per drafted section, 1-based index, and a
    # partial_report that grows (exec summary NOT included — it is last).
    assert [i for i, _t, _h, _txt, _p in sections] == [1, 2]
    assert [t for _i, t, _h, _txt, _p in sections] == [2, 2]
    assert sections[0][2] == "Section One"
    assert sections[0][4] == sections[0][3]  # first partial = first section
    assert sections[1][4] == f"{sections[0][3]}\n\n{sections[1][3]}"
    assert "Synthesized executive summary" not in sections[1][4]

    # The run itself is unaffected: full report assembled as usual.
    assert "## Section One" in result["final_answer"]
    assert "## References" in result["final_answer"]


def test_raising_callbacks_do_not_fail_the_run(monkeypatch, capsys):
    def bad_stage(s, d):
        raise RuntimeError("stage callback boom")

    def bad_section(i, t, h, txt, p):
        raise RuntimeError("section callback boom")

    result = _run(monkeypatch, _basic_env(), on_stage=bad_stage, on_section=bad_section,
                  output_format="markdown")

    # A bad callback must never kill the run (P1 never-crash contract).
    assert "## Section One" in result["final_answer"]
    assert "## References" in result["final_answer"]
    out = capsys.readouterr().out
    assert "on_stage callback failed" in out
    assert "on_section callback failed" in out


# ---------------------------------------------------------------------------
# T2 — UI deep handler: streaming generator + inner-state handoff
# ---------------------------------------------------------------------------

def test_ui_deep_handler_streams_and_hands_inner_state(monkeypatch):
    handlers = importlib.import_module("ui.gradio_handlers")

    def fake_deep_research(query, verbose=False, on_stage=None, on_section=None):
        assert on_stage is not None and on_section is not None
        on_stage(1, "decomposing query: fake")
        on_stage(2, "investigating sub-question 1/2: one")
        on_section(1, 2, "Part One", "## Part One text", "## Part One text")
        time.sleep(0.6)  # let the generator tick and yield the first partial
        on_stage(3, "drafting 2 sections")
        on_stage(5, "assembling final report")
        time.sleep(0.6)  # let the generator tick again before the final
        on_section(
            2, 2, "Part Two", "## Part Two text",
            "## Part One text\n\n## Part Two text",
        )
        on_stage(5, "complete: 2 sections, 3 references")
        return {
            "final_answer": "FULL FINAL REPORT containing Part One and Part Two",
            "state": {
                "evidence_json": (
                    '{"web_evidence": {"results": '
                    '[{"title": "One", "url": "https://ex/u1", "score": 0.9}]},'
                    ' "document_evidence": {"chunks": []}}'
                ),
                "verification_status": {
                    "confidence": "high",
                    "coverage": "comprehensive",
                    "gaps": [],
                },
                "verification": "FULL FINAL REPORT containing Part One and Part Two",
            },
            "stats": {"llm_calls": 7, "sections": 2},
        }

    monkeypatch.setattr(handlers, "deep_research", fake_deep_research)
    state = handlers.build_app_state(ready=True, source="docs/")
    # The handler yields the SAME list/dict objects mutated in place, so
    # snapshot each frame at yield time (that is what Gradio renders).
    frames = []
    for message, history, _st, status in handlers.chat("fake query", [], state, None, False, "deep"):
        assert message == ""  # query box cleared on every yield
        # First frame is the pre-streaming status yield (no assistant entry
        # yet); later frames end in the assistant entry.
        last = history[-1] if history else None
        content = last["content"] if last and last["role"] == "assistant" else ""
        frames.append((content, status))

    assert len(frames) >= 3  # heartbeat-yielded partials before the final
    contents = [c for c, _s in frames]
    statuses = [s for _c, s in frames]
    # Order: stage-status yields, then a partial report, then the final full
    # report (which only appears in the LAST yield).
    assert any("Stage" in s for s in statuses[:-1])
    assert any("## Part One text" in c for c in contents[:-1])
    assert "FULL FINAL REPORT" in contents[-1]
    assert not any("FULL FINAL REPORT" in c for c in contents[:-1])
    assert "Deep research complete" in statuses[-1]

    # State handoff for the Save action: the INNER state (top-level
    # evidence_json), NOT the {"final_answer","state","stats"} envelope.
    # (state is the same dict object the generator mutated in place.)
    handed = state["last_report_state"]
    assert "evidence_json" in handed
    assert "verification_status" in handed
    assert "final_answer" not in handed and "stats" not in handed
    assert state["last_report"] == (
        "FULL FINAL REPORT containing Part One and Part Two"
    )
    assert state["last_report_query"] == "fake query"


def test_ui_deep_handler_error_is_one_line(monkeypatch):
    handlers = importlib.import_module("ui.gradio_handlers")

    def broken_deep_research(*a, **k):
        raise RuntimeError("simulated 500 from the LLM server")

    monkeypatch.setattr(handlers, "deep_research", broken_deep_research)
    state = handlers.build_app_state(ready=True, source="docs/")
    frames = []
    for message, history, _st, status in handlers.chat("fake query", [], state, None, False, "deep"):
        frames.append((history[-1]["content"], status))

    last_content = frames[-1][0]
    assert last_content == "Error: simulated 500 from the LLM server"
    assert "Traceback" not in last_content
    assert "last_report" not in state


# ---------------------------------------------------------------------------
# T3 — UI standard path unchanged
# ---------------------------------------------------------------------------

def test_ui_standard_handler_behavior_unchanged(monkeypatch):
    handlers = importlib.import_module("ui.gradio_handlers")
    calls: dict = {}

    def fake_orchestrator(message, session_id=None, verbose=False,
                           debug_enabled=False):
        calls.update(
            message=message, session_id=session_id,
            verbose=verbose, debug_enabled=debug_enabled,
        )
        return {"final_answer": "STANDARD FINAL ANSWER", "evidence_json": "{}"}

    monkeypatch.setattr(handlers, "orchestrator_agent", fake_orchestrator)
    state = handlers.build_app_state(ready=True, source="docs/")
    frames = []
    for message, history, _st, status in handlers.chat("standard question", [], state, None, False):
        frames.append((message, history[-1]["content"], status))

    # Same invocation as before the mode parameter existed (default mode).
    assert calls == {
        "message": "standard question",
        "session_id": state["session_id"],
        "verbose": True,
        "debug_enabled": False,
    }
    _message, last_content, _status = frames[-1]
    assert last_content == "STANDARD FINAL ANSWER"
    assert state["last_report"] == "STANDARD FINAL ANSWER"
    assert state["last_report_query"] == "standard question"
    # Standard success explicitly clears any stale deep state: the key now
    # holds None, and the save branch treats None the same as absent.
    assert state.get("last_report_state") is None


# ---------------------------------------------------------------------------
# T4 — Gradio app builds (import-level smoke)
# ---------------------------------------------------------------------------

def test_gradio_app_imports():
    app = importlib.import_module("ui.gradio_app")
    assert app.demo is not None  # gr.Blocks instance built at import time


# ---------------------------------------------------------------------------
# T5 — a standard success must clear any stale deep-run state (Save parity)
# ---------------------------------------------------------------------------

def test_standard_success_clears_stale_deep_state(monkeypatch):
    handlers = importlib.import_module("ui.gradio_handlers")

    def fake_deep_research(query, verbose=False, on_stage=None, on_section=None):
        on_section(1, 1, "Part One", "## Part One text", "## Part One text")
        on_stage(5, "complete: 1 section, 1 reference")
        return {
            "final_answer": "DEEP ANSWER",
            "state": {
                "evidence_json": "{}",
                "verification_status": {"confidence": "high"},
            },
            "stats": {"llm_calls": 3},
        }

    def fake_orchestrator(message, session_id=None, verbose=False, debug_enabled=False):
        return {"final_answer": "STANDARD ANSWER", "evidence_json": "{}"}

    state = handlers.build_app_state(ready=True, source="docs/")

    monkeypatch.setattr(handlers, "deep_research", fake_deep_research)
    for _m, _h, _s, _st in handlers.chat("deep q", [], state, None, False, "deep"):
        pass
    assert state["last_report"] == "DEEP ANSWER"
    assert isinstance(state.get("last_report_state"), dict)

    # A subsequent STANDARD success must not leave the deep state behind,
    # or Save would attach the old run's evidence/side-file to the new report.
    monkeypatch.setattr(handlers, "orchestrator_agent", fake_orchestrator)
    for _m, _h, _s, _st in handlers.chat("standard q", [], state, None, False):
        pass
    assert state["last_report"] == "STANDARD ANSWER"
    assert state["last_report_query"] == "standard q"
    assert state["last_report_state"] is None


# ---------------------------------------------------------------------------
# T6 — deep-run lock: concurrent deep runs are serialized / rejected
# ---------------------------------------------------------------------------

def test_deep_run_lock_serializes_concurrent_runs(monkeypatch):
    handlers = importlib.import_module("ui.gradio_handlers")
    release_first = threading.Event()

    def fake_deep_research(query, verbose=False, on_stage=None, on_section=None):
        on_stage(1, "decomposing query")
        release_first.wait(timeout=10)  # hold the run (and the lock) briefly
        on_stage(5, "complete: 1 section, 1 reference")
        return {
            "final_answer": "DEEP ANSWER",
            "state": {"evidence_json": "{}"},
            "stats": {"llm_calls": 3},
        }

    monkeypatch.setattr(handlers, "deep_research", fake_deep_research)

    state1 = handlers.build_app_state(ready=True, source="docs/")
    gen1 = handlers.chat("first", [], state1, None, False, "deep")
    next(gen1)  # pre-streaming status frame
    next(gen1)  # first deep frame — this is what starts the worker thread
    for _ in range(100):
        if handlers._deep_run_lock.locked():
            break
        time.sleep(0.02)
    assert handlers._deep_run_lock.locked()

    # A second deep run while the first is in flight: its final frame is the
    # one-line "already in progress" error, and no report state is set.
    state2 = handlers.build_app_state(ready=True, source="docs/")
    frames2 = []
    for _m, h, _s, _st in handlers.chat("second", [], state2, None, False, "deep"):
        last = h[-1] if h else None
        frames2.append(last["content"] if last and last["role"] == "assistant" else "")
    assert frames2[-1] == (
        "Error: A deep research run is already in progress — please wait "
        "for it to finish."
    )
    assert "last_report" not in state2

    # Release the first run: it completes normally and the lock is free.
    release_first.set()
    for _m, h, _s, _st in gen1:
        last = h[-1] if h else None
        final = last["content"] if last and last["role"] == "assistant" else ""
    assert final == "DEEP ANSWER"
    assert state1["last_report"] == "DEEP ANSWER"
    for _ in range(100):
        if not handlers._deep_run_lock.locked():
            break
        time.sleep(0.02)
    assert handlers._deep_run_lock.locked() is False


# ---------------------------------------------------------------------------
# T7 — clear_chat preserves the deep state alongside the other report fields
# ---------------------------------------------------------------------------

def test_clear_chat_preserves_last_report_state():
    handlers = importlib.import_module("ui.gradio_handlers")
    inner = {
        "evidence_json": "{}",
        "verification_status": {"confidence": "high"},
    }
    state = handlers.build_app_state(ready=True, source="docs/")
    state["last_report"] = "DEEP ANSWER"
    state["last_report_query"] = "deep q"
    state["last_report_state"] = inner

    _history, next_state, _status = handlers.clear_chat(state)
    assert next_state["last_report"] == "DEEP ANSWER"
    assert next_state["last_report_query"] == "deep q"
    assert next_state["last_report_state"] == inner

    # A standard state has no deep state: clear yields None, not a stale dict.
    _h2, next_state2, _s2 = handlers.clear_chat(
        handlers.build_app_state(ready=True, source="docs/")
    )
    assert next_state2["last_report_state"] is None


# ---------------------------------------------------------------------------
# T8 — watchdog: an outlived deep run yields a timeout frame and returns
# ---------------------------------------------------------------------------

def test_deep_run_watchdog_times_out(monkeypatch):
    handlers = importlib.import_module("ui.gradio_handlers")
    # Shrink the watchdog window (read at loop time, so monkeypatching the
    # module attribute is effective) below the fake run's duration.
    monkeypatch.setattr(handlers, "MAX_DEEP_RUN_SECONDS", 0.1)

    def fake_deep_research(query, verbose=False, on_stage=None, on_section=None):
        time.sleep(0.3)  # outlives the shrunken watchdog window
        on_stage(5, "complete: 1 section, 1 reference")
        return {
            "final_answer": "DEEP ANSWER",
            "state": {"evidence_json": "{}"},
            "stats": {"llm_calls": 3},
        }

    monkeypatch.setattr(handlers, "deep_research", fake_deep_research)
    state = handlers.build_app_state(ready=True, source="docs/")

    frames = []
    for _m, h, _s, _st in handlers.chat("slow q", [], state, None, False, "deep"):
        last = h[-1] if h else None
        frames.append(last["content"] if last and last["role"] == "assistant" else "")

    # The generator gave up with a timeout frame instead of waiting for the
    # worker, and stored no report (the run is treated as failed).
    assert frames[-1] == "Error: Deep research run timed out after 45 minutes"
    assert "last_report" not in state

    # The orphaned worker finishes in the background and frees the lock.
    for _ in range(100):
        if not handlers._deep_run_lock.locked():
            break
        time.sleep(0.02)
    assert handlers._deep_run_lock.locked() is False
