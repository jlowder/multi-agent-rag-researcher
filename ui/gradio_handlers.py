from __future__ import annotations

import sys
import traceback
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from queue import Empty, Queue
from shutil import copy2, rmtree
from tempfile import mkdtemp
from threading import Lock, Thread
from time import monotonic, sleep
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deep_research_orchestrator import deep_research
from memory import init_memory, set_debug_mode
from memory.evidence_cache import clear_evidence_cache
from orchestrator_agent import orchestrator_agent
from qdrant_vector_database import ingest_documents, reconcile_corpus

DEFAULT_DOCS_DIR = ROOT_DIR / "docs"

# One deep run at a time: a second submission while a run is in flight
# (e.g. after a UI stop orphaned its worker) must not start a concurrent
# run_model burst that races the LLM call budget.
_deep_run_lock = Lock()

# Watchdog: a deep run outliving this is treated as hung — the UI yields a
# timeout frame and returns (the worker keeps running and frees the lock).
MAX_DEEP_RUN_SECONDS = 45 * 60


def build_app_state(
    ready: bool = False,
    source: str = "",
    source_key: str = "docs",
) -> dict:
    return {
        "session_id": str(uuid4()),
        "ready": ready,
        "source": source,
        "source_key": source_key,
    }


def stage_uploaded_pdfs(file_paths: list[str]) -> Path:
    temp_dir = Path(mkdtemp(prefix="ragagent_pdfs_"))
    for file_path in file_paths:
        source = Path(getattr(file_path, "name", file_path))
        copy2(source, temp_dir / source.name)
    return temp_dir


def upload_signature(file_paths: list[str] | None) -> str:
    if not file_paths:
        return "docs"
    signatures = []
    for file_path in file_paths:
        path = Path(getattr(file_path, "name", file_path))
        try:
            stats = path.stat()
            signatures.append(f"{path.name}:{stats.st_size}:{stats.st_mtime_ns}")
        except OSError:
            signatures.append(path.name)
    return "|".join(sorted(signatures))


def status_text(state: dict) -> str:
    if state.get("ready"):
        return f"Research source: `{state.get('source') or 'docs/'}`"
    return "Research source: `docs/` until you upload your own PDFs to replace the current indexed documents"


def _elapsed_seconds(start: float) -> str:
    total = int(monotonic() - start)
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m {total % 60}s"


def capture_logs(func, *args, **kwargs):
    buffer = StringIO()
    with redirect_stdout(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue().strip()


class LiveLogWriter:
    def __init__(self, queue: Queue):
        self.queue = queue

    def write(self, text: str) -> int:
        if text:
            self.queue.put(text)
        return len(text)

    def flush(self) -> None:
        return None


def format_trace_message(logs: str) -> str:
    logs = logs.strip()
    if not logs:
        return "Working..."
    return f"**Agent Trace**\n\n```text\n{logs}\n```"


def ingest_status_message(file_paths: list[str] | None) -> str:
    if file_paths:
        return "Ingesting uploaded documents..."
    return "Ingesting documents..."


def set_trace_entry(history: list[dict], logs: str) -> None:
    history[-1] = {"role": "assistant", "content": format_trace_message(logs)}


def append_trace_entry(history: list[dict], logs: str) -> None:
    history.append({"role": "assistant", "content": format_trace_message(logs)})


def ingest_source_documents(file_paths: list[str] | None) -> tuple[str, dict]:
    init_memory()
    staged_dir: Path | None = None

    try:
        if file_paths:
            staged_dir = stage_uploaded_pdfs(file_paths)
            pdf_dir = staged_dir
            pdf_names = sorted(Path(getattr(file_path, "name", file_path)).name for file_path in file_paths)
            source = ", ".join(pdf_names)
            source_key = upload_signature(file_paths)
        else:
            pdf_dir = DEFAULT_DOCS_DIR
            source = "docs/"
            source_key = "docs"

        # Drop corpus entries for PDFs deleted since the last ingest (and the
        # evidence packs built on them) before re-ingesting. Never blocks
        # startup: a reconcile failure just leaves the old corpus in place.
        try:
            vanished = reconcile_corpus(DEFAULT_DOCS_DIR)
            if vanished:
                clear_evidence_cache()
        except Exception as exc:
            print(f"WARNING: corpus reconcile failed: {type(exc).__name__}: {exc}")

        info = ingest_documents(pdf_dir)
        next_state = build_app_state(ready=True, source=source, source_key=source_key)
        status = f"{status_text(next_state)} · {info['num_pdfs']} PDF(s)"
        return status, next_state
    finally:
        if staged_dir is not None:
            rmtree(staged_dir, ignore_errors=True)


def ingest_with_trace(file_paths: list[str] | None) -> tuple[str, dict, str]:
    (status, next_state), index_logs = capture_logs(ingest_source_documents, file_paths)
    logs = index_logs.strip()
    trace_message = ingest_status_message(file_paths)
    if logs:
        trace_message = f"{trace_message}\n{logs}".strip()
    return status, next_state, f"{trace_message}\nDocuments ingested.".strip()


def load_default_docs(state: dict) -> tuple[dict, str]:
    try:
        status, indexed_state, _ = ingest_with_trace(None)
        return {**indexed_state, "session_id": state["session_id"]}, status
    except Exception as exc:
        return state, f"Default docs ingestion failed: {exc}"


def clear_chat(state: dict) -> tuple[list, dict, str]:
    next_state = build_app_state(
        ready=state.get("ready", False),
        source=state.get("source", ""),
        source_key=state.get("source_key", "docs"),
    )
    # Preserve report fields on clear
    next_state["last_report"] = state.get("last_report", "")
    next_state["last_report_query"] = state.get("last_report_query", "")
    next_state["last_report_title"] = state.get("last_report_title", "")
    next_state["last_report_state"] = state.get("last_report_state")
    return [], next_state, status_text(next_state)


def ingest_uploaded_documents(
    file_paths: list[str] | None,
    history: list[dict] | None,
    state: dict,
):
    history = history or []
    if not file_paths:
        yield history, state, status_text(state)
        return

    streamed_logs = ingest_status_message(file_paths)
    append_trace_entry(history, streamed_logs)
    yield history, {**state, "ready": False}, "Ingesting documents..."

    try:
        status, next_state, streamed_logs = ingest_with_trace(file_paths)
        set_trace_entry(history, streamed_logs)
        yield history, next_state, status
    except Exception as exc:
        history[-1] = {"role": "assistant", "content": f"Error: {exc}"}
        yield history, state, status_text(state)


def handle_save_report(state: dict) -> str:
    """Save the latest research report to a markdown file and return the path."""
    report = state.get("last_report", "") or ""
    query = state.get("last_report_query", "")
    if not report.strip():
        return "No report to save. Run a query first."
    from memory.save_report import save_report, ReportConfig
    session_id = state.get("session_id", "default")
    report_state = state.get("last_report_state")
    if isinstance(report_state, dict) and report_state.get("report_json"):
        # Deep-mode structured output (Phase 5): save the canonical
        # structured document (JSON + sources + Markdown side export).
        from memory.save_report import save_structured_report
        from models.report_schema import ResearchReport

        config = ReportConfig.research()
        config.include_evidence_dump = True
        path = save_structured_report(
            ResearchReport.model_validate_json(report_state["report_json"]),
            state=report_state,
            config=config,
        )
        return f"Report saved to: {path}"
    if isinstance(report_state, dict) and report_state:
        # Deep mode: the NESTED pipeline state (top-level evidence_json,
        # verification, verification_status) enriches the saved report with
        # the verification header and the raw-evidence side file.
        config = ReportConfig.research()
        config.include_evidence_dump = True
        path = save_report(
            report,
            query=query,
            session_id=session_id,
            state=report_state,
            config=config,
            title=state.get("last_report_title", ""),
        )
        return f"Report saved to: {path}"
    path = save_report(
        report,
        query=query,
        session_id=session_id,
        config=ReportConfig.research(),
        title=state.get("last_report_title", ""),
    )
    return f"Report saved to: {path}"


def _chat_deep(message: str, history: list[dict], state: dict):
    """Deep-mode generator (P2-5): a worker thread runs deep_research while
    its on_stage/on_section callbacks push events onto a queue; the generator
    yields a status line plus the growing partial report on a ~0.5s heartbeat
    and, on completion, hands the NESTED pipeline state (not the
    {"final_answer","state","stats"} envelope) to the save flow via
    state["last_report_state"]."""
    started = monotonic()
    event_queue: Queue = Queue()
    result_box: dict[str, object] = {"result": None, "error": None}

    def on_stage(stage_no: int, detail: str) -> None:
        event_queue.put(("stage", stage_no, detail))

    def on_section(index: int, total: int, heading: str,
                   section_text: str, partial_report: str) -> None:
        event_queue.put(("section", partial_report))

    def run_deep() -> None:
        if not _deep_run_lock.acquire(blocking=False):
            # An orphaned prior run is still executing: never start a second
            # concurrent deep run — fail fast with a one-line error instead.
            result_box["error"] = RuntimeError(
                "A deep research run is already in progress — please wait "
                "for it to finish."
            )
            event_queue.put(("done", None))
            return
        try:
            result_box["result"] = deep_research(
                message,
                verbose=True,
                on_stage=on_stage,
                on_section=on_section,
            )
        except Exception as exc:
            traceback.print_exc()  # full detail to the server log; the UI
            # gets a one-line error (raised via the "done" event below)
            result_box["error"] = exc
        finally:
            _deep_run_lock.release()
            event_queue.put(("done", None))

    Thread(target=run_deep, daemon=True).start()

    stage_label = "starting deep research"
    partial = ""
    history.append({"role": "assistant", "content": f"⏳ {stage_label}…"})
    while True:
        if monotonic() - started > MAX_DEEP_RUN_SECONDS:
            # Watchdog: the UI gives up; the worker thread may finish in the
            # background (the run lock releases in its finally).
            history[-1] = {
                "role": "assistant",
                "content": "Error: Deep research run timed out after 45 minutes",
            }
            yield "", history, state, "⏱ Deep research timed out after 45 minutes"
            return
        while True:
            try:
                event = event_queue.get_nowait()
            except Empty:
                break
            kind = event[0]
            if kind == "stage":
                stage_label = f"⏳ Stage {event[1]} — {event[2]}"
            elif kind == "section":
                partial = event[1]
            elif kind == "done":
                if result_box["error"] is not None:
                    raise result_box["error"]
                result = result_box["result"] or {}
                answer = result.get("final_answer", "")
                state["last_report"] = answer
                state["last_report_query"] = message
                # Deep runs title via metadata.title (structured save); clear
                # any stale standard-run title so the legacy-deep markdown
                # branch below falls back to the query.
                state["last_report_title"] = ""
                state["last_report_state"] = result.get("state") or {}
                history[-1] = {"role": "assistant", "content": answer}
                yield "", history, state, (
                    f"✅ Deep research complete · {_elapsed_seconds(started)} — "
                    "open 'Save Report' to store it as markdown"
                )
                return
        if partial:
            history[-1] = {"role": "assistant", "content": partial}
        else:
            history[-1] = {"role": "assistant", "content": f"{stage_label}…"}
        yield "", history, state, f"{stage_label} · {_elapsed_seconds(started)}"
        sleep(0.5)


def chat(message: str, history: list[dict] | None, state: dict, file_paths: list[str] | None, debug: bool = False, mode: str = "standard"):
    message = (message or "").strip()
    if not message:
        yield "", history, state, status_text(state)
        return

    set_debug_mode(debug)

    history = history or []
    logs = ""
    streamed_logs = ""

    try:
        current_upload_key = upload_signature(file_paths) if file_paths else None
        needs_reindex = (
            not state.get("ready")
            or (
                current_upload_key is not None
                and current_upload_key != state.get("source_key")
            )
        )

        history.append({"role": "user", "content": message})

        if needs_reindex:
            streamed_logs = ingest_status_message(file_paths)
            append_trace_entry(history, streamed_logs)
            yield "", history, state, "Ingesting documents..."

            status, state, streamed_logs = ingest_with_trace(file_paths)
            set_trace_entry(history, streamed_logs)
            yield "", history, state, status
        else:
            status = status_text(state)
            # Deep mode deliberately skips the empty "Working…" trace entry:
            # _chat_deep supplies its own ⏳ starting bubble (logs are empty).
            if mode != "deep":
                append_trace_entry(history, logs)
            yield "", history, state, status

        if mode == "deep":
            yield from _chat_deep(message, history, state)
            return

        log_queue = Queue()
        result_box: dict[str, object] = {"result": None, "error": None}

        def run_agent() -> None:
            try:
                with redirect_stdout(LiveLogWriter(log_queue)):
                    result_box["result"] = orchestrator_agent(
                        message,
                        session_id=state["session_id"],
                        verbose=True,
                        debug_enabled=debug,
                    )
            except Exception as exc:
                result_box["error"] = exc
            finally:
                log_queue.put(None)

        worker = Thread(target=run_agent, daemon=True)
        worker.start()

        if not streamed_logs:
            streamed_logs = logs
        while True:
            updated = False
            while True:
                try:
                    chunk = log_queue.get_nowait()
                except Empty:
                    break

                if chunk is None:
                    if result_box["error"] is not None:
                        raise result_box["error"]

                    result = result_box["result"]
                    if result is not None:
                        if streamed_logs.strip():
                            set_trace_entry(history, streamed_logs)
                            history.append({"role": "assistant", "content": result["final_answer"]})
                        else:
                            history[-1] = {"role": "assistant", "content": result["final_answer"]}
                        # Store report for save button
                        state["last_report"] = result.get("final_answer", "")
                        state["last_report_query"] = message
                        # LLM title for the save filename ("" on any failure
                        # -> save_report keeps the query-based name).
                        try:
                            from worker_agents.decomposition_agent import (
                                generate_report_title,
                            )
                            state["last_report_title"] = generate_report_title(message)
                        except Exception:
                            state["last_report_title"] = ""
                        # A standard run has no nested pipeline state: clear any
                        # stale deep-run state so Save doesn't attach the old
                        # run's evidence/side-file to this report.
                        state["last_report_state"] = None
                    yield "", history, state, status
                    return

                streamed_logs = f"{streamed_logs}{chunk}"
                updated = True

            if updated:
                set_trace_entry(history, streamed_logs)
                yield "", history, state, status

            sleep(0.1)
    except Exception as exc:
        if history and history[-1].get("role") == "assistant":
            if history[-1].get("content") != "Working...":
                history.append({"role": "assistant", "content": f"Error: {exc}"})
            else:
                history[-1] = {"role": "assistant", "content": f"Error: {exc}"}
        else:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Error: {exc}"})
        yield "", history, state, status_text(state)


INITIAL_STATE = build_app_state()
INITIAL_STATUS = status_text(INITIAL_STATE)
