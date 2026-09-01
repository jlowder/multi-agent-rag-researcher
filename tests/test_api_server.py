"""
Tests for the FastAPI research service (api_server.py).

Unit tests drive the app with a fake run_fn (the create_app seam) — no LLM,
no network. One integration test reuses test_deep_pipeline._install_stubs
to run the REAL deep_research through the API with every LLM / retrieval
surface stubbed.
"""

import json
import shutil
import threading
import time

import pytest
from fastapi.testclient import TestClient

import api_server
import test_deep_pipeline as tdp


# ---------------------------------------------------------------------------
# Fakes + helpers
# ---------------------------------------------------------------------------


def _fake_report_json(topic: str) -> str:
    """Plausible canonical ResearchReport JSON for the fake run_fn."""
    return json.dumps(
        {
            "schema_version": "1.0",
            "report": {
                "metadata": {"topic": topic},
                "executive_summary": ["fake executive summary"],
                "sections": [
                    {
                        "id": "s1",
                        "heading": "One",
                        "blocks": [
                            {
                                "type": "paragraph",
                                "spans": [
                                    {"text": "fake section body", "citations": []}
                                ],
                            }
                        ],
                    }
                ],
                "sources": [],
            },
            "quality": {"confidence_level": "high"},
        }
    )


def _make_fake_run_fn():
    """Fast fake run_fn: sleeps ~0.05s per stage via the app-provided
    on_stage (exercising step recording) and returns a canned
    deep_research-shaped result. Records calls on .calls."""

    def run_fn(topic, **kwargs):
        on_stage = kwargs.get("on_stage")
        run_fn.calls.append(
            {
                "topic": topic,
                "budgets": {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("on_stage", "on_section")
                },
            }
        )
        for n, detail in (
            (1, "decomposing query"),
            (2, "investigating 2 sub-question(s)"),
            (3, "drafting 2 section(s)"),
            (4, "critic pass: checking every drafted section"),
            (5, "assembling final report"),
        ):
            time.sleep(0.05)
            if on_stage is not None:
                on_stage(n, detail)
        return {
            "final_answer": f"fake answer for {topic}",
            "state": {"report_json": _fake_report_json(topic)},
            "stats": {"llm_calls": 7, "wall_s": 0.3, "sections": 2},
        }

    run_fn.calls = []
    return run_fn


@pytest.fixture()
def client():
    return TestClient(api_server.create_app(run_fn=_make_fake_run_fn()))


def _wait(client: TestClient, task_id: str, timeout: float = 10.0) -> dict:
    """Poll GET /research/{id} until the task leaves 'running'."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/research/{task_id}").json()
        if last["status"] != "running":
            return last
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} still running after {timeout}s")


# ---------------------------------------------------------------------------
# POST /research + lifecycle
# ---------------------------------------------------------------------------


def test_post_returns_202_and_lists_task(client):
    resp = client.post("/research", json={"topic": "tiny topic"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "running"
    assert body["task_id"]
    assert body["current_step"]
    assert body["links"] == {
        "status": f"/research/{body['task_id']}",
        "report": f"/research/{body['task_id']}/report",
    }
    listed = client.get("/research").json()["tasks"]
    assert [t["id"] for t in listed] == [body["task_id"]]
    assert listed[0]["topic"] == "tiny topic"


def test_run_completes_and_steps_track_all_stages(client):
    task_id = client.post("/research", json={"topic": "tiny topic"}).json()["task_id"]
    last = _wait(client, task_id)
    assert last["status"] == "completed"
    assert "error" not in last  # omitted when the run did not fail
    assert last["finished_at"] is not None
    stages = [s["stage"] for s in last["steps"]]
    assert stages == ["decompose", "investigate", "draft", "critique", "assemble"]
    assert all(s["detail"] and s["ts"] > 0 for s in last["steps"])
    assert last["current_step"].startswith("assemble:")
    assert last["stats"] == {"llm_calls": 7, "wall_s": 0.3, "sections": 2}


def test_budgets_forwarded_to_run_fn():
    fn = _make_fake_run_fn()
    client = TestClient(api_server.create_app(run_fn=fn))
    task_id = client.post(
        "/research",
        json={"topic": "t", "max_rounds": 2, "budget_doc": 3, "budget_web": 1},
    ).json()["task_id"]
    _wait(client, task_id)
    assert fn.calls[0]["budgets"] == {
        "max_rounds": 2,
        "budget_doc": 3,
        "budget_web": 1,
    }


def test_second_post_while_running_is_409():
    release = threading.Event()

    def run_fn(topic, **kwargs):
        release.wait(timeout=10)
        return {
            "final_answer": f"done {topic}",
            "state": {"report_json": _fake_report_json(topic)},
            "stats": {"llm_calls": 1},
        }

    client = TestClient(api_server.create_app(run_fn=run_fn))
    first = client.post("/research", json={"topic": "one"}).json()["task_id"]
    resp = client.post("/research", json={"topic": "two"})
    assert resp.status_code == 409
    assert resp.json() == {
        "error": "a research run is already in progress",
        "running_task_id": first,
    }
    release.set()
    _wait(client, first)
    # Once the run finishes, a new run is accepted again.
    assert client.post("/research", json={"topic": "three"}).status_code == 202


def test_run_failure_is_recorded():
    def run_fn(topic, **kwargs):
        raise RuntimeError("boom: simulated failure")

    client = TestClient(api_server.create_app(run_fn=run_fn))
    task_id = client.post("/research", json={"topic": "tiny topic"}).json()["task_id"]
    last = _wait(client, task_id)
    assert last["status"] == "failed"
    assert "boom: simulated failure" in last["error"]
    assert "stats" not in last
    assert client.get(f"/research/{task_id}/report").status_code == 409


# ---------------------------------------------------------------------------
# GET /research/{id} + report endpoint
# ---------------------------------------------------------------------------


def test_report_after_completed(client):
    task_id = client.post("/research", json={"topic": "tiny topic"}).json()["task_id"]
    _wait(client, task_id)
    resp = client.get(f"/research/{task_id}/report")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    data = json.loads(resp.text)
    assert data["report"]["sections"]
    assert data["report"]["sections"][0]["heading"] == "One"


def test_report_while_running_is_409():
    release = threading.Event()

    def run_fn(topic, **kwargs):
        on_stage = kwargs.get("on_stage")
        if on_stage is not None:
            on_stage(1, "decomposing query")
        release.wait(timeout=10)
        return {
            "final_answer": f"done {topic}",
            "state": {"report_json": _fake_report_json(topic)},
            "stats": {"llm_calls": 1},
        }

    client = TestClient(api_server.create_app(run_fn=run_fn))
    task_id = client.post("/research", json={"topic": "tiny topic"}).json()["task_id"]
    resp = client.get(f"/research/{task_id}/report")
    assert resp.status_code == 409
    assert resp.json() == {"status": "running"}
    release.set()
    _wait(client, task_id)
    assert client.get(f"/research/{task_id}/report").status_code == 200


def test_unknown_task_is_404(client):
    assert client.get("/research/deadbeef").status_code == 404
    assert client.get("/research/deadbeef/report").status_code == 404


def test_empty_topic_is_422(client):
    assert client.post("/research", json={"topic": ""}).status_code == 422


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "multi-agent-rag-researcher"
    assert body["running"] is False
    assert isinstance(body["deep_configured"], bool)


# ---------------------------------------------------------------------------
# Integration: REAL deep_research through the API (all LLM surfaces stubbed)
# ---------------------------------------------------------------------------


def test_real_deep_research_through_api(monkeypatch):
    tdp._install_stubs(monkeypatch, tdp._basic_env())
    # test_deep_pipeline's autouse temp-dir cleanup does not apply here;
    # collect the temp cache DB dir(s) this test created for manual cleanup.
    created = list(tdp._CACHE_TMP_DIRS)
    client = TestClient(api_server.create_app())  # default run_fn = real deep_research
    try:
        task_id = client.post(
            "/research",
            json={
                "topic": "what is a small thing",
                "max_rounds": 1,
                "budget_doc": 1,
                "budget_web": 1,
            },
        ).json()["task_id"]
        last = _wait(client, task_id, timeout=60)
        assert last["status"] == "completed", last.get("error")
        assert last["stats"]["sections"] >= 1
        rep = client.get(f"/research/{task_id}/report")
        assert rep.status_code == 200
        data = json.loads(rep.text)
        assert {"schema_version", "report", "quality"} <= set(data)
        assert data["schema_version"] == "1.0"
        assert data["report"]["sections"]
    finally:
        for d in created:
            shutil.rmtree(d, ignore_errors=True)
        tdp._CACHE_TMP_DIRS[:] = [p for p in tdp._CACHE_TMP_DIRS if p not in created]
