"""
Tests for the F1 guard in ui/gradio_handlers.ingest_source_documents:
the corpus reconcile (and the evidence-cache clear it can trigger) must
run on the page-load path (no upload files) but NOT on the upload path —
an upload ingest does a full reset_collection, making a docs/-based
reconcile redundant and able to desync the catalog.
"""

import ui.gradio_handlers as gh


def _patch_env(monkeypatch, tmp_path):
    """Neutralize side effects; return spies for the reconcile path."""
    calls = {"reconcile": [], "clear": [], "ingest": []}
    monkeypatch.setattr(gh, "init_memory", lambda: None)
    monkeypatch.setattr(gh, "ingest_documents", lambda pdf_dir: (
        calls["ingest"].append(pdf_dir) or {"num_pdfs": 0, "collection_name": "x"}
    ))
    monkeypatch.setattr(
        gh, "reconcile_corpus",
        lambda pdf_dir=None: calls["reconcile"].append(pdf_dir) or [],
    )
    monkeypatch.setattr(gh, "clear_evidence_cache", lambda: calls["clear"].append(1))
    monkeypatch.setattr(gh, "stage_uploaded_pdfs",
                        lambda files: tmp_path / "staged")
    return calls


def test_upload_path_does_not_reconcile(monkeypatch, tmp_path):
    calls = _patch_env(monkeypatch, tmp_path)
    gh.ingest_source_documents(["upload.pdf"])
    assert calls["reconcile"] == []
    assert calls["clear"] == []
    # Ingest still runs, against the staged dir.
    assert calls["ingest"] == [tmp_path / "staged"]


def test_page_load_path_reconciles(monkeypatch, tmp_path):
    calls = _patch_env(monkeypatch, tmp_path)
    gh.ingest_source_documents(None)
    assert calls["reconcile"] == [gh.DEFAULT_DOCS_DIR]
    assert calls["ingest"] == [gh.DEFAULT_DOCS_DIR]


def test_empty_file_paths_list_reconciles(monkeypatch, tmp_path):
    calls = _patch_env(monkeypatch, tmp_path)
    gh.ingest_source_documents([])
    assert calls["reconcile"] == [gh.DEFAULT_DOCS_DIR]


def test_reconcile_vanish_clears_cache_on_page_load(monkeypatch, tmp_path):
    calls = _patch_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        gh, "reconcile_corpus",
        lambda pdf_dir=None: calls["reconcile"].append(pdf_dir) or ["gone.pdf"],
    )
    gh.ingest_source_documents(None)
    assert calls["reconcile"] == [gh.DEFAULT_DOCS_DIR]
    assert calls["clear"] == [1]
