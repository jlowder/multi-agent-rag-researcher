"""
Tests for qdrant_vector_database.vector_store.reconcile_corpus (C1).

No real Qdrant: a fake client object is monkeypatched over
get_qdrant_client, and the catalog path is monkeypatched to a tmp file.
"""

import json

import qdrant_vector_database.vector_store as vs


class _FakePoint:
    def __init__(self, payload: dict):
        self.payload = payload


class FakeQdrantClient:
    """Records create/delete calls; scroll returns the in-memory points."""

    def __init__(self, indexed: dict[str, str] | None = None, exists: bool = True):
        self._indexed = indexed or {}
        self.exists = exists
        self.created = []
        self.deleted_filters = []

    @property
    def points(self) -> list[_FakePoint]:
        return [
            _FakePoint({"document_name": n, "document_title": t})
            for n, t in sorted(self._indexed.items())
        ]

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists and collection_name == vs.COLLECTION_NAME

    def create_collection(self, collection_name: str, vectors_config=None) -> None:
        self.created.append(collection_name)

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        return (self.points, None)

    def delete(self, collection_name: str, filter) -> None:
        self.deleted_filters.append(filter)
        # simulate the filtered delete for subsequent scroll/inspection
        names = set(filter.must[0].match.any)
        self._indexed = {
            n: t for n, t in self._indexed.items() if n not in names
        }


def _write_catalog(path, documents: list[dict]) -> None:
    path.write_text(
        json.dumps({"version": 2, "documents": documents}, indent=2),
        encoding="utf-8",
    )


def _read_catalog(path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["documents"]


def _setup(monkeypatch, tmp_path, fake: FakeQdrantClient, documents: list[dict] | None):
    catalog_path = tmp_path / "indexed_documents.json"
    if documents is not None:
        _write_catalog(catalog_path, documents)
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    monkeypatch.setattr(vs, "INDEXED_DOCUMENTS_PATH", catalog_path)
    return catalog_path


def test_reconcile_removes_vanished_doc_from_collection_and_catalog(
    monkeypatch, tmp_path
):
    # (a) 2 indexed docs; one file deleted from disk -> exactly that doc is
    # purged from the collection (one batched delete filter) and the catalog,
    # the other doc untouched.
    pdf_dir = tmp_path / "docs"
    pdf_dir.mkdir()
    (pdf_dir / "kept.pdf").write_bytes(b"%PDF")
    # "gone.pdf" is indexed but no longer on disk.
    fake = FakeQdrantClient({"gone.pdf": "Gone Title", "kept.pdf": "Kept Title"})
    catalog = _setup(
        monkeypatch, tmp_path, fake,
        [{"file_name": "gone.pdf", "title": "Gone Title"},
         {"file_name": "kept.pdf", "title": "Kept Title"}],
    )

    vanished = vs.reconcile_corpus(pdf_dir)

    assert vanished == ["gone.pdf"]
    assert len(fake.deleted_filters) == 1
    condition = fake.deleted_filters[0].must[0]
    assert condition.key == "document_name"
    assert condition.match.any == ["gone.pdf"]
    assert [p.payload["document_name"] for p in fake.points] == ["kept.pdf"]
    assert _read_catalog(catalog) == [{"file_name": "kept.pdf", "title": "Kept Title"}]


def test_reconcile_purges_everything_when_dir_empty(monkeypatch, tmp_path):
    # (b) all files deleted (empty dir) -> all indexed docs purged, catalog [].
    pdf_dir = tmp_path / "docs"
    pdf_dir.mkdir()
    fake = FakeQdrantClient({"a.pdf": "A", "b.pdf": "B"})
    catalog = _setup(
        monkeypatch, tmp_path, fake,
        [{"file_name": "a.pdf", "title": "A"}, {"file_name": "b.pdf", "title": "B"}],
    )

    vanished = vs.reconcile_corpus(pdf_dir)

    assert vanished == ["a.pdf", "b.pdf"]
    condition = fake.deleted_filters[0].must[0]
    assert condition.match.any == ["a.pdf", "b.pdf"]
    assert fake.points == []
    assert _read_catalog(catalog) == []


def test_reconcile_noop_when_all_files_present(monkeypatch, tmp_path):
    # (c) corpus consistent -> no delete call, catalog file byte-identical.
    pdf_dir = tmp_path / "docs"
    pdf_dir.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (pdf_dir / name).write_bytes(b"%PDF")
    fake = FakeQdrantClient({"a.pdf": "A", "b.pdf": "B"})
    catalog = _setup(
        monkeypatch, tmp_path, fake,
        [{"file_name": "a.pdf", "title": "A"}, {"file_name": "b.pdf", "title": "B"}],
    )
    before = catalog.read_bytes()

    assert vs.reconcile_corpus(pdf_dir) == []
    assert fake.deleted_filters == []
    assert catalog.read_bytes() == before


def test_reconcile_missing_collection_ensures_and_returns_empty(monkeypatch, tmp_path):
    # (d) collection missing -> created, returns [], no error, no catalog write.
    pdf_dir = tmp_path / "docs"
    pdf_dir.mkdir()
    (pdf_dir / "a.pdf").write_bytes(b"%PDF")
    fake = FakeQdrantClient(exists=False)
    monkeypatch.setattr(vs, "get_qdrant_client", lambda: fake)
    catalog = tmp_path / "indexed_documents.json"
    monkeypatch.setattr(vs, "INDEXED_DOCUMENTS_PATH", catalog)

    assert vs.reconcile_corpus(pdf_dir) == []
    assert fake.created == [vs.COLLECTION_NAME]
    assert fake.deleted_filters == []
    assert not catalog.exists()


def test_reconcile_empty_collection_returns_empty(monkeypatch, tmp_path):
    # (d) collection exists but holds no points -> [] with no writes.
    pdf_dir = tmp_path / "docs"
    pdf_dir.mkdir()
    fake = FakeQdrantClient({})
    catalog = _setup(monkeypatch, tmp_path, fake, None)

    assert vs.reconcile_corpus(pdf_dir) == []
    assert fake.deleted_filters == []
    assert not catalog.exists()


def test_reconcile_missing_pdf_dir_purges_all_indexed(monkeypatch, tmp_path):
    # The live stale case: docs/ gone entirely -> every indexed doc is
    # treated as vanished (empty on-disk set).
    fake = FakeQdrantClient({"a.pdf": "A"})
    catalog = _setup(monkeypatch, tmp_path, fake,
                     [{"file_name": "a.pdf", "title": "A"}])

    assert vs.reconcile_corpus(tmp_path / "no_such_dir") == ["a.pdf"]
    assert _read_catalog(catalog) == []
