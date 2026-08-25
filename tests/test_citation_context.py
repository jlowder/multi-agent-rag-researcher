"""Unit tests for deterministic citation-keyed evidence (P1-5).

Covers build_citation_context (key assignment order, registry fields,
exclusions) and format_references (first-appearance numbering, orphan key
dropping, contiguous numbering, graceful omission of missing parts).
No LLM calls; everything is synthetic.
"""

from memory.helpers import build_citation_context, format_references


def _synthetic_pack() -> dict:
    """Evidence pack shaped like the retriever's output.

    - 3 doc chunks: one high-score, one low-score, one with NO identity
      (both document_name and document_title missing) but the HIGHEST score.
    - 2 web results: one well-formed (with published_date), one missing
      `url` but with the highest web score.
    """
    return {
        "route_used": "both",
        "summary": "synthetic",
        "document_evidence": {
            "query": "genetic programming",
            "chunks": [
                {
                    "document_name": "gp_manual.pdf",
                    "document_title": "Genetic Programming Manual",
                    "page_number": 3,
                    "chunk_id": "chunk_10",
                    "citation": "[gp_manual.pdf p.3]",
                    "content": "Manual content about representation and terminals.",
                    "score": 0.91,
                },
                {
                    "document_name": "koza_1992.pdf",
                    "document_title": "Genetic Programming",
                    "page_number": 1,
                    "chunk_id": "chunk_0",
                    "citation": "[koza_1992.pdf p.1]",
                    "content": "Koza content about fitness and automatic definition.",
                    "score": 0.87,
                },
                {
                    # Missing both identity fields: highest score, must get NO key.
                    "page_number": 5,
                    "chunk_id": "chunk_99",
                    "citation": "",
                    "content": "Orphan content with no document identity.",
                    "score": 0.99,
                },
            ],
        },
        "web_evidence": {
            "query": "genetic programming",
            "results": [
                {
                    "title": "GP Survey",
                    "url": "https://example.org/survey",
                    "content": "Survey content on GP applications.",
                    "score": 0.8,
                    "published_date": "2025-01-15",
                },
                {
                    # Missing url: highest web score, must get NO key.
                    "title": "No URL result",
                    "content": "Result without a URL cannot be cited.",
                    "score": 0.95,
                },
            ],
        },
    }


class TestBuildCitationContext:
    def test_key_assignment_score_descending(self):
        _, registry = build_citation_context(_synthetic_pack())
        assert sorted(registry.keys()) == ["D1", "D2", "W1"]
        # D1 is the higher-scoring doc chunk, D2 the lower one.
        assert registry["D1"]["document_name"] == "gp_manual.pdf"
        assert registry["D2"]["document_name"] == "koza_1992.pdf"
        # The only well-formed web result takes W1.
        assert registry["W1"]["url"] == "https://example.org/survey"

    def test_registry_doc_fields(self):
        _, registry = build_citation_context(_synthetic_pack())
        assert registry["D1"] == {
            "kind": "doc",
            "title": "Genetic Programming Manual",
            "document_name": "gp_manual.pdf",
            "page_number": 3,
            "citation": "[gp_manual.pdf p.3]",
            "score": 0.91,
        }
        assert registry["D2"]["page_number"] == 1
        assert registry["D2"]["citation"] == "[koza_1992.pdf p.1]"

    def test_registry_web_fields(self):
        _, registry = build_citation_context(_synthetic_pack())
        assert registry["W1"] == {
            "kind": "web",
            "title": "GP Survey",
            "url": "https://example.org/survey",
            "published_date": "2025-01-15",
            "score": 0.8,
        }

    def test_excluded_chunks_receive_no_key(self):
        text, registry = build_citation_context(_synthetic_pack())
        # No D3 (orphan doc) and no W2 (url-less web) exist...
        assert "D3" not in registry and "W2" not in registry
        assert "[D3]" not in text and "[W2]" not in text
        # ...but their content still appears as uncited context.
        assert "Orphan content with no document identity." in text
        assert "(uncited)" in text
        assert "Result without a URL cannot be cited." in text

    def test_evidence_text_block_formats(self):
        text, _ = build_citation_context(_synthetic_pack())
        assert (
            "[D1] Genetic Programming Manual — gp_manual.pdf p. 3\n"
            "Manual content about representation and terminals.\n" in text
        )
        assert (
            "[W1] GP Survey (https://example.org/survey, 2025-01-15)\n"
            "Survey content on GP applications.\n" in text
        )
        # Docs come before web blocks.
        assert text.index("[D1]") < text.index("[W1]")

    def test_missing_page_number_omits_page_suffix(self):
        pack = {
            "document_evidence": {
                "chunks": [
                    {
                        "document_name": "file.pdf",
                        "document_title": "Title Only",
                        "page_number": None,
                        "chunk_id": "chunk_1",
                        "citation": "",
                        "content": "Content without page info.",
                        "score": 0.5,
                    }
                ]
            },
            "web_evidence": {"results": []},
        }
        text, registry = build_citation_context(pack)
        assert "[D1] Title Only — file.pdf\nContent without page info.\n" in text
        assert " p. " not in text
        assert registry["D1"]["page_number"] is None

    def test_equal_scores_tiebreak_by_name(self):
        pack = {
            "document_evidence": {"chunks": []},
            "web_evidence": {
                "results": [
                    {"title": "B", "url": "https://b.org", "content": "b content", "score": 0.8},
                    {"title": "A", "url": "https://a.org", "content": "a content", "score": 0.8},
                ]
            },
        }
        _, registry = build_citation_context(pack)
        assert registry["W1"]["url"] == "https://a.org"
        assert registry["W2"]["url"] == "https://b.org"

    def test_empty_pack(self):
        text, registry = build_citation_context({})
        assert text == ""
        assert registry == {}


class TestFormatReferences:
    def setup_method(self):
        self._pack = _synthetic_pack()
        self._text, self._registry = build_citation_context(self._pack)

    def test_first_appearance_order_and_numbering(self):
        refs = format_references(self._registry, ["W1", "D2", "D1", "W1"])
        assert refs == (
            "[1] GP Survey. https://example.org/survey (2025-01-15).\n"
            "[2] Genetic Programming. *koza_1992.pdf* p. 1.\n"
            "[3] Genetic Programming Manual. *gp_manual.pdf* p. 3."
        )

    def test_orphan_keys_dropped_and_numbering_contiguous(self):
        refs = format_references(self._registry, ["D9", "W42", "D1"])
        assert refs == "[1] Genetic Programming Manual. *gp_manual.pdf* p. 3."

    def test_duplicate_keys_collapse(self):
        refs = format_references(self._registry, ["D1", "D1", "D2"])
        lines = refs.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("[1] Genetic Programming Manual.")
        assert lines[1].startswith("[2] Genetic Programming.")

    def test_empty_input_returns_empty_string(self):
        assert format_references({}, []) == ""
        assert format_references(self._registry, []) == ""
        assert format_references(None, None) == ""

    def test_missing_parts_omitted_gracefully(self):
        registry = {
            "D1": {
                "kind": "doc",
                "title": "Title Only",
                "document_name": "file.pdf",
                "page_number": None,
                "citation": "",
                "score": 0.5,
            },
            "W1": {
                "kind": "web",
                "title": "",
                "url": "https://x.org/page",
                "published_date": None,
                "score": 0.5,
            },
        }
        refs = format_references(registry, ["D1", "W1"])
        assert refs == (
            "[1] Title Only. *file.pdf*.\n"
            "[2] https://x.org/page."
        )

    def test_round_trip_from_cited_keys_in_text(self):
        # Simulate the post-hoc pass: keys found in the body, in appearance
        # order, must resolve to references that all exist in the registry.
        cited = []
        for key in ("D2", "D1", "D2", "W1"):
            if key not in cited:
                cited.append(key)
        refs = format_references(self._registry, cited)
        lines = refs.splitlines()
        assert [line.split("]")[0] + "]" for line in lines] == ["[1]", "[2]", "[3]"]
        assert all(f"[{key}]" in self._text for key in cited)


class TestFormatReferencesFilenameTitles:
    """Doc titles that are bare filenames must not render as the title part
    (the document_name already carries that info)."""

    @staticmethod
    def _doc_registry(title: str, name: str, page=None) -> dict:
        return {
            "D1": {
                "kind": "doc",
                "title": title,
                "document_name": name,
                "page_number": page,
                "citation": "",
                "score": 0.9,
            }
        }

    def test_filename_like_title_dropped(self):
        refs = format_references(
            self._doc_registry("gpem.dvi", "04_grammar_based_genetic_programming.pdf", 3),
            ["D1"],
        )
        assert refs == "[1] *04_grammar_based_genetic_programming.pdf* p. 3."

    def test_normal_title_kept(self):
        refs = format_references(
            self._doc_registry(
                "Grammar-based Genetic Programming: a survey", "survey.pdf", 12
            ),
            ["D1"],
        )
        assert refs == "[1] Grammar-based Genetic Programming: a survey. *survey.pdf* p. 12."

    def test_title_equal_to_filename_stem_dropped(self):
        refs = format_references(
            self._doc_registry(
                "04_grammar_based_genetic_programming",
                "04_grammar_based_genetic_programming.pdf",
                17,
            ),
            ["D1"],
        )
        assert refs == "[1] *04_grammar_based_genetic_programming.pdf* p. 17."

    def test_title_equal_to_filename_basename_dropped_case_insensitive(self):
        refs = format_references(
            self._doc_registry("report.PDF", "REPORT.pdf", 1), ["D1"]
        )
        assert refs == "[1] *REPORT.pdf* p. 1."

    def test_web_entries_unchanged(self):
        registry = {
            "W1": {
                "kind": "web",
                "title": "notes.pdf",
                "url": "https://x.org/notes.pdf",
                "published_date": None,
                "score": 0.5,
            }
        }
        assert (
            format_references(registry, ["W1"])
            == "[1] notes.pdf. https://x.org/notes.pdf."
        )
