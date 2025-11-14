import pytest

pytest.importorskip("unstructured.partition.auto")

from catalystindex.parsers.unstructured_adapter import UnstructuredParserAdapter
from catalystindex.models.common import SectionText  # noqa: F401 (imported for type hints)


class FakeMetadata:
    def __init__(self, page_number: int | None = None, category_depth: int | None = None) -> None:
        self.page_number = page_number
        self.category_depth = category_depth


class FakeElement:
    def __init__(self, text: str, category: str = "", page_number: int | None = None) -> None:
        self.text = text
        self.category = category
        self.metadata = FakeMetadata(page_number=page_number)


def test_unstructured_parser_groups_headings(monkeypatch):
    import unstructured.partition.auto as auto  # type: ignore

    elements = [
        FakeElement("Section 1", category="Title", page_number=1),
        FakeElement("Paragraph A", category="NarrativeText", page_number=1),
        FakeElement("Section 2", category="Heading", page_number=2),
        FakeElement("Paragraph B", category="NarrativeText", page_number=2),
    ]

    monkeypatch.setattr(auto, "partition", lambda **kwargs: elements)

    parser = UnstructuredParserAdapter()
    sections = list(parser.parse(b"ignored", document_title="Test Doc"))

    assert len(sections) == 2
    assert sections[0].title == "Section 1"
    assert sections[0].metadata["parser"] == "unstructured"
    assert sections[1].start_page == 2


def test_unstructured_parser_fallback(monkeypatch):
    import unstructured.partition.auto as auto  # type: ignore

    monkeypatch.setattr(auto, "partition", lambda **kwargs: [])

    parser = UnstructuredParserAdapter()
    sections = list(parser.parse(b"fallback text", document_title="Doc"))


def test_unstructured_parser_uses_content_type_hint(monkeypatch):
    parser = UnstructuredParserAdapter()

    captured = {}

    def fake_partition(self, kind, payload, document_title):  # pragma: no cover - patched helper
        captured["kind"] = kind
        return [
            FakeElement("Heading", category="Title", page_number=1),
            FakeElement("Body", category="NarrativeText", page_number=1),
        ]

    monkeypatch.setattr(UnstructuredParserAdapter, "_partition_elements", fake_partition)

    sections = list(
        parser.parse(b"dummy", document_title="Doc", content_type="application/pdf")
    )

    assert captured["kind"] == "pdf"
    assert sections[0].title == "Heading"

    assert len(sections) == 1
    assert sections[0].text.endswith("Body")
