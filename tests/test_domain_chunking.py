import pytest
from pydantic import ValidationError

from packages.omni_rag_core.chunking import chunk_pages
from packages.omni_rag_core.domain import (
    ChunkingConfig,
    ChunkingStrategy,
    Company,
    ImportRequest,
    KnowledgeBase,
    LLMSelection,
    PageSelection,
)


def test_page_selection_includes_excludes_and_clamps():
    assert PageSelection(included_pages=[1, 2, 20]).resolve(3) == [1, 2]
    assert PageSelection(excluded_pages=[2]).resolve(3) == [1, 3]
    with pytest.raises(ValidationError, match="either included pages or excluded pages"):
        PageSelection(included_pages=[1], excluded_pages=[2])


def test_chunking_config_validates_overlap_and_hierarchy():
    with pytest.raises(ValidationError, match="overlap"):
        ChunkingConfig(chunk_size=100, overlap=100)
    with pytest.raises(ValidationError, match="parent_size"):
        ChunkingConfig(strategy="hierarchical", chunk_size=500, parent_size=400)


def test_import_request_normalizes_and_validates_name():
    request = ImportRequest(document_id="doc", company_id="company", name="  Product   guide  ")
    assert request.name == "PRODUCT GUIDE"
    with pytest.raises(ValidationError, match="at least 2 characters"):
        ImportRequest(document_id="doc", company_id="company", name="   ")


def test_knowledge_base_uses_one_source_for_files_and_urls():
    common = {
        "company_id": "company",
        "name": "Guide",
        "chunking": ChunkingConfig(),
    }
    assert KnowledgeBase(
        **common,
        source="guide.pdf",
        mime_type="application/pdf",
        total_pages=1,
        selected_pages=1,
    ).source == "guide.pdf"
    assert KnowledgeBase(
        **common, source="https://example.com", mime_type="text/html"
    ).source == "https://example.com"
    with pytest.raises(ValidationError, match="source"):
        KnowledgeBase(**common, source="", mime_type="text/html")
    with pytest.raises(ValidationError, match="cannot contain PDF page information"):
        KnowledgeBase(
            **common,
            source="https://example.com",
            mime_type="text/html",
            total_pages=1,
            selected_pages=1,
        )


def test_company_normalizes_and_validates_profile():
    company = Company(
        name="  Acme   Corp ",
        email="hello@acme.example",
        bot_avatar="brain",
        bot_greet_message="Hello {{company_name}}, I'm {{bot_alias}}.",
    )
    assert company.name == "Acme Corp"
    assert company.bot_avatar == "brain"
    with pytest.raises(ValidationError, match="valid email"):
        Company(name="Acme", email="invalid")
    with pytest.raises(ValidationError, match="at most 1000"):
        Company(name="Acme", about="x" * 1001)
    with pytest.raises(ValidationError, match="at most 400"):
        Company(name="Acme", bot_greet_message="x" * 401)
    with pytest.raises(ValidationError, match="Unsupported greeting placeholder"):
        Company(name="Acme", bot_greet_message="Hello {{user_name}}")
    with pytest.raises(ValidationError, match="malformed placeholder syntax"):
        Company(name="Acme", bot_greet_message="Hello {{company_name}")


def test_llm_selection_requires_an_explicit_provider_and_model():
    assert LLMSelection(provider="openai", model="gpt-5.6-sol").model == "gpt-5.6-sol"
    assert LLMSelection(provider="ollama", model="gemma4:12b").provider == "ollama"
    assert LLMSelection(provider="openai", model="custom-model").model == "custom-model"
    with pytest.raises(ValidationError):
        LLMSelection()
    with pytest.raises(ValidationError, match="at least 1 character"):
        LLMSelection(provider="openai", model="")


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_every_chunking_strategy_returns_metadata(strategy):
    text = (
        "INTRODUCTION:\n\nThis is the first sentence. This is a second sentence.\n\nA separate idea follows with useful details. "
        * 8
    )
    config = ChunkingConfig(strategy=strategy, chunk_size=120, overlap=20, parent_size=320)
    chunks = chunk_pages([(2, text)], "kb-1", "company-1", config)
    assert chunks
    assert all(chunk.page_number == 2 and chunk.knowledge_base_id == "kb-1" for chunk in chunks)
    assert all(chunk.company_id == "company-1" for chunk in chunks)
    assert all("section" in chunk.metadata for chunk in chunks)
    assert all("index" not in chunk.metadata for chunk in chunks)
    assert all("strategy" not in chunk.metadata for chunk in chunks)
    if strategy == ChunkingStrategy.HIERARCHICAL:
        assert {chunk.metadata["level"] for chunk in chunks} == {"parent", "child"}
        parents = {chunk.id: chunk for chunk in chunks if chunk.metadata["level"] == "parent"}
        children = [chunk for chunk in chunks if chunk.metadata["level"] == "child"]
        assert children
        assert all(parent.breadcrumb == ["INTRODUCTION"] for parent in parents.values())
        assert all(
            child.parent_id in parents and child.breadcrumb == ["INTRODUCTION"]
            for child in children
        )
    else:
        assert all("breadcrumb" not in chunk.metadata for chunk in chunks)


def test_chunking_handles_empty_text_and_plain_first_line():
    config = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=0)
    assert chunk_pages([(1, "")], "kb-id", "company-id", config) == []
    chunks = chunk_pages(
        [(1, "A normal long first line without a heading because it continues here. Text.")],
        "id",
        "company-1",
        config,
    )
    assert chunks[0].section is None
    assert chunks[0].metadata["section"] is None
    assert chunks[0].breadcrumb == []
    assert "breadcrumb" not in chunks[0].metadata


@pytest.mark.parametrize(
    "strategy",
    [
        ChunkingStrategy.FIXED,
        ChunkingStrategy.RECURSIVE,
        ChunkingStrategy.SEMANTIC,
    ],
)
def test_overlap_keeps_complete_sentence_boundaries(strategy):
    first = "Alpha workers review every request before final approval."
    overlapping = "Bravo managers record each decision for careful auditing."
    last = "Charlie operators publish the completed result to customers."
    chunks = chunk_pages(
        [(1, f"{first} {overlapping} {last}")],
        "kb-id",
        "company-id",
        ChunkingConfig(strategy=strategy, chunk_size=115, overlap=20),
    )

    assert [chunk.text for chunk in chunks] == [
        f"{first} {overlapping}",
        f"{overlapping} {last}",
    ]
    assert len(overlapping) > 20


def test_hierarchical_child_overlap_keeps_complete_sentence_boundaries():
    first = "Alpha workers review every request before final approval."
    overlapping = "Bravo managers record each decision for careful auditing."
    last = "Charlie operators publish the completed result to customers."
    chunks = chunk_pages(
        [(1, f"{first} {overlapping} {last}")],
        "kb-id",
        "company-id",
        ChunkingConfig(
            strategy=ChunkingStrategy.HIERARCHICAL,
            chunk_size=115,
            overlap=20,
            parent_size=400,
        ),
    )

    children = [chunk.text for chunk in chunks if chunk.metadata["level"] == "child"]
    assert children == [f"{first} {overlapping}", f"{overlapping} {last}"]


def test_overlap_stops_at_available_sentence_boundaries():
    long_sentence = "A" * 130 + "."
    final_sentence = "The final sentence stays complete even at the document end."
    chunks = chunk_pages(
        [(1, f"{long_sentence} {final_sentence}")],
        "kb-id",
        "company-id",
        ChunkingConfig(strategy=ChunkingStrategy.FIXED, chunk_size=100, overlap=90),
    )

    assert [chunk.text for chunk in chunks] == [long_sentence, final_sentence]


def test_hierarchical_breadcrumb_contains_only_heading_path():
    text = """# Employee Handbook

## Leave Policy

### Exceptions

Special leave requires approval."""
    chunks = chunk_pages(
        [(7, text)],
        "kb-id",
        "company-id",
        ChunkingConfig(strategy="hierarchical", chunk_size=100, overlap=0, parent_size=300),
    )
    exception_chunks = [chunk for chunk in chunks if "Special leave" in chunk.text]
    assert exception_chunks
    assert all(
        chunk.breadcrumb == ["Employee Handbook", "Leave Policy", "Exceptions"]
        for chunk in exception_chunks
    )
    assert all("Page 7" not in chunk.breadcrumb for chunk in chunks)
    assert all(chunk.id not in chunk.breadcrumb for chunk in chunks)


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_url_chunks_omit_page_number_and_preserve_sections(strategy):
    chunks = chunk_pages(
        [(None, "# Guide\n\n## Installation\n\nInstall the package. Restart the service.")],
        "web-kb",
        "company",
        ChunkingConfig(strategy=strategy, chunk_size=100, overlap=0, parent_size=300),
    )
    assert chunks
    assert all("page_number" not in chunk.metadata and chunk.page_number == 0 for chunk in chunks)
    assert any(chunk.section == "Installation" for chunk in chunks)
