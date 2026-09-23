import asyncio
from pathlib import Path

import pytest

from packages.omni_rag_core.domain import (
    Citation,
    ChatRequest,
    Chunk,
    ChunkingConfig,
    Company,
    ImportRequest,
    KnowledgeBase,
)
from packages.omni_rag_core.services import (
    ChatService,
    CompanyIndexingError,
    CompanyService,
    ImportService,
    citations_for_answer,
    normalize_answer_citations,
)


class Documents:
    def path_for(self, document_id):
        if document_id == "missing":
            raise FileNotFoundError
        return Path("fake.pdf")

    def extract(self, path):
        return [
            "HEADING:\nA refund is available within thirty days.",
            "Contact support for assistance.",
        ]


class Repository:
    def __init__(self):
        self.saved = []

    def save(self, kb):
        self.saved.append(kb.model_copy(deep=True))
        return kb

    def get(self, knowledge_base_id):
        return next((item for item in reversed(self.saved) if item.id == knowledge_base_id), None)

    def replace(self, knowledge_base_id, kb):
        self.saved = [item for item in self.saved if item.id != knowledge_base_id]
        self.saved.append(kb.model_copy(deep=True))
        return kb

    def get_company(self, company_id):
        return Company(id=company_id, name="Acme")

    def save_company(self, company):
        self.company = company
        return company


class Vectors:
    def __init__(self):
        self.chunks = []
        self.searched_sources = []

    def upsert(self, chunks, embeddings):
        self.chunks = chunks

    def upsert_strict(self, chunks, embeddings):
        self.upsert(chunks, embeddings)

    def search(self, query, embedding, ids, company, limit=6):
        self.searched_sources = ids
        return [
            (
                Chunk(
                    knowledge_base_id="kb",
                    company_id="company",
                    text="Refunds take 30 days.",
                    metadata={"page_number": 4, "section": "Refunds"},
                ),
                0.81234,
            )
        ]

    def delete_company_metadata(self, company_id):
        self.deleted_metadata = company_id

    def delete_knowledge_base(self, company_id, knowledge_base_id):
        self.deleted_knowledge_base = (company_id, knowledge_base_id)


class AI:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def chat(self, messages):
        self.messages = messages
        return "Use the refund form [Source 1]."

    async def stream_chat(self, messages):
        yield "Use "
        yield "the form."


def test_import_service_indexes_selected_pages():
    repository, vectors, ai = Repository(), Vectors(), AI()
    service = ImportService(Documents(), repository, vectors, ai)
    request = ImportRequest(
        document_id="doc", company_id="company", name="Support guide", pages={"included_pages": [1]}
    )
    result = asyncio.run(service.import_document(request, "guide.pdf", "a" * 64))
    assert result.status == "enabled"
    assert result.selected_pages == 1
    assert result.excluded_pages == [2]
    assert result.chunk_count == len(vectors.chunks) > 0
    assert result.document_checksum == "a" * 64
    assert [item.status for item in repository.saved] == ["indexing", "enabled"]


def test_import_service_requires_pages_and_propagates_missing():
    service = ImportService(Documents(), Repository(), Vectors(), AI())
    with pytest.raises(ValueError, match="At least one"):
        asyncio.run(
            service.import_document(
                ImportRequest(
                    document_id="doc",
                    company_id="company",
                    name="Guide",
                    pages={"included_pages": [99]},
                ),
                "x.pdf",
                "b" * 64,
            )
        )
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            service.import_document(
                ImportRequest(document_id="missing", company_id="company", name="Guide"),
                "x.pdf",
                "c" * 64,
            )
        )


def test_import_service_indexes_and_refreshes_url_without_page_metadata():
    repository, vectors, ai = Repository(), Vectors(), AI()
    service = ImportService(Documents(), repository, vectors, ai)
    request = ImportRequest(
        document_id="web",
        company_id="company",
        name="Web guide",
        chunking={
            "strategy": "hierarchical",
            "chunk_size": 100,
            "overlap": 20,
            "parent_size": 300,
        },
    )
    result = asyncio.run(
        service.import_web_page(
            request,
            "https://example.com/guide",
            "d" * 64,
            "# Guide\n\n## Setup\n\nInstall the application before continuing.",
        )
    )
    assert result.source == "https://example.com/guide"
    assert result.total_pages is None
    assert result.selected_pages is None
    assert result.excluded_pages is None
    assert result.chunk_count == len(vectors.chunks) > 0
    assert all("page_number" not in chunk.metadata for chunk in vectors.chunks)
    assert any(chunk.breadcrumb == ["Guide", "Setup"] for chunk in vectors.chunks)

    existing = KnowledgeBase(
        id="old-web",
        company_id="company",
        name="Old web guide",
        source="https://example.com/old",
        mime_type="text/html",
        chunking=ChunkingConfig(),
        status="disabled",
    )
    repository.saved.append(existing)
    refreshed = asyncio.run(
        service.import_web_page(
            request.model_copy(update={"name": "Refreshed guide"}),
            existing.source,
            "e" * 64,
            "Updated content is now available.",
            existing.id,
        )
    )
    assert refreshed.name == "REFRESHED GUIDE"
    assert refreshed.status == "disabled"
    assert repository.get(existing.id) is None
    assert vectors.deleted_knowledge_base == ("company", existing.id)


@pytest.mark.parametrize("strategy", ["fixed", "semantic"])
def test_import_service_rejects_unsupported_url_chunking(strategy):
    service = ImportService(Documents(), Repository(), Vectors(), AI())
    request = ImportRequest(
        document_id="web",
        company_id="company",
        name="Web guide",
        chunking={"strategy": strategy},
    )
    with pytest.raises(ValueError, match="recursive or hierarchical"):
        asyncio.run(
            service.import_web_page(
                request,
                "https://example.com/guide",
                "f" * 64,
                "Extracted content.",
            )
        )


def test_company_service_indexes_separate_labeled_chunks():
    repository, vectors, ai = Repository(), Vectors(), AI()
    service = CompanyService(repository, vectors, ai)
    company = Company(
        name="Acme",
        about="Rocket systems",
        phone="+1 555 0100",
        email="hello@acme.example",
        address="1 Orbit Way",
        maps_url="https://maps.example/acme",
    )
    saved = asyncio.run(service.index(company))
    assert saved.name == "Acme"
    assert vectors.deleted_metadata == company.id
    assert [chunk.section for chunk in vectors.chunks] == [
        "Company Name",
        "About",
        "Contact Details",
        "Address",
    ]
    assert all(chunk.source_type == "company_metadata" for chunk in vectors.chunks)
    assert all(chunk.company_id == company.id for chunk in vectors.chunks)
    assert all(chunk.knowledge_base_id == company.id for chunk in vectors.chunks)
    assert all(
        set(chunk.metadata) == {"section"} and chunk.page_number == 0 for chunk in vectors.chunks
    )


def test_company_service_does_not_save_when_vector_refresh_fails():
    class FailingVectors(Vectors):
        def __init__(self):
            super().__init__()
            self.delete_attempts = 0

        def delete_company_metadata(self, company_id):
            self.delete_attempts += 1
            raise RuntimeError("Qdrant offline")

    repository, vectors = Repository(), FailingVectors()
    service = CompanyService(repository, vectors, AI())
    with pytest.raises(CompanyIndexingError, match="could not be synchronized"):
        asyncio.run(service.index(Company(name="New name")))
    assert not hasattr(repository, "company")
    assert vectors.delete_attempts == 2


def test_chat_service_answer_and_stream():
    ai = AI()
    vectors = Vectors()
    service = ChatService(vectors, ai)
    request = ChatRequest(
        message="How do refunds work?",
        session_id="session-1",
        knowledge_base_ids=["kb"],
        history=[{"role": "user", "content": "Earlier question"}],
    )
    response = asyncio.run(
        service.answer(
            request,
            {"kb": "GUIDE"},
            "company",
            {"kb": "https://example.com/guide"},
        )
    )
    assert response.answer.startswith("Use")
    assert response.session_id
    assert response.citations[0].page_number == 4
    assert response.citations[0].knowledge_base_id == "kb"
    assert response.citations[0].knowledge_base == "GUIDE"
    assert response.citations[0].source_url == "https://example.com/guide"
    assert vectors.searched_sources == ["kb"]
    assert response.suggested_replies == []

    session, citations, stream = asyncio.run(
        service.stream(
            ChatRequest(
                message="More",
                session_id="session",
                knowledge_base_ids=["kb"],
            ),
            {"kb": "GUIDE"},
            "company",
        )
    )

    async def collect():
        return "".join([part async for part in stream])

    assert session == "session"
    assert citations[0].score == 0.8123
    assert asyncio.run(collect()) == "Use the form."

    empty = asyncio.run(
        service.answer(ChatRequest(message="No sources", session_id="empty-session"))
    )
    assert empty.citations == []


def test_citations_for_answer_returns_only_inline_citations():
    citations = [
        Citation(
            source_number=number,
            knowledge_base_id=f"kb-{number}",
            knowledge_base=f"Guide {number}",
            page_number=number,
            excerpt="Supporting text",
            score=0.9,
        )
        for number in range(1, 4)
    ]

    filtered = citations_for_answer(
        "The policy is supported here [Source 1] and here [Sources 3, 1].", citations
    )

    assert [citation.source_number for citation in filtered] == [1, 2]
    assert [citation.knowledge_base_id for citation in filtered] == ["kb-1", "kb-3"]
    assert citations_for_answer("No inline citation was generated.", citations) == []

    answer, normalized = normalize_answer_citations(
        "This uses only the second candidate [Source 2].", citations
    )
    assert answer == "This uses only the second candidate [Source 1]."
    assert [citation.source_number for citation in normalized] == [1]
    assert [citation.knowledge_base_id for citation in normalized] == ["kb-2"]
