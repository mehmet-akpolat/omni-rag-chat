import re
from collections.abc import AsyncIterator
from uuid import uuid4

from .ai import ChatClient, OllamaClient
from .chunking import chunk_pages
from .documents import DocumentStore
from .domain import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Chunk,
    ChunkingStrategy,
    Citation,
    Company,
    ImportRequest,
    KnowledgeBase,
)
from .repository import KnowledgeRepository
from .vector_store import VectorStore

URL_CHUNKING_STRATEGIES = frozenset(
    {ChunkingStrategy.RECURSIVE, ChunkingStrategy.HIERARCHICAL}
)


class CompanyIndexingError(RuntimeError):
    pass


class ImportService:
    def __init__(
        self,
        documents: DocumentStore,
        repository: KnowledgeRepository,
        vectors: VectorStore,
        ai: OllamaClient,
    ):
        self.documents = documents
        self.repository = repository
        self.vectors = vectors
        self.ai = ai

    async def import_document(
        self, request: ImportRequest, source: str, document_checksum: str
    ) -> KnowledgeBase:
        pages = self.documents.extract(self.documents.path_for(request.document_id))
        selected = request.pages.resolve(len(pages))
        if not selected:
            raise ValueError("At least one page must be selected")
        excluded = sorted(set(range(1, len(pages) + 1)) - set(selected))
        company = self.repository.get_company(request.company_id)
        if not company:
            raise ValueError("Company not found")
        kb = KnowledgeBase(
            company_id=company.id,
            name=request.name,
            source=source,
            mime_type="application/pdf",
            total_pages=len(pages),
            excluded_pages=excluded,
            selected_pages=len(selected),
            chunking=request.chunking,
            status="indexing",
            document_checksum=document_checksum,
        )
        self.repository.save(kb)
        chosen_pages = [(number, pages[number - 1]) for number in selected]
        chunks = chunk_pages(chosen_pages, kb.id, company.id, request.chunking)
        embeddings = await self.ai.embed([chunk.text for chunk in chunks])
        self.vectors.upsert(chunks, embeddings)
        kb.chunk_count = len(chunks)
        kb.status = "enabled"
        return self.repository.save(kb)

    async def import_web_page(
        self,
        request: ImportRequest,
        source: str,
        document_checksum: str,
        text: str,
        replaces_knowledge_base_id: str | None = None,
    ) -> KnowledgeBase:
        if request.chunking.strategy not in URL_CHUNKING_STRATEGIES:
            raise ValueError(
                "URL knowledge bases support only recursive or hierarchical chunking"
            )
        company = self.repository.get_company(request.company_id)
        if not company:
            raise ValueError("Company not found")
        existing = None
        if replaces_knowledge_base_id:
            existing = self.repository.get(replaces_knowledge_base_id)
            if not existing or existing.company_id != company.id:
                raise ValueError("Knowledge base to refresh was not found")
        kb = KnowledgeBase(
            company_id=company.id,
            name=request.name,
            source=source,
            mime_type="text/html",
            total_pages=None,
            excluded_pages=None,
            selected_pages=None,
            chunking=request.chunking,
            status="indexing",
            document_checksum=document_checksum,
        )
        chunks = chunk_pages([(None, text)], kb.id, company.id, request.chunking)
        if not chunks:
            raise ValueError("The URL does not contain enough text to index")
        embeddings = await self.ai.embed([chunk.text for chunk in chunks])
        try:
            self.vectors.upsert(chunks, embeddings)
            kb.chunk_count = len(chunks)
            kb.status = existing.status if existing else "enabled"
            if existing:
                saved = self.repository.replace(existing.id, kb)
            else:
                saved = self.repository.save(kb)
        except Exception:
            self.vectors.delete_knowledge_base(company.id, kb.id)
            raise
        if existing:
            try:
                self.vectors.delete_knowledge_base(company.id, existing.id)
            except Exception:
                pass
        return saved


class CompanyService:
    def __init__(self, repository: KnowledgeRepository, vectors: VectorStore, ai: OllamaClient):
        self.repository = repository
        self.vectors = vectors
        self.ai = ai

    @staticmethod
    def _chunks(company: Company) -> list[Chunk]:
        fields = [
            ("Company Name", company.name),
            ("About", company.about),
            (
                "Contact Details",
                "; ".join(
                    value
                    for value in [
                        f"Phone: {company.phone}" if company.phone else "",
                        f"Email: {company.email}" if company.email else "",
                    ]
                    if value
                ),
            ),
            (
                "Address",
                "; ".join(
                    value
                    for value in [
                        company.address,
                        f"Map: {company.maps_url}" if company.maps_url else "",
                    ]
                    if value
                ),
            ),
        ]
        return [
            Chunk(
                knowledge_base_id=company.id,
                company_id=company.id,
                text=f"{label}: {value}",
                metadata={"section": label},
                source_type="company_metadata",
            )
            for label, value in fields
            if value
        ]

    async def index(self, company: Company) -> Company:
        chunks = self._chunks(company)
        embeddings = await self.ai.embed([chunk.text for chunk in chunks])
        try:
            self.vectors.delete_company_metadata(company.id)
            self.vectors.upsert_strict(chunks, embeddings)
        except Exception as exc:
            try:
                self.vectors.delete_company_metadata(company.id)
            except Exception:
                pass
            raise CompanyIndexingError(
                "Company metadata could not be synchronized with the vector store"
            ) from exc
        return self.repository.save_company(company)


class CitationRetagger:
    """Renumber cited retrieval candidates in their first-use order."""

    def __init__(self, citations: list[Citation]):
        self._candidates = {
            citation.source_number or position: citation
            for position, citation in enumerate(citations, 1)
        }
        self._number_mapping: dict[int, int] = {}

    def transform(self, text: str) -> str:
        def replace_marker(match: re.Match[str]) -> str:
            original_numbers = [
                int(number) for number in re.findall(r"\d+", match.group(1))
            ]
            new_numbers: list[int] = []
            for original_number in original_numbers:
                if original_number in self._candidates:
                    if original_number not in self._number_mapping:
                        self._number_mapping[original_number] = len(self._number_mapping) + 1
                    new_numbers.append(self._number_mapping[original_number])
                else:
                    new_numbers.append(original_number)
            label = "Source" if len(new_numbers) == 1 else "Sources"
            return f"[{label} {', '.join(str(number) for number in new_numbers)}]"

        return re.sub(
            r"\[\s*Sources?\s+([^\]]+)\]",
            replace_marker,
            text,
            flags=re.IGNORECASE,
        )

    def citations(self) -> list[Citation]:
        cited_numbers = sorted(self._number_mapping, key=self._number_mapping.get)
        return [
            self._candidates[original_number].model_copy(
                update={"source_number": self._number_mapping[original_number]}
            )
            for original_number in cited_numbers
        ]


def citations_for_answer(answer: str, citations: list[Citation]) -> list[Citation]:
    """Return only retrieval candidates referenced by an inline [Source N] marker."""
    retagger = CitationRetagger(citations)
    retagger.transform(answer)
    return retagger.citations()


def normalize_answer_citations(
    answer: str, citations: list[Citation]
) -> tuple[str, list[Citation]]:
    """Filter citations and renumber both the answer markers and returned sources."""
    retagger = CitationRetagger(citations)
    normalized_answer = retagger.transform(answer)
    return normalized_answer, retagger.citations()


class ChatService:
    def __init__(self, vectors: VectorStore, ai: OllamaClient):
        self.vectors = vectors
        self.ai = ai

    async def _context(
        self,
        request: ChatRequest,
        source_names_by_id: dict[str, str] | None = None,
        company_id: str = "",
        source_urls_by_id: dict[str, str] | None = None,
    ) -> tuple[list[Citation], str]:
        if not company_id:
            return [], ""
        source_names_by_id = source_names_by_id or {}
        source_urls_by_id = source_urls_by_id or {}
        source_ids = list(source_names_by_id) or request.knowledge_base_ids
        embedding = (await self.ai.embed([request.message]))[0]
        matches = self.vectors.search(request.message, embedding, source_ids, company_id)
        citations = [
            Citation(
                source_number=index,
                knowledge_base_id=chunk.knowledge_base_id,
                knowledge_base=source_names_by_id.get(chunk.knowledge_base_id, "COMPANY PROFILE"),
                page_number=chunk.page_number,
                excerpt=chunk.text[:220],
                score=round(score, 4),
                section=chunk.section,
                source_url=source_urls_by_id.get(chunk.knowledge_base_id),
            )
            for index, (chunk, score) in enumerate(matches, 1)
        ]
        context = "\n\n".join(
            f"[Source {index}: "
            f"{source_names_by_id.get(chunk.knowledge_base_id, 'COMPANY PROFILE')}, "
            f"{f'page {chunk.page_number}' if chunk.page_number else chunk.section or 'web page'}]"
            f"\n{chunk.text}"
            for index, (chunk, _) in enumerate(matches, 1)
        )
        return citations, context

    @staticmethod
    def _messages(request: ChatRequest, context: str) -> list[ChatMessage]:
        system = (
            "You are a careful knowledge assistant. Answer only from the supplied context. "
            "When context is insufficient, say so and suggest contacting a human. "
            "Cite every supported claim inline using the exact marker [Source N], and only "
            "use source numbers present in the context.\n\nCONTEXT:\n"
            + (context or "No relevant context found.")
        )
        return [
            ChatMessage(role="system", content=system),
            *request.history[-12:],
            ChatMessage(role="user", content=request.message),
        ]

    async def answer(
        self,
        request: ChatRequest,
        source_names_by_id: dict[str, str] | None = None,
        company_id: str = "",
        source_urls_by_id: dict[str, str] | None = None,
        chat_client: ChatClient | None = None,
    ) -> ChatResponse:
        citations, context = await self._context(
            request, source_names_by_id, company_id, source_urls_by_id
        )
        answer = await (chat_client or self.ai).chat(self._messages(request, context))
        answer, citations = normalize_answer_citations(answer, citations)
        return ChatResponse(
            session_id=request.session_id or str(uuid4()),
            answer=answer,
            citations=citations,
        )

    async def stream(
        self,
        request: ChatRequest,
        source_names_by_id: dict[str, str] | None = None,
        company_id: str = "",
        source_urls_by_id: dict[str, str] | None = None,
        chat_client: ChatClient | None = None,
    ) -> tuple[str, list[Citation], AsyncIterator[str]]:
        citations, context = await self._context(
            request, source_names_by_id, company_id, source_urls_by_id
        )
        session_id = request.session_id or str(uuid4())
        return session_id, citations, (chat_client or self.ai).stream_chat(
            self._messages(request, context)
        )
