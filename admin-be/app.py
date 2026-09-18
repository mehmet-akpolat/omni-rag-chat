import hashlib
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from uuid import uuid4
from xml.etree import ElementTree

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from packages.omni_rag_core.ai import OllamaClient
from packages.omni_rag_core.documents import DocumentStore, InvalidDocument
from packages.omni_rag_core.domain import (
    ChunkingStrategy,
    ChatSession,
    ChatSessionMessage,
    Company,
    CompanyInput,
    CompanyView,
    DocumentPreview,
    ImportRequest,
    KnowledgeBase,
)
from packages.omni_rag_core.repository import (
    DuplicateCompanyName,
    DuplicateDocumentContent,
    DuplicateKnowledgeBaseName,
    SqlKnowledgeRepository,
)
from packages.omni_rag_core.services import CompanyIndexingError, CompanyService, ImportService
from packages.omni_rag_core.settings import Settings, get_settings
from packages.omni_rag_core.vector_store import QdrantVectorStore, ResilientVectorStore

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_LOGO_BYTES = 512 * 1024
MAX_CHAT_SESSION_DATE_RANGE_DAYS = 14
LOGO_MIME_TYPES = {"image/png", "image/jpeg", "image/svg+xml"}


def valid_logo(content: bytes, mime_type: str) -> bool:
    if mime_type == "image/png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return content.startswith(b"\xff\xd8\xff")
    if mime_type != "image/svg+xml":
        return False
    lowered = content.lower()
    if any(token in lowered for token in (b"<!doctype", b"<!entity", b"<script")):
        return False
    try:
        root = ElementTree.fromstring(content)
    except (ElementTree.ParseError, ValueError):
        return False
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        return False
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag in {"script", "foreignobject", "iframe", "object", "embed"}:
            return False
        for attribute, value in element.attrib.items():
            if attribute.rsplit("}", 1)[-1].lower().startswith("on"):
                return False
            if value.strip().lower().startswith("javascript:"):
                return False
    return True


class KnowledgeBaseState(BaseModel):
    enabled: bool


class KnowledgeBasePage(BaseModel):
    items: list[KnowledgeBase]
    page: int
    page_size: int
    total: int
    total_pages: int


class ChatSessionPage(BaseModel):
    items: list[ChatSession]
    page: int
    page_size: int
    total: int
    total_pages: int


class ChatSessionDetail(BaseModel):
    session: ChatSession
    messages: list[ChatSessionMessage]


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        documents = DocumentStore(config.upload_dir)
        repository = SqlKnowledgeRepository(config.database_url)
        vectors = ResilientVectorStore(QdrantVectorStore(config))
        app.state.documents = documents
        app.state.repository = repository
        app.state.vectors = vectors
        app.state.importer = ImportService(documents, repository, vectors, OllamaClient(config))
        app.state.companies = CompanyService(repository, vectors, OllamaClient(config))
        app.state.uploads = {}
        try:
            yield
        finally:
            repository.engine.dispose()

    app = FastAPI(
        title="Omni RAG Admin API",
        version="1.0.0",
        description="PDF preview, configuration, and knowledge-base ingestion.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "admin-backend"}

    @app.get("/api/v1/chunking-strategies")
    def strategies() -> list[dict]:
        return [
            {
                "id": ChunkingStrategy.FIXED,
                "name": "Fixed-size",
                "description": "Predictable character windows with overlap.",
            },
            {
                "id": ChunkingStrategy.RECURSIVE,
                "name": "Recursive",
                "description": "Preserves paragraphs and sentence boundaries.",
            },
            {
                "id": ChunkingStrategy.SEMANTIC,
                "name": "Semantic",
                "description": "Groups related paragraphs into coherent passages.",
            },
            {
                "id": ChunkingStrategy.HIERARCHICAL,
                "name": "Hierarchical",
                "description": "Indexes parent sections alongside smaller child chunks.",
            },
        ]

    @app.post("/api/v1/documents/preview", response_model=DocumentPreview)
    async def preview(
        request: Request,
        file: UploadFile = File(...),
        company_id: str = Form(...),
    ) -> DocumentPreview:
        if not request.app.state.repository.get_company(company_id):
            raise HTTPException(404, "Company not found")
        if file.content_type not in {"application/pdf", "application/x-pdf"} or not file.filename:
            raise HTTPException(415, "Only PDF files are supported")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "PDF exceeds the 40 MB upload limit")
        if not content.startswith(b"%PDF"):
            raise HTTPException(422, "The selected file is not a valid PDF")
        checksum = hashlib.sha256(content).hexdigest()
        duplicate = request.app.state.repository.get_by_checksum(checksum, company_id)
        if duplicate:
            raise HTTPException(
                409,
                f'This document content is already imported as "{duplicate.name}"',
            )
        document_id = str(uuid4())
        path = request.app.state.documents.save(document_id, content)
        try:
            pages = request.app.state.documents.extract(path)
        except InvalidDocument as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(422, str(exc)) from exc
        request.app.state.uploads[document_id] = {
            "filename": file.filename,
            "checksum": checksum,
            "company_id": company_id,
        }
        return request.app.state.documents.preview(document_id, file.filename, pages)

    @app.post("/api/v1/knowledge-bases", response_model=KnowledgeBase, status_code=201)
    async def create_knowledge_base(payload: ImportRequest, request: Request) -> KnowledgeBase:
        upload = request.app.state.uploads.get(payload.document_id)
        if not upload:
            raise HTTPException(404, "Uploaded document not found; upload it again")
        filename = upload["filename"]
        checksum = upload["checksum"]
        if upload["company_id"] != payload.company_id:
            raise HTTPException(422, "Uploaded document belongs to a different company")
        company = request.app.state.repository.get_company(payload.company_id)
        if not company:
            raise HTTPException(404, "Company not found")
        duplicate = request.app.state.repository.get_by_checksum(checksum, payload.company_id)
        if duplicate:
            raise HTTPException(
                409,
                f'This document content is already imported as "{duplicate.name}"',
            )
        if request.app.state.repository.name_exists(payload.name, payload.company_id):
            raise HTTPException(409, f'A knowledge base named "{payload.name}" already exists')
        try:
            result = await request.app.state.importer.import_document(payload, filename, checksum)
            request.app.state.uploads.pop(payload.document_id, None)
            return result
        except DuplicateDocumentContent as exc:
            raise HTTPException(409, str(exc)) from exc
        except DuplicateKnowledgeBaseName as exc:
            raise HTTPException(409, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "Uploaded document not found; upload it again") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/v1/knowledge-bases", response_model=KnowledgeBasePage)
    def list_knowledge_bases(
        request: Request,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=10, ge=1, le=100),
        search: str | None = Query(default=None, max_length=120),
        company_id: str | None = Query(default=None),
    ) -> KnowledgeBasePage:
        repository = request.app.state.repository
        total = repository.count(search=search, company_id=company_id)
        return KnowledgeBasePage(
            items=repository.list(
                offset=(page - 1) * page_size,
                limit=page_size,
                search=search,
                company_id=company_id,
            ),
            page=page,
            page_size=page_size,
            total=total,
            total_pages=(total + page_size - 1) // page_size,
        )

    @app.get("/api/v1/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBase)
    def get_knowledge_base(knowledge_base_id: str, request: Request) -> KnowledgeBase:
        item = request.app.state.repository.get(knowledge_base_id)
        if not item:
            raise HTTPException(404, "Knowledge base not found")
        return item

    @app.get("/api/v1/chat-sessions", response_model=ChatSessionPage)
    def list_chat_sessions(
        request: Request,
        company_id: str = Query(...),
        date_from: date = Query(...),
        date_to: date = Query(...),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
    ) -> ChatSessionPage:
        if not request.app.state.repository.get_company(company_id):
            raise HTTPException(404, "Company not found")
        if date_from > date_to:
            raise HTTPException(422, "date_from must not be after date_to")
        if (date_to - date_from).days >= MAX_CHAT_SESSION_DATE_RANGE_DAYS:
            raise HTTPException(
                422,
                f"Chat session date range cannot exceed {MAX_CHAT_SESSION_DATE_RANGE_DAYS} days",
            )
        created_from = datetime.combine(date_from, time.min, tzinfo=timezone.utc)
        created_to = datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=timezone.utc)
        repository = request.app.state.repository
        total = repository.count_chat_sessions(
            company_id, created_from=created_from, created_to=created_to
        )
        return ChatSessionPage(
            items=repository.list_chat_sessions(
                company_id,
                offset=(page - 1) * page_size,
                limit=page_size,
                created_from=created_from,
                created_to=created_to,
            ),
            page=page,
            page_size=page_size,
            total=total,
            total_pages=(total + page_size - 1) // page_size,
        )

    @app.get("/api/v1/chat-sessions/{session_id}", response_model=ChatSessionDetail)
    def get_chat_session(session_id: str, request: Request) -> ChatSessionDetail:
        chat_session = request.app.state.repository.get_chat_session(session_id)
        if not chat_session:
            raise HTTPException(404, "Chat session not found")
        return ChatSessionDetail(
            session=chat_session,
            messages=request.app.state.repository.list_chat_messages(session_id),
        )

    @app.patch("/api/v1/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBase)
    def update_knowledge_base(
        knowledge_base_id: str, payload: KnowledgeBaseState, request: Request
    ) -> KnowledgeBase:
        item = request.app.state.repository.get(knowledge_base_id)
        if not item:
            raise HTTPException(404, "Knowledge base not found")
        item.status = "enabled" if payload.enabled else "disabled"
        return request.app.state.repository.save(item)

    @app.delete("/api/v1/knowledge-bases/{knowledge_base_id}", status_code=204)
    def delete_knowledge_base(knowledge_base_id: str, request: Request) -> Response:
        item = request.app.state.repository.get(knowledge_base_id)
        if not item:
            raise HTTPException(404, "Knowledge base not found")
        try:
            request.app.state.vectors.delete_knowledge_base(item.company_id, item.id)
        except Exception as exc:
            raise HTTPException(
                503,
                "Vector-store deletion failed; knowledge-base metadata was retained",
            ) from exc
        request.app.state.repository.delete(knowledge_base_id)
        return Response(status_code=204)

    @app.get("/api/v1/companies", response_model=list[CompanyView])
    def list_companies(request: Request) -> list[Company]:
        return request.app.state.repository.list_companies()

    @app.get("/api/v1/companies/{company_id}/logo")
    def get_company_logo(company_id: str, request: Request) -> Response:
        company = request.app.state.repository.get_company(company_id)
        if not company or not company.logo_data:
            raise HTTPException(404, "Company logo not found")
        return Response(
            content=company.logo_data,
            media_type=company.logo_mime_type or "application/octet-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
            },
        )

    @app.put("/api/v1/companies/{company_id}/logo", response_model=CompanyView)
    async def update_company_logo(
        company_id: str, request: Request, file: UploadFile = File(...)
    ) -> Company:
        company = request.app.state.repository.get_company(company_id)
        if not company:
            raise HTTPException(404, "Company not found")
        mime_type = file.content_type or ""
        if mime_type not in LOGO_MIME_TYPES:
            raise HTTPException(415, "Logo must be a PNG, SVG, or JPEG image")
        content = await file.read(MAX_LOGO_BYTES + 1)
        if len(content) > MAX_LOGO_BYTES:
            raise HTTPException(413, "Company logo exceeds the 512 KB upload limit")
        if not valid_logo(content, mime_type):
            raise HTTPException(422, "The selected file is not a valid image")
        company = company.model_copy(update={"logo_data": content, "logo_mime_type": mime_type})
        return request.app.state.repository.save_company(company)

    @app.delete("/api/v1/companies/{company_id}/logo", status_code=204)
    def delete_company_logo(company_id: str, request: Request) -> Response:
        company = request.app.state.repository.get_company(company_id)
        if not company:
            raise HTTPException(404, "Company not found")
        request.app.state.repository.save_company(
            company.model_copy(update={"logo_data": None, "logo_mime_type": None})
        )
        return Response(status_code=204)

    @app.post("/api/v1/companies", response_model=CompanyView, status_code=201)
    async def create_company(payload: CompanyInput, request: Request) -> Company:
        if request.app.state.repository.company_name_exists(payload.name):
            raise HTTPException(409, f'A company named "{payload.name}" already exists')
        try:
            return await request.app.state.companies.index(Company(**payload.model_dump()))
        except DuplicateCompanyName as exc:
            raise HTTPException(409, str(exc)) from exc
        except CompanyIndexingError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.put("/api/v1/companies/{company_id}", response_model=CompanyView)
    async def update_company(company_id: str, payload: CompanyInput, request: Request) -> Company:
        existing = request.app.state.repository.get_company(company_id)
        if not existing:
            raise HTTPException(404, "Company not found")
        company = Company(
            **payload.model_dump(),
            id=company_id,
            created_at=existing.created_at,
            logo_data=existing.logo_data,
            logo_mime_type=existing.logo_mime_type,
        )
        if request.app.state.repository.company_name_exists(company.name, exclude_id=company_id):
            raise HTTPException(409, f'A company named "{company.name}" already exists')
        try:
            return await request.app.state.companies.index(company)
        except DuplicateCompanyName as exc:
            raise HTTPException(409, str(exc)) from exc
        except CompanyIndexingError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.delete("/api/v1/companies/{company_id}", status_code=204)
    def delete_company(company_id: str, request: Request) -> Response:
        company = request.app.state.repository.get_company(company_id)
        if not company:
            raise HTTPException(404, "Company not found")
        try:
            request.app.state.vectors.delete_company(company.id)
        except Exception as exc:
            raise HTTPException(
                503, "Vector-store deletion failed; company metadata was retained"
            ) from exc
        request.app.state.repository.delete_company(company_id)
        return Response(status_code=204)

    return app


app = create_app()
