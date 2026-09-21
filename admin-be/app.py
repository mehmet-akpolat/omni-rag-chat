import hashlib
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from uuid import uuid4
from xml.etree import ElementTree

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from packages.omni_rag_core.ai import OllamaClient
from packages.omni_rag_core.credentials import CredentialEncryptionError
from packages.omni_rag_core.documents import DocumentStore, InvalidDocument
from packages.omni_rag_core.domain import (
    ChunkingStrategy,
    ChatSession,
    ChatSessionMessage,
    Company,
    CompanyInput,
    CompanyLLMMapping,
    CompanyView,
    DocumentPreview,
    ImportRequest,
    KnowledgeBase,
    LLMProvider,
    LLMSelection,
    PagePreview,
)
from packages.omni_rag_core.repository import (
    DuplicateCompanyName,
    DuplicateDocumentContent,
    DuplicateKnowledgeBaseName,
    SqlKnowledgeRepository,
)
from packages.omni_rag_core.services import (
    URL_CHUNKING_STRATEGIES,
    CompanyIndexingError,
    CompanyService,
    ImportService,
)
from packages.omni_rag_core.settings import Settings, get_settings
from packages.omni_rag_core.vector_store import QdrantVectorStore, ResilientVectorStore
from packages.omni_rag_core.web_documents import WebDocumentFetcher

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


class CompanyLLMInput(LLMSelection):
    api_key: str | None = Field(default=None, max_length=4096)

    @field_validator("api_key")
    @classmethod
    def normalize_api_key(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class CompanyLLMView(LLMSelection):
    has_api_key: bool = False
    api_key_masked: str | None = None


class CompanyConfigurationInput(CompanyInput):
    llm: CompanyLLMInput | None = None


class CompanyConfigurationView(CompanyView):
    llm: CompanyLLMView | None = None


class UrlPreviewRequest(BaseModel):
    company_id: str
    url: str = Field(min_length=8, max_length=2048)


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


class ChatSessionDeleteRequest(BaseModel):
    company_id: str
    session_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)
    date_from: date | None = None
    date_to: date | None = None


class ChatSessionDeleteResult(BaseModel):
    deleted: int


def is_url_mime(mime_type: str) -> bool:
    return mime_type in {"text/html", "application/xhtml+xml"}


def masked_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    if len(api_key) < 4:
        return "•" * len(api_key)
    return f"{api_key[:2]}{'•' * 8}{api_key[-2:]}"


def validate_llm_selection(selection: LLMSelection, config: Settings) -> None:
    if selection.model not in config.llm_models[selection.provider.value]:
        raise HTTPException(
            422,
            f'{selection.model} is not configured for provider "{selection.provider.value}"',
        )


def company_configuration(company: Company, repository) -> CompanyConfigurationView:
    mapping = repository.get_company_llm_mapping(company.id)
    return CompanyConfigurationView(
        **CompanyView.model_validate(company.model_dump()).model_dump(),
        llm=(
            CompanyLLMView(
                provider=mapping.provider,
                model=mapping.model,
                has_api_key=mapping.has_api_key,
                api_key_masked=masked_api_key(mapping.api_key),
            )
            if mapping
            else None
        ),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        documents = DocumentStore(config.upload_dir)
        repository = SqlKnowledgeRepository(config.database_url, config.llm_credentials_key)
        vectors = ResilientVectorStore(QdrantVectorStore(config))
        app.state.documents = documents
        app.state.repository = repository
        app.state.vectors = vectors
        app.state.importer = ImportService(documents, repository, vectors, OllamaClient(config))
        app.state.companies = CompanyService(repository, vectors, OllamaClient(config))
        app.state.web_documents = WebDocumentFetcher(
            timeout=config.web_fetch_timeout,
            max_bytes=config.web_fetch_max_bytes,
            max_redirects=config.web_fetch_max_redirects,
        )
        app.state.uploads = {}
        try:
            yield
        finally:
            repository.engine.dispose()

    app = FastAPI(
        title="Omni RAG Admin API",
        version="1.0.0",
        description="PDF and URL preview, configuration, and knowledge-base ingestion.",
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
                "description": "Keeps neighboring paragraphs together in coherent passages.",
            },
            {
                "id": ChunkingStrategy.HIERARCHICAL,
                "name": "Hierarchical",
                "description": "Indexes parent sections alongside smaller child chunks.",
            },
        ]

    @app.get("/api/v1/llm-options")
    def llm_options() -> list[dict[str, object]]:
        return [
            {"provider": provider.value, "models": list(config.llm_models[provider.value])}
            for provider in LLMProvider
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
            "source": file.filename,
            "mime_type": "application/pdf",
            "checksum": checksum,
            "company_id": company_id,
        }
        return request.app.state.documents.preview(document_id, file.filename, pages)

    @app.post("/api/v1/urls/preview", response_model=DocumentPreview)
    async def preview_url(payload: UrlPreviewRequest, request: Request) -> DocumentPreview:
        repository = request.app.state.repository
        if not repository.get_company(payload.company_id):
            raise HTTPException(404, "Company not found")
        try:
            page = await request.app.state.web_documents.fetch(payload.url)
        except InvalidDocument as exc:
            raise HTTPException(422, str(exc)) from exc
        checksum = hashlib.sha256(page.text.encode("utf-8")).hexdigest()
        duplicate = repository.get_by_checksum(checksum, payload.company_id)
        if duplicate:
            raise HTTPException(
                409,
                f'This URL content is already imported as "{duplicate.name}"',
            )
        existing = repository.get_by_url_source(page.url, payload.company_id)
        document_id = str(uuid4())
        request.app.state.uploads[document_id] = {
            "source": page.url,
            "title": page.title,
            "mime_type": "text/html",
            "checksum": checksum,
            "company_id": payload.company_id,
            "text": page.text,
            "replaces_knowledge_base_id": existing.id if existing else None,
        }
        return DocumentPreview(
            document_id=document_id,
            source=page.url,
            mime_type="text/html",
            total_pages=1,
            pages=[PagePreview(page_number=1, excerpt=" ".join(page.text.split())[:420])],
            title=page.title,
            replaces_knowledge_base_id=existing.id if existing else None,
        )

    @app.post("/api/v1/knowledge-bases", response_model=KnowledgeBase, status_code=201)
    async def create_knowledge_base(payload: ImportRequest, request: Request) -> KnowledgeBase:
        upload = request.app.state.uploads.get(payload.document_id)
        if not upload:
            raise HTTPException(404, "Uploaded document not found; upload it again")
        source = upload["source"]
        checksum = upload["checksum"]
        source_is_url = is_url_mime(upload["mime_type"])
        replacement_id = upload.get("replaces_knowledge_base_id")
        if (
            source_is_url
            and payload.chunking.strategy not in URL_CHUNKING_STRATEGIES
        ):
            raise HTTPException(
                422,
                "URL knowledge bases support only recursive or hierarchical chunking",
            )
        if upload["company_id"] != payload.company_id:
            raise HTTPException(422, "Uploaded document belongs to a different company")
        company = request.app.state.repository.get_company(payload.company_id)
        if not company:
            raise HTTPException(404, "Company not found")
        duplicate = request.app.state.repository.get_by_checksum(checksum, payload.company_id)
        if duplicate and duplicate.id != replacement_id:
            raise HTTPException(
                409,
                f'This document content is already imported as "{duplicate.name}"',
            )
        existing_name = request.app.state.repository.get(replacement_id) if replacement_id else None
        if (
            request.app.state.repository.name_exists(payload.name, payload.company_id)
            and (not existing_name or existing_name.name != payload.name)
        ):
            raise HTTPException(409, f'A knowledge base named "{payload.name}" already exists')
        try:
            if source_is_url:
                result = await request.app.state.importer.import_web_page(
                    payload,
                    source,
                    checksum,
                    upload["text"],
                    replacement_id,
                )
            else:
                result = await request.app.state.importer.import_document(
                    payload, source, checksum
                )
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

    @app.delete("/api/v1/chat-sessions", response_model=ChatSessionDeleteResult)
    def delete_chat_sessions(
        payload: ChatSessionDeleteRequest, request: Request
    ) -> ChatSessionDeleteResult:
        repository = request.app.state.repository
        if not repository.get_company(payload.company_id):
            raise HTTPException(404, "Company not found")
        if payload.session_ids is not None:
            if payload.date_from is not None or payload.date_to is not None:
                raise HTTPException(422, "Use session_ids or a date range, not both")
            return ChatSessionDeleteResult(
                deleted=repository.delete_chat_sessions(
                    payload.company_id,
                    payload.session_ids,
                )
            )
        if payload.date_from is None or payload.date_to is None:
            raise HTTPException(422, "Provide session_ids or both date_from and date_to")
        if payload.date_from > payload.date_to:
            raise HTTPException(422, "date_from must not be after date_to")
        if (payload.date_to - payload.date_from).days >= MAX_CHAT_SESSION_DATE_RANGE_DAYS:
            raise HTTPException(
                422,
                f"Chat session date range cannot exceed {MAX_CHAT_SESSION_DATE_RANGE_DAYS} days",
            )
        created_from = datetime.combine(payload.date_from, time.min, tzinfo=timezone.utc)
        created_to = datetime.combine(
            payload.date_to + timedelta(days=1), time.min, tzinfo=timezone.utc
        )
        return ChatSessionDeleteResult(
            deleted=repository.delete_chat_sessions_in_range(
                payload.company_id,
                created_from=created_from,
                created_to=created_to,
            )
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

    @app.get("/api/v1/companies", response_model=list[CompanyConfigurationView])
    def list_companies(request: Request) -> list[CompanyConfigurationView]:
        repository = request.app.state.repository
        return [company_configuration(company, repository) for company in repository.list_companies()]

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

    @app.put("/api/v1/companies/{company_id}/logo", response_model=CompanyConfigurationView)
    async def update_company_logo(
        company_id: str, request: Request, file: UploadFile = File(...)
    ) -> CompanyConfigurationView:
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
        company = request.app.state.repository.save_company(company)
        return company_configuration(company, request.app.state.repository)

    @app.delete("/api/v1/companies/{company_id}/logo", status_code=204)
    def delete_company_logo(company_id: str, request: Request) -> Response:
        company = request.app.state.repository.get_company(company_id)
        if not company:
            raise HTTPException(404, "Company not found")
        request.app.state.repository.save_company(
            company.model_copy(update={"logo_data": None, "logo_mime_type": None})
        )
        return Response(status_code=204)

    @app.post("/api/v1/companies", response_model=CompanyConfigurationView, status_code=201)
    async def create_company(
        payload: CompanyConfigurationInput, request: Request
    ) -> CompanyConfigurationView:
        selection = payload.llm
        if selection:
            validate_llm_selection(selection, config)
        if request.app.state.repository.company_name_exists(payload.name):
            raise HTTPException(409, f'A company named "{payload.name}" already exists')
        if selection and selection.provider != LLMProvider.OLLAMA:
            if not selection.api_key:
                raise HTTPException(422, "An API key is required for the selected provider")
            if not config.llm_credentials_key:
                raise HTTPException(503, "LLM credential encryption is not configured")
        try:
            company = await request.app.state.companies.index(
                Company(**payload.model_dump(exclude={"llm"}))
            )
            if selection:
                request.app.state.repository.save_company_llm_mapping(
                    CompanyLLMMapping(
                        company_id=company.id,
                        provider=selection.provider,
                        model=selection.model,
                        api_key=selection.api_key,
                    )
                )
            return company_configuration(company, request.app.state.repository)
        except DuplicateCompanyName as exc:
            raise HTTPException(409, str(exc)) from exc
        except CredentialEncryptionError as exc:
            raise HTTPException(503, str(exc)) from exc
        except CompanyIndexingError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.put("/api/v1/companies/{company_id}", response_model=CompanyConfigurationView)
    async def update_company(
        company_id: str, payload: CompanyConfigurationInput, request: Request
    ) -> CompanyConfigurationView:
        existing = request.app.state.repository.get_company(company_id)
        if not existing:
            raise HTTPException(404, "Company not found")
        selection = payload.llm
        stored_mapping = request.app.state.repository.get_company_llm_mapping(
            company_id, include_api_key=False
        )
        if "llm" not in payload.model_fields_set:
            if stored_mapping:
                selection = CompanyLLMInput(
                    provider=stored_mapping.provider,
                    model=stored_mapping.model,
                )
        if selection:
            validate_llm_selection(selection, config)
        if selection and selection.provider != LLMProvider.OLLAMA:
            has_preserved_key = bool(
                stored_mapping
                and stored_mapping.provider == selection.provider
                and stored_mapping.has_api_key
            )
            if not selection.api_key and not has_preserved_key:
                raise HTTPException(422, "An API key is required for the selected provider")
            if not config.llm_credentials_key:
                raise HTTPException(503, "LLM credential encryption is not configured")
        company = Company(
            **payload.model_dump(exclude={"llm"}),
            id=company_id,
            created_at=existing.created_at,
            logo_data=existing.logo_data,
            logo_mime_type=existing.logo_mime_type,
            updated_at=existing.updated_at,
        )
        if request.app.state.repository.company_name_exists(company.name, exclude_id=company_id):
            raise HTTPException(409, f'A company named "{company.name}" already exists')
        try:
            company = await request.app.state.companies.index(company)
            if selection:
                request.app.state.repository.save_company_llm_mapping(
                    CompanyLLMMapping(
                        company_id=company.id,
                        provider=selection.provider,
                        model=selection.model,
                        api_key=selection.api_key,
                    )
                )
            return company_configuration(company, request.app.state.repository)
        except DuplicateCompanyName as exc:
            raise HTTPException(409, str(exc)) from exc
        except CredentialEncryptionError as exc:
            raise HTTPException(503, str(exc)) from exc
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
