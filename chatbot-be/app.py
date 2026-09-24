import json
import asyncio
from contextlib import asynccontextmanager
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from packages.omni_rag_core.ai import LLMClientFactory, OllamaClient
from packages.omni_rag_core.domain import (
    ChatRequest,
    ChatResponse,
    ChatSession,
    ChatSessionMessage,
    ChatSessionOwner,
    ChatSessionStatus,
    Company,
    CompanyView,
    KnowledgeBase,
)
from packages.omni_rag_core.repository import SqlKnowledgeRepository
from packages.omni_rag_core.services import (
    ChatService,
    CitationRetagger,
)
from packages.omni_rag_core.settings import Settings, get_settings
from packages.omni_rag_core.vector_store import QdrantVectorStore, ResilientVectorStore


class EscalationRequest(BaseModel):
    session_id: str
    reason: str = Field(default="User requested human support", max_length=500)


class ChatSessionStart(BaseModel):
    company_id: str
    previous_session_id: str | None = Field(default=None, max_length=64)


class ChatSessionEnd(BaseModel):
    status: ChatSessionStatus


class ChatSessionContext(BaseModel):
    session: ChatSession
    company: CompanyView
    messages: list[ChatSessionMessage]
    expires_at: datetime


def render_greeting(company: Company) -> str:
    return company.bot_greet_message.replace("{{bot_alias}}", company.bot_alias).replace(
        "{{company_name}}", company.name
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        vectors = ResilientVectorStore(QdrantVectorStore(config))
        app.state.repository = SqlKnowledgeRepository(
            config.database_url, config.llm_credentials_key
        )
        app.state.chat = ChatService(vectors, OllamaClient(config))
        app.state.llm_factory = LLMClientFactory(config)
        app.state.escalations = []

        async def expire_sessions() -> None:
            while True:
                cutoff = datetime.now(timezone.utc) - timedelta(minutes=config.chat_session_minutes)
                app.state.repository.timeout_expired_chat_sessions(cutoff)
                await asyncio.sleep(config.chat_session_sweep_seconds)

        expiry_task = asyncio.create_task(expire_sessions())
        try:
            yield
        finally:
            expiry_task.cancel()
            with suppress(asyncio.CancelledError):
                await expiry_task
            app.state.repository.engine.dispose()

    app = FastAPI(
        title="Omni RAG Chat API",
        version="1.0.0",
        description="Grounded question answering with citations and streaming.",
        docs_url="/chat",
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
        return {"status": "ok", "service": "chatbot-backend"}

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

    @app.get("/api/v1/knowledge-bases", response_model=list[KnowledgeBase])
    def list_knowledge_bases(request: Request, session_id: str = Query(...)) -> list[KnowledgeBase]:
        _, company = resolve_session(session_id, request)
        return [
            item
            for item in request.app.state.repository.list(company_id=company.id)
            if item.status == "enabled"
        ]

    def client_ip(request: Request) -> str:
        return request.client.host if request.client else ""

    def expires_at(chat_session: ChatSession) -> datetime:
        created_at = chat_session.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return created_at + timedelta(minutes=config.chat_session_minutes)

    def resolve_session(
        session_id: str, request: Request, *, require_active: bool = True
    ) -> tuple[ChatSession, Company]:
        repository = request.app.state.repository
        chat_session = repository.get_chat_session(session_id)
        if not chat_session:
            raise HTTPException(404, "Chat session not found")
        if chat_session.status == ChatSessionStatus.ACTIVE and datetime.now(
            timezone.utc
        ) >= expires_at(chat_session):
            chat_session = repository.close_chat_session(session_id, ChatSessionStatus.TIMEOUT)
            assert chat_session is not None
        if require_active and chat_session.status != ChatSessionStatus.ACTIVE:
            raise HTTPException(409, "Chat session is not active")
        company = repository.get_company(chat_session.company_id)
        if not company:
            raise HTTPException(404, "Session company not found")
        return chat_session, company

    def session_context(
        chat_session: ChatSession, company: Company, request: Request
    ) -> ChatSessionContext:
        return ChatSessionContext(
            session=chat_session,
            company=CompanyView.model_validate(company.model_dump()),
            messages=request.app.state.repository.list_chat_messages(chat_session.id),
            expires_at=expires_at(chat_session),
        )

    @app.post("/api/v1/chat-sessions", response_model=ChatSessionContext, status_code=201)
    def start_chat_session(payload: ChatSessionStart, request: Request) -> ChatSessionContext:
        company = request.app.state.repository.get_company(payload.company_id)
        if not company:
            raise HTTPException(404, "Company not found")
        if payload.previous_session_id:
            previous, _ = resolve_session(
                payload.previous_session_id, request, require_active=False
            )
            if previous.company_id != company.id:
                raise HTTPException(409, "Previous session belongs to another company")
            if previous.status == ChatSessionStatus.ACTIVE:
                request.app.state.repository.close_chat_session(
                    previous.id, ChatSessionStatus.CLOSED
                )
        chat_session, _ = request.app.state.repository.create_chat_session(
            ChatSession(company_id=company.id, ip_address=client_ip(request)),
            render_greeting(company),
        )
        return session_context(chat_session, company, request)

    @app.get("/api/v1/chat-sessions/{session_id}", response_model=ChatSessionContext)
    def get_chat_session(session_id: str, request: Request) -> ChatSessionContext:
        chat_session, company = resolve_session(session_id, request, require_active=False)
        return session_context(chat_session, company, request)

    @app.get("/api/v1/chat-sessions/{session_id}/company-logo")
    def get_session_company_logo(session_id: str, request: Request) -> Response:
        _, company = resolve_session(session_id, request, require_active=False)
        if not company.logo_data:
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

    @app.post("/api/v1/chat-sessions/{session_id}/close", status_code=204)
    def close_chat_session_on_unload(session_id: str, request: Request) -> Response:
        current, _ = resolve_session(session_id, request, require_active=False)
        if current.status == ChatSessionStatus.ACTIVE:
            request.app.state.repository.close_chat_session(session_id, ChatSessionStatus.CLOSED)
        return Response(status_code=204)

    @app.patch("/api/v1/chat-sessions/{session_id}", response_model=ChatSession)
    def end_chat_session(session_id: str, payload: ChatSessionEnd, request: Request) -> ChatSession:
        if payload.status == ChatSessionStatus.ACTIVE:
            raise HTTPException(422, "Session can only be closed or timed out")
        current, _ = resolve_session(session_id, request, require_active=False)
        if current.status != ChatSessionStatus.ACTIVE:
            return current
        chat_session = request.app.state.repository.close_chat_session(session_id, payload.status)
        assert chat_session is not None
        return chat_session

    def enabled_sources(
        payload: ChatRequest, request: Request
    ) -> tuple[ChatRequest, dict[str, str], str, dict[str, str]]:
        _, company = resolve_session(payload.session_id, request)
        enabled = {
            item.id: item
            for item in request.app.state.repository.list(company_id=company.id)
            if item.status == "enabled"
        }
        enabled_ids = set(enabled)
        requested_ids = (
            set(payload.knowledge_base_ids) if payload.knowledge_base_ids else enabled_ids
        )
        selected_ids = sorted(requested_ids & enabled_ids)
        return (
            payload.model_copy(update={"knowledge_base_ids": selected_ids}),
            {source_id: enabled[source_id].name for source_id in selected_ids},
            company.id,
            {
                source_id: enabled[source_id].source
                for source_id in selected_ids
                if enabled[source_id].mime_type
                in {"text/html", "application/xhtml+xml"}
            },
        )

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        filtered, source_names_by_id, company_id, source_urls_by_id = enabled_sources(
            payload, request
        )
        mapping = request.app.state.repository.get_company_llm_mapping(company_id)
        if not mapping:
            raise HTTPException(503, "The company does not have a language model configured")
        if mapping.model not in config.llm_models[mapping.provider.value]:
            raise HTTPException(503, "The company's language model is not configured")
        request.app.state.repository.add_chat_message(
            ChatSessionMessage(
                session_id=filtered.session_id,
                owner=ChatSessionOwner.USER,
                content=filtered.message,
            )
        )
        chat_client = request.app.state.llm_factory.create(mapping, mapping.api_key)
        result = await request.app.state.chat.answer(
            filtered,
            source_names_by_id,
            company_id,
            source_urls_by_id,
            chat_client,
        )
        result = result.model_copy(update={"session_id": filtered.session_id})
        request.app.state.repository.add_chat_message(
            ChatSessionMessage(
                session_id=result.session_id,
                owner=ChatSessionOwner.BOT,
                content=result.answer,
                input_tokens=chat_client.usage.input_tokens,
                output_tokens=chat_client.usage.output_tokens,
            )
        )
        return result

    @app.post("/api/v1/chat/stream")
    async def stream_chat(payload: ChatRequest, request: Request) -> StreamingResponse:
        filtered, source_names_by_id, company_id, source_urls_by_id = enabled_sources(
            payload, request
        )
        mapping = request.app.state.repository.get_company_llm_mapping(company_id)
        if not mapping:
            raise HTTPException(503, "The company does not have a language model configured")
        if mapping.model not in config.llm_models[mapping.provider.value]:
            raise HTTPException(503, "The company's language model is not configured")
        request.app.state.repository.add_chat_message(
            ChatSessionMessage(
                session_id=filtered.session_id,
                owner=ChatSessionOwner.USER,
                content=filtered.message,
            )
        )
        chat_client = request.app.state.llm_factory.create(mapping, mapping.api_key)
        _, citations, tokens = await request.app.state.chat.stream(
            filtered,
            source_names_by_id,
            company_id,
            source_urls_by_id,
            chat_client,
        )
        session_id = filtered.session_id

        async def events():
            answer_parts: list[str] = []
            marker_buffer = ""
            retagger = CitationRetagger(citations)
            yield f"event: meta\ndata: {json.dumps({'session_id': session_id})}\n\n"
            try:
                async for token in tokens:
                    marker_buffer += token
                    while marker_buffer:
                        marker_start = marker_buffer.find("[")
                        if marker_start < 0:
                            emitted = marker_buffer
                            marker_buffer = ""
                        elif marker_start > 0:
                            emitted = marker_buffer[:marker_start]
                            marker_buffer = marker_buffer[marker_start:]
                        else:
                            marker_end = marker_buffer.find("]")
                            if marker_end < 0:
                                break
                            emitted = retagger.transform(marker_buffer[: marker_end + 1])
                            marker_buffer = marker_buffer[marker_end + 1 :]
                        answer_parts.append(emitted)
                        yield f"event: token\ndata: {json.dumps({'token': emitted})}\n\n"
                if marker_buffer:
                    emitted = retagger.transform(marker_buffer)
                    marker_buffer = ""
                    answer_parts.append(emitted)
                    yield f"event: token\ndata: {json.dumps({'token': emitted})}\n\n"
                cited = retagger.citations()
                yield f"event: citations\ndata: {json.dumps([item.model_dump() for item in cited])}\n\n"
                yield "event: done\ndata: {}\n\n"
            finally:
                if marker_buffer:
                    answer_parts.append(retagger.transform(marker_buffer))
                stored_answer = "".join(answer_parts).strip()
                if stored_answer:
                    request.app.state.repository.add_chat_message(
                        ChatSessionMessage(
                            session_id=session_id,
                            owner=ChatSessionOwner.BOT,
                            content=stored_answer,
                            input_tokens=chat_client.usage.input_tokens,
                            output_tokens=chat_client.usage.output_tokens,
                        )
                    )

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/api/v1/escalations", status_code=202)
    def escalate(payload: EscalationRequest, request: Request) -> dict:
        if not payload.session_id.strip():
            raise HTTPException(422, "A session ID is required")
        resolve_session(payload.session_id, request)
        ticket = {
            "ticket_id": f"HUM-{uuid4().hex[:8].upper()}",
            "session_id": payload.session_id,
            "reason": payload.reason,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        request.app.state.escalations.append(ticket)
        return ticket

    return app


app = create_app()
