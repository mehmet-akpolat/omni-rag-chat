from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from packages.omni_rag_core.ai import OpenAIClient
from packages.omni_rag_core.domain import (
    ChatResponse,
    ChatSession,
    ChatSessionOwner,
    ChatSessionStatus,
    ChunkingConfig,
    Citation,
    Company,
    CompanyLLMMapping,
    KnowledgeBase,
)
from packages.omni_rag_core.settings import Settings
from tests.service_loader import load_service

chat_api = load_service("chatbot-be", "omni_chatbot_api")
create_app = chat_api.create_app


def config(tmp_path):
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'chat.db'}",
        llm_credentials_key=Fernet.generate_key().decode(),
    )


class Chat:
    payload = None
    sources = None
    company = None
    source_urls = None

    async def answer(self, payload, sources, company, source_urls, chat_client=None):
        self.payload = payload
        self.sources = sources
        self.company = company
        self.source_urls = source_urls
        self.chat_client = chat_client
        chat_client.usage.input_tokens = 31
        chat_client.usage.output_tokens = 12
        return ChatResponse(session_id="s1", answer="Answer", citations=[])

    async def stream(self, payload, sources, company, source_urls, chat_client=None):
        self.payload = payload
        self.sources = sources
        self.company = company
        self.source_urls = source_urls
        self.chat_client = chat_client

        async def tokens():
            yield "Hello [Source "
            yield "2]"
            chat_client.usage.input_tokens = 40
            chat_client.usage.output_tokens = 15

        return (
            "s2",
            [
                Citation(
                    source_number=1,
                    knowledge_base_id="kb-1",
                    knowledge_base="First guide",
                    page_number=1,
                    excerpt="First text",
                    score=0.9,
                ),
                Citation(
                    source_number=2,
                    knowledge_base_id="kb-2",
                    knowledge_base="Second guide",
                    page_number=2,
                    excerpt="Second text",
                    score=0.8,
                ),
            ],
            tokens(),
        )


def test_chat_routes_stream_list_and_escalation(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        client.app.state.chat = Chat()
        logo = b"\x89PNG\r\n\x1a\ncompany-logo"
        company = client.app.state.repository.save_company(
            Company(name="Acme", logo_data=logo, logo_mime_type="image/png")
        )
        client.app.state.repository.save_company_llm_mapping(
            CompanyLLMMapping(
                company_id=company.id,
                provider="openai",
                model="gpt-5.6-luna",
                api_key="company-openai-key",
            )
        )
        assert client.get("/health").json()["service"] == "chatbot-backend"
        assert client.get("/chat").status_code == 200
        assert client.get("/bot").status_code == 404
        assert client.get("/docs").status_code == 404
        ready = KnowledgeBase(
            company_id=company.id,
            name="Ready",
            source="r.pdf",
            mime_type="application/pdf",
            total_pages=1,
            selected_pages=1,
            chunking=ChunkingConfig(),
        )
        indexing = ready.model_copy(update={"id": "other", "name": "Busy", "status": "indexing"})
        disabled = ready.model_copy(update={"id": "disabled", "name": "Off", "status": "disabled"})
        client.app.state.repository.save(ready)
        client.app.state.repository.save(indexing)
        client.app.state.repository.save(disabled)
        assert client.get("/api/v1/companies").json()[0]["name"] == "Acme"
        assert "{{bot_alias}}" in client.get("/api/v1/companies").json()[0]["bot_greet_message"]
        logo_response = client.get(f"/api/v1/companies/{company.id}/logo")
        assert logo_response.content == logo
        assert logo_response.headers["content-type"] == "image/png"
        assert client.get("/api/v1/companies/missing/logo").status_code == 404
        started = client.post(
            "/api/v1/chat-sessions",
            json={"company_id": company.id},
        )
        assert started.status_code == 201
        context = started.json()
        session_id = context["session"]["id"]
        assert context["session"]["status"] == "active"
        assert context["session"]["message_count"] == 1
        assert context["company"]["id"] == company.id
        assert context["expires_at"] > context["session"]["created_at"]
        assert context["messages"][0]["owner"] == "bot"
        assert client.get(f"/api/v1/chat-sessions/{session_id}").json()["company"]["name"] == "Acme"
        assert client.get(f"/api/v1/chat-sessions/{session_id}/company-logo").content == logo
        listed = client.get("/api/v1/knowledge-bases", params={"session_id": session_id}).json()
        assert [item["name"] for item in listed] == ["READY"]
        assert (
            client.post(
                "/api/v1/chat-sessions",
                json={"company_id": "missing"},
            ).status_code
            == 404
        )
        assert client.get("/api/v1/chat-sessions/missing").status_code == 404
        assert (
            client.post(
                "/api/v1/chat-sessions",
                json={"company_id": company.id, "previous_session_id": "missing"},
            ).status_code
            == 404
        )
        assert (
            client.get("/api/v1/knowledge-bases", params={"session_id": "missing"}).status_code
            == 404
        )
        assert (
            client.patch("/api/v1/chat-sessions/missing", json={"status": "closed"}).status_code
            == 404
        )
        assert (
            client.patch(
                f"/api/v1/chat-sessions/{session_id}", json={"status": "active"}
            ).status_code
            == 422
        )
        answer = client.post(
            "/api/v1/chat",
            json={
                "message": "Hi",
                "session_id": session_id,
            },
        ).json()
        assert answer["answer"] == "Answer"
        assert answer["session_id"] == session_id
        assert isinstance(client.app.state.chat.chat_client, OpenAIClient)
        assert client.app.state.chat.payload.knowledge_base_ids == [ready.id]
        assert client.app.state.chat.sources == {ready.id: "READY"}
        assert client.app.state.chat.company == company.id
        assert client.app.state.chat.source_urls == {}
        client.post(
            "/api/v1/chat",
            json={
                "message": "Hi",
                "session_id": session_id,
                "knowledge_base_ids": [disabled.id],
            },
        )
        assert client.app.state.chat.payload.knowledge_base_ids == []
        streamed = client.post(
            "/api/v1/chat/stream",
            json={
                "message": "Hi",
                "session_id": session_id,
            },
        )
        assert streamed.status_code == 200
        assert (
            "event: meta" in streamed.text
            and "Hello" in streamed.text
            and "[Source 1]" in streamed.text
            and "[Source 2]" not in streamed.text
            and "event: citations" in streamed.text
            and "event: done" in streamed.text
        )
        messages = client.app.state.repository.list_chat_messages(session_id)
        assert (
            messages[0].content
            == "Hello! I'm AIBot, the AI assistant for Acme. How can I help you today?"
        )
        assert [message.owner for message in messages] == [
            ChatSessionOwner.BOT,
            ChatSessionOwner.USER,
            ChatSessionOwner.BOT,
            ChatSessionOwner.USER,
            ChatSessionOwner.BOT,
            ChatSessionOwner.USER,
            ChatSessionOwner.BOT,
        ]
        assert (messages[2].input_tokens, messages[2].output_tokens) == (31, 12)
        assert (messages[-1].input_tokens, messages[-1].output_tokens) == (40, 15)
        assert messages[-1].content == "Hello [Source 1]"
        escalation = client.post("/api/v1/escalations", json={"session_id": session_id})
        assert escalation.status_code == 202 and escalation.json()["ticket_id"].startswith("HUM-")
        assert len(client.app.state.escalations) == 1
        assert client.post("/api/v1/escalations", json={"session_id": "  "}).status_code == 422
        assert client.post("/api/v1/escalations", json={"session_id": "missing"}).status_code == 404
        assert client.post("/api/v1/chat", json={"message": "Hi"}).status_code == 422
        assert (
            client.post("/api/v1/chat", json={"message": "Hi", "session_id": "missing"}).status_code
            == 404
        )

        unload_session = client.post(
            "/api/v1/chat-sessions", json={"company_id": company.id}
        ).json()["session"]["id"]
        assert client.post(f"/api/v1/chat-sessions/{unload_session}/close").status_code == 204
        assert (
            client.app.state.repository.get_chat_session(unload_session).status
            == ChatSessionStatus.CLOSED
        )
        assert client.post(f"/api/v1/chat-sessions/{unload_session}/close").status_code == 204

        other = client.app.state.repository.save_company(Company(name="Other"))
        other_session = client.post(
            "/api/v1/chat-sessions",
            json={"company_id": other.id},
        ).json()["session"]["id"]
        assert (
            client.post(
                "/api/v1/chat-sessions",
                json={"company_id": company.id, "previous_session_id": other_session},
            ).status_code
            == 409
        )

        replacement = client.post(
            "/api/v1/chat-sessions",
            json={"company_id": company.id, "previous_session_id": session_id},
        )
        assert replacement.status_code == 201
        replacement_id = replacement.json()["session"]["id"]
        assert (
            client.app.state.repository.get_chat_session(session_id).status
            == ChatSessionStatus.CLOSED
        )
        assert (
            client.post(
                "/api/v1/chat",
                json={"message": "Too late", "session_id": session_id},
            ).status_code
            == 409
        )

        timed_out = client.patch(
            f"/api/v1/chat-sessions/{replacement_id}", json={"status": "timeout"}
        )
        assert timed_out.status_code == 200
        assert timed_out.json()["status"] == ChatSessionStatus.TIMEOUT
        assert timed_out.json()["ended_at"] is not None
        ended_at = timed_out.json()["ended_at"]
        closed_again = client.patch(
            f"/api/v1/chat-sessions/{replacement_id}", json={"status": "closed"}
        )
        assert closed_again.json()["status"] == ChatSessionStatus.TIMEOUT
        assert closed_again.json()["ended_at"] == ended_at
        assert (
            client.get(f"/api/v1/chat-sessions/{replacement_id}").json()["session"]["status"]
            == ChatSessionStatus.TIMEOUT
        )

        expired, _ = client.app.state.repository.create_chat_session(
            ChatSession(
                id="expired-session",
                company_id=company.id,
                created_at=datetime.now(timezone.utc) - timedelta(minutes=11),
            ),
            "Old greeting",
        )
        assert expired.status == ChatSessionStatus.ACTIVE
        expired_context = client.get("/api/v1/chat-sessions/expired-session")
        assert expired_context.status_code == 200
        assert expired_context.json()["session"]["status"] == ChatSessionStatus.TIMEOUT
        assert (
            client.get(
                "/api/v1/knowledge-bases", params={"session_id": "expired-session"}
            ).status_code
            == 409
        )
        assert (
            client.post("/api/v1/escalations", json={"session_id": "expired-session"}).status_code
            == 409
        )


def test_chat_rejects_company_without_language_model_before_storing_question(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        client.app.state.chat = Chat()
        company = client.app.state.repository.save_company(Company(name="Unconfigured"))
        started = client.post("/api/v1/chat-sessions", json={"company_id": company.id})
        session_id = started.json()["session"]["id"]
        before = len(client.app.state.repository.list_chat_messages(session_id))
        response = client.post(
            "/api/v1/chat", json={"message": "Hello", "session_id": session_id}
        )
        assert response.status_code == 503
        assert "does not have a language model configured" in response.json()["detail"]
        assert len(client.app.state.repository.list_chat_messages(session_id)) == before
