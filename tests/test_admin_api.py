import hashlib

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from packages.omni_rag_core.domain import (
    ChatSession,
    ChatSessionMessage,
    ChatSessionOwner,
    ChunkingConfig,
    Company,
    KnowledgeBase,
)
from packages.omni_rag_core.documents import InvalidDocument
from packages.omni_rag_core.services import CompanyIndexingError
from packages.omni_rag_core.settings import Settings
from packages.omni_rag_core.web_documents import WebPage
from tests.service_loader import load_service

admin_api = load_service("admin-be", "omni_admin_api")
MAX_UPLOAD_BYTES = admin_api.MAX_UPLOAD_BYTES
MAX_LOGO_BYTES = admin_api.MAX_LOGO_BYTES
create_app = admin_api.create_app


def config(tmp_path):
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        upload_dir=tmp_path / "uploads",
        llm_credentials_key=Fernet.generate_key().decode(),
    )


def pdf_bytes(tmp_path):
    path = tmp_path / "a.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as output:
        writer.write(output)
    return path.read_bytes()


def test_company_logo_validation_supports_only_safe_png_svg_and_jpeg():
    assert admin_api.valid_logo(b"\x89PNG\r\n\x1a\nbody", "image/png")
    assert admin_api.valid_logo(b"\xff\xd8\xffbody", "image/jpeg")
    assert admin_api.valid_logo(b"<svg><circle/></svg>", "image/svg+xml")
    assert not admin_api.valid_logo(b"GIF89a", "image/gif")
    assert not admin_api.valid_logo(b"<svg onload='run()'/>", "image/svg+xml")
    assert not admin_api.valid_logo(b"<svg><a href='javascript:run()'/></svg>", "image/svg+xml")
    assert not admin_api.valid_logo(b"not xml", "image/svg+xml")
    assert admin_api.masked_api_key("company-openai-key") == "co••••••••ey"
    assert admin_api.masked_api_key(None) is None
    assert admin_api.masked_api_key("abc") == "•••"


class Importer:
    async def import_document(self, payload, source, document_checksum):
        if payload.name == "NO PAGES":
            raise ValueError("At least one page must be selected")
        return KnowledgeBase(
            company_id=payload.company_id,
            name=payload.name,
            source=source,
            mime_type="application/pdf",
            total_pages=1,
            selected_pages=1,
            chunking=payload.chunking,
            chunk_count=2,
            document_checksum=document_checksum,
        )

    async def import_web_page(
        self,
        payload,
        source,
        document_checksum,
        text,
        replaces_knowledge_base_id=None,
    ):
        assert text
        return KnowledgeBase(
            company_id=payload.company_id,
            name=payload.name,
            source=source,
            mime_type="text/html",
            chunking=payload.chunking,
            chunk_count=3,
            document_checksum=document_checksum,
        )


class WebDocuments:
    def __init__(self, page=None, error=None):
        self.page = page
        self.error = error

    async def fetch(self, url):
        if self.error:
            raise self.error
        return self.page


class Vectors:
    def __init__(self, fail=False):
        self.fail = fail
        self.deleted = []

    def delete_knowledge_base(self, company_id, knowledge_base_id):
        if self.fail:
            raise RuntimeError("Qdrant offline")
        self.deleted.append((company_id, knowledge_base_id))

    def delete_company(self, company_id):
        if self.fail:
            raise RuntimeError("Qdrant offline")
        self.deleted.append((company_id, "*"))


class Companies:
    def __init__(self, repository):
        self.repository = repository

    async def index(self, company):
        return self.repository.save_company(company)


class FailingCompanies:
    async def index(self, company):
        raise CompanyIndexingError("Company metadata could not be synchronized")


def test_new_company_has_no_default_language_model_mapping(tmp_path):
    custom_config = config(tmp_path)
    custom_config.openai_models = "environment-model"
    with TestClient(create_app(custom_config)) as client:
        client.app.state.companies = Companies(client.app.state.repository)
        created = client.post("/api/v1/companies", json={"name": "Unconfigured"})
        assert created.status_code == 201
        assert created.json()["llm"] is None
        assert (
            client.app.state.repository.get_company_llm_mapping(created.json()["id"])
            is None
        )
        options = client.get("/api/v1/llm-options").json()
        assert next(item for item in options if item["provider"] == "openai")["models"] == [
            "environment-model"
        ]


def test_admin_health_strategies_upload_and_import(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        company = client.app.state.repository.save_company(Company(name="Acme"))
        assert client.get("/health").json()["status"] == "ok"
        assert len(client.get("/api/v1/chunking-strategies").json()) == 4
        assert (
            client.post(
                "/api/v1/documents/preview",
                files={
                    "file": ("x.txt", b"x", "text/plain"),
                    "company_id": (None, company.id),
                },
            ).status_code
            == 415
        )
        assert (
            client.post(
                "/api/v1/documents/preview",
                files={
                    "file": ("x.pdf", b"bad", "application/pdf"),
                    "company_id": (None, company.id),
                },
            ).status_code
            == 422
        )
        response = client.post(
            "/api/v1/documents/preview",
            files={
                "file": ("guide.pdf", pdf_bytes(tmp_path), "application/pdf"),
                "company_id": (None, company.id),
            },
        )
        assert response.status_code == 200
        doc = response.json()
        assert doc["total_pages"] == 1
        client.app.state.importer = Importer()
        payload = {
            "document_id": doc["document_id"],
            "company_id": company.id,
            "name": "Guide",
            "chunking": {"strategy": "recursive"},
        }
        created = client.post("/api/v1/knowledge-bases", json=payload)
        assert created.status_code == 201 and created.json()["chunk_count"] == 2
        assert created.json()["selected_pages"] == 1
        assert created.json()["excluded_pages"] == []
        client.app.state.repository.save(
            KnowledgeBase(
                company_id=company.id,
                name="Guide",
                source="guide.pdf",
                mime_type="application/pdf",
                total_pages=1,
                selected_pages=1,
                chunking=ChunkingConfig(),
            )
        )
        client.app.state.uploads[doc["document_id"]] = {
            "source": "guide-v2.pdf",
            "mime_type": "application/pdf",
            "checksum": "f" * 64,
            "company_id": company.id,
        }
        duplicate = client.post(
            "/api/v1/knowledge-bases",
            json={
                "document_id": doc["document_id"],
                "company_id": company.id,
                "name": "  guide  ",
            },
        )
        assert duplicate.status_code == 409
        assert "already exists" in duplicate.json()["detail"]
        payload["name"] = "No pages"
        assert client.post("/api/v1/knowledge-bases", json=payload).status_code == 422


def test_admin_previews_and_imports_url_knowledge_base(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        company = client.app.state.repository.save_company(Company(name="Acme"))
        page = WebPage(
            url="https://example.com/guide",
            title="Product Guide",
            text="# Welcome\n\nThis guide explains the product.",
        )
        client.app.state.web_documents = WebDocuments(page=page)
        preview = client.post(
            "/api/v1/urls/preview",
            json={"company_id": company.id, "url": "https://example.com/guide"},
        )
        assert preview.status_code == 200
        body = preview.json()
        assert body["source"] == page.url
        assert body["title"] == page.title
        assert body["pages"][0]["excerpt"].startswith("# Welcome")

        client.app.state.importer = Importer()
        unsupported = client.post(
            "/api/v1/knowledge-bases",
            json={
                "document_id": body["document_id"],
                "company_id": company.id,
                "name": "Website guide",
                "chunking": {"strategy": "fixed"},
            },
        )
        assert unsupported.status_code == 422
        assert "recursive or hierarchical" in unsupported.json()["detail"]

        imported = client.post(
            "/api/v1/knowledge-bases",
            json={
                "document_id": body["document_id"],
                "company_id": company.id,
                "name": "Website guide",
                "chunking": {"strategy": "hierarchical"},
            },
        )
        assert imported.status_code == 201
        assert imported.json()["source"] == page.url
        assert imported.json()["total_pages"] is None
        assert imported.json()["selected_pages"] is None
        assert imported.json()["excluded_pages"] is None
        assert imported.json()["chunk_count"] == 3

        assert client.post(
            "/api/v1/urls/preview",
            json={"company_id": "missing", "url": page.url},
        ).status_code == 404


def test_admin_url_preview_handles_fetch_errors_duplicates_and_refresh(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        company = client.app.state.repository.save_company(Company(name="Acme"))
        client.app.state.web_documents = WebDocuments(error=InvalidDocument("Blocked URL"))
        rejected = client.post(
            "/api/v1/urls/preview",
            json={"company_id": company.id, "url": "http://127.0.0.1/private"},
        )
        assert rejected.status_code == 422 and rejected.json()["detail"] == "Blocked URL"

        existing = KnowledgeBase(
            company_id=company.id,
            name="Web guide",
            source="https://example.com/guide",
            mime_type="text/html",
            chunking=ChunkingConfig(),
            document_checksum="a" * 64,
        )
        client.app.state.repository.save(existing)
        changed = WebPage(
            url=existing.source,
            title="Updated guide",
            text="Changed public content with a new checksum.",
        )
        client.app.state.web_documents = WebDocuments(page=changed)
        refreshed = client.post(
            "/api/v1/urls/preview",
            json={"company_id": company.id, "url": existing.source},
        )
        assert refreshed.status_code == 200
        assert refreshed.json()["replaces_knowledge_base_id"] == existing.id

        duplicate_page = WebPage(
            url="https://example.com/copy",
            title="Copy",
            text="same content",
        )
        checksum = hashlib.sha256(duplicate_page.text.encode()).hexdigest()
        duplicate = existing.model_copy(
            update={
                "id": "duplicate-content",
                "name": "Copy",
                "source": duplicate_page.url,
                "document_checksum": checksum,
            }
        )
        client.app.state.repository.save(duplicate)
        client.app.state.web_documents = WebDocuments(page=duplicate_page)
        response = client.post(
            "/api/v1/urls/preview",
            json={"company_id": company.id, "url": duplicate_page.url},
        )
        assert response.status_code == 409


def test_admin_rejects_duplicate_document_content(tmp_path):
    content = pdf_bytes(tmp_path)
    with TestClient(create_app(config(tmp_path))) as client:
        company = client.app.state.repository.save_company(Company(name="Acme"))
        uploaded = client.post(
            "/api/v1/documents/preview",
            files={
                "file": ("original.pdf", content, "application/pdf"),
                "company_id": (None, company.id),
            },
        ).json()
        checksum = client.app.state.uploads[uploaded["document_id"]]["checksum"]
        client.app.state.repository.save(
            KnowledgeBase(
                company_id=company.id,
                name="Original",
                source="original.pdf",
                mime_type="application/pdf",
                total_pages=1,
                selected_pages=1,
                chunking=ChunkingConfig(),
                document_checksum=checksum,
            )
        )

        renamed = client.post(
            "/api/v1/documents/preview",
            files={
                "file": ("renamed.pdf", content, "application/pdf"),
                "company_id": (None, company.id),
            },
        )
        assert renamed.status_code == 409
        assert "ORIGINAL" in renamed.json()["detail"]

        duplicate_import = client.post(
            "/api/v1/knowledge-bases",
            json={
                "document_id": uploaded["document_id"],
                "company_id": company.id,
                "name": "Different name",
            },
        )
        assert duplicate_import.status_code == 409
        assert "document content" in duplicate_import.json()["detail"]


def test_admin_missing_import_list_and_get(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        company = client.app.state.repository.save_company(Company(name="Acme"))
        assert (
            client.post(
                "/api/v1/knowledge-bases",
                json={
                    "document_id": "gone",
                    "company_id": company.id,
                    "name": "Guide",
                },
            ).status_code
            == 404
        )
        kb = KnowledgeBase(
            company_id=company.id,
            name="Saved",
            source="x.pdf",
            mime_type="application/pdf",
            total_pages=1,
            selected_pages=1,
            chunking=ChunkingConfig(),
        )
        client.app.state.repository.save(kb)
        listing = client.get(
            f"/api/v1/knowledge-bases?page=1&page_size=10&company_id={company.id}"
        ).json()
        assert listing["items"][0]["name"] == "SAVED"
        assert listing["page"] == 1
        assert listing["page_size"] == 10
        assert listing["total"] == 1
        assert listing["total_pages"] == 1
        assert client.get("/api/v1/knowledge-bases?search=pdf").json()["total"] == 1
        assert client.get("/api/v1/knowledge-bases?search=missing").json()["items"] == []
        assert client.get("/api/v1/knowledge-bases?page=0").status_code == 422
        assert client.get(f"/api/v1/knowledge-bases/{kb.id}").status_code == 200
        assert client.get("/api/v1/knowledge-bases/missing").status_code == 404
        assert (
            client.patch("/api/v1/knowledge-bases/missing", json={"enabled": False}).status_code
            == 404
        )
        assert client.delete("/api/v1/knowledge-bases/missing").status_code == 404

        disabled = client.patch(f"/api/v1/knowledge-bases/{kb.id}", json={"enabled": False})
        assert disabled.status_code == 200
        assert disabled.json()["status"] == "disabled"
        enabled = client.patch(f"/api/v1/knowledge-bases/{kb.id}", json={"enabled": True})
        assert enabled.json()["status"] == "enabled"

        vectors = Vectors()
        client.app.state.vectors = vectors
        assert client.delete(f"/api/v1/knowledge-bases/{kb.id}").status_code == 204
        assert vectors.deleted == [(company.id, kb.id)]
        assert client.get(f"/api/v1/knowledge-bases/{kb.id}").status_code == 404


def test_admin_retains_metadata_when_vector_deletion_fails(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        company = client.app.state.repository.save_company(Company(name="Acme"))
        kb = KnowledgeBase(
            company_id=company.id,
            name="Keep me",
            source="x.pdf",
            mime_type="application/pdf",
            total_pages=1,
            selected_pages=1,
            chunking=ChunkingConfig(),
        )
        client.app.state.repository.save(kb)
        client.app.state.vectors = Vectors(fail=True)
        response = client.delete(f"/api/v1/knowledge-bases/{kb.id}")
        assert response.status_code == 503
        assert client.app.state.repository.get(kb.id) is not None


def test_admin_rejects_oversized_file(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        company = client.app.state.repository.save_company(Company(name="Acme"))
        response = client.post(
            "/api/v1/documents/preview",
            files={
                "file": ("huge.pdf", b"%PDF" + b"x" * MAX_UPLOAD_BYTES, "application/pdf"),
                "company_id": (None, company.id),
            },
        )
        assert response.status_code == 413


def test_admin_company_crud_and_cascade(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        client.app.state.companies = Companies(client.app.state.repository)
        client.app.state.vectors = Vectors()
        missing_key = client.post(
            "/api/v1/companies",
            json={
                "name": "Missing key",
                "llm": {"provider": "openai", "model": "gpt-5.6-luna"},
            },
        )
        assert missing_key.status_code == 422
        assert client.app.state.repository.company_name_exists("Missing key") is False
        created = client.post(
            "/api/v1/companies",
            json={
                "name": "Acme",
                "about": "We make rockets.",
                "email": "hello@acme.example",
                "bot_alias": "Orbit",
                "bot_avatar": "brain",
                "bot_greet_message": "Welcome to {{company_name}}. I'm {{bot_alias}}.",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-5.6-terra",
                    "api_key": "company-openai-key",
                },
            },
        )
        assert created.status_code == 201
        company = created.json()
        assert company["name"] == "Acme"
        assert company["bot_greet_message"] == "Welcome to {{company_name}}. I'm {{bot_alias}}."
        assert company["llm"] == {
            "provider": "openai",
            "model": "gpt-5.6-terra",
            "has_api_key": True,
            "api_key_masked": "co••••••••ey",
        }
        assert "api_key" not in company["llm"]
        options = client.get("/api/v1/llm-options").json()
        assert next(item for item in options if item["provider"] == "ollama")["models"] == [
            "gemma4:12b"
        ]
        invalid_template = client.post(
            "/api/v1/companies",
            json={"name": "Invalid template", "bot_greet_message": "Hello {{user_name}}"},
        )
        assert invalid_template.status_code == 422
        assert "Unsupported greeting placeholder" in invalid_template.text
        assert client.post("/api/v1/companies", json={"name": " acme "}).status_code == 409
        assert [item["name"] for item in client.get("/api/v1/companies").json()] == ["Acme"]

        updated = client.put(
            f"/api/v1/companies/{company['id']}",
            json={
                "name": "Acme Corp",
                "bot_alias": "Nova",
                "bot_greet_message": company["bot_greet_message"],
                "llm": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5",
                    "api_key": "company-anthropic-key",
                },
            },
        )
        assert updated.status_code == 200
        assert updated.json()["name"] == "Acme Corp"
        assert updated.json()["bot_alias"] == "Nova"
        assert updated.json()["bot_greet_message"] == company["bot_greet_message"]
        assert updated.json()["llm"]["model"] == "claude-sonnet-5"
        assert updated.json()["llm"]["has_api_key"] is True

        preserved_key = client.put(
            f"/api/v1/companies/{company['id']}",
            json={
                "name": "Acme Corp",
                "bot_alias": "Nova",
                "bot_greet_message": company["bot_greet_message"],
                "llm": {"provider": "anthropic", "model": "claude-opus-5"},
            },
        )
        assert preserved_key.status_code == 200
        assert preserved_key.json()["llm"] == {
            "provider": "anthropic",
            "model": "claude-opus-5",
            "has_api_key": True,
            "api_key_masked": "co••••••••ey",
        }

        local_model = client.put(
            f"/api/v1/companies/{company['id']}",
            json={
                "name": "Acme Corp",
                "bot_alias": "Nova",
                "bot_greet_message": company["bot_greet_message"],
                "llm": {"provider": "ollama", "model": "gemma4:12b"},
            },
        )
        assert local_model.status_code == 200
        assert local_model.json()["llm"] == {
            "provider": "ollama",
            "model": "gemma4:12b",
            "has_api_key": False,
            "api_key_masked": None,
        }

        client.app.state.repository.save(
            KnowledgeBase(
                company_id=company["id"],
                name="Guide",
                source="guide.pdf",
                mime_type="application/pdf",
                total_pages=1,
                selected_pages=1,
                chunking=ChunkingConfig(),
            )
        )
        assert client.delete(f"/api/v1/companies/{company['id']}").status_code == 204
        assert client.app.state.repository.list(company_id=company["id"]) == []
        assert client.app.state.vectors.deleted == [(company["id"], "*")]
        assert client.delete(f"/api/v1/companies/{company['id']}").status_code == 404


def test_admin_rejects_company_write_when_vector_sync_fails(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        client.app.state.companies = FailingCompanies()
        response = client.post("/api/v1/companies", json={"name": "Acme"})
        assert response.status_code == 503
        assert client.app.state.repository.list_companies() == []

        existing = client.app.state.repository.save_company(Company(name="Existing"))
        response = client.put(f"/api/v1/companies/{existing.id}", json={"name": "Updated"})
        assert response.status_code == 503
        assert client.app.state.repository.get_company(existing.id).name == "Existing"


def test_admin_company_logo_upload_read_preserve_and_delete(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        client.app.state.companies = Companies(client.app.state.repository)
        company = client.app.state.repository.save_company(Company(name="Acme"))
        logo_url = f"/api/v1/companies/{company.id}/logo"

        assert client.get(logo_url).status_code == 404
        assert (
            client.put(logo_url, files={"file": ("logo.txt", b"text", "text/plain")}).status_code
            == 415
        )
        assert (
            client.put(logo_url, files={"file": ("logo.png", b"not-png", "image/png")}).status_code
            == 422
        )
        unsafe_svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>'
        assert (
            client.put(
                logo_url,
                files={"file": ("unsafe.svg", unsafe_svg, "image/svg+xml")},
            ).status_code
            == 422
        )
        safe_svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>'
        svg_upload = client.put(logo_url, files={"file": ("logo.svg", safe_svg, "image/svg+xml")})
        assert svg_upload.status_code == 200
        assert client.get(logo_url).headers["content-type"] == "image/svg+xml"
        assert (
            client.put(
                logo_url,
                files={
                    "file": (
                        "huge.png",
                        b"\x89PNG\r\n\x1a\n" + b"x" * MAX_LOGO_BYTES,
                        "image/png",
                    )
                },
            ).status_code
            == 413
        )

        content = b"\x89PNG\r\n\x1a\ncompany-logo"
        uploaded = client.put(logo_url, files={"file": ("logo.png", content, "image/png")})
        assert uploaded.status_code == 200
        assert uploaded.json()["has_logo"] is True
        assert "logo_data" not in uploaded.json()
        logo = client.get(logo_url)
        assert logo.content == content
        assert logo.headers["content-type"] == "image/png"
        assert logo.headers["cache-control"] == "no-store"

        updated = client.put(
            f"/api/v1/companies/{company.id}",
            json={"name": "Acme", "about": "Updated profile"},
        )
        assert updated.status_code == 200
        assert client.get(logo_url).content == content

        assert client.delete(logo_url).status_code == 204
        assert client.get(logo_url).status_code == 404


def test_admin_lists_and_reviews_company_chat_sessions(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        repository = client.app.state.repository
        company = repository.save_company(Company(name="Acme"))
        chat_session, _ = repository.create_chat_session(
            ChatSession(id="session-1", company_id=company.id, ip_address="127.0.0.1"),
            "Welcome to Acme",
        )
        repository.add_chat_message(
            ChatSessionMessage(
                session_id=chat_session.id,
                owner=ChatSessionOwner.USER,
                content="Hello",
            )
        )
        created_date = chat_session.created_at.date().isoformat()

        listed = client.get(
            "/api/v1/chat-sessions",
            params={
                "company_id": company.id,
                "date_from": created_date,
                "date_to": created_date,
                "page": 1,
                "page_size": 10,
            },
        )
        assert listed.status_code == 200
        assert listed.json()["total"] == 1
        assert listed.json()["items"][0]["message_count"] == 2

        for session_id in ("session-2", "session-3"):
            repository.create_chat_session(ChatSession(id=session_id, company_id=company.id))
        first_page = client.get(
            "/api/v1/chat-sessions",
            params={
                "company_id": company.id,
                "date_from": created_date,
                "date_to": created_date,
                "page": 1,
                "page_size": 1,
            },
        ).json()
        second_page = client.get(
            "/api/v1/chat-sessions",
            params={
                "company_id": company.id,
                "date_from": created_date,
                "date_to": created_date,
                "page": 2,
                "page_size": 1,
            },
        ).json()
        assert first_page["total"] == 3
        assert first_page["total_pages"] == 3
        assert len(first_page["items"]) == len(second_page["items"]) == 1
        assert first_page["items"][0]["id"] != second_page["items"][0]["id"]

        detail = client.get(f"/api/v1/chat-sessions/{chat_session.id}")
        assert detail.status_code == 200
        assert [message["content"] for message in detail.json()["messages"]] == [
            "Welcome to Acme",
            "Hello",
        ]
        repository.add_chat_message(
            ChatSessionMessage(
                session_id=chat_session.id,
                owner=ChatSessionOwner.BOT,
                content="How can I help?",
                input_tokens=21,
                output_tokens=8,
            )
        )
        analytics = client.get(
            "/api/v1/analytics",
            params={
                "company_id": company.id,
                "date_from": created_date,
                "date_to": created_date,
            },
        )
        assert analytics.status_code == 200
        assert analytics.json()["session_count"] == 3
        assert analytics.json()["message_count"] == 3
        assert analytics.json()["average_messages_per_session"] == 1
        assert analytics.json()["input_tokens"] == 21
        assert analytics.json()["output_tokens"] == 8
        assert analytics.json()["daily"][0]["date"] == created_date
        assert client.get(
            "/api/v1/analytics",
            params={
                "company_id": "missing",
                "date_from": created_date,
                "date_to": created_date,
            },
        ).status_code == 404
        assert client.get(
            "/api/v1/analytics",
            params={
                "company_id": company.id,
                "date_from": "2026-01-01",
                "date_to": "2026-04-01",
            },
        ).status_code == 422

        other_company = repository.save_company(Company(name="Other company"))
        other_session, _ = repository.create_chat_session(
            ChatSession(id="other-company-session", company_id=other_company.id),
            "Keep me",
        )
        deleted = client.request(
            "DELETE",
            "/api/v1/chat-sessions",
            json={
                "company_id": company.id,
                "session_ids": ["session-2", other_session.id, "missing"],
            },
        )
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": 1}
        assert client.get("/api/v1/chat-sessions/session-2").status_code == 404
        assert client.get(f"/api/v1/chat-sessions/{other_session.id}").status_code == 200
        assert client.request(
            "DELETE",
            "/api/v1/chat-sessions",
            json={"company_id": "missing", "session_ids": [chat_session.id]},
        ).status_code == 404
        assert client.request(
            "DELETE",
            "/api/v1/chat-sessions",
            json={"company_id": company.id, "session_ids": []},
        ).status_code == 422
        deleted_filtered = client.request(
            "DELETE",
            "/api/v1/chat-sessions",
            json={
                "company_id": company.id,
                "date_from": created_date,
                "date_to": created_date,
            },
        )
        assert deleted_filtered.status_code == 200
        assert deleted_filtered.json() == {"deleted": 2}
        assert client.get(f"/api/v1/chat-sessions/{chat_session.id}").status_code == 404
        assert client.get("/api/v1/chat-sessions/session-3").status_code == 404
        assert client.get(f"/api/v1/chat-sessions/{other_session.id}").status_code == 200
        assert client.request(
            "DELETE",
            "/api/v1/chat-sessions",
            json={
                "company_id": company.id,
                "session_ids": ["missing"],
                "date_from": created_date,
                "date_to": created_date,
            },
        ).status_code == 422
        assert client.request(
            "DELETE",
            "/api/v1/chat-sessions",
            json={"company_id": company.id},
        ).status_code == 422

        assert (
            client.get(
                "/api/v1/chat-sessions",
                params={
                    "company_id": company.id,
                    "date_from": "2000-01-01",
                    "date_to": "2000-01-02",
                },
            ).json()["total"]
            == 0
        )
        assert (
            client.get(
                "/api/v1/chat-sessions",
                params={
                    "company_id": company.id,
                    "date_from": "2026-09-18",
                    "date_to": "2026-09-17",
                },
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/api/v1/chat-sessions",
                params={
                    "company_id": company.id,
                    "date_from": "2026-09-01",
                    "date_to": "2026-09-15",
                },
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/api/v1/chat-sessions",
                params={
                    "company_id": "missing",
                    "date_from": created_date,
                    "date_to": created_date,
                },
            ).status_code
            == 404
        )
        assert (
            client.get("/api/v1/chat-sessions", params={"company_id": company.id}).status_code
            == 422
        )
        assert client.get("/api/v1/chat-sessions/missing").status_code == 404
