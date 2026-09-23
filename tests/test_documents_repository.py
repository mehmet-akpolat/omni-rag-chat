from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.fernet import Fernet
from pypdf import PdfWriter
from sqlalchemy import inspect, select, update

from packages.omni_rag_core.documents import DocumentStore, InvalidDocument
from packages.omni_rag_core.credentials import CredentialEncryptionError
from packages.omni_rag_core.domain import (
    DEFAULT_BOT_GREET_MESSAGE,
    ChatSession,
    ChatSessionMessage,
    ChatSessionOwner,
    ChatSessionStatus,
    ChunkingConfig,
    Company,
    CompanyLLMMapping,
    KnowledgeBase,
)
from packages.omni_rag_core.repository import (
    CompanyRow,
    CompanyLLMMappingRow,
    DuplicateCompanyName,
    DuplicateDocumentContent,
    DuplicateKnowledgeBaseName,
    KnowledgeBaseRow,
    SqlKnowledgeRepository,
)


def make_pdf(path: Path, pages: int = 2):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as output:
        writer.write(output)


def test_document_store_round_trip_and_preview(tmp_path):
    source = tmp_path / "source.pdf"
    make_pdf(source)
    store = DocumentStore(tmp_path / "uploads")
    path = store.save("doc", source.read_bytes())
    assert store.path_for("doc") == path
    pages = store.extract(path)
    assert pages == ["", ""]
    preview = store.preview("doc", "unsafe/<report>.pdf", [" hello   world ", "next"])
    assert preview.source == "_report_.pdf"
    assert preview.total_pages == 2
    assert preview.pages[0].excerpt == "hello world"


def test_document_store_rejects_missing_and_invalid(tmp_path):
    store = DocumentStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.path_for("missing")
    bad = store.save("bad", b"not a PDF")
    with pytest.raises(InvalidDocument, match="readable PDF"):
        store.extract(bad)


def test_sql_repository_crud_and_order(tmp_path):
    encryption_key = Fernet.generate_key().decode()
    repo = SqlKnowledgeRepository(
        f"sqlite:///{tmp_path / 'test.db'}", encryption_key
    )
    company = repo.save_company(
        Company(name="Acme", logo_data=b"logo-bytes", logo_mime_type="image/png")
    )
    other_company = repo.save_company(Company(name="Other"))
    first = KnowledgeBase(
        company_id=company.id,
        name="One",
        source="one.pdf",
        mime_type="application/pdf",
        total_pages=3,
        excluded_pages=[2],
        selected_pages=2,
        chunking=ChunkingConfig(),
        document_checksum="a" * 64,
    )
    second = KnowledgeBase(
        company_id=company.id,
        name="Two",
        source="two.pdf",
        mime_type="application/pdf",
        total_pages=1,
        selected_pages=1,
        chunking=ChunkingConfig(strategy="fixed"),
        chunk_count=4,
        document_checksum="b" * 64,
    )
    assert repo.save(first).id == first.id
    repo.save(second)
    assert repo.get(first.id).name == "ONE"
    assert repo.get(first.id).excluded_pages == [2]
    assert repo.get(first.id).selected_pages == 2
    assert repo.name_exists(" one ", company.id) is True
    assert repo.name_exists("missing", company.id) is False
    assert repo.name_exists(" one ", other_company.id) is False
    duplicate = second.model_copy(
        update={"id": "duplicate", "name": "  oNe  ", "document_checksum": "c" * 64}
    )
    with pytest.raises(DuplicateKnowledgeBaseName, match="already exists"):
        repo.save(duplicate)
    duplicate_content = second.model_copy(
        update={"id": "duplicate-content", "name": "Three", "document_checksum": "a" * 64}
    )
    with pytest.raises(DuplicateDocumentContent, match="already been imported"):
        repo.save(duplicate_content)
    assert repo.get_by_checksum("a" * 64, company.id).id == first.id
    assert repo.get_by_checksum("f" * 64, company.id) is None
    indexes = inspect(repo.engine).get_indexes("knowledge_bases")
    assert [(index["name"], index["unique"]) for index in indexes] == [
        ("uq_knowledge_bases_company_checksum", 1),
        ("uq_knowledge_bases_company_name", 1),
        ("uq_knowledge_bases_company_url_source", 1),
    ]
    columns = [column["name"] for column in inspect(repo.engine).get_columns("knowledge_bases")]
    assert columns == [
        "id",
        "name",
        "company_id",
        "source",
        "mime_type",
        "document_checksum",
        "total_pages",
        "selected_pages",
        "excluded_pages_json",
        "chunking_json",
        "chunk_count",
        "status",
        "created_at",
        "updated_at",
    ]
    company_columns = [column["name"] for column in inspect(repo.engine).get_columns("companies")]
    assert company_columns == [
        "id",
        "name",
        "about",
        "phone",
        "email",
        "address",
        "maps_url",
        "logo_mime_type",
        "logo_data",
        "bot_alias",
        "bot_avatar",
        "bot_greet_message",
        "created_at",
        "updated_at",
    ]
    mapping_columns = [
        column["name"]
        for column in inspect(repo.engine).get_columns("company_llm_mappings")
    ]
    assert mapping_columns == [
        "id",
        "company_id",
        "provider",
        "model",
        "api_key_encrypted",
        "created_at",
        "updated_at",
    ]
    message_columns = [
        column["name"]
        for column in inspect(repo.engine).get_columns("chat_session_messages")
    ]
    assert message_columns == [
        "id",
        "session_id",
        "owner",
        "content",
        "input_tokens",
        "output_tokens",
        "created_at",
    ]
    assert repo.get("missing") is None
    assert [item.id for item in repo.list()] == [second.id, first.id]
    assert [item.id for item in repo.list(offset=1, limit=1)] == [first.id]
    assert [item.id for item in repo.list(search="TWO")] == [second.id]
    assert [item.id for item in repo.list(company_id=company.id)] == [second.id, first.id]
    assert repo.list(company_id=other_company.id) == []
    assert repo.count() == 2
    assert repo.count(company_id=company.id) == 2
    assert repo.count(search="one.pdf") == 1
    first.status = "updated"
    repo.save(first)
    assert repo.get(first.id).status == "updated"
    assert repo.delete(second.id) is True
    assert repo.delete(second.id) is False
    assert repo.get(second.id) is None
    assert [item.name for item in repo.list_companies()] == ["Acme", "Other"]
    stored_company = repo.get_company(company.id)
    assert stored_company.name == "Acme"
    assert stored_company.logo_data == b"logo-bytes"
    assert stored_company.has_logo is True
    assert stored_company.bot_greet_message == DEFAULT_BOT_GREET_MESSAGE
    assert repo.get_company_llm_mapping(company.id) is None
    configured_llm = repo.save_company_llm_mapping(
        CompanyLLMMapping(
            company_id=company.id,
            provider="openai",
            model="gpt-5.6-terra",
            api_key="company-openai-secret",
        )
    )
    assert configured_llm.model == "gpt-5.6-terra"
    loaded_llm = repo.get_company_llm_mapping(company.id)
    assert loaded_llm.provider == "openai"
    assert loaded_llm.api_key == "company-openai-secret"
    assert loaded_llm.has_api_key is True
    with repo.engine.connect() as connection:
        encrypted = connection.scalar(
            select(CompanyLLMMappingRow.api_key_encrypted).where(
                CompanyLLMMappingRow.company_id == company.id
            )
        )
    assert encrypted and encrypted != "company-openai-secret"
    preserved = repo.save_company_llm_mapping(
        CompanyLLMMapping(
            company_id=company.id,
            provider="openai",
            model="gpt-5.6-sol",
        )
    )
    assert preserved.has_api_key is True
    assert repo.get_company_llm_mapping(company.id).api_key == "company-openai-secret"
    unconfigured_reader = SqlKnowledgeRepository(f"sqlite:///{tmp_path / 'test.db'}")
    public_mapping = unconfigured_reader.get_company_llm_mapping(
        company.id, include_api_key=False
    )
    assert public_mapping.has_api_key is True and public_mapping.api_key is None
    with pytest.raises(CredentialEncryptionError, match="required before using"):
        unconfigured_reader.get_company_llm_mapping(company.id)
    unconfigured_reader.engine.dispose()
    cleared = repo.save_company_llm_mapping(
        CompanyLLMMapping(company_id=company.id, provider="ollama", model="gemma4:12b")
    )
    assert cleared.has_api_key is False and cleared.api_key is None
    with repo.engine.connect() as connection:
        assert connection.scalar(
            select(CompanyLLMMappingRow.api_key_encrypted).where(
                CompanyLLMMappingRow.company_id == company.id
            )
        ) is None
    with pytest.raises(ValueError, match="Company not found"):
        repo.save_company_llm_mapping(
            CompanyLLMMapping(company_id="missing", provider="ollama", model="gemma4:12b")
        )
    assert repo.company_name_exists(" acme ") is True
    assert repo.company_name_exists("acme", exclude_id=company.id) is False
    with pytest.raises(DuplicateCompanyName, match="already exists"):
        repo.save_company(Company(name="acme"))
    with repo.engine.connect() as connection:
        company_indexes = connection.exec_driver_sql(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name = 'companies' ORDER BY name"
        ).scalars()
        assert list(company_indexes) == ["sqlite_autoindex_companies_1", "uq_companies_name_ci"]
    assert repo.delete_company(company.id) is True
    assert repo.delete_company(company.id) is False
    assert repo.get_company_llm_mapping(company.id) is None
    assert repo.get(first.id) is None
    repo.engine.dispose()


def test_repository_stores_searches_and_replaces_url_sources(tmp_path):
    repo = SqlKnowledgeRepository(f"sqlite:///{tmp_path / 'urls.db'}")
    company = repo.save_company(Company(name="Acme"))
    for identifier, name, checksum in (
        ("pdf-v1", "PDF v1", "b" * 64),
        ("pdf-v2", "PDF v2", "c" * 64),
    ):
        repo.save(
            KnowledgeBase(
                id=identifier,
                company_id=company.id,
                name=name,
                source="revised-guide.pdf",
                mime_type="application/pdf",
                total_pages=1,
                selected_pages=1,
                chunking=ChunkingConfig(),
                document_checksum=checksum,
            )
        )
    assert len(repo.list(search="revised-guide.pdf")) == 2

    existing = KnowledgeBase(
        id="old-url",
        company_id=company.id,
        name="Web guide",
        source="https://example.com/guide",
        mime_type="text/html",
        chunking=ChunkingConfig(),
        document_checksum="d" * 64,
    )
    repo.save(existing)
    with repo.engine.connect() as connection:
        stored_page_fields = connection.exec_driver_sql(
            "SELECT total_pages, selected_pages, excluded_pages_json "
            "FROM knowledge_bases WHERE id = ?",
            (existing.id,),
        ).one()
    assert tuple(stored_page_fields) == (None, None, None)
    assert repo.get_by_url_source(existing.source, company.id).id == existing.id
    assert repo.get_by_url_source(existing.source, "other") is None
    assert repo.list(search="example.com")[0].id == existing.id

    replacement = existing.model_copy(
        update={
            "id": "new-url",
            "document_checksum": "e" * 64,
        }
    )
    replaced = repo.replace(existing.id, replacement)
    assert replaced.id == replacement.id
    assert replaced.updated_at >= replaced.created_at
    assert repo.get(existing.id) is None
    assert repo.get(replacement.id).source == "https://example.com/guide"
    with pytest.raises(ValueError, match="not found"):
        repo.replace("missing", replacement.model_copy(update={"id": "another"}))
    repo.engine.dispose()


def test_chat_session_repository_lifecycle_filters_and_company_cascade(tmp_path):
    repo = SqlKnowledgeRepository(f"sqlite:///{tmp_path / 'sessions.db'}")
    with repo.engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    company = repo.save_company(Company(name="Session company"))
    other = repo.save_company(Company(name="Other session company"))
    created_at = datetime.now(timezone.utc) - timedelta(hours=1)
    chat_session, created = repo.create_chat_session(
        ChatSession(
            id="session-1",
            company_id=company.id,
            ip_address="127.0.0.1",
            created_at=created_at,
        ),
        "Welcome",
    )
    assert created is True and chat_session.message_count == 1
    duplicate, created = repo.create_chat_session(
        ChatSession(id="session-1", company_id=company.id), "Ignored"
    )
    assert created is False and duplicate.message_count == 1
    with pytest.raises(ValueError, match="different company"):
        repo.create_chat_session(ChatSession(id="session-1", company_id=other.id))
    user_message = repo.add_chat_message(
        ChatSessionMessage(
            session_id="session-1",
            owner=ChatSessionOwner.USER,
            content="Hello",
        )
    )
    assert user_message.content == "Hello"
    answer_message = repo.add_chat_message(
        ChatSessionMessage(
            session_id="session-1",
            owner=ChatSessionOwner.BOT,
            content="Hello back",
            input_tokens=12,
            output_tokens=5,
        )
    )
    assert answer_message.input_tokens == 12 and answer_message.output_tokens == 5
    assert [message.owner for message in repo.list_chat_messages("session-1")] == [
        ChatSessionOwner.BOT,
        ChatSessionOwner.USER,
        ChatSessionOwner.BOT,
    ]
    assert repo.get_chat_session("session-1").message_count == 3
    assert repo.get_chat_session("missing") is None
    assert repo.count_chat_sessions(company.id) == 1
    assert repo.count_chat_sessions(company.id, created_from=created_at - timedelta(minutes=1)) == 1
    assert repo.count_chat_sessions(company.id, created_to=created_at - timedelta(minutes=1)) == 0
    analytics_from = created_at.replace(hour=0, minute=0, second=0, microsecond=0)
    analytics = repo.get_chat_analytics(
        company.id,
        created_from=analytics_from,
        created_to=analytics_from + timedelta(days=1),
    )
    assert analytics.session_count == 1
    assert analytics.message_count == 3
    assert analytics.average_messages_per_session == 3
    assert (analytics.input_tokens, analytics.output_tokens) == (12, 5)
    assert len(analytics.daily) == 1 and analytics.daily[0].session_count == 1
    assert [item.id for item in repo.list_chat_sessions(company.id, limit=1)] == ["session-1"]
    assert repo.list_chat_sessions(company.id, created_from=created_at + timedelta(minutes=1)) == []
    with pytest.raises(ValueError, match="closed or timed out"):
        repo.close_chat_session("session-1", ChatSessionStatus.ACTIVE)
    closed = repo.close_chat_session("session-1", ChatSessionStatus.TIMEOUT)
    assert closed.status == ChatSessionStatus.TIMEOUT and closed.ended_at is not None
    ended_at = closed.ended_at
    unchanged = repo.close_chat_session("session-1", ChatSessionStatus.CLOSED)
    assert unchanged.status == ChatSessionStatus.TIMEOUT and unchanged.ended_at == ended_at
    assert repo.close_chat_session("missing", ChatSessionStatus.CLOSED) is None
    deleted_session, _ = repo.create_chat_session(
        ChatSession(id="delete-me", company_id=company.id), "Delete this transcript"
    )
    other_session, _ = repo.create_chat_session(
        ChatSession(id="other-session", company_id=other.id), "Keep this transcript"
    )
    assert repo.delete_chat_sessions(
        company.id,
        [deleted_session.id, deleted_session.id, other_session.id, "missing"],
    ) == 1
    assert repo.get_chat_session(deleted_session.id) is None
    assert repo.list_chat_messages(deleted_session.id) == []
    assert repo.get_chat_session(other_session.id) is not None
    assert repo.delete_chat_sessions(company.id, []) == 0
    range_created_at = created_at - timedelta(days=2)
    range_session, _ = repo.create_chat_session(
        ChatSession(
            id="delete-by-range",
            company_id=company.id,
            created_at=range_created_at,
        ),
        "Delete this ranged transcript",
    )
    assert repo.delete_chat_sessions_in_range(
        company.id,
        created_from=range_created_at - timedelta(minutes=1),
        created_to=range_created_at + timedelta(minutes=1),
    ) == 1
    assert repo.get_chat_session(range_session.id) is None
    assert repo.list_chat_messages(range_session.id) == []
    assert repo.get_chat_session("session-1") is not None
    assert repo.get_chat_session(other_session.id) is not None
    auto_expired, _ = repo.create_chat_session(
        ChatSession(
            id="auto-expired",
            company_id=other.id,
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
    )
    assert auto_expired.status == ChatSessionStatus.ACTIVE
    assert repo.timeout_expired_chat_sessions(datetime.now(timezone.utc) - timedelta(hours=1)) == 1
    assert repo.get_chat_session("auto-expired").status == ChatSessionStatus.TIMEOUT
    with pytest.raises(ValueError, match="not found"):
        repo.add_chat_message(
            ChatSessionMessage(
                session_id="missing",
                owner=ChatSessionOwner.BOT,
                content="No session",
            )
        )
    repo.add_chat_message(
        ChatSessionMessage(
            session_id="session-1",
            owner=ChatSessionOwner.BOT,
            content="Completed during close",
        )
    )
    assert repo.delete_company(company.id) is True
    assert repo.get_chat_session("session-1") is None
    assert repo.list_chat_messages("session-1") == []
    repo.engine.dispose()


def test_repository_migrates_legacy_ready_status(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'legacy.db'}"
    repo = SqlKnowledgeRepository(database_url)
    company = repo.save_company(Company(name="Legacy company"))
    item = KnowledgeBase(
        company_id=company.id,
        name="Legacy",
        source="legacy.pdf",
        mime_type="application/pdf",
        total_pages=1,
        selected_pages=1,
        chunking=ChunkingConfig(),
        status="ready",
    )
    repo.save(item)
    with repo.engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE companies DROP COLUMN logo_data")
        connection.exec_driver_sql("ALTER TABLE companies DROP COLUMN logo_mime_type")
        connection.exec_driver_sql("ALTER TABLE companies DROP COLUMN bot_greet_message")
        connection.exec_driver_sql("ALTER TABLE companies ADD COLUMN logo_url VARCHAR(500)")
        connection.exec_driver_sql("DROP INDEX uq_knowledge_bases_company_checksum")
        connection.exec_driver_sql("DROP INDEX uq_knowledge_bases_company_name")
        connection.exec_driver_sql("DROP INDEX uq_knowledge_bases_company_url_source")
        connection.exec_driver_sql("ALTER TABLE knowledge_bases RENAME COLUMN source TO filename")
        connection.exec_driver_sql(
            "ALTER TABLE knowledge_bases ADD COLUMN source_type VARCHAR(16) DEFAULT 'pdf'"
        )
        connection.exec_driver_sql(
            "ALTER TABLE knowledge_bases ADD COLUMN source_url VARCHAR(2048)"
        )
        connection.exec_driver_sql(
            "CREATE UNIQUE INDEX uq_knowledge_bases_company_source_url "
            "ON knowledge_bases (company_id, source_url)"
        )
        connection.exec_driver_sql("ALTER TABLE knowledge_bases DROP COLUMN document_checksum")
        connection.exec_driver_sql("CREATE INDEX ix_knowledge_bases_name ON knowledge_bases (name)")
        connection.exec_driver_sql(
            "CREATE UNIQUE INDEX uq_knowledge_bases_name_ci ON knowledge_bases (lower(name))"
        )
        connection.execute(
            update(KnowledgeBaseRow).where(KnowledgeBaseRow.id == item.id).values(name=" Legacy ")
        )
        connection.execute(
            update(CompanyRow).where(CompanyRow.id == company.id).values(bot_avatar="orbit")
        )
    repo.engine.dispose()
    migrated = SqlKnowledgeRepository(database_url)
    assert migrated.get(item.id).status == "enabled"
    assert migrated.get(item.id).name == "LEGACY"
    assert migrated.get_company(company.id).bot_avatar == "book"
    assert migrated.get_company(company.id).bot_greet_message == DEFAULT_BOT_GREET_MESSAGE
    indexes = inspect(migrated.engine).get_indexes("knowledge_bases")
    assert [(index["name"], index["unique"]) for index in indexes] == [
        ("uq_knowledge_bases_company_checksum", 1),
        ("uq_knowledge_bases_company_name", 1),
        ("uq_knowledge_bases_company_url_source", 1),
    ]
    company_columns = {
        column["name"] for column in inspect(migrated.engine).get_columns("companies")
    }
    assert company_columns >= {
        "logo_data",
        "logo_mime_type",
        "bot_greet_message",
    }
    assert "logo_url" not in company_columns
    knowledge_columns = {
        column["name"] for column in inspect(migrated.engine).get_columns("knowledge_bases")
    }
    assert "source" in knowledge_columns
    assert {"filename", "source_type", "source_url"}.isdisjoint(knowledge_columns)
    migrated.engine.dispose()
