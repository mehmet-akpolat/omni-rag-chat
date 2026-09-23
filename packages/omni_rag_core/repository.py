from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    delete,
    func,
    inspect,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .credentials import CredentialCipher
from .domain import (
    DEFAULT_BOT_GREET_MESSAGE,
    ChatAnalytics,
    ChatAnalyticsDaily,
    ChatSession,
    ChatSessionMessage,
    ChatSessionOwner,
    ChatSessionStatus,
    Company,
    CompanyLLMMapping,
    KnowledgeBase,
    LLMProvider,
    WEB_MIME_TYPES,
)


class Base(DeclarativeBase):
    pass


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class CompanyRow(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    about: Mapped[str] = mapped_column(Text, default="")
    phone: Mapped[str] = mapped_column(String(50), default="")
    email: Mapped[str] = mapped_column(String(254), default="")
    address: Mapped[str] = mapped_column(Text, default="")
    maps_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    logo_mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    logo_data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    bot_alias: Mapped[str] = mapped_column(String(80), default="AIBot")
    bot_avatar: Mapped[str] = mapped_column(String(32), default="bot")
    bot_greet_message: Mapped[str] = mapped_column(Text, default=DEFAULT_BOT_GREET_MESSAGE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


company_name_unique_index = Index("uq_companies_name_ci", func.lower(CompanyRow.name), unique=True)


class CompanyLLMMappingRow(Base):
    __tablename__ = "company_llm_mappings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    company_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("companies.id", ondelete="CASCADE"), unique=True
    )
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(120))
    api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class KnowledgeBaseRow(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    company_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("companies.id", ondelete="CASCADE")
    )
    source: Mapped[str] = mapped_column(String(2048))
    mime_type: Mapped[str] = mapped_column(String(100))
    document_checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    excluded_pages_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunking_json: Mapped[str] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="enabled")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


knowledge_base_name_unique_index = Index(
    "uq_knowledge_bases_company_name",
    KnowledgeBaseRow.company_id,
    KnowledgeBaseRow.name,
    unique=True,
)
knowledge_base_checksum_unique_index = Index(
    "uq_knowledge_bases_company_checksum",
    KnowledgeBaseRow.company_id,
    KnowledgeBaseRow.document_checksum,
    unique=True,
)
knowledge_base_url_source_unique_index = Index(
    "uq_knowledge_bases_company_url_source",
    KnowledgeBaseRow.company_id,
    KnowledgeBaseRow.source,
    unique=True,
    sqlite_where=KnowledgeBaseRow.mime_type.in_(WEB_MIME_TYPES),
    postgresql_where=KnowledgeBaseRow.mime_type.in_(WEB_MIME_TYPES),
)


class ChatSessionRow(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    company_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("companies.id", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(String(16), default="active")
    ip_address: Mapped[str] = mapped_column(String(45), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


chat_session_company_created_index = Index(
    "ix_chat_sessions_company_created",
    ChatSessionRow.company_id,
    ChatSessionRow.created_at,
)


class ChatSessionMessageRow(Base):
    __tablename__ = "chat_session_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE")
    )
    owner: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


chat_session_message_order_index = Index(
    "ix_chat_session_messages_session_created",
    ChatSessionMessageRow.session_id,
    ChatSessionMessageRow.created_at,
)


class DuplicateKnowledgeBaseName(ValueError):
    pass


class DuplicateDocumentContent(ValueError):
    pass


class DuplicateCompanyName(ValueError):
    pass


class KnowledgeRepository(ABC):
    @abstractmethod
    def save(self, knowledge_base: KnowledgeBase) -> KnowledgeBase: ...

    @abstractmethod
    def list(
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        search: str | None = None,
        company_id: str | None = None,
    ) -> list[KnowledgeBase]: ...

    @abstractmethod
    def count(self, *, search: str | None = None, company_id: str | None = None) -> int: ...

    @abstractmethod
    def get(self, knowledge_base_id: str) -> KnowledgeBase | None: ...

    @abstractmethod
    def delete(self, knowledge_base_id: str) -> bool: ...

    @abstractmethod
    def name_exists(self, name: str, company_id: str) -> bool: ...

    @abstractmethod
    def get_by_checksum(self, checksum: str, company_id: str) -> KnowledgeBase | None: ...

    @abstractmethod
    def get_by_url_source(self, source: str, company_id: str) -> KnowledgeBase | None: ...

    @abstractmethod
    def replace(self, knowledge_base_id: str, knowledge_base: KnowledgeBase) -> KnowledgeBase: ...

    @abstractmethod
    def save_company(self, company: Company) -> Company: ...

    @abstractmethod
    def list_companies(self) -> list[Company]: ...

    @abstractmethod
    def get_company(self, company_id: str) -> Company | None: ...

    @abstractmethod
    def save_company_llm_mapping(self, mapping: CompanyLLMMapping) -> CompanyLLMMapping: ...

    @abstractmethod
    def get_company_llm_mapping(
        self, company_id: str, *, include_api_key: bool = True
    ) -> CompanyLLMMapping | None: ...

    @abstractmethod
    def delete_company(self, company_id: str) -> bool: ...

    @abstractmethod
    def company_name_exists(self, name: str, exclude_id: str | None = None) -> bool: ...

    @abstractmethod
    def create_chat_session(
        self, chat_session: ChatSession, greeting: str | None = None
    ) -> tuple[ChatSession, bool]: ...

    @abstractmethod
    def close_chat_session(
        self, session_id: str, status: ChatSessionStatus
    ) -> ChatSession | None: ...

    @abstractmethod
    def timeout_expired_chat_sessions(self, created_before: datetime) -> int: ...

    @abstractmethod
    def add_chat_message(self, message: ChatSessionMessage) -> ChatSessionMessage: ...

    @abstractmethod
    def get_chat_session(self, session_id: str) -> ChatSession | None: ...

    @abstractmethod
    def list_chat_sessions(
        self,
        company_id: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[ChatSession]: ...

    @abstractmethod
    def count_chat_sessions(
        self,
        company_id: str,
        *,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> int: ...

    @abstractmethod
    def get_chat_analytics(
        self,
        company_id: str,
        *,
        created_from: datetime,
        created_to: datetime,
    ) -> ChatAnalytics: ...

    @abstractmethod
    def delete_chat_sessions(self, company_id: str, session_ids: list[str]) -> int: ...

    @abstractmethod
    def delete_chat_sessions_in_range(
        self,
        company_id: str,
        *,
        created_from: datetime,
        created_to: datetime,
    ) -> int: ...

    @abstractmethod
    def list_chat_messages(self, session_id: str) -> list[ChatSessionMessage]: ...


class SqlKnowledgeRepository(KnowledgeRepository):
    def __init__(self, database_url: str, llm_credentials_key: str | None = None):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args)
        self.credentials = CredentialCipher(llm_credentials_key)
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            message_columns = {
                column["name"]
                for column in inspect(connection).get_columns("chat_session_messages")
            }
            for column_name in ("input_tokens", "output_tokens"):
                if column_name not in message_columns:
                    connection.exec_driver_sql(
                        f"ALTER TABLE chat_session_messages ADD COLUMN "
                        f"{column_name} INTEGER NOT NULL DEFAULT 0"
                    )
            company_columns = {
                column["name"] for column in inspect(connection).get_columns("companies")
            }
            if "logo_data" not in company_columns:
                binary_type = "BYTEA" if connection.dialect.name == "postgresql" else "BLOB"
                connection.exec_driver_sql(
                    f"ALTER TABLE companies ADD COLUMN logo_data {binary_type}"
                )
            if "logo_mime_type" not in company_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE companies ADD COLUMN logo_mime_type VARCHAR(100)"
                )
            if "bot_greet_message" not in company_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE companies ADD COLUMN bot_greet_message TEXT"
                )
                connection.execute(
                    update(CompanyRow).values(bot_greet_message=DEFAULT_BOT_GREET_MESSAGE)
                )
            if "updated_at" not in company_columns:
                timestamp_type = (
                    "TIMESTAMP WITH TIME ZONE"
                    if connection.dialect.name == "postgresql"
                    else "DATETIME"
                )
                connection.exec_driver_sql(
                    f"ALTER TABLE companies ADD COLUMN updated_at {timestamp_type}"
                )
                connection.exec_driver_sql(
                    "UPDATE companies SET updated_at = created_at WHERE updated_at IS NULL"
                )
                if connection.dialect.name == "postgresql":
                    connection.exec_driver_sql(
                        "ALTER TABLE companies ALTER COLUMN updated_at SET NOT NULL"
                    )
            if "logo_url" in company_columns:
                connection.exec_driver_sql("ALTER TABLE companies DROP COLUMN logo_url")
            avatar_migrations = {
                "sparkles": "brain",
                "message": "headset",
                "orbit": "book",
            }
            for previous_avatar, current_avatar in avatar_migrations.items():
                connection.execute(
                    update(CompanyRow)
                    .where(CompanyRow.bot_avatar == previous_avatar)
                    .values(bot_avatar=current_avatar)
                )
            duplicate_company_name = connection.execute(
                select(func.lower(CompanyRow.name))
                .group_by(func.lower(CompanyRow.name))
                .having(func.count() > 1)
                .limit(1)
            ).scalar_one_or_none()
            if duplicate_company_name:
                raise DuplicateCompanyName(
                    "Existing companies contain names that differ only by letter case"
                )
            connection.exec_driver_sql("DROP INDEX IF EXISTS uq_companies_name")
            connection.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_companies_name_ci ON companies (lower(name))"
            )
            columns = {
                column["name"] for column in inspect(connection).get_columns("knowledge_bases")
            }
            connection.exec_driver_sql(
                "DROP INDEX IF EXISTS uq_knowledge_bases_company_source_url"
            )
            connection.exec_driver_sql(
                "DROP INDEX IF EXISTS uq_knowledge_bases_company_url_source"
            )
            if "source" not in columns and "filename" in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE knowledge_bases RENAME COLUMN filename TO source"
                )
                columns = (columns - {"filename"}) | {"source"}
            elif "source" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE knowledge_bases ADD COLUMN source VARCHAR(2048)"
                )
                columns.add("source")
            if connection.dialect.name == "postgresql":
                connection.exec_driver_sql(
                    "ALTER TABLE knowledge_bases "
                    "ALTER COLUMN source TYPE VARCHAR(2048)"
                )
            if "source_url" in columns:
                connection.exec_driver_sql(
                    "UPDATE knowledge_bases SET source = source_url "
                    "WHERE source_url IS NOT NULL"
                )
                connection.exec_driver_sql(
                    "ALTER TABLE knowledge_bases DROP COLUMN source_url"
                )
                columns.remove("source_url")
            if "source_type" in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE knowledge_bases DROP COLUMN source_type"
                )
                columns.remove("source_type")
            if "filename" in columns:
                connection.exec_driver_sql(
                    "UPDATE knowledge_bases SET source = filename "
                    "WHERE source IS NULL OR source = ''"
                )
                connection.exec_driver_sql(
                    "ALTER TABLE knowledge_bases DROP COLUMN filename"
                )
                columns.remove("filename")
            if "document_checksum" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE knowledge_bases ADD COLUMN document_checksum VARCHAR(64)"
                )
            if "company_id" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE knowledge_bases ADD COLUMN company_id VARCHAR(36)"
                )
            unassigned_count = connection.scalar(
                select(func.count())
                .select_from(KnowledgeBaseRow)
                .where(KnowledgeBaseRow.company_id.is_(None))
            )
            if unassigned_count:
                legacy_company_id = connection.scalar(
                    select(CompanyRow.id)
                    .where(func.lower(CompanyRow.name) == "default company")
                    .limit(1)
                )
                if not legacy_company_id:
                    legacy_company_id = str(uuid4())
                    connection.execute(
                        CompanyRow.__table__.insert().values(
                            id=legacy_company_id,
                            name="Default Company",
                            logo_data=None,
                            logo_mime_type=None,
                            about="Migrated knowledge bases",
                            phone="",
                            email="",
                            address="",
                            maps_url=None,
                            bot_alias="AIBot",
                            bot_avatar="bot",
                            bot_greet_message=DEFAULT_BOT_GREET_MESSAGE,
                            created_at=datetime.now(timezone.utc),
                            updated_at=datetime.now(timezone.utc),
                        )
                    )
                connection.execute(
                    update(KnowledgeBaseRow)
                    .where(KnowledgeBaseRow.company_id.is_(None))
                    .values(company_id=legacy_company_id)
                )
            if connection.dialect.name == "postgresql":
                connection.exec_driver_sql(
                    "ALTER TABLE knowledge_bases ALTER COLUMN company_id SET NOT NULL"
                )
                foreign_keys = inspect(connection).get_foreign_keys("knowledge_bases")
                if not any(
                    foreign_key.get("constrained_columns") == ["company_id"]
                    for foreign_key in foreign_keys
                ):
                    connection.exec_driver_sql(
                        "ALTER TABLE knowledge_bases "
                        "ADD CONSTRAINT fk_knowledge_bases_company_id "
                        "FOREIGN KEY (company_id) REFERENCES companies (id) ON DELETE CASCADE"
                    )
            rows = connection.execute(
                select(KnowledgeBaseRow.id, KnowledgeBaseRow.company_id, KnowledgeBaseRow.name)
            ).all()
            normalized_names: dict[tuple[str, str], str] = {}
            for knowledge_base_id, company_id, name in rows:
                normalized = " ".join(name.split()).upper()
                key = (company_id, normalized)
                existing_id = normalized_names.get(key)
                if existing_id and existing_id != knowledge_base_id:
                    raise DuplicateKnowledgeBaseName(
                        f'Existing knowledge bases normalize to the duplicate name "{normalized}"'
                    )
                normalized_names[key] = knowledge_base_id
            for knowledge_base_id, _, name in rows:
                normalized = " ".join(name.split()).upper()
                if name != normalized:
                    connection.execute(
                        update(KnowledgeBaseRow)
                        .where(KnowledgeBaseRow.id == knowledge_base_id)
                        .values(name=normalized)
                    )
            if connection.dialect.name == "postgresql":
                for column_name in (
                    "total_pages",
                    "selected_pages",
                    "excluded_pages_json",
                ):
                    connection.exec_driver_sql(
                        "ALTER TABLE knowledge_bases "
                        f"ALTER COLUMN {column_name} DROP NOT NULL"
                    )
            if "updated_at" not in columns:
                timestamp_type = (
                    "TIMESTAMP WITH TIME ZONE"
                    if connection.dialect.name == "postgresql"
                    else "DATETIME"
                )
                connection.exec_driver_sql(
                    f"ALTER TABLE knowledge_bases ADD COLUMN updated_at {timestamp_type}"
                )
                connection.exec_driver_sql(
                    "UPDATE knowledge_bases SET updated_at = created_at WHERE updated_at IS NULL"
                )
                if connection.dialect.name == "postgresql":
                    connection.exec_driver_sql(
                        "ALTER TABLE knowledge_bases ALTER COLUMN updated_at SET NOT NULL"
                    )
            if connection.dialect.name != "postgresql":
                page_columns = {
                    column["name"]: column
                    for column in inspect(connection).get_columns("knowledge_bases")
                }
                if any(
                    not page_columns[column_name]["nullable"]
                    for column_name in (
                        "total_pages",
                        "selected_pages",
                        "excluded_pages_json",
                    )
                ):
                    connection.exec_driver_sql(
                        "DROP TABLE IF EXISTS knowledge_bases_nullable"
                    )
                    connection.exec_driver_sql(
                        "CREATE TABLE knowledge_bases_nullable ("
                        "id VARCHAR(36) NOT NULL PRIMARY KEY, "
                        "name VARCHAR(120) NOT NULL, "
                        "company_id VARCHAR(36) NOT NULL, "
                        "source VARCHAR(2048) NOT NULL, "
                        "mime_type VARCHAR(100) NOT NULL, "
                        "document_checksum VARCHAR(64), "
                        "total_pages INTEGER, "
                        "selected_pages INTEGER, "
                        "excluded_pages_json TEXT, "
                        "chunking_json TEXT NOT NULL, "
                        "chunk_count INTEGER NOT NULL, "
                        "status VARCHAR(32) NOT NULL, "
                        "created_at DATETIME NOT NULL, "
                        "updated_at DATETIME NOT NULL, "
                        "FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE"
                        ")"
                    )
                    connection.exec_driver_sql(
                        "INSERT INTO knowledge_bases_nullable ("
                        "id, name, company_id, source, mime_type, document_checksum, "
                        "total_pages, selected_pages, excluded_pages_json, chunking_json, "
                        "chunk_count, status, created_at, updated_at"
                        ") SELECT "
                        "id, name, company_id, source, mime_type, document_checksum, "
                        "total_pages, selected_pages, excluded_pages_json, chunking_json, "
                        "chunk_count, status, created_at, updated_at FROM knowledge_bases"
                    )
                    connection.exec_driver_sql("DROP TABLE knowledge_bases")
                    connection.exec_driver_sql(
                        "ALTER TABLE knowledge_bases_nullable RENAME TO knowledge_bases"
                    )
            connection.exec_driver_sql(
                "UPDATE knowledge_bases SET "
                "total_pages = NULL, selected_pages = NULL, excluded_pages_json = NULL "
                "WHERE mime_type IN ('text/html', 'application/xhtml+xml')"
            )
            connection.exec_driver_sql("DROP INDEX IF EXISTS ix_knowledge_bases_name")
            connection.exec_driver_sql("DROP INDEX IF EXISTS uq_knowledge_bases_name_ci")
            connection.exec_driver_sql("DROP INDEX IF EXISTS uq_knowledge_bases_name")
            connection.exec_driver_sql("DROP INDEX IF EXISTS uq_knowledge_bases_document_checksum")
            connection.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_bases_company_name "
                "ON knowledge_bases (company_id, name)"
            )
            connection.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_bases_company_checksum "
                "ON knowledge_bases (company_id, document_checksum)"
            )
            connection.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_bases_company_url_source "
                "ON knowledge_bases (company_id, source) "
                "WHERE mime_type IN ('text/html', 'application/xhtml+xml')"
            )
            connection.execute(
                update(KnowledgeBaseRow)
                .where(KnowledgeBaseRow.status == "ready")
                .values(status="enabled")
            )
            mapping_columns = [
                column["name"]
                for column in inspect(connection).get_columns("company_llm_mappings")
            ]
            expected_mapping_columns = [
                "id",
                "company_id",
                "provider",
                "model",
                "api_key_encrypted",
                "created_at",
                "updated_at",
            ]
            if mapping_columns != expected_mapping_columns:
                legacy_mappings = [
                    dict(row)
                    for row in connection.exec_driver_sql(
                        "SELECT company_id, provider, model, created_at, updated_at "
                        "FROM company_llm_mappings"
                    ).mappings()
                ]
                connection.exec_driver_sql("DROP TABLE company_llm_mappings")
                CompanyLLMMappingRow.__table__.create(connection)
                if legacy_mappings:
                    connection.execute(
                        CompanyLLMMappingRow.__table__.insert(),
                        [
                            {
                                "id": str(uuid4()),
                                **mapping,
                                "api_key_encrypted": None,
                            }
                            for mapping in legacy_mappings
                        ],
                    )

    @staticmethod
    def _to_domain(row: KnowledgeBaseRow) -> KnowledgeBase:
        return KnowledgeBase(
            id=row.id,
            company_id=row.company_id,
            name=row.name,
            source=row.source,
            mime_type=row.mime_type,
            total_pages=row.total_pages,
            excluded_pages=(
                json.loads(row.excluded_pages_json)
                if row.excluded_pages_json is not None
                else None
            ),
            selected_pages=row.selected_pages,
            chunking=json.loads(row.chunking_json),
            chunk_count=row.chunk_count,
            status=row.status,
            created_at=_as_utc(row.created_at),
            updated_at=_as_utc(row.updated_at),
            document_checksum=row.document_checksum,
        )

    def save(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        knowledge_base = KnowledgeBase.model_validate(knowledge_base.model_dump())
        with Session(self.engine) as session:
            try:
                existing = session.get(KnowledgeBaseRow, knowledge_base.id)
                stored = knowledge_base.model_copy(
                    update={
                        "created_at": (
                            _as_utc(existing.created_at) if existing else knowledge_base.created_at
                        ),
                        "updated_at": datetime.now(timezone.utc) if existing else knowledge_base.created_at,
                    }
                )
                values = stored.model_dump()
                excluded_pages = values.pop("excluded_pages")
                values["excluded_pages_json"] = (
                    json.dumps(excluded_pages) if excluded_pages is not None else None
                )
                values["chunking_json"] = json.dumps(values.pop("chunking"), default=str)
                session.merge(KnowledgeBaseRow(**values))
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                if knowledge_base.document_checksum and self.get_by_checksum(
                    knowledge_base.document_checksum, knowledge_base.company_id
                ):
                    raise DuplicateDocumentContent(
                        "This document content has already been imported"
                    ) from exc
                if self.name_exists(knowledge_base.name, knowledge_base.company_id):
                    raise DuplicateKnowledgeBaseName(
                        f'A knowledge base named "{knowledge_base.name}" already exists'
                    ) from exc
                raise
        return stored

    @staticmethod
    def _search_filter(search: str | None):
        term = search.strip() if search else ""
        if not term:
            return None
        pattern = f"%{term}%"
        return or_(
            KnowledgeBaseRow.name.ilike(pattern),
            KnowledgeBaseRow.source.ilike(pattern),
        )

    def list(
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        search: str | None = None,
        company_id: str | None = None,
    ) -> list[KnowledgeBase]:
        statement = select(KnowledgeBaseRow)
        search_filter = self._search_filter(search)
        if search_filter is not None:
            statement = statement.where(search_filter)
        if company_id:
            statement = statement.where(KnowledgeBaseRow.company_id == company_id)
        statement = statement.order_by(
            KnowledgeBaseRow.created_at.desc(), KnowledgeBaseRow.id.desc()
        ).offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
        with Session(self.engine) as session:
            rows = session.scalars(statement)
            return [self._to_domain(row) for row in rows]

    def count(self, *, search: str | None = None, company_id: str | None = None) -> int:
        statement = select(func.count()).select_from(KnowledgeBaseRow)
        search_filter = self._search_filter(search)
        if search_filter is not None:
            statement = statement.where(search_filter)
        if company_id:
            statement = statement.where(KnowledgeBaseRow.company_id == company_id)
        with Session(self.engine) as session:
            return session.scalar(statement) or 0

    def get(self, knowledge_base_id: str) -> KnowledgeBase | None:
        with Session(self.engine) as session:
            row = session.get(KnowledgeBaseRow, knowledge_base_id)
            return self._to_domain(row) if row else None

    def delete(self, knowledge_base_id: str) -> bool:
        with Session(self.engine) as session:
            row = session.get(KnowledgeBaseRow, knowledge_base_id)
            if not row:
                return False
            session.delete(row)
            session.commit()
            return True

    def name_exists(self, name: str, company_id: str) -> bool:
        normalized = " ".join(name.split()).upper()
        with Session(self.engine) as session:
            statement = (
                select(KnowledgeBaseRow.id)
                .where(
                    KnowledgeBaseRow.company_id == company_id,
                    KnowledgeBaseRow.name == normalized,
                )
                .limit(1)
            )
            return session.scalar(statement) is not None

    def get_by_checksum(self, checksum: str, company_id: str) -> KnowledgeBase | None:
        with Session(self.engine) as session:
            row = session.scalar(
                select(KnowledgeBaseRow)
                .where(
                    KnowledgeBaseRow.company_id == company_id,
                    KnowledgeBaseRow.document_checksum == checksum,
                )
                .limit(1)
            )
            return self._to_domain(row) if row else None

    def get_by_url_source(self, source: str, company_id: str) -> KnowledgeBase | None:
        with Session(self.engine) as session:
            row = session.scalar(
                select(KnowledgeBaseRow)
                .where(
                    KnowledgeBaseRow.company_id == company_id,
                    KnowledgeBaseRow.source == source,
                    KnowledgeBaseRow.mime_type.in_(WEB_MIME_TYPES),
                )
                .limit(1)
            )
            return self._to_domain(row) if row else None

    def replace(self, knowledge_base_id: str, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        knowledge_base = KnowledgeBase.model_validate(knowledge_base.model_dump())
        with Session(self.engine) as session:
            existing = session.get(KnowledgeBaseRow, knowledge_base_id)
            if not existing:
                raise ValueError("Knowledge base to replace was not found")
            stored = knowledge_base.model_copy(
                update={
                    "created_at": _as_utc(existing.created_at),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            values = stored.model_dump()
            excluded_pages = values.pop("excluded_pages")
            values["excluded_pages_json"] = (
                json.dumps(excluded_pages) if excluded_pages is not None else None
            )
            values["chunking_json"] = json.dumps(values.pop("chunking"), default=str)
            session.delete(existing)
            session.flush()
            session.add(KnowledgeBaseRow(**values))
            session.commit()
        return stored

    @staticmethod
    def _company_to_domain(row: CompanyRow) -> Company:
        return Company(
            id=row.id,
            name=row.name,
            logo_data=row.logo_data,
            logo_mime_type=row.logo_mime_type,
            about=row.about,
            phone=row.phone,
            email=row.email,
            address=row.address,
            maps_url=row.maps_url,
            bot_alias=row.bot_alias,
            bot_avatar=row.bot_avatar,
            bot_greet_message=row.bot_greet_message,
            created_at=_as_utc(row.created_at),
            updated_at=_as_utc(row.updated_at),
        )

    def save_company(self, company: Company) -> Company:
        company = company.model_copy(deep=True)
        with Session(self.engine) as session:
            try:
                existing = session.get(CompanyRow, company.id)
                stored = company.model_copy(
                    update={
                        "created_at": (
                            _as_utc(existing.created_at) if existing else company.created_at
                        ),
                        "updated_at": datetime.now(timezone.utc) if existing else company.created_at,
                    }
                )
                values = stored.model_dump(exclude={"logo_data", "has_logo"})
                values["logo_data"] = stored.logo_data
                session.merge(CompanyRow(**values))
                session.flush()
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise DuplicateCompanyName(
                    f'A company named "{company.name}" already exists'
                ) from exc
        return stored

    def list_companies(self) -> list[Company]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(CompanyRow).order_by(func.lower(CompanyRow.name), CompanyRow.name)
            )
            return [self._company_to_domain(row) for row in rows]

    def get_company(self, company_id: str) -> Company | None:
        with Session(self.engine) as session:
            row = session.get(CompanyRow, company_id)
            return self._company_to_domain(row) if row else None

    def _llm_mapping_to_domain(
        self, row: CompanyLLMMappingRow, *, include_api_key: bool = True
    ) -> CompanyLLMMapping:
        api_key = (
            self.credentials.decrypt(row.api_key_encrypted)
            if row.api_key_encrypted and include_api_key
            else None
        )
        return CompanyLLMMapping(
            id=row.id,
            company_id=row.company_id,
            provider=row.provider,
            model=row.model,
            api_key=api_key,
            has_api_key=bool(row.api_key_encrypted),
            created_at=_as_utc(row.created_at),
            updated_at=_as_utc(row.updated_at),
        )

    def save_company_llm_mapping(self, mapping: CompanyLLMMapping) -> CompanyLLMMapping:
        mapping = CompanyLLMMapping.model_validate(
            {**mapping.model_dump(), "api_key": mapping.api_key}
        )
        with Session(self.engine) as session:
            if not session.get(CompanyRow, mapping.company_id):
                raise ValueError("Company not found")
            existing = session.scalar(
                select(CompanyLLMMappingRow).where(
                    CompanyLLMMappingRow.company_id == mapping.company_id
                )
            )
            created_at = _as_utc(existing.created_at) if existing else mapping.created_at
            mapping_id = existing.id if existing else mapping.id
            provider = mapping.provider.value
            if mapping.provider == LLMProvider.OLLAMA:
                api_key_encrypted = None
            elif mapping.api_key and mapping.api_key.strip():
                api_key_encrypted = self.credentials.encrypt(mapping.api_key.strip())
            elif existing and existing.provider == provider and existing.api_key_encrypted:
                api_key_encrypted = existing.api_key_encrypted
            else:
                raise ValueError(f'An API key is required for provider "{provider}"')
            stored = mapping.model_copy(
                update={
                    "id": mapping_id,
                    "api_key": mapping.api_key,
                    "has_api_key": bool(api_key_encrypted),
                    "created_at": created_at,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            session.merge(
                CompanyLLMMappingRow(
                    id=stored.id,
                    company_id=stored.company_id,
                    provider=provider,
                    model=stored.model,
                    api_key_encrypted=api_key_encrypted,
                    created_at=stored.created_at,
                    updated_at=stored.updated_at,
                )
            )
            session.commit()
            return stored

    def get_company_llm_mapping(
        self, company_id: str, *, include_api_key: bool = True
    ) -> CompanyLLMMapping | None:
        with Session(self.engine) as session:
            row = session.scalar(
                select(CompanyLLMMappingRow).where(
                    CompanyLLMMappingRow.company_id == company_id
                )
            )
            return (
                self._llm_mapping_to_domain(row, include_api_key=include_api_key)
                if row
                else None
            )

    def delete_company(self, company_id: str) -> bool:
        with Session(self.engine) as session:
            company = session.get(CompanyRow, company_id)
            if not company:
                return False
            session_ids = select(ChatSessionRow.id).where(ChatSessionRow.company_id == company_id)
            session.execute(
                delete(ChatSessionMessageRow).where(
                    ChatSessionMessageRow.session_id.in_(session_ids)
                )
            )
            session.execute(delete(ChatSessionRow).where(ChatSessionRow.company_id == company_id))
            session.execute(
                delete(CompanyLLMMappingRow).where(
                    CompanyLLMMappingRow.company_id == company_id
                )
            )
            for knowledge_base in session.scalars(
                select(KnowledgeBaseRow).where(KnowledgeBaseRow.company_id == company_id)
            ):
                session.delete(knowledge_base)
            session.delete(company)
            session.commit()
            return True

    def company_name_exists(self, name: str, exclude_id: str | None = None) -> bool:
        normalized = " ".join(name.split()).lower()
        statement = select(CompanyRow.id).where(func.lower(CompanyRow.name) == normalized)
        if exclude_id:
            statement = statement.where(CompanyRow.id != exclude_id)
        with Session(self.engine) as session:
            return session.scalar(statement.limit(1)) is not None

    @staticmethod
    def _chat_session_to_domain(row: ChatSessionRow, message_count: int = 0) -> ChatSession:
        return ChatSession(
            id=row.id,
            company_id=row.company_id,
            status=row.status,
            ip_address=row.ip_address,
            created_at=row.created_at,
            ended_at=row.ended_at,
            message_count=message_count,
        )

    @staticmethod
    def _chat_message_to_domain(row: ChatSessionMessageRow) -> ChatSessionMessage:
        return ChatSessionMessage(
            id=row.id,
            session_id=row.session_id,
            owner=row.owner,
            content=row.content,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            created_at=row.created_at,
        )

    def create_chat_session(
        self, chat_session: ChatSession, greeting: str | None = None
    ) -> tuple[ChatSession, bool]:
        chat_session = ChatSession.model_validate(chat_session.model_dump())
        with Session(self.engine) as session:
            existing = session.get(ChatSessionRow, chat_session.id)
            if existing:
                if existing.company_id != chat_session.company_id:
                    raise ValueError("Chat session belongs to a different company")
                count = (
                    session.scalar(
                        select(func.count())
                        .select_from(ChatSessionMessageRow)
                        .where(ChatSessionMessageRow.session_id == existing.id)
                    )
                    or 0
                )
                return self._chat_session_to_domain(existing, count), False
            values = chat_session.model_dump(exclude={"message_count"})
            values["status"] = str(chat_session.status)
            row = ChatSessionRow(**values)
            session.add(row)
            # Persist the parent first; without mapped ORM relationships SQLAlchemy
            # cannot infer that the greeting row depends on this session row.
            session.flush()
            if greeting and greeting.strip():
                session.add(
                    ChatSessionMessageRow(
                        id=str(uuid4()),
                        session_id=chat_session.id,
                        owner=str(ChatSessionOwner.BOT),
                        content=greeting.strip(),
                        created_at=chat_session.created_at,
                    )
                )
            session.commit()
            return self._chat_session_to_domain(
                row, 1 if greeting and greeting.strip() else 0
            ), True

    def close_chat_session(self, session_id: str, status: ChatSessionStatus) -> ChatSession | None:
        if status == ChatSessionStatus.ACTIVE:
            raise ValueError("A completed session must be closed or timed out")
        with Session(self.engine) as session:
            row = session.get(ChatSessionRow, session_id)
            if not row:
                return None
            if row.status == str(ChatSessionStatus.ACTIVE):
                row.status = str(status)
                row.ended_at = datetime.now(timezone.utc)
                session.commit()
            count = (
                session.scalar(
                    select(func.count())
                    .select_from(ChatSessionMessageRow)
                    .where(ChatSessionMessageRow.session_id == row.id)
                )
                or 0
            )
            return self._chat_session_to_domain(row, count)

    def timeout_expired_chat_sessions(self, created_before: datetime) -> int:
        with Session(self.engine) as session:
            result = session.execute(
                update(ChatSessionRow)
                .where(
                    ChatSessionRow.status == str(ChatSessionStatus.ACTIVE),
                    ChatSessionRow.created_at <= created_before,
                )
                .values(
                    status=str(ChatSessionStatus.TIMEOUT),
                    ended_at=datetime.now(timezone.utc),
                )
            )
            session.commit()
            return result.rowcount or 0

    def add_chat_message(self, message: ChatSessionMessage) -> ChatSessionMessage:
        message = ChatSessionMessage.model_validate(message.model_dump())
        with Session(self.engine) as session:
            chat_session = session.get(ChatSessionRow, message.session_id)
            if not chat_session:
                raise ValueError("Chat session not found")
            values = message.model_dump()
            values["owner"] = str(message.owner)
            session.add(ChatSessionMessageRow(**values))
            session.commit()
        return message

    def get_chat_session(self, session_id: str) -> ChatSession | None:
        with Session(self.engine) as session:
            row = session.get(ChatSessionRow, session_id)
            if not row:
                return None
            count = (
                session.scalar(
                    select(func.count())
                    .select_from(ChatSessionMessageRow)
                    .where(ChatSessionMessageRow.session_id == row.id)
                )
                or 0
            )
            return self._chat_session_to_domain(row, count)

    def list_chat_sessions(
        self,
        company_id: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[ChatSession]:
        statement = (
            select(ChatSessionRow, func.count(ChatSessionMessageRow.id))
            .outerjoin(
                ChatSessionMessageRow,
                ChatSessionMessageRow.session_id == ChatSessionRow.id,
            )
            .where(ChatSessionRow.company_id == company_id)
            .group_by(ChatSessionRow.id)
            .order_by(ChatSessionRow.created_at.desc(), ChatSessionRow.id.desc())
            .offset(offset)
        )
        if created_from:
            statement = statement.where(ChatSessionRow.created_at >= created_from)
        if created_to:
            statement = statement.where(ChatSessionRow.created_at < created_to)
        if limit is not None:
            statement = statement.limit(limit)
        with Session(self.engine) as session:
            return [
                self._chat_session_to_domain(row, count)
                for row, count in session.execute(statement)
            ]

    def count_chat_sessions(
        self,
        company_id: str,
        *,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> int:
        statement = (
            select(func.count())
            .select_from(ChatSessionRow)
            .where(ChatSessionRow.company_id == company_id)
        )
        if created_from:
            statement = statement.where(ChatSessionRow.created_at >= created_from)
        if created_to:
            statement = statement.where(ChatSessionRow.created_at < created_to)
        with Session(self.engine) as session:
            return session.scalar(statement) or 0

    def get_chat_analytics(
        self,
        company_id: str,
        *,
        created_from: datetime,
        created_to: datetime,
    ) -> ChatAnalytics:
        day_expression = func.date(ChatSessionRow.created_at)
        statement = (
            select(
                day_expression.label("day"),
                func.count(func.distinct(ChatSessionRow.id)).label("session_count"),
                func.count(ChatSessionMessageRow.id).label("message_count"),
                func.coalesce(func.sum(ChatSessionMessageRow.input_tokens), 0).label(
                    "input_tokens"
                ),
                func.coalesce(func.sum(ChatSessionMessageRow.output_tokens), 0).label(
                    "output_tokens"
                ),
            )
            .select_from(ChatSessionRow)
            .outerjoin(
                ChatSessionMessageRow,
                ChatSessionMessageRow.session_id == ChatSessionRow.id,
            )
            .where(
                ChatSessionRow.company_id == company_id,
                ChatSessionRow.created_at >= created_from,
                ChatSessionRow.created_at < created_to,
            )
            .group_by(day_expression)
            .order_by(day_expression)
        )
        with Session(self.engine) as session:
            rows = session.execute(statement).all()
        by_day = {
            str(day): (int(session_count), int(message_count), int(input_tokens), int(output_tokens))
            for day, session_count, message_count, input_tokens, output_tokens in rows
        }
        daily: list[ChatAnalyticsDaily] = []
        current = created_from.date()
        final = created_to.date()
        while current < final:
            values = by_day.get(current.isoformat(), (0, 0, 0, 0))
            daily.append(
                ChatAnalyticsDaily(
                    date=current.isoformat(),
                    session_count=values[0],
                    message_count=values[1],
                    input_tokens=values[2],
                    output_tokens=values[3],
                )
            )
            current += timedelta(days=1)
        session_count = sum(point.session_count for point in daily)
        message_count = sum(point.message_count for point in daily)
        return ChatAnalytics(
            company_id=company_id,
            date_from=created_from.date().isoformat(),
            date_to=(created_to.date() - timedelta(days=1)).isoformat(),
            session_count=session_count,
            message_count=message_count,
            average_messages_per_session=(
                round(message_count / session_count, 2) if session_count else 0
            ),
            input_tokens=sum(point.input_tokens for point in daily),
            output_tokens=sum(point.output_tokens for point in daily),
            daily=daily,
        )

    def delete_chat_sessions(self, company_id: str, session_ids: list[str]) -> int:
        unique_ids = list(dict.fromkeys(session_ids))
        if not unique_ids:
            return 0
        with Session(self.engine) as session:
            owned_ids = list(
                session.scalars(
                    select(ChatSessionRow.id).where(
                        ChatSessionRow.company_id == company_id,
                        ChatSessionRow.id.in_(unique_ids),
                    )
                )
            )
            if not owned_ids:
                return 0
            session.execute(
                delete(ChatSessionMessageRow).where(
                    ChatSessionMessageRow.session_id.in_(owned_ids)
                )
            )
            session.execute(
                delete(ChatSessionRow).where(ChatSessionRow.id.in_(owned_ids))
            )
            session.commit()
            return len(owned_ids)

    def delete_chat_sessions_in_range(
        self,
        company_id: str,
        *,
        created_from: datetime,
        created_to: datetime,
    ) -> int:
        with Session(self.engine) as session:
            session_ids = list(
                session.scalars(
                    select(ChatSessionRow.id).where(
                        ChatSessionRow.company_id == company_id,
                        ChatSessionRow.created_at >= created_from,
                        ChatSessionRow.created_at < created_to,
                    )
                )
            )
            if not session_ids:
                return 0
            session.execute(
                delete(ChatSessionMessageRow).where(
                    ChatSessionMessageRow.session_id.in_(session_ids)
                )
            )
            session.execute(
                delete(ChatSessionRow).where(ChatSessionRow.id.in_(session_ids))
            )
            session.commit()
            return len(session_ids)

    def list_chat_messages(self, session_id: str) -> list[ChatSessionMessage]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(ChatSessionMessageRow)
                .where(ChatSessionMessageRow.session_id == session_id)
                .order_by(
                    ChatSessionMessageRow.created_at,
                    ChatSessionMessageRow.id,
                )
            )
            return [self._chat_message_to_domain(row) for row in rows]
