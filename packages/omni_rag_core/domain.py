from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

DEFAULT_BOT_GREET_MESSAGE = (
    "Hello! I'm {{bot_alias}}, the AI assistant for {{company_name}}. How can I help you today?"
)
GREETING_PLACEHOLDERS = frozenset({"bot_alias", "company_name"})
WEB_MIME_TYPES = frozenset({"text/html", "application/xhtml+xml"})


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class LLMSelection(BaseModel):
    provider: LLMProvider
    model: str = Field(min_length=1, max_length=200)


class CompanyLLMMapping(LLMSelection):
    id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    has_api_key: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _normalized_name(value: str) -> str:
    normalized = " ".join(value.split()).upper()
    if len(normalized) < 2:
        raise ValueError("Knowledge base name must contain at least 2 characters")
    if len(normalized) > 120:
        raise ValueError("Knowledge base name must contain at most 120 characters")
    return normalized


def _normalized_company_name(value: str) -> str:
    normalized = " ".join(value.split())
    if len(normalized) < 2 or len(normalized) > 120:
        raise ValueError("Company name must contain between 2 and 120 characters")
    return normalized


class BotAvatar(StrEnum):
    BOT = "bot"
    BRAIN = "brain"
    HEADSET = "headset"
    BOOK = "book"


class CompanyFields(BaseModel):
    name: str
    about: str = Field(default="", max_length=1000)
    phone: str = Field(default="", max_length=50)
    email: str = Field(default="", max_length=254)
    address: str = Field(default="", max_length=500)
    maps_url: str | None = Field(default=None, max_length=500)
    bot_alias: str = Field(default="AIBot", min_length=2, max_length=80)
    bot_avatar: BotAvatar = BotAvatar.BOT
    bot_greet_message: str = Field(default=DEFAULT_BOT_GREET_MESSAGE, max_length=400)

    @field_validator("name")
    @classmethod
    def normalize_company_name(cls, value: str) -> str:
        return _normalized_company_name(value)

    @field_validator("bot_greet_message")
    @classmethod
    def validate_greeting_template(cls, value: str) -> str:
        cursor = 0
        while cursor < len(value):
            opening = value.find("{{", cursor)
            closing = value.find("}}", cursor)
            if closing >= 0 and (opening < 0 or closing < opening):
                raise ValueError("Greeting template contains malformed placeholder syntax")
            if opening < 0:
                break
            closing = value.find("}}", opening + 2)
            if closing < 0:
                raise ValueError("Greeting template contains malformed placeholder syntax")
            keyword = value[opening + 2 : closing]
            if keyword not in GREETING_PLACEHOLDERS:
                raise ValueError(
                    f'Unsupported greeting placeholder "{keyword}"; '
                    "use only {{bot_alias}} or {{company_name}}"
                )
            cursor = closing + 2
        return value

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip()
        if value and ("@" not in value or value.startswith("@") or value.endswith("@")):
            raise ValueError("Enter a valid email address")
        return value


class CompanyInput(CompanyFields):
    pass


class Company(CompanyFields):
    id: str = Field(default_factory=lambda: str(uuid4()))
    logo_data: bytes | None = None
    logo_mime_type: str | None = Field(default=None, max_length=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def has_logo(self) -> bool:
        return bool(self.logo_data)


class CompanyView(CompanyFields):
    id: str
    logo_mime_type: str | None = None
    has_logo: bool
    created_at: datetime
    updated_at: datetime


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = Field(default=700, ge=100, le=4000)
    overlap: int = Field(default=100, ge=0, le=1000)
    similarity_threshold: float = Field(default=0.72, ge=0, le=1)
    parent_size: int = Field(default=1600, ge=300, le=8000)

    @model_validator(mode="after")
    def validate_sizes(self) -> "ChunkingConfig":
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if self.strategy == ChunkingStrategy.HIERARCHICAL and self.parent_size <= self.chunk_size:
            raise ValueError("parent_size must be larger than chunk_size")
        return self


class PageSelection(BaseModel):
    included_pages: list[int] = Field(default_factory=list)
    excluded_pages: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection_mode(self) -> "PageSelection":
        if self.included_pages and self.excluded_pages:
            raise ValueError("Use either included pages or excluded pages, not both")
        return self

    def resolve(self, total_pages: int) -> list[int]:
        valid = set(range(1, total_pages + 1))
        selected = set(self.included_pages) if self.included_pages else valid
        return sorted((selected & valid) - set(self.excluded_pages))


class ImportRequest(BaseModel):
    document_id: str
    company_id: str
    name: str = Field(min_length=2, max_length=120)
    pages: PageSelection = Field(default_factory=PageSelection)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _normalized_name(value)


class PagePreview(BaseModel):
    page_number: int
    excerpt: str


class DocumentPreview(BaseModel):
    document_id: str
    source: str
    mime_type: str
    total_pages: int
    pages: list[PagePreview]
    title: str | None = None
    replaces_knowledge_base_id: str | None = None


class KnowledgeBase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    source: str = Field(min_length=1, max_length=2048)
    mime_type: str
    total_pages: int | None = Field(default=None, ge=1)
    excluded_pages: list[int] | None = None
    selected_pages: int | None = Field(default=None, ge=0)
    chunking: ChunkingConfig
    chunk_count: int = 0
    status: str = "enabled"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    document_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    company_id: str

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _normalized_name(value)

    @model_validator(mode="after")
    def validate_page_summary(self) -> "KnowledgeBase":
        if self.mime_type in WEB_MIME_TYPES:
            if any(
                value is not None
                for value in (
                    self.total_pages,
                    self.selected_pages,
                    self.excluded_pages,
                )
            ):
                raise ValueError("URL knowledge bases cannot contain PDF page information")
            return self
        if self.total_pages is None or self.selected_pages is None:
            raise ValueError("PDF knowledge bases require total and selected page counts")
        excluded = sorted(set(self.excluded_pages or []))
        if any(page < 1 or page > self.total_pages for page in excluded):
            raise ValueError("Excluded pages must be within the document page range")
        if self.selected_pages != self.total_pages - len(excluded):
            raise ValueError("Selected page count does not match excluded pages")
        self.excluded_pages = excluded
        return self


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    knowledge_base_id: str
    company_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_type: str = "knowledge_base"

    @property
    def page_number(self) -> int:
        return int(self.metadata.get("page_number", 0))

    @property
    def section(self) -> str | None:
        return self.metadata.get("section")

    @property
    def breadcrumb(self) -> list[str]:
        return self.metadata.get("breadcrumb", [])

    @property
    def parent_id(self) -> str | None:
        return self.metadata.get("parent_id")


class Citation(BaseModel):
    source_number: int | None = Field(default=None, ge=1)
    knowledge_base_id: str
    knowledge_base: str
    page_number: int
    excerpt: str
    score: float
    section: str | None = None
    source_url: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatSessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"
    TIMEOUT = "timeout"


class ChatSessionOwner(StrEnum):
    BOT = "bot"
    USER = "user"


class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=64)
    company_id: str
    status: ChatSessionStatus = ChatSessionStatus.ACTIVE
    ip_address: str = Field(default="", max_length=45)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    message_count: int = 0


class ChatSessionMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    owner: ChatSessionOwner
    content: str = Field(min_length=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatAnalyticsDaily(BaseModel):
    date: str
    session_count: int = Field(ge=0)
    message_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ChatAnalytics(BaseModel):
    company_id: str
    date_from: str
    date_to: str
    session_count: int = Field(ge=0)
    message_count: int = Field(ge=0)
    average_messages_per_session: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    daily: list[ChatAnalyticsDaily] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(min_length=1, max_length=64)
    knowledge_base_ids: list[str] = Field(default_factory=list)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    suggested_replies: list[str] = Field(default_factory=list)
