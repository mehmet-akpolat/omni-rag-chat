import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter

from qdrant_client import QdrantClient, models

from .domain import Chunk
from .settings import Settings


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def upsert_strict(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist vectors and propagate failures to consistency-sensitive callers."""
        self.upsert(chunks, embeddings)

    @abstractmethod
    def search(
        self,
        query: str,
        embedding: list[float],
        knowledge_base_ids: list[str],
        company_id: str,
        limit: int = 6,
    ) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def delete_knowledge_base(self, company_id: str, knowledge_base_id: str) -> None: ...

    @abstractmethod
    def delete_company(self, company_id: str) -> None: ...

    @abstractmethod
    def delete_company_metadata(self, company_id: str) -> None: ...


def sparse_vector(text: str, dimensions: int = 2**18) -> tuple[list[int], list[float]]:
    counts = Counter(re.findall(r"[a-z0-9]+", text.lower()))
    entries: dict[int, float] = {}
    for token, count in counts.items():
        index = (
            int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            % dimensions
        )
        entries[index] = entries.get(index, 0.0) + 1.0 + math.log(count)
    ordered = sorted(entries.items())
    return [item[0] for item in ordered], [item[1] for item in ordered]


class MemoryVectorStore(VectorStore):
    def __init__(self):
        self.items: list[tuple[Chunk, list[float]]] = []

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ids = {chunk.id for chunk in chunks}
        self.items = [item for item in self.items if item[0].id not in ids]
        self.items.extend(zip(chunks, embeddings, strict=True))

    def search(
        self,
        query: str,
        embedding: list[float],
        knowledge_base_ids: list[str],
        company_id: str,
        limit: int = 6,
    ) -> list[tuple[Chunk, float]]:
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked: list[tuple[Chunk, float]] = []
        for chunk, vector in self.items:
            if chunk.company_id != company_id:
                continue
            if (
                chunk.source_type != "company_metadata"
                and chunk.knowledge_base_id not in knowledge_base_ids
            ):
                continue
            dense = sum(a * b for a, b in zip(embedding, vector, strict=False))
            words = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
            lexical = len(tokens & words) / max(1, len(tokens))
            ranked.append((chunk, 0.7 * dense + 0.3 * lexical))
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]

    def delete_knowledge_base(self, company_id: str, knowledge_base_id: str) -> None:
        self.items = [
            item
            for item in self.items
            if not (
                item[0].company_id == company_id and item[0].knowledge_base_id == knowledge_base_id
            )
        ]

    def delete_company(self, company_id: str) -> None:
        self.items = [item for item in self.items if item[0].company_id != company_id]

    def delete_company_metadata(self, company_id: str) -> None:
        self.items = [
            item
            for item in self.items
            if not (item[0].company_id == company_id and item[0].source_type == "company_metadata")
        ]


class QdrantVectorStore(VectorStore):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.settings.qdrant_collection):
            return
        self.client.create_collection(
            self.settings.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"lexical": models.SparseVectorParams()},
        )

    @staticmethod
    def _payload(chunk: Chunk) -> dict:
        metadata = {
            key: value for key, value in chunk.metadata.items() if key not in {"index", "strategy"}
        }
        if chunk.source_type == "company_metadata":
            for key in ("page_number", "breadcrumb", "parent_id"):
                metadata.pop(key, None)
        return {
            "company_id": chunk.company_id,
            "knowledge_base_id": chunk.knowledge_base_id,
            "text": chunk.text,
            "source_type": chunk.source_type,
            "metadata": metadata,
        }

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._ensure_collection(len(embeddings[0]))
        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            indices, values = sparse_vector(chunk.text)
            points.append(
                models.PointStruct(
                    id=chunk.id,
                    vector={
                        "dense": embedding,
                        "lexical": models.SparseVector(indices=indices, values=values),
                    },
                    payload=self._payload(chunk),
                )
            )
        self.client.upsert(self.settings.qdrant_collection, points=points, wait=True)

    def search(
        self,
        query: str,
        embedding: list[float],
        knowledge_base_ids: list[str],
        company_id: str,
        limit: int = 6,
    ) -> list[tuple[Chunk, float]]:
        indices, values = sparse_vector(query)
        must = [models.FieldCondition(key="company_id", match=models.MatchValue(value=company_id))]
        if knowledge_base_ids:
            must.append(
                models.Filter(
                    should=[
                        models.FieldCondition(
                            key="knowledge_base_id",
                            match=models.MatchAny(any=knowledge_base_ids),
                        ),
                        models.FieldCondition(
                            key="source_type",
                            match=models.MatchValue(value="company_metadata"),
                        ),
                    ]
                )
            )
        else:
            must.append(
                models.FieldCondition(
                    key="source_type", match=models.MatchValue(value="company_metadata")
                )
            )
        query_filter = models.Filter(must=must)
        result = self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=embedding, using="dense", limit=limit * 2, filter=query_filter
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="lexical",
                    limit=limit * 2,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        ).points
        return [
            (
                Chunk.model_validate(
                    {
                        **point.payload,
                        **({"id": str(point.id)} if getattr(point, "id", None) is not None else {}),
                    }
                ),
                float(point.score),
            )
            for point in result
        ]

    def _delete_filter(self, query_filter: models.Filter) -> None:
        if not self.client.collection_exists(self.settings.qdrant_collection):
            return
        self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=models.FilterSelector(filter=query_filter),
            wait=True,
        )

    def delete_knowledge_base(self, company_id: str, knowledge_base_id: str) -> None:
        self._delete_filter(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="company_id", match=models.MatchValue(value=company_id)
                    ),
                    models.FieldCondition(
                        key="knowledge_base_id",
                        match=models.MatchValue(value=knowledge_base_id),
                    ),
                ]
            )
        )

    def delete_company(self, company_id: str) -> None:
        self._delete_filter(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="company_id", match=models.MatchValue(value=company_id)
                    )
                ]
            )
        )

    def delete_company_metadata(self, company_id: str) -> None:
        self._delete_filter(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="company_id", match=models.MatchValue(value=company_id)
                    ),
                    models.FieldCondition(
                        key="source_type", match=models.MatchValue(value="company_metadata")
                    ),
                ]
            )
        )


class ResilientVectorStore(VectorStore):
    """Uses Qdrant when available and keeps a process-local fallback for development."""

    def __init__(self, primary: VectorStore, fallback: MemoryVectorStore | None = None):
        self.primary = primary
        self.fallback = fallback or MemoryVectorStore()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.fallback.upsert(chunks, embeddings)
        try:
            self.primary.upsert(chunks, embeddings)
        except Exception:
            return

    def upsert_strict(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.fallback.upsert(chunks, embeddings)
        self.primary.upsert(chunks, embeddings)

    def search(
        self,
        query: str,
        embedding: list[float],
        knowledge_base_ids: list[str],
        company_id: str,
        limit: int = 6,
    ) -> list[tuple[Chunk, float]]:
        try:
            return self.primary.search(query, embedding, knowledge_base_ids, company_id, limit)
        except Exception:
            return self.fallback.search(query, embedding, knowledge_base_ids, company_id, limit)

    def delete_knowledge_base(self, company_id: str, knowledge_base_id: str) -> None:
        self.primary.delete_knowledge_base(company_id, knowledge_base_id)
        self.fallback.delete_knowledge_base(company_id, knowledge_base_id)

    def delete_company(self, company_id: str) -> None:
        self.primary.delete_company(company_id)
        self.fallback.delete_company(company_id)

    def delete_company_metadata(self, company_id: str) -> None:
        try:
            self.primary.delete_company_metadata(company_id)
        finally:
            self.fallback.delete_company_metadata(company_id)
