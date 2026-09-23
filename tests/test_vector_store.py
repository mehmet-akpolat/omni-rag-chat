import pytest

from packages.omni_rag_core.domain import Chunk
from packages.omni_rag_core.vector_store import (
    MemoryVectorStore,
    ResilientVectorStore,
    VectorStore,
    sparse_vector,
)


def a_chunk(identifier="one", kb_id="guide-id", company_id="company-id"):
    return Chunk(
        id=identifier,
        knowledge_base_id=kb_id,
        company_id=company_id,
        text="refund policy details",
        metadata={"page_number": 3},
    )


def test_sparse_vector_is_sorted_and_counts_terms():
    indices, values = sparse_vector("alpha beta alpha")
    assert indices == sorted(indices)
    assert len(indices) == len(values) == 2
    assert max(values) > 1


def test_memory_store_upsert_search_filter_and_replace():
    store = MemoryVectorStore()
    store.upsert([a_chunk(), a_chunk("two", "other-id")], [[1, 0], [0, 1]])
    results = store.search("refund", [1, 0], ["guide-id"], "company-id", 3)
    assert results[0][0].id == "one"
    store.upsert([a_chunk()], [[0, 1]])
    assert len(store.items) == 2
    assert store.search("none", [1, 0], ["missing-id"], "company-id") == []


def test_memory_store_always_searches_company_profile_with_tenant_isolation():
    store = MemoryVectorStore()
    profile = a_chunk("profile", "company-id").model_copy(
        update={
            "source_type": "company_metadata",
            "text": "Address: 1 Orbit Way",
            "metadata": {"section": "Address"},
        }
    )
    other = profile.model_copy(update={"id": "other-profile", "company_id": "other-company"})
    store.upsert([profile, other], [[1, 0], [1, 0]])
    results = store.search("address", [1, 0], [], "company-id")
    assert [chunk.id for chunk, _ in results] == ["profile"]
    store.delete_company_metadata("company-id")
    assert [chunk.company_id for chunk, _ in store.items] == ["other-company"]


class StubStore(VectorStore):
    def __init__(self, fail=False):
        self.fail = fail
        self.upserts = 0

    def upsert(self, chunks, embeddings):
        self.upserts += 1
        if self.fail:
            raise RuntimeError("offline")

    def search(self, query, embedding, knowledge_base_ids, company_id, limit=6):
        if self.fail:
            raise RuntimeError("offline")
        return [(a_chunk(), 0.9)]

    def delete_knowledge_base(self, company_id, knowledge_base_id):
        if self.fail:
            raise RuntimeError("offline")
        self.deleted = knowledge_base_id

    def delete_company(self, company_id):
        if self.fail:
            raise RuntimeError("offline")
        self.deleted_company = company_id

    def delete_company_metadata(self, company_id):
        if self.fail:
            raise RuntimeError("offline")
        self.deleted_metadata = company_id


def test_resilient_store_primary_and_fallback():
    chunk = a_chunk()
    online = StubStore()
    store = ResilientVectorStore(online)
    store.upsert([chunk], [[1, 0]])
    assert online.upserts == 1
    assert store.search("refund", [1, 0], ["guide-id"], "company-id")[0][1] == 0.9
    store.delete_knowledge_base("company-id", "guide-id")
    assert online.deleted == "guide-id"
    assert store.fallback.items == []
    offline = StubStore(True)
    fallback = ResilientVectorStore(offline)
    fallback.upsert([chunk], [[1, 0]])
    assert fallback.search("refund", [1, 0], ["guide-id"], "company-id")[0][0].id == "one"
    with pytest.raises(RuntimeError, match="offline"):
        fallback.delete_knowledge_base("company-id", "guide-id")


def test_resilient_metadata_delete_always_cleans_fallback():
    profile = a_chunk("profile", "company-id").model_copy(
        update={"source_type": "company_metadata"}
    )
    store = ResilientVectorStore(StubStore(True))
    store.fallback.upsert([profile], [[1, 0]])
    with pytest.raises(RuntimeError, match="offline"):
        store.delete_company_metadata("company-id")
    assert store.fallback.items == []


def test_resilient_strict_upsert_propagates_primary_failure():
    chunk = a_chunk()
    store = ResilientVectorStore(StubStore(True))
    with pytest.raises(RuntimeError, match="offline"):
        store.upsert_strict([chunk], [[1, 0]])
    assert store.fallback.items[0][0].id == chunk.id
