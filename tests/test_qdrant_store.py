from types import SimpleNamespace

from packages.omni_rag_core.domain import Chunk
from packages.omni_rag_core.settings import Settings
from packages.omni_rag_core.vector_store import QdrantVectorStore


class FakeClient:
    def __init__(self):
        self.exists = False
        self.created = 0
        self.points = []
        self.deleted = None

    def collection_exists(self, name):
        return self.exists

    def create_collection(self, *args, **kwargs):
        self.created += 1
        self.exists = True

    def upsert(self, name, points, wait):
        self.points = points

    def query_points(self, **kwargs):
        self.query_points_filter = kwargs["prefetch"][0].filter
        payload = {
            "knowledge_base_id": "kb-id",
            "company_id": "company-id",
            "text": "answer",
            "source_type": "knowledge_base",
            "metadata": {"page_number": 1, "section": "Refunds"},
        }
        return SimpleNamespace(points=[SimpleNamespace(id="point-id", payload=payload, score=0.8)])

    def delete(self, **kwargs):
        self.deleted = kwargs


def test_qdrant_upsert_ensure_and_hybrid_query():
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.settings = Settings(_env_file=None, qdrant_collection="test")
    store.client = FakeClient()
    chunk = Chunk(
        knowledge_base_id="kb-id",
        company_id="company-id",
        text="refund refund",
        metadata={
            "page_number": 1,
            "section": "Refunds",
            "breadcrumb": ["kb-id", "Refunds"],
        },
    )
    store.upsert([], [])
    store.upsert([chunk], [[1.0, 0.0]])
    assert store.client.created == 1
    assert len(store.client.points) == 1
    assert store.client.points[0].payload == {
        "company_id": "company-id",
        "knowledge_base_id": "kb-id",
        "text": "refund refund",
        "source_type": "knowledge_base",
        "metadata": {
            "page_number": 1,
            "section": "Refunds",
            "breadcrumb": ["kb-id", "Refunds"],
        },
    }
    store._ensure_collection(2)
    assert store.client.created == 1
    results = store.search("refund", [1.0, 0.0], ["kb-id"], "company-id", 2)
    assert results[0][0].text == "answer"
    assert results[0][1] == 0.8
    query_filter = store.client.query_points_filter
    assert query_filter.must[0].key == "company_id"
    store.delete_knowledge_base("company-id", "kb-id")
    assert store.client.deleted["wait"] is True
    delete_filter = store.client.deleted["points_selector"].filter
    assert [condition.key for condition in delete_filter.must] == [
        "company_id",
        "knowledge_base_id",
    ]
    store.client.exists = False
    store.client.deleted = None
    store.delete_knowledge_base("company-id", "kb-id")
    assert store.client.deleted is None


def test_qdrant_company_metadata_filter_and_delete():
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.settings = Settings(_env_file=None, qdrant_collection="test")
    store.client = FakeClient()
    store.client.exists = True

    profile = Chunk(
        knowledge_base_id="company-id",
        company_id="company-id",
        text="Address: 1 Orbit Way",
        source_type="company_metadata",
        metadata={
            "section": "Address",
            "page_number": 0,
            "breadcrumb": ["Address"],
            "parent_id": "unused",
            "index": 2,
            "strategy": "fixed",
        },
    )
    store.upsert([profile], [[1.0, 0.0]])
    assert store.client.points[0].payload == {
        "company_id": "company-id",
        "knowledge_base_id": "company-id",
        "text": "Address: 1 Orbit Way",
        "source_type": "company_metadata",
        "metadata": {"section": "Address"},
    }

    store.search("where", [1.0, 0.0], [], "company-id")
    query_filter = store.client.query_points_filter
    assert query_filter.must[0].key == "company_id"
    assert query_filter.must[1].key == "source_type"

    store.delete_company_metadata("company-id")
    delete_filter = store.client.deleted["points_selector"].filter
    assert [condition.key for condition in delete_filter.must] == ["company_id", "source_type"]
    store.delete_company("company-id")
    assert store.client.deleted["points_selector"].filter.must[0].key == "company_id"
