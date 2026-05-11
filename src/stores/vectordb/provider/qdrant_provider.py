"""
Phase 1 — Vector DB Wrapper: Qdrant Provider
"""

import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

class QdrantProvider:
    def __init__(
        self,
        db_path: str = "assets/db/qdrant_data",
        collection_name: str = "job_documents",
        vector_size: int = 768,          # changed from 3072 to 768 (nomic-embed-text)
        distance: Distance = Distance.COSINE,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.client = QdrantClient(path=db_path)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )
            print(f"[qdrant] Created collection '{self.collection_name}' "
                  f"(dim={self.vector_size}, metric={self.distance})")
        else:
            count = self.client.count(self.collection_name).count
            print(f"[qdrant] Using existing collection '{self.collection_name}' "
                  f"({count} points already stored)")

    def upsert_chunks(self, chunks: list, vectors: list[list[float]]):
        assert len(chunks) == len(vectors), "chunks and vectors must have same length"

        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source_file": chunk.source_file,
                    "page": chunk.page,
                    "language": getattr(chunk, 'language', 'en'),
                    "chunk_index": getattr(chunk, 'chunk_index', 0),
                    **getattr(chunk, 'metadata', {})
                }
            ))

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        print(f"[qdrant] Stored {len(points)} points in '{self.collection_name}'")

    def count(self) -> int:
        """Return the total number of stored points."""
        try:
            return self.client.count(self.collection_name).count
        except:
            return 0

    def search(self, query_vector: list[float], top_k: int = 5, language_filter: Optional[str] = None):
        search_filter = None
        if language_filter:
            search_filter = Filter(
                must=[FieldCondition(key="language", match=MatchValue(value=language_filter))]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )

        return [
            {
                "score": r.score,
                "text": r.payload.get("text", ""),
                "source_file": r.payload.get("source_file", ""),
                "page": r.payload.get("page"),
                "language": r.payload.get("language", ""),
            }
            for r in results
        ]

    def reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        self._ensure_collection()
        print(f"[qdrant] Collection '{self.collection_name}' has been reset.")