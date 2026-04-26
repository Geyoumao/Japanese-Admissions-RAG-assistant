from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from app.models import ChunkMetadata, DocumentChunk


class KeywordStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    pdf_name TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    section_title TEXT NOT NULL,
                    university TEXT NOT NULL,
                    year TEXT NOT NULL,
                    text_ja TEXT NOT NULL,
                    tokens_ja TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    text_ja,
                    tokens_ja,
                    pdf_name,
                    section_title
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_university ON chunks(university)")
            conn.commit()

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunk_fts")
            conn.execute("DELETE FROM chunks")
            conn.commit()

    def delete_document(self, pdf_name: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS chunk_count FROM chunks WHERE pdf_name = ?",
                (pdf_name,),
            ).fetchone()
            chunk_count = int(row["chunk_count"]) if row else 0
            if chunk_count == 0:
                return 0
            conn.execute(
                "DELETE FROM chunk_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE pdf_name = ?)",
                (pdf_name,),
            )
            conn.execute("DELETE FROM chunks WHERE pdf_name = ?", (pdf_name,))
            conn.commit()
            return chunk_count

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        with self._connect() as conn:
            for chunk in chunks:
                conn.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk.chunk_id,))
                conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,))
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, pdf_name, page, section_title, university, year, text_ja, tokens_ja
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.metadata.doc_id,
                        chunk.metadata.pdf_name,
                        chunk.metadata.page,
                        chunk.metadata.section_title,
                        chunk.metadata.university,
                        chunk.metadata.year,
                        chunk.text_ja,
                        chunk.token_string(),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO chunk_fts (chunk_id, text_ja, tokens_ja, pdf_name, section_title)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.text_ja,
                        chunk.token_string(),
                        chunk.metadata.pdf_name,
                        chunk.metadata.section_title,
                    ),
                )
            conn.commit()

    def fetch_chunks(self, chunk_ids: list[str]) -> dict[str, DocumentChunk]:
        if not chunk_ids:
            return {}
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT chunk_id, doc_id, pdf_name, page, section_title, university, year, text_ja, tokens_ja
                FROM chunks
                WHERE chunk_id IN ({placeholders})
                """,
                tuple(chunk_ids),
            ).fetchall()
        chunks = {}
        for row in rows:
            metadata = ChunkMetadata(
                doc_id=row["doc_id"],
                pdf_name=row["pdf_name"],
                page=row["page"],
                section_title=row["section_title"],
                university=row["university"],
                year=row["year"],
            )
            chunks[row["chunk_id"]] = DocumentChunk(
                chunk_id=row["chunk_id"],
                text_ja=row["text_ja"],
                tokens_ja=row["tokens_ja"].split(),
                metadata=metadata,
            )
        return chunks

    def list_documents(self) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pdf_name, university, year, COUNT(*) AS chunk_count
                FROM chunks
                GROUP BY pdf_name, university, year
                ORDER BY pdf_name
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def search(self, query: str, limit: int = 20, filters: dict[str, str] | None = None) -> list[tuple[str, float]]:
        match_query = self._build_match_query(query)
        if not match_query:
            return []

        filters = filters or {}
        where_clauses = []
        parameters: list[object] = [match_query]
        if filters.get("university"):
            where_clauses.append("c.university = ?")
            parameters.append(filters["university"])
        if filters.get("year"):
            where_clauses.append("c.year = ?")
            parameters.append(filters["year"])
        if filters.get("pdf_name"):
            where_clauses.append("c.pdf_name = ?")
            parameters.append(filters["pdf_name"])
        where_sql = f"AND {' AND '.join(where_clauses)}" if where_clauses else ""
        parameters.append(limit)

        sql = f"""
            SELECT c.chunk_id, bm25(chunk_fts, 1.0, 2.0, 0.3, 0.2) AS bm25_score
            FROM chunk_fts
            JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id
            WHERE chunk_fts MATCH ?
            {where_sql}
            ORDER BY bm25_score ASC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(parameters)).fetchall()
        return [(row["chunk_id"], float(-row["bm25_score"])) for row in rows]

    def _build_match_query(self, query: str) -> str:
        tokens = [token.strip() for token in query.split() if token.strip()]
        if not tokens:
            tokens = [query.strip()]
        safe_tokens = [token.replace('"', "").replace("'", "") for token in tokens if token]
        return " OR ".join(f'"{token}"' for token in safe_tokens if token)


class QdrantVectorStore:
    def __init__(self, storage_path: str | Path, collection_name: str, vector_size: int) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(path=str(self.storage_path))

        collections = {item.name for item in self.client.get_collections().collections}
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def clear(self) -> None:
        from qdrant_client.http.models import Distance, VectorParams

        self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def delete_document(self, pdf_name: str) -> None:
        from qdrant_client.http import models as rest

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[rest.FieldCondition(key="pdf_name", match=rest.MatchValue(value=pdf_name))]
                )
            ),
            wait=True,
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        from qdrant_client.http.models import PointStruct

        points = [
            PointStruct(id=self._point_id(chunk.chunk_id), vector=vector, payload=chunk.to_payload())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], limit: int = 20, filters: dict[str, str] | None = None) -> list[tuple[str, float]]:
        from qdrant_client.http import models as rest

        query_filter = None
        filters = filters or {}
        must: list[rest.FieldCondition] = []
        for key in ("university", "year", "pdf_name"):
            value = filters.get(key)
            if value:
                must.append(rest.FieldCondition(key=key, match=rest.MatchValue(value=value)))
        if must:
            query_filter = rest.Filter(must=must)
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        results: list[tuple[str, float]] = []
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                results.append((str(chunk_id), float(point.score)))
        return results

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
