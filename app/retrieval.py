from __future__ import annotations

from dataclasses import dataclass

from app.embedder import E5Embedder
from app.models import QueryRewrite, RetrievedChunk
from app.ranking import reciprocal_rank_fusion
from app.reranker import TransformerReranker
from app.stores import KeywordStore, QdrantVectorStore
from app.text import JapaneseTokenizer, normalize_for_match


@dataclass(slots=True)
class HybridRetriever:
    keyword_store: KeywordStore
    vector_store: QdrantVectorStore
    embedder: E5Embedder
    reranker: TransformerReranker
    tokenizer: JapaneseTokenizer
    bm25_top_k: int = 20
    dense_top_k: int = 20
    coarse_top_k: int = 30
    final_top_n: int = 5

    def search(self, rewrite: QueryRewrite) -> list[RetrievedChunk]:
        effective_filters = self._resolve_filters(rewrite)
        rewrite.filters = effective_filters
        keyword_query = self._build_keyword_query(rewrite)
        bm25_hits = self.keyword_store.search(keyword_query, limit=self.bm25_top_k, filters=effective_filters)

        dense_query_text = " ".join(
            piece for piece in [rewrite.rewritten_ja, " ".join(rewrite.expanded_keywords), rewrite.original_zh] if piece
        )
        dense_vector = self.embedder.embed_query(dense_query_text)
        dense_hits = self.vector_store.search(dense_vector, limit=self.dense_top_k, filters=effective_filters)

        fused = reciprocal_rank_fusion([[item_id for item_id, _ in bm25_hits], [item_id for item_id, _ in dense_hits]])
        candidate_ids = [
            item_id for item_id, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)[: self.coarse_top_k]
        ]
        chunks = self.keyword_store.fetch_chunks(candidate_ids)

        merged: list[RetrievedChunk] = []
        bm25_map = dict(bm25_hits)
        dense_map = dict(dense_hits)
        for chunk_id in candidate_ids:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            merged.append(
                RetrievedChunk(
                    chunk=chunk,
                    bm25_score=bm25_map.get(chunk_id),
                    dense_score=dense_map.get(chunk_id),
                    rrf_score=fused.get(chunk_id),
                )
            )

        rerank_query = f"{rewrite.original_zh}\n{rewrite.rewritten_ja}".strip()
        rerank_scores = self.reranker.score(rerank_query, [item.chunk.text_ja for item in merged])
        for item, score in zip(merged, rerank_scores, strict=False):
            item.rerank_score = score
        merged.sort(key=lambda item: item.rerank_score if item.rerank_score is not None else -1.0, reverse=True)
        return merged[: self.final_top_n]

    def _build_keyword_query(self, rewrite: QueryRewrite) -> str:
        parts = [rewrite.rewritten_ja] + rewrite.expanded_keywords + self.tokenizer.tokenize(rewrite.rewritten_ja)
        seen: set[str] = set()
        ordered: list[str] = []
        for part in parts:
            cleaned = part.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            ordered.append(cleaned)
        return " ".join(ordered)

    def _resolve_filters(self, rewrite: QueryRewrite) -> dict[str, str]:
        effective = dict(rewrite.filters)
        catalog = self.keyword_store.list_documents()
        if not catalog:
            return effective

        combined_text = " ".join(
            part
            for part in [rewrite.original_zh, rewrite.rewritten_ja, *rewrite.expanded_keywords]
            if part
        )
        normalized_query = normalize_for_match(combined_text)

        matched_pdf = self._match_catalog_value(normalized_query, catalog, "pdf_name")
        matched_university = self._match_catalog_value(normalized_query, catalog, "university")
        matched_year = self._match_catalog_value(normalized_query, catalog, "year")

        if matched_pdf:
            effective.setdefault("pdf_name", matched_pdf["pdf_name"])
            if matched_pdf.get("university"):
                effective.setdefault("university", matched_pdf["university"])
            if matched_pdf.get("year"):
                effective.setdefault("year", matched_pdf["year"])
        if matched_university:
            effective.setdefault("university", matched_university["university"])
        if matched_year:
            effective.setdefault("year", matched_year["year"])
        return effective

    @staticmethod
    def _match_catalog_value(
        normalized_query: str,
        catalog: list[dict[str, str]],
        field_name: str,
    ) -> dict[str, str] | None:
        best_row: dict[str, str] | None = None
        best_length = 0
        for row in catalog:
            value = (row.get(field_name) or "").strip()
            if not value:
                continue
            normalized_value = normalize_for_match(value.rsplit(".", 1)[0] if field_name == "pdf_name" else value)
            if not normalized_value:
                continue
            if normalized_value in normalized_query:
                if len(normalized_value) > best_length:
                    best_row = row
                    best_length = len(normalized_value)
        return best_row
