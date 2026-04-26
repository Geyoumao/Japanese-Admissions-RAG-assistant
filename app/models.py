from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkMetadata:
    doc_id: str
    pdf_name: str
    page: int
    section_title: str = ""
    university: str = ""
    year: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    text_ja: str
    tokens_ja: list[str]
    metadata: ChunkMetadata

    def token_string(self) -> str:
        return " ".join(self.tokens_ja)

    def to_payload(self) -> dict[str, Any]:
        payload = self.metadata.to_dict()
        payload["chunk_id"] = self.chunk_id
        payload["text_ja"] = self.text_ja
        payload["tokens_ja"] = self.token_string()
        return payload


@dataclass(slots=True)
class QueryRewrite:
    original_zh: str
    rewritten_ja: str
    expanded_keywords: list[str] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: DocumentChunk
    bm25_score: float | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass(slots=True)
class AnswerCitation:
    pdf_name: str
    page: int
    section_title: str
    quote_ja: str


@dataclass(slots=True)
class AnswerResponse:
    answer_zh: str
    citations: list[AnswerCitation]
    rewritten_query: QueryRewrite
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
