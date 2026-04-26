from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from app.config import AppSettings
from app.embedder import E5Embedder
from app.llm import AnswerGenerator, QueryRewriteService
from app.models import AnswerResponse, ChunkMetadata, DocumentChunk
from app.parser import PDFParser
from app.retrieval import HybridRetriever
from app.reranker import TransformerReranker
from app.stores import KeywordStore, QdrantVectorStore
from app.text import JapaneseTokenizer, chunk_japanese_text, infer_university_and_year, normalize_text


@dataclass(slots=True)
class IngestionStats:
    pdf_count: int
    page_count: int
    chunk_count: int
    ocr_pages: int


class IngestionService:
    def __init__(
        self,
        parser: PDFParser,
        tokenizer: JapaneseTokenizer,
        embedder: E5Embedder,
        keyword_store: KeywordStore,
        vector_store: QdrantVectorStore,
        settings: AppSettings,
    ) -> None:
        self.parser = parser
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.keyword_store = keyword_store
        self.vector_store = vector_store
        self.settings = settings

    def rebuild_from_directory(self, input_dir: str | Path) -> IngestionStats:
        pdf_paths = sorted(Path(input_dir).glob("*.pdf"))
        self.keyword_store.clear()
        self.vector_store.clear()
        return self.ingest_paths(pdf_paths)

    def remove_document(self, pdf_name: str) -> int:
        removed_chunks = self.keyword_store.delete_document(pdf_name)
        if removed_chunks:
            self.vector_store.delete_document(pdf_name)
        return removed_chunks

    def ingest_paths(self, pdf_paths: list[Path]) -> IngestionStats:
        all_chunks: list[DocumentChunk] = []
        page_count = 0
        ocr_pages = 0
        for pdf_path in pdf_paths:
            pages = self.parser.extract_pages(pdf_path)
            full_text = normalize_text("\n".join(page.text for page in pages))
            first_page_text = pages[0].text if pages else ""
            university, year = infer_university_and_year(pdf_path.name, full_text, first_page_text=first_page_text)
            doc_id = self._build_doc_id(pdf_path, full_text)
            page_count += len(pages)
            ocr_pages += sum(1 for page in pages if page.used_ocr)
            for page in pages:
                if not page.text.strip():
                    continue
                chunks = chunk_japanese_text(
                    page.text,
                    chunk_size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                )
                for index, (section_title, chunk_text) in enumerate(chunks, start=1):
                    chunk_id = f"{doc_id}:{page.page_number}:{index}"
                    metadata = ChunkMetadata(
                        doc_id=doc_id,
                        pdf_name=pdf_path.name,
                        page=page.page_number,
                        section_title=section_title,
                        university=university,
                        year=year,
                    )
                    all_chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text_ja=chunk_text,
                            tokens_ja=self.tokenizer.tokenize(chunk_text),
                            metadata=metadata,
                        )
                    )
        vectors = self.embedder.embed_documents([chunk.text_ja for chunk in all_chunks]) if all_chunks else []
        self.keyword_store.upsert_chunks(all_chunks)
        self.vector_store.upsert_chunks(all_chunks, vectors)
        return IngestionStats(
            pdf_count=len(pdf_paths),
            page_count=page_count,
            chunk_count=len(all_chunks),
            ocr_pages=ocr_pages,
        )

    @staticmethod
    def _build_doc_id(pdf_path: Path, full_text: str) -> str:
        digest = hashlib.sha1()
        digest.update(str(pdf_path.resolve()).encode("utf-8"))
        digest.update(full_text[:2000].encode("utf-8"))
        return digest.hexdigest()


class RAGAssistant:
    def __init__(
        self,
        retriever: HybridRetriever,
        rewriter: QueryRewriteService,
        answerer: AnswerGenerator,
        keyword_store: KeywordStore,
    ) -> None:
        self.retriever = retriever
        self.rewriter = rewriter
        self.answerer = answerer
        self.keyword_store = keyword_store

    def answer(self, question_zh: str) -> AnswerResponse:
        rewrite = self.rewriter.rewrite(question_zh)
        if self._should_split_by_university(rewrite):
            return self._answer_by_university(question_zh, rewrite)
        evidence = self.retriever.search(rewrite)
        return self.answerer.answer(question_zh, rewrite, evidence)

    def _should_split_by_university(self, rewrite) -> bool:
        if rewrite.filters.get("university") or rewrite.filters.get("pdf_name"):
            return False
        universities = [row.get("university", "").strip() for row in self.keyword_store.list_documents()]
        universities = [name for name in universities if name]
        return len(set(universities)) > 1

    def _answer_by_university(self, question_zh: str, rewrite) -> AnswerResponse:
        catalog = self.keyword_store.list_documents()
        ordered_universities: list[str] = []
        for row in catalog:
            university = (row.get("university") or "").strip()
            if university and university not in ordered_universities:
                ordered_universities.append(university)

        section_answers: list[str] = []
        merged_citations = []
        merged_retrieved = []
        answered_universities: list[str] = []

        for university in ordered_universities:
            scoped_rewrite = replace(
                rewrite,
                expanded_keywords=list(rewrite.expanded_keywords),
                filters={**rewrite.filters, "university": university},
            )
            evidence = self.retriever.search(scoped_rewrite)
            if not evidence:
                continue
            response = self.answerer.answer(question_zh, scoped_rewrite, evidence)
            answered_universities.append(university)
            section_answers.append(f"【{university}】\n{response.answer_zh}")
            merged_citations.extend(response.citations)
            merged_retrieved.extend(response.retrieved)

        if not section_answers:
            evidence = self.retriever.search(rewrite)
            return self.answerer.answer(question_zh, rewrite, evidence)

        intro = (
            "\u5df2\u6839\u636e\u5f53\u524d\u5df2\u7d22\u5f15\u7684\u5b66\u6821\u6570\u91cf\uff0c"
            "\u5206\u522b\u6574\u7406\u5404\u6821\u7684\u76f8\u5173\u4fe1\u606f\u5982\u4e0b\uff1a"
        )
        return AnswerResponse(
            answer_zh=f"{intro}\n\n" + "\n\n".join(section_answers),
            citations=merged_citations,
            rewritten_query=rewrite,
            retrieved=merged_retrieved,
            notes={"reason": "multi_university_split", "universities": answered_universities},
        )


def build_services(settings: AppSettings) -> tuple[IngestionService, RAGAssistant, KeywordStore]:
    settings.ensure_directories()
    tokenizer = JapaneseTokenizer()
    embedder = E5Embedder(settings.embedding_model)
    parser = PDFParser(ocr_text_threshold=settings.ocr_text_threshold)
    keyword_store = KeywordStore(settings.sqlite_path)
    vector_store = QdrantVectorStore(settings.qdrant_path, settings.collection_name, embedder.dimension)
    reranker = TransformerReranker(settings.reranker_model, enabled=settings.enable_reranker)
    retriever = HybridRetriever(
        keyword_store=keyword_store,
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        tokenizer=tokenizer,
        bm25_top_k=settings.bm25_top_k,
        dense_top_k=settings.dense_top_k,
        coarse_top_k=settings.coarse_top_k,
        final_top_n=settings.final_top_n,
    )
    from app.llm import DeepSeekBackend

    llm_backend = DeepSeekBackend(
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        api_key=settings.api_key,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    ingestion = IngestionService(parser, tokenizer, embedder, keyword_store, vector_store, settings)
    assistant = RAGAssistant(retriever, QueryRewriteService(llm_backend), AnswerGenerator(llm_backend), keyword_store)
    return ingestion, assistant, keyword_store
