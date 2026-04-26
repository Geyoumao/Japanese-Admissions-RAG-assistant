from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppSettings:
    project_root: Path
    data_dir: Path
    pdf_dir: Path
    index_dir: Path
    sqlite_path: Path
    qdrant_path: Path
    collection_name: str
    embedding_model: str
    embedding_dimension: int
    reranker_model: str
    enable_reranker: bool
    deepseek_base_url: str
    deepseek_model: str
    api_key: str
    llm_timeout_seconds: int
    ocr_text_threshold: int
    chunk_size: int
    chunk_overlap: int
    bm25_top_k: int
    dense_top_k: int
    coarse_top_k: int
    final_top_n: int

    @classmethod
    def from_env(cls) -> "AppSettings":
        project_root = Path(__file__).resolve().parent.parent
        load_env_file(project_root / ".env")
        data_dir = project_root / "data"
        pdf_dir = data_dir / "pdfs"
        index_dir = data_dir / "index"
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            pdf_dir=pdf_dir,
            index_dir=index_dir,
            sqlite_path=index_dir / "rag.sqlite3",
            qdrant_path=index_dir / "qdrant",
            collection_name=os.getenv("RAG_COLLECTION_NAME", "admissions_chunks"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            embedding_dimension=_env_int("RAG_EMBEDDING_DIM", 384),
            reranker_model=os.getenv("RAG_RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual"),
            enable_reranker=_env_bool("ENABLE_RERANKER", True),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=os.getenv("API_KEY", ""),
            llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 120),
            ocr_text_threshold=_env_int("OCR_TEXT_THRESHOLD", 80),
            chunk_size=_env_int("CHUNK_SIZE", 420),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 90),
            bm25_top_k=_env_int("BM25_TOP_K", 20),
            dense_top_k=_env_int("DENSE_TOP_K", 20),
            coarse_top_k=_env_int("COARSE_TOP_K", 30),
            final_top_n=_env_int("FINAL_TOP_N", 5),
        )

    def ensure_directories(self) -> None:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
