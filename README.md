# Japanese Admissions RAG Assistant

This project is a local RAG MVP for Japanese university admissions PDFs.

Pipeline:

- PDF text extraction with `PyMuPDF`
- OCR fallback with `PaddleOCR`
- Japanese tokenization with `SudachiPy`
- Dense embeddings with `intfloat/multilingual-e5-small`
- BM25 keyword search with `SQLite FTS5`
- Dense vector search with local `Qdrant`
- RRF fusion + rerank
- Final Chinese answer generation with DeepSeek API

Answers include:

- PDF file name
- Page number
- Japanese evidence text

## Project Structure

- `app/ingest.py`: CLI ingestion entrypoint
- `app/ui.py`: Streamlit UI
- `app/service.py`: end-to-end ingestion and QA flow
- `app/parser.py`: PDF parsing and OCR fallback
- `app/stores.py`: SQLite FTS5 and Qdrant storage

## 1. Create a Virtual Environment

```powershell
pip install -r requirements.txt
```

## 2. Configure `.env`

Create or edit `.env` in the project root:

```env
API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

## 3. Put PDFs into `data/pdfs`

You can either:

- copy PDF files into `data/pdfs/`
- or upload them from the Streamlit sidebar

## 4. Build the Index

```powershell
python -m app.ingest --input data/pdfs --rebuild
```

## 5. Start the UI

```powershell
streamlit run app/ui.py
```

## Optional Environment Variables

```env
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-small
RAG_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
LLM_TIMEOUT_SECONDS=120
ENABLE_RERANKER=0
CHUNK_SIZE=420
CHUNK_OVERLAP=90
BM25_TOP_K=20
DENSE_TOP_K=20
COARSE_TOP_K=30
FINAL_TOP_N=5
OCR_TEXT_THRESHOLD=80
ENABLE_RERANKER=0


```

## Notes

- Query rewrite and final answer generation both use DeepSeek API.
- If DeepSeek is unavailable, the app falls back to a simple local answer formatter instead of crashing.
- Model downloads are still needed for embeddings, reranking, and OCR on first run.
- If startup or downloads are too slow, set `ENABLE_RERANKER=0` to skip the reranker model first.
