from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppSettings
from app.service import build_services


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Japanese admissions PDFs into the local RAG index.")
    parser.add_argument("--input", required=True, help="Directory that contains PDF files.")
    parser.add_argument("--rebuild", action="store_true", help="Clear and rebuild the index before ingestion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = AppSettings.from_env()
    ingestion, _, _ = build_services(settings)
    input_dir = Path(args.input)
    if args.rebuild:
        stats = ingestion.rebuild_from_directory(input_dir)
    else:
        stats = ingestion.ingest_paths(sorted(input_dir.glob("*.pdf")))
    print(
        f"Ingested {stats.pdf_count} PDFs, {stats.page_count} pages, "
        f"{stats.chunk_count} chunks, OCR used on {stats.ocr_pages} pages."
    )


if __name__ == "__main__":
    main()
