from __future__ import annotations


def main() -> None:
    print("Run `streamlit run app/ui.py` to start the RAG assistant UI.")
    print("Run `python -m app.ingest --input data/pdfs --rebuild` to build the local index.")


if __name__ == "__main__":
    main()
