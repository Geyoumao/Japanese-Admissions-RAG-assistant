import unittest
from pathlib import Path

from app.models import ChunkMetadata, DocumentChunk
from app.stores import KeywordStore


class KeywordStoreDeleteTests(unittest.TestCase):
    def test_delete_document_removes_only_selected_pdf(self) -> None:
        db_path = Path(__file__).resolve().parent / "delete_test.sqlite3"
        try:
            store = KeywordStore(db_path)
            store.clear()

            chunk_a = DocumentChunk(
                chunk_id="a:1:1",
                text_ja="試験日 A",
                tokens_ja=["試験日", "A"],
                metadata=ChunkMetadata(doc_id="a", pdf_name="a.pdf", page=1, university="A大学", year="2027"),
            )
            chunk_b = DocumentChunk(
                chunk_id="b:1:1",
                text_ja="試験日 B",
                tokens_ja=["試験日", "B"],
                metadata=ChunkMetadata(doc_id="b", pdf_name="b.pdf", page=1, university="B大学", year="2027"),
            )
            store.upsert_chunks([chunk_a, chunk_b])

            removed = store.delete_document("a.pdf")

            self.assertEqual(removed, 1)
            documents = store.list_documents()
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["pdf_name"], "b.pdf")
        finally:
            try:
                db_path.unlink(missing_ok=True)
            except PermissionError:
                pass


if __name__ == "__main__":
    unittest.main()
