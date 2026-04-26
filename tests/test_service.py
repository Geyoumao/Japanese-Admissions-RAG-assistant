import unittest
import sys
import types

from app.models import AnswerCitation, AnswerResponse, ChunkMetadata, DocumentChunk, QueryRewrite, RetrievedChunk

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.post = None
    sys.modules["requests"] = requests_stub

from app.service import RAGAssistant


class FakeRetriever:
    def __init__(self, payload):
        self.payload = payload

    def search(self, rewrite):
        university = rewrite.filters.get("university", "")
        return self.payload.get(university, [])


class FakeRewriter:
    def rewrite(self, question_zh):
        return QueryRewrite(original_zh=question_zh, rewritten_ja=question_zh, expanded_keywords=[], filters={})


class FakeAnswerer:
    def answer(self, question_zh, rewrite, evidence):
        university = rewrite.filters.get("university") or evidence[0].chunk.metadata.university
        return AnswerResponse(
            answer_zh=f"{university} answer",
            citations=[
                AnswerCitation(
                    pdf_name=item.chunk.metadata.pdf_name,
                    page=item.chunk.metadata.page,
                    section_title=item.chunk.metadata.section_title,
                    quote_ja=item.chunk.text_ja,
                )
                for item in evidence
            ],
            rewritten_query=rewrite,
            retrieved=evidence,
        )


class FakeKeywordStore:
    def list_documents(self):
        return [
            {"pdf_name": "nitech_joho.pdf", "university": "名古屋工業大学", "year": "2027", "chunk_count": 10},
            {"pdf_name": "omu_joho.pdf", "university": "大阪公立大学", "year": "2027", "chunk_count": 12},
        ]


def _make_chunk(university, pdf_name):
    chunk = DocumentChunk(
        chunk_id=f"{pdf_name}:1:1",
        text_ja="試験日 2027-02-01",
        tokens_ja=["試験日"],
        metadata=ChunkMetadata(
            doc_id=pdf_name,
            pdf_name=pdf_name,
            page=1,
            section_title="試験日",
            university=university,
            year="2027",
        ),
    )
    return RetrievedChunk(chunk=chunk, bm25_score=1.0)


class RagAssistantTests(unittest.TestCase):
    def test_splits_answer_by_university_when_multiple_docs_exist(self):
        payload = {
            "名古屋工業大学": [_make_chunk("名古屋工業大学", "nitech_joho.pdf")],
            "大阪公立大学": [_make_chunk("大阪公立大学", "omu_joho.pdf")],
        }
        assistant = RAGAssistant(FakeRetriever(payload), FakeRewriter(), FakeAnswerer(), FakeKeywordStore())

        response = assistant.answer("考试时间是什么时候？")

        self.assertIn("【名古屋工業大学】", response.answer_zh)
        self.assertIn("【大阪公立大学】", response.answer_zh)
        self.assertEqual(response.notes.get("reason"), "multi_university_split")


if __name__ == "__main__":
    unittest.main()
