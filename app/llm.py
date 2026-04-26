from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

from app.models import AnswerCitation, AnswerResponse, QueryRewrite, RetrievedChunk


def _extract_json_block(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


@dataclass(slots=True)
class DeepSeekBackend:
    base_url: str
    model: str
    api_key: str
    timeout_seconds: int = 120

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("Missing API_KEY for DeepSeek API.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
            # Keep the assistant in non-thinking mode for lower latency and cost.
            "thinking": {"type": "disabled"},
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


class QueryRewriteService:
    def __init__(self, llm_backend: DeepSeekBackend) -> None:
        self.llm_backend = llm_backend

    def rewrite(self, question_zh: str) -> QueryRewrite:
        system_prompt = (
            "You rewrite Chinese questions for Japanese university admissions PDF retrieval. "
            "Return valid JSON only with keys: original_zh, rewritten_ja, expanded_keywords, filters. "
            "filters may only include university, year, pdf_name."
        )
        user_prompt = f"""
Analyze the Chinese question below and expand it into a better retrieval query for Japanese source documents.

Question:
{question_zh}

Requirements:
1. rewritten_ja should be short Japanese retrieval text or keywords.
2. expanded_keywords should include likely Japanese terms, synonyms, and key phrases.
3. filters should only be filled when the question explicitly mentions a university, year, or file name.
4. Output JSON only.
""".strip()
        try:
            raw = self.llm_backend.generate(
                system_prompt,
                user_prompt,
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=400,
            )
            data = _extract_json_block(raw)
            keywords = [str(item).strip() for item in data.get("expanded_keywords", []) if str(item).strip()]
            filters = {str(key): str(value) for key, value in dict(data.get("filters", {})).items() if str(value).strip()}
            return QueryRewrite(
                original_zh=str(data.get("original_zh") or question_zh).strip(),
                rewritten_ja=str(data.get("rewritten_ja") or question_zh).strip(),
                expanded_keywords=keywords,
                filters=filters,
            )
        except Exception:
            fallback_keywords = [
                token
                for token in question_zh.replace("\uFF0C", " ").replace("\u3002", " ").split()
                if token
            ]
            return QueryRewrite(
                original_zh=question_zh,
                rewritten_ja=question_zh,
                expanded_keywords=fallback_keywords,
                filters={},
            )


class AnswerGenerator:
    def __init__(self, llm_backend: DeepSeekBackend) -> None:
        self.llm_backend = llm_backend

    def answer(self, question_zh: str, rewrite: QueryRewrite, evidence: list[RetrievedChunk]) -> AnswerResponse:
        citations = [
            AnswerCitation(
                pdf_name=item.chunk.metadata.pdf_name,
                page=item.chunk.metadata.page,
                section_title=item.chunk.metadata.section_title,
                quote_ja=item.chunk.text_ja,
            )
            for item in evidence
        ]
        if not evidence:
            return AnswerResponse(
                answer_zh=(
                    "\u5f53\u524d\u8d44\u6599\u672a\u660e\u786e\u7ed9\u51fa\u8fd9\u4e2a\u95ee\u9898\u7684\u7b54\u6848\uff0c"
                    "\u8bf7\u6362\u4e00\u79cd\u95ee\u6cd5\uff0c\u6216\u5bfc\u5165\u66f4\u5b8c\u6574\u7684\u52df\u96c6\u8981\u9879 PDF\u3002"
                ),
                citations=[],
                rewritten_query=rewrite,
                retrieved=[],
                notes={"reason": "no_evidence"},
            )
        universities = sorted({item.chunk.metadata.university for item in evidence if item.chunk.metadata.university})
        if not rewrite.filters.get("university") and len(universities) > 1:
            ambiguous = "、".join(universities[:5])
            return AnswerResponse(
                answer_zh=(
                    f"\u5f53\u524d\u547d\u4e2d\u4e86\u591a\u6240\u5b66\u6821\u7684\u52df\u96c6\u8981\u9879\uff0c"
                    f"\u5305\u62ec\uff1a{ambiguous}\u3002"
                    "\u4e3a\u907f\u514d\u628a\u4e0d\u540c\u5b66\u6821\u7684\u6761\u4ef6\u6df7\u5728\u4e00\u8d77\uff0c"
                    "\u8bf7\u5728\u95ee\u9898\u91cc\u76f4\u63a5\u5199\u51fa\u5b66\u6821\u540d\u540e\u518d\u95ee\u4e00\u6b21\u3002"
                ),
                citations=citations,
                rewritten_query=rewrite,
                retrieved=evidence,
                notes={"reason": "ambiguous_university"},
            )

        evidence_blocks = []
        for index, item in enumerate(evidence, start=1):
            evidence_blocks.append(
                "\n".join(
                    [
                        f"[Evidence {index}]",
                        f"PDF: {item.chunk.metadata.pdf_name}",
                        f"Page: {item.chunk.metadata.page}",
                        f"Section: {item.chunk.metadata.section_title or 'N/A'}",
                        f"Japanese Source: {item.chunk.text_ja}",
                    ]
                )
            )

        system_prompt = (
            "You are a Japanese university admissions assistant. "
            "Answer in Simplified Chinese only, based strictly on the provided Japanese evidence. "
            "Do not invent facts. If evidence is insufficient, say so clearly. "
            "End the answer with a short section titled \u51fa\u5904 listing PDF names and page numbers."
        )
        evidence_text = "\n\n".join(evidence_blocks)
        user_prompt = f"""
User question: {question_zh}

Retrieval rewrite:
- rewritten_ja: {rewrite.rewritten_ja}
- expanded_keywords: {", ".join(rewrite.expanded_keywords) or "none"}

Evidence:
{evidence_text}

Please answer in Simplified Chinese.
Give the conclusion first, then brief explanation, then a final section titled \u51fa\u5904.
""".strip()
        try:
            answer = self.llm_backend.generate(system_prompt, user_prompt, temperature=0.0, max_tokens=1200)
        except Exception:
            answer = self._fallback_answer(question_zh, citations)
        return AnswerResponse(
            answer_zh=answer,
            citations=citations,
            rewritten_query=rewrite,
            retrieved=evidence,
        )

    @staticmethod
    def _fallback_answer(question_zh: str, citations: list[AnswerCitation]) -> str:
        lines = [
            (
                f"\u5173\u4e8e\u201c{question_zh}\u201d\uff0c"
                "\u6211\u627e\u5230\u4e86\u4ee5\u4e0b\u6700\u76f8\u5173\u7684\u65e5\u6587\u539f\u6587\uff0c"
                "\u8bf7\u4f18\u5148\u4ee5\u51fa\u5904\u4e3a\u51c6\uff1a"
            ),
            "",
        ]
        for citation in citations[:3]:
            lines.append(
                f"- {citation.pdf_name} "
                f"\u7b2c {citation.page} \u9875\uff1a{citation.quote_ja[:180]}"
            )
        lines.append("")
        lines.append("\u51fa\u5904\uff1a")
        for citation in citations[:3]:
            lines.append(f"- {citation.pdf_name} / \u7b2c {citation.page} \u9875")
        return "\n".join(lines)
