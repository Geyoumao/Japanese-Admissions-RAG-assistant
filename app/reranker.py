from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TransformerReranker:
    model_name: str
    batch_size: int = 8
    enabled: bool = True
    _tokenizer: object | None = field(init=False, default=None, repr=False)
    _model: object | None = field(init=False, default=None, repr=False)
    _device: str = field(init=False, default="cpu", repr=False)
    _fallback_reason: str = field(init=False, default="", repr=False)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if not self.enabled or not self.model_name:
            return [self._lexical_overlap(query, document) for document in documents]
        self._ensure_model()
        if self._tokenizer is None or self._model is None:
            return [self._lexical_overlap(query, document) for document in documents]

        import torch

        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(documents), self.batch_size):
                batch_docs = documents[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                outputs = self._model(**encoded)
                logits = outputs.logits.squeeze(-1)
                if logits.ndim == 0:
                    scores.append(float(logits.item()))
                else:
                    scores.extend(float(item) for item in logits.detach().cpu().tolist())
        return scores

    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
        except Exception as exc:
            self._fallback_reason = str(exc)
            self._tokenizer = None
            self._model = None

    @staticmethod
    def _lexical_overlap(query: str, document: str) -> float:
        query_tokens = {token for token in query.lower().split() if token}
        document_tokens = {token for token in document.lower().split() if token}
        if not query_tokens or not document_tokens:
            return 0.0
        return len(query_tokens & document_tokens) / len(query_tokens)
