from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class E5Embedder:
    model_name: str
    _model: object | None = field(init=False, default=None, repr=False)

    @property
    def dimension(self) -> int:
        model = self._ensure_model()
        return int(model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        prompts = [f"passage: {text}" for text in texts]
        embeddings = model.encode(prompts, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        embedding = model.encode([f"query: {text}"], normalize_embeddings=True, convert_to_numpy=True)
        return embedding[0].tolist()

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model
