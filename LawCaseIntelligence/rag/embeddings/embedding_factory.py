"""
rag/embeddings/embedding_factory.py
Factory that returns the configured embedding provider.
Only BGE embeddings are supported (Gemini removed).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

BASE_DIR_NOTE = "Embeddings: BAAI/bge-large-en-v1.5 (downloads ~1.3GB on first run)"


class EmbeddingProvider:
    """Base class for embedding providers."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    def get_langchain_embeddings(self):
        raise NotImplementedError


class BGEEmbeddings(EmbeddingProvider):
    """
    HuggingFace BGE-large embeddings via sentence-transformers.
    Model: BAAI/bge-large-en-v1.5  (dimension: 1024)
    Downloads once to ~/.cache/huggingface and is cached locally.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        self._model_name = model_name
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"BGEEmbeddings ready: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)

    def get_langchain_embeddings(self):
        """Return the raw LangChain HuggingFaceEmbeddings object for ChromaDB."""
        return self._model

    @property
    def dimension(self) -> int:
        return 1024

    def __repr__(self) -> str:
        return f"BGEEmbeddings(model={self._model_name})"


@lru_cache(maxsize=1)
def get_embeddings() -> BGEEmbeddings:
    """
    Return the singleton BGEEmbeddings instance.
    Cached after first call — safe to call from anywhere.
    """
    from backend.config.settings import get_settings
    settings = get_settings()
    model_name = getattr(settings, "bge_model_name", "BAAI/bge-large-en-v1.5")
    logger.info(f"Initialising embeddings: {model_name}")
    return BGEEmbeddings(model_name=model_name)
