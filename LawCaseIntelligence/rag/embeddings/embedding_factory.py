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

    def __init__(self, model_name: str = None) -> None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # Auto-select smallest model for low-memory environments (Render free tier = 512 MB)
        if model_name is None:
            import os
            # Default: sentence-transformers/all-MiniLM-L6-v2 (~80 MB, fastest, fits 512 MB)
            # Alternatives:
            #   BAAI/bge-small-en-v1.5  (~130 MB, 384 dim, good quality)
            #   BAAI/bge-base-en-v1.5   (~440 MB, 768 dim, better)
            #   BAAI/bge-large-en-v1.5  (~1.3 GB, 1024 dim, best — needs 2 GB+ RAM)
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._model_name = model_name
        self._model = None  # lazy-load on first use to avoid memory spikes at init
        logger.info(f"BGEEmbeddings configured: {model_name} (lazy-load on first use)")

    def _ensure_loaded(self):
        """Lazy-load the model only when first needed (saves memory + faster startup)."""
        if self._model is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            import gc
            gc.collect()  # free unused memory before loading
            logger.info(f"BGEEmbeddings loading model: {self._model_name}")
            self._model = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"BGEEmbeddings ready: {self._model_name}")
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._ensure_loaded().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._ensure_loaded().embed_query(text)

    def get_langchain_embeddings(self):
        """Return the raw LangChain HuggingFaceEmbeddings object for ChromaDB."""
        return self._ensure_loaded()

    @property
    def dimension(self) -> int:
        # all-MiniLM-L6-v2 = 384, bge-small = 384, bge-base = 768, bge-large = 1024
        if "base" in self._model_name:
            return 768
        elif "large" in self._model_name:
            return 1024
        return 384  # default for small models including all-MiniLM-L6-v2

    def __repr__(self) -> str:
        return f"BGEEmbeddings(model={self._model_name})"


@lru_cache(maxsize=1)
def get_embeddings() -> BGEEmbeddings:
    """
    Return the singleton BGEEmbeddings instance.
    Cached after first call — safe to call from anywhere.
    """
    import os
    # Read directly from env var first (works in both dev and production)
    model_name = os.getenv("EMBEDDING_MODEL")
    if not model_name:
        # Fallback to settings
        from backend.config.settings import get_settings
        settings = get_settings()
        model_name = getattr(settings, "bge_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    logger.info(f"Initialising embeddings: {model_name}")
    return BGEEmbeddings(model_name=model_name)
