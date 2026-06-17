"""
rag/embeddings/bge_embeddings.py
BGE-large embeddings wrapper with batch processing support.
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


class BGELargeEmbeddings:
    """
    BAAI/bge-large-en-v1.5 embeddings via sentence-transformers.
    Dimension: 1024. Optimized for retrieval with normalize_embeddings=True.
    """

    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        logger.info(f"BGELargeEmbeddings loaded: {model_name}")

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed a list of documents in batches."""
        all_embeddings: list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs  = self._model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(vecs.tolist())
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        # BGE instruction prefix improves retrieval quality
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        vec = self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()

    @property
    def dimension(self) -> int:
        return 1024
