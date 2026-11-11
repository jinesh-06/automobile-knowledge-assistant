from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            try:
                # Load the model once
                cls._model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load sentence-transformers model: {e}")
                cls._model = None
        return cls._instance

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        """
        if not self._model:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True).tolist()
        return embeddings

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query string.
        """
        if not self._model:
            return []
        embedding = self._model.encode([query], convert_to_numpy=True).tolist()[0]
        return embedding
