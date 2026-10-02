import logging
from typing import List
from sentence_transformers import SentenceTransformer
from ..config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            model_name = settings.EMBEDDING_MODEL
            logger.info(f"Loading embedding model: {model_name}")
            try:
                self._model = SentenceTransformer(model_name)
            except Exception as e:
                logger.error(f"Failed to load embedding model {model_name}: {e}")
                raise RuntimeError(f"Could not initialize embedding model {model_name}: {str(e)}")
        return self._model

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates a 384-dimensional embedding vector for a single text query.
        """
        if not text:
            return [0.0] * 384
        vec = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates 384-dimensional embedding vectors for a batch of texts.
        """
        if not texts:
            return []
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

embedding_service = EmbeddingService()
