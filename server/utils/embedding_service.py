import logging

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

embedding_model = None


def load_model():
    global embedding_model

    try:
        # Try local cache first
        embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
        logger.info("Loaded model from local cache")

        return embedding_model

    except Exception:
        logger.warning("Model not found locally. Downloading...")

        embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=False
        )

        logger.info("Model downloaded and cached")

        return embedding_model


def load_rerank_model():
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    return reranker