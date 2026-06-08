import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

embedding_model = None


def load_model():
    global embedding_model

    if embedding_model is None:
        logger.info("Loading embedding model...")

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

        logger.info("Embedding model loaded.")

    return embedding_model
