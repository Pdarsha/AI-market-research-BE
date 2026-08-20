import logging
from server.db.base_class import Base
from server.db.session import engine, ensure_vector_extension

logger = logging.getLogger(__name__)


def init_db():
    logger.info("Initializing database...")
    ensure_vector_extension()
    Base.metadata.create_all(engine)
    logger.info("Database initialization completed.")
