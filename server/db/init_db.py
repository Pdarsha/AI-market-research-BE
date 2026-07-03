from server.db.base_class import Base
from server.db.session import engine, ensure_vector_extension


def init_db():
    print("____creating database_________")
    ensure_vector_extension()
    Base.metadata.create_all(engine)
    print("______database created________")
