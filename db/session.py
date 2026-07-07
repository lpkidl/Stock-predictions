"""
Engine / session management for the results database.

SQLite by default (results/stocks.db); switch to Postgres by setting
DATABASE_URL in .env — nothing else in the codebase needs to change.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from config import settings

logger = logging.getLogger(__name__)

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

engine = create_engine(
    settings.DATABASE_URL,
    future=True,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create the SQLite file's parent directory and all tables (idempotent)."""
    if _is_sqlite:
        # sqlite:///results/stocks.db -> ensure ./results exists
        db_path = settings.DATABASE_URL.replace("sqlite:///", "", 1)
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

    from db.models import Base

    Base.metadata.create_all(engine)


@contextmanager
def session_scope():
    """Provide a transactional scope: commit on success, rollback on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
