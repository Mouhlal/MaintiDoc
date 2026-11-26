"""
database.py — Async SQLAlchemy engine and session setup for MaintiDoc.

Provides:
- A configured async engine backed by PostgreSQL.
- An async session factory for use in FastAPI dependencies.
- Table initialisation helper called on application startup.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.config import settings
from backend.db.models import Base

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_async_engine(
    settings.database_url,
    echo=settings.is_development,  # SQL logging in dev mode
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # detect stale connections
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # keep ORM objects usable after commit
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session, then close it on completion.

    Usage::

        @router.get("/")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """
    Create all database tables defined in the ORM models.

    Should be called once on application startup (e.g. from lifespan).
    Safe to run against an already-initialised database (uses CREATE TABLE
    IF NOT EXISTS semantics from SQLAlchemy).
    """
    logger.info("Initialising database schema …")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema ready.")
