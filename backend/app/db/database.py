"""
Async SQLAlchemy engine and session factory for PostgreSQL.

Usage:
    async with get_db() as session:
        session.add(...)
        await session.commit()
"""

import os
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

engine = create_async_engine(os.getenv("DATABASE_URL"), echo=False, future=True)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


@asynccontextmanager
async def get_db():
    """Yield an async database session, committing on success and rolling back on error."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Create all tables defined in models (safe to call on startup)."""
    from app.db import models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
