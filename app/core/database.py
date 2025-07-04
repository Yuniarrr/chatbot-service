import contextlib
import json

from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import Dialect, types
from typing import Any, Optional, AsyncIterator
from sqlalchemy.sql.type_api import _T
from typing_extensions import Self

from app.env import DATABASE_URL, PGVECTOR_DB_URL

Base = declarative_base()


class DatabaseSessionManager:
    def __init__(self, db_url: str, engine_kwargs: dict[str, Any] = {}):
        self.engine_kwargs = engine_kwargs
        self._engine: AsyncEngine = None
        self._sessionmaker = None
        self._database_url = db_url

    async def initialize(self):
        # Local environment: use standard connection
        self._engine = create_async_engine(
            "postgresql+asyncpg://" + self._database_url,
            **self.engine_kwargs,
        )

        self._sessionmaker = async_sessionmaker(
            autocommit=False, bind=self._engine, class_=AsyncSession
        )

    async def close(self):
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        await self._engine.dispose()

        self._engine = None
        self._sessionmaker = None

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")

        async with self._engine.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise Exception("DatabaseSessionManager is not initialized")

        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


session_manager = DatabaseSessionManager(
    db_url=DATABASE_URL, engine_kwargs={"echo": True}
)
pgvector_session_manager = DatabaseSessionManager(
    db_url=PGVECTOR_DB_URL, engine_kwargs={"echo": True}
)


class JSONField(types.TypeDecorator):
    impl = types.Text
    cache_ok = True

    def process_bind_param(self, value: Optional[_T], dialect: Dialect) -> Any:
        return json.dumps(value)

    def process_result_value(self, value: Optional[_T], dialect: Dialect) -> Any:
        if value is not None:
            return json.loads(value)

    def copy(self, **kw: Any) -> Self:
        return JSONField(self.impl.length)

    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        if value is not None:
            return json.loads(value)
