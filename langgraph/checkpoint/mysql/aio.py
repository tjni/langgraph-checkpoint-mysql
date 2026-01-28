from __future__ import annotations

import logging
import urllib.parse
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import aiomysql  # type: ignore
from typing_extensions import Self, override

from langgraph.checkpoint.mysql import _ainternal
from langgraph.checkpoint.mysql.aio_base import BaseAsyncMySQLSaver
from langgraph.checkpoint.mysql.shallow import BaseShallowAsyncMySQLSaver
from langgraph.checkpoint.serde.base import SerializerProtocol

logger = logging.getLogger(__name__)

Conn = _ainternal.Conn[aiomysql.Connection]  # For backward compatibility

# Global pool cache to prevent duplicate pools for the same connection string
_pool_cache: dict[str, aiomysql.Pool] = {}


class AIOMySQLSaver(BaseAsyncMySQLSaver[aiomysql.Connection, aiomysql.DictCursor]):
    @staticmethod
    def parse_conn_string(conn_string: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(conn_string)

        # In order to provide additional params via the connection string,
        # we convert the parsed.query to a dict so we can access the values.
        # This is necessary when using a unix socket, for example.
        params_as_dict = dict(urllib.parse.parse_qsl(parsed.query))

        return {
            "host": parsed.hostname or "localhost",
            "user": parsed.username,
            "password": parsed.password or "",
            "db": parsed.path[1:] or None,
            "port": parsed.port or 3306,
            "unix_socket": params_as_dict.get("unix_socket"),
        }

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: SerializerProtocol | None = None,
    ) -> AsyncIterator[Self]:
        """Create a new AIOMySQLSaver instance from a connection string.

        Args:
            conn_string: The MySQL connection info string.

        Returns:
            AIOMySQLSaver: A new AIOMySQLSaver instance.

        Example:
            conn_string=mysql+aiomysql://user:password@localhost/db?unix_socket=/path/to/socket
        """
        async with aiomysql.connect(
            **cls.parse_conn_string(conn_string),
            autocommit=True,
        ) as conn:
            yield cls(conn=conn, serde=serde)

    @classmethod
    async def create_pool(
        cls,
        conn_string: str,
        *,
        minsize: int = 5,
        maxsize: int = 20,
        pool_recycle: int = 1800,
    ) -> aiomysql.Pool:
        """Create a persistent connection pool for the given connection string.

        Args:
            conn_string: The MySQL connection info string.
            minsize: Minimum number of connections in the pool (default: 5).
            maxsize: Maximum number of connections in the pool (default: 20).
            pool_recycle: Time in seconds to recycle connections (default: 1800 = 30 min).

        Returns:
            aiomysql.Pool: A connection pool instance.

        Note:
            Pools are cached per connection string to prevent duplicate pools.
        """
        if conn_string not in _pool_cache:
            parsed = cls.parse_conn_string(conn_string)
            _pool_cache[conn_string] = await aiomysql.create_pool(
                host=parsed["host"],
                port=parsed["port"],
                user=parsed["user"],
                password=parsed["password"],
                db=parsed["db"],
                unix_socket=parsed.get("unix_socket"),
                autocommit=True,
                minsize=minsize,
                maxsize=maxsize,
                pool_recycle=pool_recycle,
            )
            logger.info(
                f"Created connection pool for {parsed['host']}:{parsed['port']}/{parsed['db']} "
                f"(minsize={minsize}, maxsize={maxsize}, pool_recycle={pool_recycle}s)"
            )
        return _pool_cache[conn_string]

    @classmethod
    @asynccontextmanager
    async def from_pool(
        cls,
        conn_string: str,
        *,
        serde: SerializerProtocol | None = None,
        minsize: int = 5,
        maxsize: int = 20,
        pool_recycle: int = 1800,
    ) -> AsyncIterator[Self]:
        """Create a new AIOMySQLSaver instance using a connection pool.

        This is the recommended method for production applications as it:
        - Reuses connections instead of creating new ones
        - Automatically recycles stale connections
        - Handles connection failures gracefully
        - Reduces connection overhead

        Args:
            conn_string: The MySQL connection info string.
            serde: Optional serializer protocol.
            minsize: Minimum number of connections in the pool (default: 5).
            maxsize: Maximum number of connections in the pool (default: 20).
            pool_recycle: Time in seconds to recycle connections (default: 1800 = 30 min).

        Returns:
            AIOMySQLSaver: A new AIOMySQLSaver instance using pooled connections.

        Example:
            async with AIOMySQLSaver.from_pool(
                conn_string="mysql+aiomysql://user:password@localhost/db",
                minsize=5,
                maxsize=20,
            ) as checkpointer:
                # Use checkpointer for long-running operations
                await checkpointer.aput(...)
        """
        pool = await cls.create_pool(
            conn_string,
            minsize=minsize,
            maxsize=maxsize,
            pool_recycle=pool_recycle,
        )
        yield cls(conn=pool, serde=serde)

    @classmethod
    async def close_pool(cls, conn_string: str) -> None:
        """Close and remove a connection pool from the cache.

        Args:
            conn_string: The connection string whose pool should be closed.
        """
        if conn_string in _pool_cache:
            pool = _pool_cache.pop(conn_string)
            pool.close()
            await pool.wait_closed()
            logger.info(f"Closed connection pool for {conn_string}")

    @classmethod
    async def close_all_pools(cls) -> None:
        """Close all cached connection pools.

        This should be called during application shutdown.
        """
        for conn_string in list(_pool_cache.keys()):
            await cls.close_pool(conn_string)

    @override
    @staticmethod
    def _get_cursor_from_connection(conn: aiomysql.Connection) -> aiomysql.DictCursor:
        return cast(aiomysql.DictCursor, conn.cursor(aiomysql.DictCursor))


class ShallowAIOMySQLSaver(
    BaseShallowAsyncMySQLSaver[aiomysql.Connection, aiomysql.DictCursor]
):
    def __init__(
        self,
        conn: aiomysql.Connection,
        serde: SerializerProtocol | None = None,
    ) -> None:
        warnings.warn(
            "ShallowAIOMySQLSaver is deprecated as of version 2.0.15 and will be removed in 3.0.0. "
            "Use AIOMysqlSaver instead, and invoke the graph with `await graph.ainvoke(..., durability='exit')`.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(conn, serde=serde)

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: SerializerProtocol | None = None,
    ) -> AsyncIterator[Self]:
        """Create a new ShallowAIOMySQLSaver instance from a connection string.

        Args:
            conn_string: The MySQL connection info string.

        Returns:
            ShallowAIOMySQLSaver: A new ShallowAIOMySQLSaver instance.

        Example:
            conn_string=mysql+aiomysql://user:password@localhost/db?unix_socket=/path/to/socket
        """
        async with aiomysql.connect(
            **AIOMySQLSaver.parse_conn_string(conn_string),
            autocommit=True,
        ) as conn:
            yield cls(conn=conn, serde=serde)

    @override
    @staticmethod
    def _get_cursor_from_connection(conn: aiomysql.Connection) -> aiomysql.DictCursor:
        return cast(aiomysql.DictCursor, conn.cursor(aiomysql.DictCursor))


__all__ = ["AIOMySQLSaver", "ShallowAIOMySQLSaver", "Conn"]
