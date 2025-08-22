from __future__ import annotations

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

Conn = _ainternal.Conn[aiomysql.Connection]  # For backward compatibility


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
