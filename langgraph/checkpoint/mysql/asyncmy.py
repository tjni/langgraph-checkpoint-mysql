from __future__ import annotations

import urllib.parse
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from asyncmy import Connection, connect  # type: ignore
from asyncmy.cursors import DictCursor  # type: ignore
from typing_extensions import Self, override

from langgraph.checkpoint.mysql.aio_base import BaseAsyncMySQLSaver
from langgraph.checkpoint.mysql.shallow import BaseShallowAsyncMySQLSaver
from langgraph.checkpoint.serde.base import SerializerProtocol


class AsyncMySaver(BaseAsyncMySQLSaver[Connection, DictCursor]):
    """Asynchronous checkpointer that stores checkpoints in a MySQL database."""

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
        """Create a new AsyncMySaver instance from a connection string.

        Args:
            conn_string: The MySQL connection info string.

        Returns:
            AsyncMySaver: A new AsyncMySaver instance.

        Example:
            conn_string=mysql+asyncmy://user:password@localhost/db?unix_socket=/path/to/socket
        """
        async with connect(
            **cls.parse_conn_string(conn_string),
            autocommit=True,
        ) as conn:
            yield cls(conn=conn, serde=serde)

    @override
    @staticmethod
    def _get_cursor_from_connection(conn: Connection) -> DictCursor:
        return cast(DictCursor, conn.cursor(DictCursor))


class ShallowAsyncMySaver(BaseShallowAsyncMySQLSaver[Connection, DictCursor]):
    def __init__(
        self,
        conn: Connection,
        serde: SerializerProtocol | None = None,
    ) -> None:
        warnings.warn(
            "ShallowAsyncMySaver is deprecated as of version 2.0.15 and will be removed in 3.0.0. "
            "Use AsyncMysqlSaver instead, and invoke the graph with `await graph.ainvoke(..., durability='exit')`.",
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
        """Create a new ShallowAsyncMySaver instance from a connection string.

        Args:
            conn_string: The MySQL connection info string.

        Returns:
            ShallowAsyncMySaver: A new ShallowAsyncMySaver instance.

        Example:
            conn_string=mysql+asyncmy://user:password@localhost/db?unix_socket=/path/to/socket
        """
        async with connect(
            **AsyncMySaver.parse_conn_string(conn_string),
            autocommit=True,
        ) as conn:
            yield cls(conn=conn, serde=serde)

    @override
    @staticmethod
    def _get_cursor_from_connection(conn: Connection) -> DictCursor:
        return cast(DictCursor, conn.cursor(DictCursor))


__all__ = ["AsyncMySaver", "ShallowAsyncMySaver"]
