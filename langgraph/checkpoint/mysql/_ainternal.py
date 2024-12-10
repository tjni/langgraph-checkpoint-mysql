"""Shared async utility functions for the MySQL checkpoint & storage classes."""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Union

import aiomysql  # type: ignore
import pymysql.connections

Conn = Union[aiomysql.Connection, aiomysql.Pool]


@asynccontextmanager
async def get_connection(
    conn: Conn,
) -> AsyncIterator[aiomysql.Connection]:
    if isinstance(conn, aiomysql.Connection):
        yield conn
    elif isinstance(conn, aiomysql.Pool):
        async with conn.acquire() as _conn:
            # This seems necessary until https://github.com/PyMySQL/PyMySQL/pull/1119
            # is merged into aiomysql.
            await _conn.set_charset(pymysql.connections.DEFAULT_CHARSET)
            yield _conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
