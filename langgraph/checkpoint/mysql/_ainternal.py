"""Shared async utility functions for the AIOMYSQL checkpoint & storage classes."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Union

import aiomysql
import logging

logger = logging.getLogger(__name__)
Conn = Union[aiomysql.Connection, aiomysql.Pool]

@asynccontextmanager
async def get_connection(
    conn: Conn,
) -> AsyncIterator[aiomysql.Connection]:
    if isinstance(conn, aiomysql.Connection):
        yield conn
    elif isinstance(conn, aiomysql.Pool):
        async with conn.acquire() as _conn:
            await _conn.set_charset("utf8mb4")
            yield _conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
