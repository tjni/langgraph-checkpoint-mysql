import urllib.parse
from typing import AsyncIterator

import aiomysql  # type: ignore
import pymysql
import pymysql.constants.ER
import pytest

DEFAULT_URI = "mysql://mysql:mysql@localhost:5441/mysql"


@pytest.fixture(scope="function")
async def conn() -> AsyncIterator[aiomysql.Connection]:
    parsed = urllib.parse.urlparse(DEFAULT_URI)
    async with await aiomysql.connect(
        user=parsed.username,
        password=parsed.password or "",
        db=parsed.path[1:],
        port=parsed.port or 3306,
        autocommit=True,
    ) as conn:
        yield conn


@pytest.fixture(scope="function", autouse=True)
async def clear_test_db(conn: aiomysql.Connection) -> None:
    """Delete all tables before each test."""
    try:
        async with conn.cursor() as cursor:
            await cursor.execute("DELETE FROM checkpoints")
            await cursor.execute("DELETE FROM checkpoint_blobs")
            await cursor.execute("DELETE FROM checkpoint_writes")
    except pymysql.ProgrammingError as e:
        if e.args[0] != pymysql.constants.ER.NO_SUCH_TABLE:
            raise
