import urllib.parse
from collections.abc import AsyncIterator

import aiomysql  # type: ignore
import pymysql
import pymysql.constants.ER
import pytest
from sqlalchemy import Engine, Pool, create_engine, create_pool_from_url

DEFAULT_BASE_URI = "mysql://mysql:mysql@localhost:5441/"
DEFAULT_URI = DEFAULT_BASE_URI + "mysql"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="function")
async def conn(anyio_backend: str) -> AsyncIterator[aiomysql.Connection]:
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

    try:
        async with conn.cursor() as cursor:
            await cursor.execute("DELETE FROM store")
    except pymysql.ProgrammingError as e:
        if e.args[0] != pymysql.constants.ER.NO_SUCH_TABLE:
            raise


def get_pymysql_sqlalchemy_engine(uri: str) -> Engine:
    updated_uri = uri.replace("mysql://", "mysql+pymysql://")
    return create_engine(updated_uri)


def get_pymysql_sqlalchemy_pool(uri: str) -> Pool:
    updated_uri = uri.replace("mysql://", "mysql+pymysql://")
    return create_pool_from_url(updated_uri)
