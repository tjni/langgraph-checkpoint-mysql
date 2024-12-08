from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from uuid import UUID, uuid4

import aiomysql  # type: ignore
import pymysql
import pymysql.constants.ER
import pytest
from langchain_core import __version__ as core_version
from packaging import version
from pytest_mock import MockerFixture

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.store.base import BaseStore
from langgraph.store.mysql.aio import AIOMySQLStore
from langgraph.store.mysql.pymysql import PyMySQLStore

DEFAULT_MYSQL_URI = "mysql://mysql:mysql@localhost:5441/"


# TODO: fix this once core is released
IS_LANGCHAIN_CORE_030_OR_GREATER = version.parse(core_version) >= version.parse(
    "0.3.0.dev0"
)
SHOULD_CHECK_SNAPSHOTS = IS_LANGCHAIN_CORE_030_OR_GREATER


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


@pytest.fixture(scope="function")
def checkpointer_pymysql():
    database = f"test_{uuid4().hex[:16]}"

    # create unique db
    with pymysql.connect(**PyMySQLSaver.parse_conn_string(DEFAULT_MYSQL_URI), autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        with PyMySQLSaver.from_conn_string(
            DEFAULT_MYSQL_URI + database
        ) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        with pymysql.connect(**PyMySQLSaver.parse_conn_string(DEFAULT_MYSQL_URI), autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"drop databASE {database}")


@asynccontextmanager
async def _checkpointer_aiomysql():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await aiomysql.connect(
        **AIOMySQLSaver.parse_conn_string(DEFAULT_MYSQL_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with AIOMySQLSaver.from_conn_string(
            DEFAULT_MYSQL_URI + database
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await aiomysql.connect(
            **AIOMySQLSaver.parse_conn_string(DEFAULT_MYSQL_URI),
            autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _checkpointer_aiomysql_pool():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await aiomysql.connect(
        **AIOMySQLSaver.parse_conn_string(DEFAULT_MYSQL_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with aiomysql.create_pool(
            **AIOMySQLSaver.parse_conn_string(DEFAULT_MYSQL_URI + database),
            maxsize=10,
            autocommit=True,
        ) as pool:
            checkpointer = AIOMySQLSaver(pool)
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await aiomysql.connect(
            **AIOMySQLSaver.parse_conn_string(DEFAULT_MYSQL_URI),
            autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def awith_checkpointer(
    checkpointer_name: Optional[str],
) -> AsyncIterator[BaseCheckpointSaver]:
    if checkpointer_name == "aiomysql":
        async with _checkpointer_aiomysql() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "aiomysql_pool":
        async with _checkpointer_aiomysql_pool() as checkpointer:
            yield checkpointer
    else:
        raise NotImplementedError(f"Unknown checkpointer: {checkpointer_name}")


@pytest.fixture(scope="function")
def store_pymysql():
    database = f"test_{uuid4().hex[:16]}"

    # create unique db
    with pymysql.connect(**PyMySQLStore.parse_conn_string(DEFAULT_MYSQL_URI), autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield store
        with PyMySQLStore.from_conn_string(DEFAULT_MYSQL_URI + database) as store:
            store.setup()
            yield store
    finally:
        # drop unique db
        with pymysql.connect(**PyMySQLStore.parse_conn_string(DEFAULT_MYSQL_URI), autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _store_aiomysql():
    database = f"test_{uuid4().hex[:16]}"
    async with await aiomysql.connect(
        **AIOMySQLStore.parse_conn_string(DEFAULT_MYSQL_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        async with AIOMySQLStore.from_conn_string(
            DEFAULT_MYSQL_URI + database
        ) as store:
            await store.setup()
            yield store
    finally:
        async with await aiomysql.connect(
            **AIOMySQLStore.parse_conn_string(DEFAULT_MYSQL_URI),
            autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def awith_store(store_name: Optional[str]) -> AsyncIterator[BaseStore]:
    if store_name == "aiomysql":
        async with _store_aiomysql() as store:
            yield store
    else:
        raise NotImplementedError(f"Unknown store {store_name}")


ALL_CHECKPOINTERS_SYNC = ["pymysql"]
ALL_CHECKPOINTERS_ASYNC = ["aiomysql", "aiomysql_pool"]
ALL_STORES_SYNC = ["pymysql"]
ALL_STORES_ASYNC = ["aiomysql"]
