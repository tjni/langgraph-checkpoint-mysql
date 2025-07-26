from contextlib import asynccontextmanager, contextmanager
from uuid import uuid4

import aiomysql  # type: ignore
import asyncmy
import pymysql
from sqlalchemy import Engine, create_engine

from langgraph.store.mysql.aio import AIOMySQLStore
from langgraph.store.mysql.asyncmy import AsyncMyStore
from langgraph.store.mysql.pymysql import PyMySQLStore

DEFAULT_MYSQL_URI = "mysql://mysql:mysql@localhost:5441/"


def get_pymysql_sqlalchemy_engine(uri: str) -> Engine:
    updated_uri = uri.replace("mysql://", "mysql+pymysql://")
    return create_engine(updated_uri)


@contextmanager
def _store_pymysql():
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


@contextmanager
def _store_pymysql_pool():
    database = f"test_{uuid4().hex[:16]}"

    # create unique db
    with pymysql.connect(**PyMySQLStore.parse_conn_string(DEFAULT_MYSQL_URI), autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield store
        engine = get_pymysql_sqlalchemy_engine(DEFAULT_MYSQL_URI + database)
        store = PyMySQLStore(engine.raw_connection)
        store.setup()
        yield store
    finally:
        # drop unique db
        with pymysql.connect(**PyMySQLStore.parse_conn_string(DEFAULT_MYSQL_URI), autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _store_asyncmy():
    database = f"test_{uuid4().hex[:16]}"
    async with await asyncmy.connect(
        **AsyncMyStore.parse_conn_string(DEFAULT_MYSQL_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        async with AsyncMyStore.from_conn_string(
            DEFAULT_MYSQL_URI + database
        ) as store:
            await store.setup()
            yield store
    finally:
        async with await asyncmy.connect(
            **AsyncMyStore.parse_conn_string(DEFAULT_MYSQL_URI),
            autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _store_asyncmy_pool():
    database = f"test_{uuid4().hex[:16]}"
    async with await asyncmy.connect(
        **AsyncMyStore.parse_conn_string(DEFAULT_MYSQL_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        async with asyncmy.create_pool(
            **AsyncMyStore.parse_conn_string(DEFAULT_MYSQL_URI + database),
            maxsize=10,
            autocommit=True,
        ) as pool:
            store = AsyncMyStore(pool)
            await store.setup()
            yield store
    finally:
        async with await asyncmy.connect(
            **AsyncMyStore.parse_conn_string(DEFAULT_MYSQL_URI),
            autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


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
async def _store_aiomysql_pool():
    database = f"test_{uuid4().hex[:16]}"
    async with await aiomysql.connect(
        **AIOMySQLStore.parse_conn_string(DEFAULT_MYSQL_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        async with aiomysql.create_pool(
            **AIOMySQLStore.parse_conn_string(DEFAULT_MYSQL_URI + database),
            maxsize=10,
            autocommit=True,
        ) as pool:
            store = AIOMySQLStore(pool)
            await store.setup()
            yield store
    finally:
        async with await aiomysql.connect(
            **AIOMySQLStore.parse_conn_string(DEFAULT_MYSQL_URI),
            autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")
