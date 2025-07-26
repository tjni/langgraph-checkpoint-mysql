from contextlib import asynccontextmanager, contextmanager
from uuid import uuid4

import aiomysql  # type: ignore
import asyncmy
import pymysql
from sqlalchemy import Engine, create_engine

from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
from langgraph.checkpoint.mysql.asyncmy import AsyncMySaver
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

DEFAULT_MYSQL_URI = "mysql://mysql:mysql@localhost:5441/"


def get_pymysql_sqlalchemy_engine(uri: str) -> Engine:
    updated_uri = uri.replace("mysql://", "mysql+pymysql://")
    return create_engine(updated_uri)


@contextmanager
def _checkpointer_pymysql():
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
                cursor.execute(f"DROP DATABASE {database}")


@contextmanager
def _checkpointer_pymysql_pool():
    database = f"test_{uuid4().hex[:16]}"

    # create unique db
    with pymysql.connect(**PyMySQLSaver.parse_conn_string(DEFAULT_MYSQL_URI), autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {database}")
    try:
        pool = get_pymysql_sqlalchemy_engine(DEFAULT_MYSQL_URI + database)
        checkpointer = PyMySQLSaver(pool.raw_connection)
        checkpointer.setup()
        yield checkpointer
    finally:
        # drop unique db
        with pymysql.connect(**PyMySQLSaver.parse_conn_string(DEFAULT_MYSQL_URI), autocommit=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _checkpointer_asyncmy():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await asyncmy.connect(
        **AsyncMySaver.parse_conn_string(DEFAULT_MYSQL_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with AsyncMySaver.from_conn_string(
            DEFAULT_MYSQL_URI + database
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await asyncmy.connect(
            **AsyncMySaver.parse_conn_string(DEFAULT_MYSQL_URI),
            autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _checkpointer_asyncmy_pool():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await asyncmy.connect(
        **AsyncMySaver.parse_conn_string(DEFAULT_MYSQL_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with asyncmy.create_pool(
            **AsyncMySaver.parse_conn_string(DEFAULT_MYSQL_URI + database),
            maxsize=10,
            autocommit=True,
        ) as pool:
            checkpointer = AsyncMySaver(pool)
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await asyncmy.connect(
            **AsyncMySaver.parse_conn_string(DEFAULT_MYSQL_URI),
            autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")

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
