from collections.abc import AsyncIterator, Iterator
from uuid import UUID

import pytest
from pytest_mock import MockerFixture
from sqlalchemy import Engine, create_engine

from langgraph.cache.base import BaseCache
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.types import Durability
from tests.conftest_checkpointer import (
    _checkpointer_aiomysql,
    _checkpointer_aiomysql_pool,
    _checkpointer_asyncmy,
    _checkpointer_asyncmy_pool,
    _checkpointer_pymysql,
    _checkpointer_pymysql_pool,
)
from tests.conftest_store import (
    _store_aiomysql,
    _store_aiomysql_pool,
    _store_asyncmy,
    _store_asyncmy_pool,
    _store_pymysql,
    _store_pymysql_pool,
)

DEFAULT_MYSQL_URI = "mysql://mysql:mysql@localhost:5441/"


def get_pymysql_sqlalchemy_engine(uri: str) -> Engine:
    updated_uri = uri.replace("mysql://", "mysql+pymysql://")
    return create_engine(updated_uri)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


@pytest.fixture(params=["sync", "async", "exit"])
def durability(request: pytest.FixtureRequest) -> Durability:
    return request.param


@pytest.fixture(scope="function", params=["sqlite", "memory"])
def cache(request: pytest.FixtureRequest) -> Iterator[BaseCache]:
    if request.param == "sqlite":
        yield SqliteCache(path=":memory:")
    elif request.param == "memory":
        yield InMemoryCache()
    else:
        raise ValueError(f"Unknown cache type: {request.param}")


@pytest.fixture(
    scope="function",
    params=["pymysql", "pymysql_pool"],
)
def sync_store(request: pytest.FixtureRequest) -> Iterator[BaseStore]:
    store_name = request.param
    if store_name == "pymysql":
        with _store_pymysql() as store:
            yield store
    elif store_name == "pymysql_pool":
        with _store_pymysql_pool() as store:
            yield store
    else:
        raise NotImplementedError(f"Unknown store {store_name}")


@pytest.fixture(
    scope="function",
    params=["aiomysql", "aiomysql_pool", "asyncmy", "asyncmy_pool"],
)
async def async_store(request: pytest.FixtureRequest) -> AsyncIterator[BaseStore]:
    store_name = request.param
    if store_name == "aiomysql":
        async with _store_aiomysql() as store:
            yield store
    elif store_name == "aiomysql_pool":
        async with _store_aiomysql_pool() as store:
            yield store
    elif store_name == "asyncmy":
        async with _store_asyncmy() as store:
            yield store
    elif store_name == "asyncmy_pool":
        async with _store_asyncmy_pool() as store:
            yield store
    else:
        raise NotImplementedError(f"Unknown store {store_name}")


@pytest.fixture(
    scope="function",
    params=["pymysql", "pymysql_pool"],
)
def sync_checkpointer(
    request: pytest.FixtureRequest,
) -> Iterator[BaseCheckpointSaver]:
    checkpointer_name = request.param
    if checkpointer_name == "pymysql":
        with _checkpointer_pymysql() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "pymysql_pool":
        with _checkpointer_pymysql_pool() as checkpointer:
            yield checkpointer
    else:
        raise NotImplementedError(f"Unknown checkpointer: {checkpointer_name}")


@pytest.fixture(
    scope="function",
    params=[
        "aiomysql",
        "aiomysql_pool",
        "asyncmy",
        "asyncmy_pool",
    ],
)
async def async_checkpointer(
    request: pytest.FixtureRequest,
) -> AsyncIterator[BaseCheckpointSaver]:
    checkpointer_name = request.param
    if checkpointer_name == "aiomysql":
        async with _checkpointer_aiomysql() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "aiomysql_pool":
        async with _checkpointer_aiomysql_pool() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "asyncmy":
        async with _checkpointer_asyncmy() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "asyncmy_pool":
        async with _checkpointer_asyncmy_pool() as checkpointer:
            yield checkpointer
    else:
        raise NotImplementedError(f"Unknown checkpointer: {checkpointer_name}")
