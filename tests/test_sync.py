from collections.abc import Iterator
from contextlib import closing, contextmanager
from copy import deepcopy
from typing import Any, cast
from uuid import uuid4

import pymysql
import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.checkpoint.serde.types import TASKS
from tests.conftest import DEFAULT_BASE_URI, get_pymysql_sqlalchemy_pool


@contextmanager
def _sqlalchemy_pool_saver() -> Iterator[PyMySQLSaver]:
    """Fixture for pool mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with pymysql.connect(
        **PyMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        checkpointer = PyMySQLSaver(
            get_pymysql_sqlalchemy_pool(DEFAULT_BASE_URI + database)
        )
        checkpointer.setup()
        yield checkpointer
    finally:
        # drop unique db
        with pymysql.connect(
            **PyMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {database}")


@contextmanager
def _callable_saver() -> Iterator[PyMySQLSaver]:
    """Fixture for pool mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with pymysql.connect(
        **PyMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        pool = get_pymysql_sqlalchemy_pool(DEFAULT_BASE_URI + database)

        def callable() -> pymysql.Connection:
            return cast(pymysql.Connection, closing(pool.connect()))

        checkpointer = PyMySQLSaver(callable)
        checkpointer.setup()
        yield checkpointer
    finally:
        # drop unique db
        with pymysql.connect(
            **PyMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {database}")


@contextmanager
def _base_saver() -> Iterator[PyMySQLSaver]:
    """Fixture for regular connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with pymysql.connect(
        **PyMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        with PyMySQLSaver.from_conn_string(DEFAULT_BASE_URI + database) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        with pymysql.connect(
            **PyMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {database}")


@contextmanager
def _saver(name: str) -> Iterator[PyMySQLSaver]:
    if name == "base":
        with _base_saver() as saver:
            yield saver
    elif name == "sqlalchemy_pool":
        with _sqlalchemy_pool_saver() as saver:
            yield saver
    elif name == "callable":
        with _callable_saver() as saver:
            yield saver


@pytest.fixture
def test_data() -> dict[str, Any]:
    """Fixture providing test data for checkpoint tests."""
    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            # for backwards compatibility testing
            "thread_ts": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }

    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()

    metadata_1: CheckpointMetadata = {
        "source": "input",
        "step": 2,
        "writes": {},
        "score": 1,
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"foo": "bar"},
        "score": None,
    }
    metadata_3: CheckpointMetadata = {}

    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


@pytest.mark.parametrize("saver_name", ["base", "sqlalchemy_pool", "callable"])
def test_search(saver_name: str, test_data: dict[str, Any]) -> None:
    with _saver(saver_name) as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        saver.put(configs[0], checkpoints[0], metadata[0], {})
        saver.put(configs[1], checkpoints[1], metadata[1], {})
        saver.put(configs[2], checkpoints[2], metadata[2], {})

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = list(saver.list(None, filter=query_1))
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == metadata[0]

        search_results_2 = list(saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == metadata[1]

        search_results_3 = list(saver.list(None, filter=query_3))
        assert len(search_results_3) == 3

        search_results_4 = list(saver.list(None, filter=query_4))
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = list(saver.list({"configurable": {"thread_id": "thread-2"}}))
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


@pytest.mark.parametrize("saver_name", ["base", "sqlalchemy_pool", "callable"])
def test_null_chars(saver_name: str, test_data: dict[str, Any]) -> None:
    with _saver(saver_name) as saver:
        config = saver.put(
            test_data["configs"][0],
            test_data["checkpoints"][0],
            {"my_key": "\x00abc"},
            {},
        )
        assert saver.get_tuple(config).metadata["my_key"] == "abc"  # type: ignore
        assert (
            list(saver.list(None, filter={"my_key": "abc"}))[0].metadata["my_key"]
            == "abc"
        )


@pytest.mark.parametrize("saver_name", ["base", "sqlalchemy_pool", "callable"])
def test_write_and_read_pending_writes_and_sends(
    saver_name: str, test_data: dict[str, Any]
) -> None:
    with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_id": "1",
                "checkpoint_ns": "",
            }
        }

        chkpnt = create_checkpoint(test_data["checkpoints"][0], {}, 1, id="1")

        saver.put(config, chkpnt, {}, {})
        saver.put_writes(config, [("w1", "w1v"), ("w2", "w2v")], "world")
        saver.put_writes(config, [(TASKS, "w3v")], "hello")

        result = next(saver.list({}))

        assert result.pending_writes == [
            ("hello", TASKS, "w3v"),
            ("world", "w1", "w1v"),
            ("world", "w2", "w2v"),
        ]

        assert result.checkpoint["pending_sends"] == ["w3v"]


@pytest.mark.parametrize("saver_name", ["base", "sqlalchemy_pool", "callable"])
@pytest.mark.parametrize(
    "channel_values",
    [
        {"channel1": "channel1v"},
        {},  # to catch regression reported in #10
    ],
)
def test_write_and_read_channel_values(
    saver_name: str, channel_values: dict[str, Any]
) -> None:
    with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-4",
                "checkpoint_id": "4",
                "checkpoint_ns": "",
            }
        }
        chkpnt = empty_checkpoint()
        chkpnt["id"] = "4"
        chkpnt["channel_values"] = channel_values

        newversions: ChannelVersions = {
            "channel1": 1,
            "channel:with:colon": 1,  # to catch regression reported in #9
        }
        chkpnt["channel_versions"] = newversions

        saver.put(config, chkpnt, {}, newversions)

        result = next(saver.list({}))
        assert result.checkpoint["channel_values"] == channel_values


@pytest.mark.parametrize("saver_name", ["base", "sqlalchemy_pool", "callable"])
def test_write_and_read_pending_writes(saver_name: str) -> None:
    with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-5",
                "checkpoint_id": "5",
                "checkpoint_ns": "",
            }
        }
        chkpnt = empty_checkpoint()
        chkpnt["id"] = "5"
        task_id = "task1"
        writes = [
            ("channel1", "somevalue"),
            ("channel2", [1, 2, 3]),
            ("channel3", None),
        ]

        saver.put(config, chkpnt, {}, {})
        saver.put_writes(config, writes, task_id)

        result = next(saver.list({}))

        assert result.pending_writes == [
            (task_id, "channel1", "somevalue"),
            (task_id, "channel2", [1, 2, 3]),
            (task_id, "channel3", None),
        ]


@pytest.mark.parametrize("saver_name", ["base", "sqlalchemy_pool", "callable"])
def test_write_with_different_checkpoint_ns_does_an_update(saver_name: str) -> None:
    with _saver(saver_name) as saver:
        config1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-6",
                "checkpoint_id": "6",
                "checkpoint_ns": "first",
            }
        }
        config2 = deepcopy(config1)
        config2["configurable"]["checkpoint_ns"] = "second"

        chkpnt = empty_checkpoint()

        saver.put(config1, chkpnt, {}, {})
        saver.put(config2, chkpnt, {}, {})

        results = list(saver.list({}))

        assert len(results) == 2
