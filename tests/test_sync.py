from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any
from uuid import uuid4

import pymysql
import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver, ShallowPyMySQLSaver
from langgraph.checkpoint.serde.types import TASKS
from tests.conftest import (
    DEFAULT_BASE_URI,
    get_pymysql_sqlalchemy_engine,
    get_pymysql_sqlalchemy_pool,
)

pytestmark = pytest.mark.anyio

SAVERS = [
    "base",
    "shallow",
    "sqlalchemy_engine",
    "sqlalchemy_pool",
]

NON_SHALLOW_SAVERS = [saver for saver in SAVERS if saver != "shallow"]


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


@contextmanager
def _database() -> Iterator[str]:
    database = f"test_{uuid4().hex[:16]}"

    # create unique db
    with pymysql.connect(
        **PyMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {database}")
    try:
        yield database
    finally:
        # drop unique db
        with pymysql.connect(
            **PyMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {database}")


@contextmanager
def _base_saver() -> Iterator[PyMySQLSaver]:
    with _database() as database:
        with PyMySQLSaver.from_conn_string(DEFAULT_BASE_URI + database) as checkpointer:
            yield checkpointer


@contextmanager
def _sqlalchemy_engine_saver() -> Iterator[PyMySQLSaver]:
    with _database() as database:
        engine = get_pymysql_sqlalchemy_engine(DEFAULT_BASE_URI + database)
        try:
            yield PyMySQLSaver(engine.raw_connection)
        finally:
            engine.dispose()


@contextmanager
def _sqlalchemy_pool_saver() -> Iterator[PyMySQLSaver]:
    with _database() as database:
        pool = get_pymysql_sqlalchemy_pool(DEFAULT_BASE_URI + database)
        try:
            yield PyMySQLSaver(pool.connect)
        finally:
            pool.dispose()


@contextmanager
def _shallow_saver() -> Iterator[ShallowPyMySQLSaver]:
    """Fixture for regular connection mode testing with a shallow checkpointer."""
    with _database() as database:
        # yield checkpointer
        with ShallowPyMySQLSaver.from_conn_string(
            DEFAULT_BASE_URI + database
        ) as checkpointer:
            yield checkpointer


@contextmanager
def _saver(name: str) -> Iterator[PyMySQLSaver | ShallowPyMySQLSaver]:
    if name == "base":
        factory = _base_saver
    elif name == "shallow":
        factory = _shallow_saver  # type: ignore
    elif name == "sqlalchemy_engine":
        factory = _sqlalchemy_engine_saver
    elif name == "sqlalchemy_pool":
        factory = _sqlalchemy_pool_saver
    else:
        raise ValueError(f"Unknown saver name: {name}")

    with factory() as saver:
        saver.setup()
        yield saver


@pytest.fixture
def test_data() -> dict[str, Any]:
    """Fixture providing test data for checkpoint tests."""
    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_id": "1",
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


@pytest.mark.parametrize("saver_name", SAVERS)
def test_combined_metadata(saver_name: str, test_data: dict[str, Any]) -> None:
    with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "__super_private_key": "super_private_value",
            },
            "metadata": {"run_id": "my_run_id"},
        }
        chkpnt: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "writes": {"foo": "bar"},
            "score": None,
        }
        saver.put(config, chkpnt, metadata, {})
        checkpoint = saver.get_tuple(config)
        assert checkpoint
        assert checkpoint.metadata == {
            **metadata,
            "run_id": "my_run_id",
        }


@pytest.mark.parametrize("saver_name", SAVERS)
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
        assert search_results_1[0].metadata == {
            **_exclude_keys(configs[0]["configurable"]),
            **metadata[0],
        }

        search_results_2 = list(saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == {
            **_exclude_keys(configs[1]["configurable"]),
            **metadata[1],
        }

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


@pytest.mark.parametrize("saver_name", SAVERS)
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


@pytest.mark.parametrize("saver_name", SAVERS)
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

        if saver_name != "shallow":
            assert result.checkpoint["channel_values"][TASKS] == ["w3v"]
        else:
            assert result.checkpoint.get("pending_sends") == ["w3v"]


@pytest.mark.parametrize("saver_name", SAVERS)
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


@pytest.mark.parametrize("saver_name", SAVERS)
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


@pytest.mark.parametrize("saver_name", SAVERS)
def test_write_with_different_checkpoint_ns_inserts(saver_name: str) -> None:
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


@pytest.mark.parametrize("saver_name", SAVERS)
def test_write_with_same_checkpoint_ns_updates(saver_name: str) -> None:
    with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-6",
                "checkpoint_id": "6",
                "checkpoint_ns": "first",
            }
        }

        chkpnt = empty_checkpoint()

        saver.put(config, chkpnt, {}, {})
        saver.put(config, chkpnt, {}, {})

        results = list(saver.list({}))

        assert len(results) == 1


def test_nonnull_migrations() -> None:
    _leading_comment_remover = re.compile(r"^/\*.*?\*/")
    for migration in PyMySQLSaver.MIGRATIONS:
        statement = _leading_comment_remover.sub("", migration).split()[0]
        assert statement.strip()


@pytest.mark.parametrize("saver_name", NON_SHALLOW_SAVERS)
def test_pending_sends_migration(saver_name: str) -> None:
    with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }

        # create the first checkpoint
        # and put some pending sends
        checkpoint_0 = empty_checkpoint()
        config = saver.put(config, checkpoint_0, {}, {})
        saver.put_writes(
            config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1"
        )
        saver.put_writes(config, [(TASKS, "send-3")], task_id="task-2")

        # check that fetching checkpoint_0 doesn't attach pending sends
        # (they should be attached to the next checkpoint)
        tuple_0 = saver.get_tuple(config)
        assert tuple_0
        assert tuple_0.checkpoint["channel_values"] == {}
        assert tuple_0.checkpoint["channel_versions"] == {}

        # create the second checkpoint
        checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
        config = saver.put(config, checkpoint_1, {}, {})

        # check that pending sends are attached to checkpoint_1
        checkpoint_1 = saver.get_tuple(config)
        assert checkpoint_1
        assert checkpoint_1.checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in checkpoint_1.checkpoint["channel_versions"]

        # check that list also applies the migration
        search_results = [
            c for c in saver.list({"configurable": {"thread_id": "thread-1"}})
        ]
        assert len(search_results) == 2
        assert search_results[-1].checkpoint["channel_values"] == {}
        assert search_results[-1].checkpoint["channel_versions"] == {}
        assert search_results[0].checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in search_results[0].checkpoint["channel_versions"]
