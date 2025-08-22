from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any
from uuid import uuid4

import aiomysql  # type: ignore
import asyncmy  # type: ignore
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
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver, ShallowAIOMySQLSaver
from langgraph.checkpoint.mysql.aio_base import BaseAsyncMySQLSaver
from langgraph.checkpoint.mysql.asyncmy import AsyncMySaver, ShallowAsyncMySaver
from langgraph.checkpoint.mysql.shallow import BaseShallowAsyncMySQLSaver
from langgraph.checkpoint.serde.types import TASKS
from langgraph.graph import END, START, MessagesState, StateGraph
from tests.conftest import DEFAULT_BASE_URI

pytestmark = pytest.mark.anyio

SAVERS = [
    "aiomysql",
    "aiomysql_pool",
    "aiomysql_shallow",
    "asyncmy",
    "asyncmy_pool",
    "asyncmy_shallow",
]

NON_SHALLOW_SAVERS = [saver for saver in SAVERS if "shallow" not in saver]


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


@asynccontextmanager
async def _aiomysql_pool_saver() -> AsyncIterator[AIOMySQLSaver]:
    """Fixture for pool mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await aiomysql.connect(
        **AIOMySQLSaver.parse_conn_string(DEFAULT_BASE_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with aiomysql.create_pool(
            **AIOMySQLSaver.parse_conn_string(DEFAULT_BASE_URI + database),
            maxsize=10,
            autocommit=True,
        ) as pool:
            checkpointer = AIOMySQLSaver(pool)
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await aiomysql.connect(
            **AIOMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _aiomysql_saver() -> AsyncIterator[AIOMySQLSaver]:
    """Fixture for regular connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await aiomysql.connect(
        **AIOMySQLSaver.parse_conn_string(DEFAULT_BASE_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        async with AIOMySQLSaver.from_conn_string(
            DEFAULT_BASE_URI + database
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await aiomysql.connect(
            **AIOMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _aiomysql_shallow_saver() -> AsyncIterator[ShallowAIOMySQLSaver]:
    """Fixture for shallow connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await aiomysql.connect(
        **AIOMySQLSaver.parse_conn_string(DEFAULT_BASE_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        async with ShallowAIOMySQLSaver.from_conn_string(
            DEFAULT_BASE_URI + database
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await aiomysql.connect(
            **AIOMySQLSaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _asyncmy_pool_saver() -> AsyncIterator[AsyncMySaver]:
    """Fixture for pool mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await asyncmy.connect(
        **AsyncMySaver.parse_conn_string(DEFAULT_BASE_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with asyncmy.create_pool(
            **AsyncMySaver.parse_conn_string(DEFAULT_BASE_URI + database),
            maxsize=10,
            autocommit=True,
        ) as pool:
            checkpointer = AsyncMySaver(pool)
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await asyncmy.connect(
            **AsyncMySaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _asyncmy_saver() -> AsyncIterator[AsyncMySaver]:
    """Fixture for regular connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await asyncmy.connect(
        **AsyncMySaver.parse_conn_string(DEFAULT_BASE_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        async with AsyncMySaver.from_conn_string(
            DEFAULT_BASE_URI + database
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await asyncmy.connect(
            **AsyncMySaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _asyncmy_shallow_saver() -> AsyncIterator[ShallowAsyncMySaver]:
    """Fixture for shallow connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await asyncmy.connect(
        **AsyncMySaver.parse_conn_string(DEFAULT_BASE_URI),
        autocommit=True,
    ) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE {database}")
    try:
        async with ShallowAsyncMySaver.from_conn_string(
            DEFAULT_BASE_URI + database
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await asyncmy.connect(
            **AsyncMySaver.parse_conn_string(DEFAULT_BASE_URI), autocommit=True
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _saver(
    name: str,
) -> AsyncIterator[BaseAsyncMySQLSaver | BaseShallowAsyncMySQLSaver]:
    if name == "aiomysql":
        async with _aiomysql_saver() as saver:
            yield saver
    elif name == "aiomysql_shallow":
        async with _aiomysql_shallow_saver() as saver:
            yield saver
    elif name == "aiomysql_pool":
        async with _aiomysql_pool_saver() as saver:
            yield saver
    elif name == "asyncmy":
        async with _asyncmy_saver() as saver:
            yield saver
    elif name == "asyncmy_shallow":
        async with _asyncmy_shallow_saver() as saver:
            yield saver
    elif name == "asyncmy_pool":
        async with _asyncmy_pool_saver() as saver:
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
async def test_combined_metadata(saver_name: str, test_data: dict[str, Any]) -> None:
    async with _saver(saver_name) as saver:
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
        await saver.aput(config, chkpnt, metadata, {})
        checkpoint = await saver.aget_tuple(config)
        assert checkpoint
        assert checkpoint.metadata == {
            **metadata,
            "run_id": "my_run_id",
        }


@pytest.mark.parametrize("saver_name", SAVERS)
async def test_asearch(saver_name: str, test_data: dict[str, Any]) -> None:
    async with _saver(saver_name) as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        await saver.aput(configs[0], checkpoints[0], metadata[0], {})
        await saver.aput(configs[1], checkpoints[1], metadata[1], {})
        await saver.aput(configs[2], checkpoints[2], metadata[2], {})

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = [c async for c in saver.alist(None, filter=query_1)]
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == {
            **_exclude_keys(configs[0]["configurable"]),
            **metadata[0],
        }

        search_results_2 = [c async for c in saver.alist(None, filter=query_2)]
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == {
            **_exclude_keys(configs[1]["configurable"]),
            **metadata[1],
        }

        search_results_3 = [c async for c in saver.alist(None, filter=query_3)]
        assert len(search_results_3) == 3

        search_results_4 = [c async for c in saver.alist(None, filter=query_4)]
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = [
            c async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})
        ]
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


@pytest.mark.parametrize("saver_name", SAVERS)
async def test_null_chars(saver_name: str, test_data: dict[str, Any]) -> None:
    async with _saver(saver_name) as saver:
        config = await saver.aput(
            test_data["configs"][0],
            test_data["checkpoints"][0],
            {"my_key": "\x00abc"},
            {},
        )
        assert (await saver.aget_tuple(config)).metadata["my_key"] == "abc"  # type: ignore
        assert [c async for c in saver.alist(None, filter={"my_key": "abc"})][
            0
        ].metadata["my_key"] == "abc"


@pytest.mark.parametrize("saver_name", SAVERS)
async def test_write_and_read_pending_writes_and_sends(
    saver_name: str, test_data: dict[str, Any]
) -> None:
    async with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_id": "1",
                "checkpoint_ns": "",
            }
        }
        chkpnt = create_checkpoint(test_data["checkpoints"][0], {}, 1, id="1")

        await saver.aput(config, chkpnt, {}, {})
        await saver.aput_writes(config, [("w1", "w1v"), ("w2", "w2v")], "world")
        await saver.aput_writes(config, [(TASKS, "w3v")], "hello")

        result = [c async for c in saver.alist({})][0]

        assert result.pending_writes == [
            ("hello", TASKS, "w3v"),
            ("world", "w1", "w1v"),
            ("world", "w2", "w2v"),
        ]

        if "shallow" not in saver_name:
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
async def test_write_and_read_channel_values(
    saver_name: str, channel_values: dict[str, Any]
) -> None:
    async with _saver(saver_name) as saver:
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

        await saver.aput(config, chkpnt, {}, newversions)

        result = [c async for c in saver.alist({})][0]
        assert result.checkpoint["channel_values"] == channel_values


@pytest.mark.parametrize("saver_name", SAVERS)
async def test_write_and_read_pending_writes(saver_name: str) -> None:
    async with _saver(saver_name) as saver:
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

        await saver.aput(config, chkpnt, {}, {})
        await saver.aput_writes(config, writes, task_id)

        result = [c async for c in saver.alist({})][0]

        assert result.pending_writes == [
            (task_id, "channel1", "somevalue"),
            (task_id, "channel2", [1, 2, 3]),
            (task_id, "channel3", None),
        ]


@pytest.mark.parametrize("saver_name", SAVERS)
async def test_write_with_different_checkpoint_ns_inserts(
    saver_name: str,
) -> None:
    async with _saver(saver_name) as saver:
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

        await saver.aput(config1, chkpnt, {}, {})
        await saver.aput(config2, chkpnt, {}, {})

        results = [c async for c in saver.alist({})]

        assert len(results) == 2


@pytest.mark.parametrize("saver_name", SAVERS)
async def test_write_with_same_checkpoint_ns_updates(
    saver_name: str,
) -> None:
    async with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-6",
                "checkpoint_id": "6",
                "checkpoint_ns": "first",
            }
        }

        chkpnt = empty_checkpoint()

        await saver.aput(config, chkpnt, {}, {})
        await saver.aput(config, chkpnt, {}, {})

        results = [c async for c in saver.alist({})]

        assert len(results) == 1


@pytest.mark.parametrize("saver_name", SAVERS)
async def test_graph_sync_get_state_history_raises(saver_name: str) -> None:
    """Regression test for https://github.com/langchain-ai/langgraph/issues/2992"""

    builder = StateGraph(MessagesState)
    builder.add_node("foo", lambda state: None)
    builder.add_edge(START, "foo")
    builder.add_edge("foo", END)

    async with _saver(saver_name) as saver:
        graph = builder.compile(checkpointer=saver)
        config: RunnableConfig = {"configurable": {"thread_id": "1"}}
        input: MessagesState = {"messages": []}
        await graph.ainvoke(input, config)

        # this method should not hang
        with pytest.raises(asyncio.exceptions.InvalidStateError):
            next(graph.get_state_history(config))


@pytest.mark.parametrize("saver_name", NON_SHALLOW_SAVERS)
async def test_pending_sends_migration(saver_name: str) -> None:
    async with _saver(saver_name) as saver:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }

        # create the first checkpoint
        # and put some pending sends
        checkpoint_0 = empty_checkpoint()
        config = await saver.aput(config, checkpoint_0, {}, {})
        await saver.aput_writes(
            config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1"
        )
        await saver.aput_writes(config, [(TASKS, "send-3")], task_id="task-2")

        # check that fetching checkpoint_0 doesn't attach pending sends
        # (they should be attached to the next checkpoint)
        tuple_0 = await saver.aget_tuple(config)
        assert tuple_0
        assert tuple_0.checkpoint["channel_values"] == {}
        assert tuple_0.checkpoint["channel_versions"] == {}

        # create the second checkpoint
        checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
        config = await saver.aput(config, checkpoint_1, {}, {})

        # check that pending sends are attached to checkpoint_1
        tuple_1 = await saver.aget_tuple(config)
        assert tuple_1
        assert tuple_1.checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in tuple_1.checkpoint["channel_versions"]

        # check that list also applies the migration
        search_results = [
            c async for c in saver.alist({"configurable": {"thread_id": "thread-1"}})
        ]
        assert len(search_results) == 2
        assert search_results[-1].checkpoint["channel_values"] == {}
        assert search_results[-1].checkpoint["channel_versions"] == {}
        assert search_results[0].checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in search_results[0].checkpoint["channel_versions"]
