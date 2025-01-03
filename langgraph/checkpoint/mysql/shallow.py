import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Generic, Optional

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.mysql import _ainternal, _internal
from langgraph.checkpoint.mysql.base import BaseMySQLSaver
from langgraph.checkpoint.mysql.utils import (
    deserialize_channel_values,
    deserialize_pending_sends,
    deserialize_pending_writes,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import TASKS

"""
To add a new migration, add a new string to the MIGRATIONS list.
The position of the migration in the list is the version number.
"""
MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);""",
    """CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id VARCHAR(150) NOT NULL,
    checkpoint_ns VARCHAR(2000) NOT NULL DEFAULT '',
    checkpoint_ns_hash BINARY(16) AS (UNHEX(MD5(checkpoint_ns))) STORED,
    type VARCHAR(150),
    checkpoint JSON NOT NULL,
    metadata JSON NOT NULL DEFAULT ('{}'),
    PRIMARY KEY (thread_id, checkpoint_ns_hash)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id VARCHAR(150) NOT NULL,
    checkpoint_ns VARCHAR(2000) NOT NULL DEFAULT '',
    checkpoint_ns_hash BINARY(16) AS (UNHEX(MD5(checkpoint_ns))) STORED,
    channel VARCHAR(150) NOT NULL,
    type VARCHAR(150) NOT NULL,
    `blob` LONGBLOB,
    PRIMARY KEY (thread_id, checkpoint_ns_hash, channel)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id VARCHAR(150) NOT NULL,
    checkpoint_ns VARCHAR(2000) NOT NULL DEFAULT '',
    checkpoint_ns_hash BINARY(16) AS (UNHEX(MD5(checkpoint_ns))) STORED,
    checkpoint_id VARCHAR(150) NOT NULL,
    task_id VARCHAR(150) NOT NULL,
    idx INTEGER NOT NULL,
    channel VARCHAR(150) NOT NULL,
    type VARCHAR(150),
    `blob` LONGBLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns_hash, checkpoint_id, task_id, idx)
);""",
    """
    CREATE INDEX checkpoints_thread_id_idx ON checkpoints (thread_id);
    """,
    """
    CREATE INDEX checkpoint_blobs_thread_id_idx ON checkpoint_blobs (thread_id);
    """,
    """
    CREATE INDEX checkpoint_writes_thread_id_idx ON checkpoint_writes (thread_id);
    """,
]

SELECT_SQL = f"""
select
    thread_id,
    checkpoint,
    checkpoint_ns,
    metadata,
    (
        select json_arrayagg(json_array(bl.channel, bl.type, bl.blob))
        from json_table(
            json_keys(checkpoint, '$.channel_versions'),
            '$[*]' columns (channel VARCHAR(150) PATH '$')
        ) as channels
        inner join checkpoint_blobs bl
            on bl.channel = channels.channel
        where bl.thread_id = checkpoints.thread_id
            and bl.checkpoint_ns_hash = checkpoints.checkpoint_ns_hash
    ) as channel_values,
    (
        select
        json_arrayagg(json_array(cw.task_id, cw.channel, cw.type, cw.blob, cw.idx))
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns_hash = checkpoints.checkpoint_ns_hash
            and cw.checkpoint_id = checkpoint->>'$.id'
    ) as pending_writes,
    (
        select json_arrayagg(json_array(cw.task_id, cw.type, cw.blob, cw.idx))
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns_hash = checkpoints.checkpoint_ns_hash
            and cw.channel = '{TASKS}'
    ) as pending_sends
from checkpoints """

UPSERT_CHECKPOINT_BLOBS_SQL = """
    INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, type, `blob`)
    VALUES (%s, %s, %s, %s, %s) AS new
    ON DUPLICATE KEY UPDATE
        type = new.type,
        `blob` = new.blob;
"""

UPSERT_CHECKPOINTS_SQL = """
    INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint, metadata)
    VALUES (%s, %s, %s, %s) AS new
    ON DUPLICATE KEY UPDATE
        checkpoint = new.checkpoint,
        metadata = new.metadata;
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, `blob`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) AS new
    ON DUPLICATE KEY UPDATE
        channel = new.channel,
        type = new.type,
        `blob` = new.blob;
"""

INSERT_CHECKPOINT_WRITES_SQL = """
    INSERT IGNORE INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, `blob`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def _dump_blobs(
    serde: SerializerProtocol,
    thread_id: str,
    checkpoint_ns: str,
    values: dict[str, Any],
    versions: ChannelVersions,
) -> list[tuple[str, str, str, str, Optional[bytes]]]:
    if not versions:
        return []

    return [
        (
            thread_id,
            checkpoint_ns,
            k,
            *(serde.dumps_typed(values[k]) if k in values else ("empty", None)),
        )
        for k in versions
    ]


class BaseShallowSyncMySQLSaver(BaseMySQLSaver, Generic[_internal.C, _internal.R]):
    """A checkpoint saver that uses MySQL to store checkpoints.
    This checkpointer ONLY stores the most recent checkpoint and does NOT retain any history.
    It is meant to be a light-weight drop-in replacement for the PostgresSaver that
    supports most of the LangGraph persistence functionality with the exception of time travel.
    """

    SELECT_SQL = SELECT_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    lock: threading.Lock

    def __init__(
        self,
        conn: _internal.Conn,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)

        self.conn = conn
        self.lock = threading.Lock()

    @staticmethod
    def _get_cursor_from_connection(conn: _internal.C) -> _internal.R:
        raise NotImplementedError

    @contextmanager
    def _cursor(self, *, pipeline: bool = False) -> Iterator[_internal.R]:
        """Create a database cursor as a context manager.

        Args:
            pipeline (bool): whether to use transaction context manager and handle concurrency
        """
        with _internal.get_connection(self.conn) as conn:
            if pipeline:
                with self.lock:
                    conn.begin()
                    try:
                        with self._get_cursor_from_connection(conn) as cur:
                            yield cur
                        conn.commit()
                    except:
                        conn.rollback()
                        raise
            else:
                with self.lock, self._get_cursor_from_connection(conn) as cur:
                    yield cur

    def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the MySQL database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        with self._cursor() as cur:
            cur.execute(self.MIGRATIONS[0])
            cur.execute("SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1")
            row = cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
            ):
                cur.execute(migration)
                cur.execute(f"INSERT INTO checkpoint_migrations (v) VALUES ({v})")
                cur.execute("COMMIT")

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the MySQL database based
        on the provided config. For shallow savers, this method returns a list with
        ONLY the most recent checkpoint.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where
        if limit:
            query += f" LIMIT {limit}"
        with self._cursor() as cur:
            cur.execute(self.SELECT_SQL + where, args)
            values = cur.fetchall()
            for value in values:
                checkpoint = self._load_checkpoint(
                    json.loads(value["checkpoint"]),
                    deserialize_channel_values(value["channel_values"]),
                    deserialize_pending_sends(value["pending_sends"]),
                )
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": checkpoint["id"],
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=self._load_metadata(value["metadata"]),
                    pending_writes=self._load_writes(
                        deserialize_pending_writes(value["pending_writes"])
                    ),
                )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the MySQL database based on the
        provided config (matching the thread ID in the config).

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.

        Examples:

            Basic:
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> checkpoint_tuple = memory.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)

            With timestamp:

            >>> config = {
            ...    "configurable": {
            ...        "thread_id": "1",
            ...        "checkpoint_ns": "",
            ...        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            ...    }
            ... }
            >>> checkpoint_tuple = memory.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)
        """  # noqa
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        args = (thread_id, checkpoint_ns)
        where = "WHERE thread_id = %s AND checkpoint_ns_hash = UNHEX(MD5(%s))"

        with self._cursor() as cur:
            cur.execute(
                self.SELECT_SQL + where,
                args,
            )
            values = cur.fetchall()
            for value in values:
                checkpoint = self._load_checkpoint(
                    json.loads(value["checkpoint"]),
                    deserialize_channel_values(value["channel_values"]),
                    deserialize_pending_sends(value["pending_sends"]),
                )
                return CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint["id"],
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=self._load_metadata(value["metadata"]),
                    pending_writes=self._load_writes(
                        deserialize_pending_writes(value["pending_writes"])
                    ),
                )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the MySQL database. The checkpoint is associated
        with the provided config. For shallow savers, this method saves ONLY the most recent
        checkpoint and overwrites a previous checkpoint, if it exists.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Examples:

            >>> from langgraph.checkpoint.mysql import PyMySQLSaver
            >>> DB_URI = "mysql://mysql:mysql@localhost:5432/mysql"
            >>> with ShallowPyMySQLSaver.from_conn_string(DB_URI) as memory:
            >>>     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            >>>     checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "channel_values": {"key": "value"}}
            >>>     saved_config = memory.put(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        with self._cursor(pipeline=True) as cur:
            cur.execute(
                """DELETE FROM checkpoint_writes
                WHERE thread_id = %s AND checkpoint_ns_hash = UNHEX(MD5(%s)) AND checkpoint_id NOT IN (%s, %s)""",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    configurable.get("checkpoint_id", ""),
                ),
            )
            cur.executemany(
                self.UPSERT_CHECKPOINT_BLOBS_SQL,
                _dump_blobs(
                    self.serde,
                    thread_id,
                    checkpoint_ns,
                    copy.pop("channel_values"),  # type: ignore[misc]
                    new_versions,
                ),
            )
            cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    json.dumps(self._dump_checkpoint(copy)),
                    self._dump_metadata(metadata),
                ),
            )
        return next_config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the MySQL database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
        """
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        with self._cursor(pipeline=True) as cur:
            cur.executemany(
                query,
                self._dump_writes(
                    config["configurable"]["thread_id"],
                    config["configurable"]["checkpoint_ns"],
                    config["configurable"]["checkpoint_id"],
                    task_id,
                    writes,
                ),
            )


class BaseShallowAsyncMySQLSaver(BaseMySQLSaver, Generic[_ainternal.C, _ainternal.R]):
    """A checkpoint saver that uses MySQL to store checkpoints asynchronously.
    This checkpointer ONLY stores the most recent checkpoint and does NOT retain any history.
    It is meant to be a light-weight drop-in replacement for the async MySQL saver that
    supports most of the LangGraph persistence functionality with the exception of time travel.
    """

    SELECT_SQL = SELECT_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL
    lock: asyncio.Lock

    def __init__(
        self,
        conn: _ainternal.Conn,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)

        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    @staticmethod
    def _get_cursor_from_connection(conn: _ainternal.C) -> _ainternal.R:
        raise NotImplementedError

    @asynccontextmanager
    async def _cursor(self, *, pipeline: bool = False) -> AsyncIterator[_ainternal.R]:
        """Create a database cursor as a context manager.

        Args:
            pipeline (bool): whether to use transaction context manager and handle concurrency
        """
        async with _ainternal.get_connection(self.conn) as conn:
            if pipeline:
                async with self.lock:
                    await conn.begin()
                    try:
                        async with self._get_cursor_from_connection(conn) as cur:
                            yield cur
                        await conn.commit()
                    except:
                        await conn.rollback()
                        raise
            else:
                async with (
                    self.lock,
                    self._get_cursor_from_connection(conn) as cur,
                ):
                    yield cur

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.
        This method creates the necessary tables in the MySQL database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        async with self._cursor() as cur:
            await cur.execute(self.MIGRATIONS[0])
            await cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            row = await cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
            ):
                await cur.execute(migration)
                await cur.execute(f"INSERT INTO checkpoint_migrations (v) VALUES ({v})")

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.
        This method retrieves a list of checkpoint tuples from the MySQL database based
        on the provided config. For shallow savers, this method returns a list with
        ONLY the most recent checkpoint.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where
        if limit:
            query += f" LIMIT {limit}"
        async with self._cursor() as cur:
            await cur.execute(self.SELECT_SQL + where, args)
            async for value in cur:
                checkpoint = await asyncio.to_thread(
                    self._load_checkpoint,
                    json.loads(value["checkpoint"]),
                    deserialize_channel_values(value["channel_values"]),
                    deserialize_pending_sends(value["pending_sends"]),
                )
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": checkpoint["id"],
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=self._load_metadata(value["metadata"]),
                    pending_writes=await asyncio.to_thread(
                        self._load_writes,
                        deserialize_pending_writes(value["pending_writes"]),
                    ),
                )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.
        This method retrieves a checkpoint tuple from the MySQL database based on the
        provided config (matching the thread ID in the config).
        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.
        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        args = (thread_id, checkpoint_ns)
        where = "WHERE thread_id = %s AND checkpoint_ns_hash = UNHEX(MD5(%s))"

        async with self._cursor() as cur:
            await cur.execute(
                self.SELECT_SQL + where,
                args,
            )

            async for value in cur:
                checkpoint = await asyncio.to_thread(
                    self._load_checkpoint,
                    json.loads(value["checkpoint"]),
                    deserialize_channel_values(value["channel_values"]),
                    deserialize_pending_sends(value["pending_sends"]),
                )
                return CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint["id"],
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=self._load_metadata(value["metadata"]),
                    pending_writes=await asyncio.to_thread(
                        self._load_writes,
                        deserialize_pending_writes(value["pending_writes"]),
                    ),
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.
        This method saves a checkpoint to the MySQL database. The checkpoint is associated
        with the provided config. For shallow savers, this method saves ONLY the most recent
        checkpoint and overwrites a previous checkpoint, if it exists.
        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.
        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")

        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        async with self._cursor(pipeline=True) as cur:
            await cur.execute(
                """DELETE FROM checkpoint_writes
                WHERE thread_id = %s AND checkpoint_ns_hash = UNHEX(MD5(%s)) AND checkpoint_id NOT IN (%s, %s)""",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    configurable.get("checkpoint_id", ""),
                ),
            )
            await cur.executemany(
                self.UPSERT_CHECKPOINT_BLOBS_SQL,
                _dump_blobs(
                    self.serde,
                    thread_id,
                    checkpoint_ns,
                    copy.pop("channel_values"),  # type: ignore[misc]
                    new_versions,
                ),
            )
            await cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    json.dumps(self._dump_checkpoint(copy)),
                    self._dump_metadata(metadata),
                ),
            )
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.
        This method saves intermediate writes associated with a checkpoint to the database.
        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        params = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            writes,
        )
        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(query, params)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.
        This method retrieves a list of checkpoint tuples from the MySQL database based
        on the provided config. For shallow savers, this method returns a list with
        ONLY the most recent checkpoint.
        """
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),  # noqa: F821
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.
        This method retrieves a checkpoint tuple from the MySQL database based on the
        provided config (matching the thread ID in the config).
        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.
        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to asynchronous shallow savers are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.
        This method saves a checkpoint to the MySQL database. The checkpoint is associated
        with the provided config. For shallow savers, this method saves ONLY the most recent
        checkpoint and overwrites a previous checkpoint, if it exists.
        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.
        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.
        This method saves intermediate writes associated with a checkpoint to the database.
        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()
