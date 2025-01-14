import asyncio
import json
import urllib.parse
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Optional, cast

import aiomysql  # type: ignore
from langchain_core.runnables import RunnableConfig
from typing_extensions import Self, override

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.mysql import _ainternal
from langgraph.checkpoint.mysql.base import BaseMySQLSaver
from langgraph.checkpoint.mysql.shallow import BaseShallowAsyncMySQLSaver
from langgraph.checkpoint.mysql.utils import (
    deserialize_channel_values,
    deserialize_pending_sends,
    deserialize_pending_writes,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

Conn = _ainternal.Conn[aiomysql.Connection]  # For backward compatibility


class AIOMySQLSaver(BaseMySQLSaver):
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
    def parse_conn_string(conn_string: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(conn_string)

        # In order to provide additional params via the connection string,
        # we convert the parsed.query to a dict so we can access the values.
        # This is necessary when using a unix socket, for example.
        params_as_dict = dict(urllib.parse.parse_qsl(parsed.query))

        return {
            "host": parsed.hostname or "localhost",
            "user": parsed.username,
            "password": parsed.password or "",
            "db": parsed.path[1:] or None,
            "port": parsed.port or 3306,
            "unix_socket": params_as_dict.get("unix_socket"),
        }

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> AsyncIterator[Self]:
        """Create a new AIOMySQLSaver instance from a connection string.

        Args:
            conn_string (str): The MySQL connection info string.

        Returns:
            AIOMySQLSaver: A new AIOMySQLSaver instance.

        Example:
            conn_string=mysql+aiomysql://user:password@localhost/db?unix_socket=/path/to/socket
        """
        async with aiomysql.connect(
            **cls.parse_conn_string(conn_string),
            autocommit=True,
        ) as conn:
            yield cls(conn=conn, serde=serde)

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
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        # if we change this to use .stream() we need to make sure to close the cursor
        async with self._cursor() as cur:
            await cur.execute(query, args)
            async for value in cur:
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        json.loads(value["checkpoint"]),
                        deserialize_channel_values(value["channel_values"]),
                        deserialize_pending_sends(value["pending_sends"]),
                    ),
                    self._load_metadata(value["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": value["thread_id"],
                                "checkpoint_ns": value["checkpoint_ns"],
                                "checkpoint_id": value["parent_checkpoint_id"],
                            }
                        }
                        if value["parent_checkpoint_id"]
                        else None
                    ),
                    await asyncio.to_thread(
                        self._load_writes,
                        deserialize_pending_writes(value["pending_writes"]),
                    ),
                )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the MySQL database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns_hash = UNHEX(MD5(%s)) AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns_hash = UNHEX(MD5(%s)) ORDER BY checkpoint_id DESC LIMIT 1"

        async with self._cursor() as cur:
            await cur.execute(
                self.SELECT_SQL + where,
                args,
            )

            async for value in cur:
                return CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        json.loads(value["checkpoint"]),
                        deserialize_channel_values(value["channel_values"]),
                        deserialize_pending_sends(value["pending_sends"]),
                    ),
                    self._load_metadata(value["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": value["parent_checkpoint_id"],
                            }
                        }
                        if value["parent_checkpoint_id"]
                        else None
                    ),
                    await asyncio.to_thread(
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
        with the provided config and its parent config (if any).

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
        checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )

        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(
                self.UPSERT_CHECKPOINT_BLOBS_SQL,
                await asyncio.to_thread(
                    self._dump_blobs,
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
                    checkpoint["id"],
                    checkpoint_id,
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

    @asynccontextmanager
    async def _cursor(
        self, *, pipeline: bool = False
    ) -> AsyncIterator[aiomysql.DictCursor]:
        """Create a database cursor as a context manager.

        Args:
            pipeline (bool): whether to use transaction context manager and handle concurrency
        """
        async with _ainternal.get_connection(self.conn) as conn:
            if pipeline:
                async with self.lock:
                    await conn.begin()
                    try:
                        async with conn.cursor(aiomysql.DictCursor) as cur:
                            yield cur
                        await conn.commit()
                    except:
                        await conn.rollback()
                        raise
            else:
                async with (
                    self.lock,
                    conn.cursor(aiomysql.DictCursor) as cur,
                ):
                    yield cur

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
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
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
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

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
                    "Synchronous calls to AIOMySQLSaver are only allowed from a "
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
        with the provided config and its parent config (if any).

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


class ShallowAIOMySQLSaver(
    BaseShallowAsyncMySQLSaver[aiomysql.Connection, aiomysql.DictCursor]
):
    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> AsyncIterator[Self]:
        """Create a new ShallowAIOMySQLSaver instance from a connection string.

        Args:
            conn_string (str): The MySQL connection info string.

        Returns:
            ShallowAIOMySQLSaver: A new ShallowAIOMySQLSaver instance.

        Example:
            conn_string=mysql+aiomysql://user:password@localhost/db?unix_socket=/path/to/socket
        """
        async with aiomysql.connect(
            **AIOMySQLSaver.parse_conn_string(conn_string),
            autocommit=True,
        ) as conn:
            yield cls(conn=conn, serde=serde)

    @override
    @staticmethod
    def _get_cursor_from_connection(conn: aiomysql.Connection) -> aiomysql.DictCursor:
        return cast(aiomysql.DictCursor, conn.cursor(aiomysql.DictCursor))


__all__ = ["AIOMySQLSaver", "ShallowAIOMySQLSaver", "Conn"]
