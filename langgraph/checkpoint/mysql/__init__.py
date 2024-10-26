import json
import threading
from contextlib import contextmanager
from typing import (
    Any,
    ContextManager,
    Generic,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    cast,
)

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.mysql.base import BaseMySQLSaver
from langgraph.checkpoint.mysql.utils import (
    deserialize_channel_values,
    deserialize_pending_sends,
    deserialize_pending_writes,
)
from langgraph.checkpoint.serde.base import SerializerProtocol


class DictCursor(Protocol):
    """
    Protocol that a cursor should implement.

    Modeled after DBAPICursor from Typeshed.
    """

    def execute(
        self,
        operation: str,
        parameters: Union[Sequence[Any], Mapping[str, Any]] = ...,
        /,
    ) -> object: ...
    def executemany(
        self, operation: str, seq_of_parameters: Sequence[Sequence[Any]], /
    ) -> object: ...
    def fetchone(self) -> Optional[dict[str, Any]]: ...
    def fetchall(self) -> Sequence[dict[str, Any]]: ...


C = TypeVar("C", bound=ContextManager, covariant=True)  # connection type
R = TypeVar("R", bound=ContextManager, covariant=True)  # cursor type


class ConnectionPool(Protocol, Generic[C]):
    """Protocol that a MySQL connection pool should implement."""

    def get_connection(self) -> C:
        """Gets a connection from the connection pool."""
        ...


Conn = Union[C, ConnectionPool[C]]


@contextmanager
def _get_connection(conn: Conn[C]) -> Iterator[C]:
    if hasattr(conn, "cursor"):
        yield cast(C, conn)
    elif hasattr(conn, "get_connection"):
        with cast(ConnectionPool[C], conn).get_connection() as _conn:
            yield _conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")


class BaseSyncMySQLSaver(BaseMySQLSaver, Generic[C, R]):
    lock: threading.Lock

    def __init__(
        self,
        conn: Conn[C],
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)

        self.conn = conn
        self.lock = threading.Lock()

    @staticmethod
    def _is_no_such_table_error(e: Exception) -> bool:
        raise NotImplementedError

    @contextmanager
    def _cursor(self) -> Iterator[R]:
        raise NotImplementedError

    def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the MySQL database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        with self._cursor() as cur_:
            cur = cast(DictCursor, cur_)
            try:
                cur.execute(
                    "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row is None:
                    version = -1
                else:
                    version = row["v"]
            except Exception as e:
                if not self._is_no_such_table_error(e):
                    raise
                version = -1
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
            ):
                cur.execute(migration)
                cur.execute(f"INSERT INTO checkpoint_migrations (v) VALUES ({v})")

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
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata. Defaults to None.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.

        Examples:
            >>> from langgraph.checkpoint.mysql import PyMySQLSaver
            >>> DB_URI = "mysql://mysql:mysql@localhost:5432/mysql"
            >>> with PyMySQLSaver.from_conn_string(DB_URI) as memory:
            ... # Run a graph, then list the checkpoints
            >>>     config = {"configurable": {"thread_id": "1"}}
            >>>     checkpoints = list(memory.list(config, limit=2))
            >>> print(checkpoints)
            [CheckpointTuple(...), CheckpointTuple(...)]

            >>> config = {"configurable": {"thread_id": "1"}}
            >>> before = {"configurable": {"checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875"}}
            >>> with PyMySQLSaver.from_conn_string(DB_URI) as memory:
            ... # Run a graph, then list the checkpoints
            >>>     checkpoints = list(memory.list(config, before=before))
            >>> print(checkpoints)
            [CheckpointTuple(...), ...]
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        # if we change this to use .stream() we need to make sure to close the cursor
        with self._cursor() as cur_:
            cur = cast(DictCursor, cur_)
            cur.execute(query, args)
            values = cur.fetchall()
            for value in values:
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    self._load_checkpoint(
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
                    self._load_writes(
                        deserialize_pending_writes(value["pending_writes"])
                    ),
                )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the MySQL database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and timestamp is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

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
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        with self._cursor() as cur_:
            cur = cast(DictCursor, cur_)
            cur.execute(
                self.SELECT_SQL + where,
                args,
            )
            values = cur.fetchall()
            for value in values:
                return CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    self._load_checkpoint(
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
                    self._load_writes(
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
        with the provided config and its parent config (if any).

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
            >>> with PyMySQLSaver.from_conn_string(DB_URI) as memory:
            >>>     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            >>>     checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "channel_values": {"key": "value"}}
            >>>     saved_config = memory.put(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
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

        with self._cursor() as cur_:
            cur = cast(DictCursor, cur_)
            cur.executemany(
                self.UPSERT_CHECKPOINT_BLOBS_SQL,
                self._dump_blobs(
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
                    checkpoint["id"],
                    checkpoint_id,
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
        with self._cursor() as cur_:
            cur = cast(DictCursor, cur_)
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
