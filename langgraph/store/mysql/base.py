import asyncio
import json
import logging
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import (
    Any,
    Callable,
    ContextManager,
    Generic,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
)

import orjson
from typing_extensions import TypedDict

from langgraph.checkpoint.mysql import _ainternal as _ainternal
from langgraph.checkpoint.mysql import _internal as _internal
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

logger = logging.getLogger(__name__)


MIGRATIONS: Sequence[str] = [
    """
CREATE TABLE IF NOT EXISTS store (
    -- 'prefix' represents the doc's 'namespace'
    prefix VARCHAR(500) NOT NULL,
    `key` VARCHAR(150) NOT NULL,
    value JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, `key`)
);
""",
    """
-- For faster lookups by prefix
CREATE INDEX store_prefix_idx ON store (prefix) USING btree;
""",
]


class DictCursor(ContextManager, Protocol):
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


C = TypeVar("C", bound=Union[_internal.Conn, _ainternal.Conn])  # connection type
R = TypeVar("R", bound=DictCursor)  # cursor type


class BaseMySQLStore(Generic[C]):
    MIGRATIONS = MIGRATIONS
    conn: C
    _deserializer: Optional[Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]]

    def _get_batch_GET_ops_queries(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
    ) -> list[tuple[str, tuple, tuple[str, ...], list]]:
        namespace_groups = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key))
        results = []
        for namespace, items in namespace_groups.items():
            _, keys = zip(*items)
            keys_to_query = ",".join(["%s"] * len(keys))
            query = f"""
                SELECT `key`, value, created_at, updated_at
                FROM store
                WHERE prefix = %s AND `key` IN ({keys_to_query})
            """
            params = (_namespace_to_text(namespace), *keys)
            results.append((query, params, namespace, items))
        return results

    def _prepare_batch_PUT_queries(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> list[tuple[str, Sequence]]:
        # Last-write wins
        dedupped_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in put_ops:
            dedupped_ops[(op.namespace, op.key)] = op

        inserts: list[PutOp] = []
        deletes: list[PutOp] = []
        for op in dedupped_ops.values():
            if op.value is None:
                deletes.append(op)
            else:
                inserts.append(op)

        queries: list[tuple[str, Sequence]] = []

        if deletes:
            namespace_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for op in deletes:
                namespace_groups[op.namespace].append(op.key)
            for namespace, keys in namespace_groups.items():
                placeholders = ",".join(["%s"] * len(keys))
                query = (
                    f"DELETE FROM store WHERE prefix = %s AND `key` IN ({placeholders})"
                )
                params = (_namespace_to_text(namespace), *keys)
                queries.append((query, params))
        if inserts:
            values = []
            insertion_params = []
            for op in inserts:
                values.append("(%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)")
                insertion_params.extend(
                    [
                        _namespace_to_text(op.namespace),
                        op.key,
                        json.dumps(op.value),
                    ]
                )
            values_str = ",".join(values)
            query = f"""
                INSERT INTO store (prefix, `key`, value, created_at, updated_at)
                VALUES {values_str} AS new
                ON DUPLICATE KEY UPDATE
                value = new.value, updated_at = CURRENT_TIMESTAMP
            """
            queries.append((query, insertion_params))

        return queries

    def _prepare_batch_search_queries(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
    ) -> list[tuple[str, Sequence]]:
        queries: list[tuple[str, Sequence]] = []
        for _, op in search_ops:
            # Build filter conditions first
            filter_params = []
            filter_conditions = []
            if op.filter:
                for key, value in op.filter.items():
                    if isinstance(value, dict):
                        for op_name, val in value.items():
                            condition, filter_params_ = self._get_filter_condition(
                                key, op_name, val
                            )
                            filter_conditions.append(condition)
                            filter_params.extend(filter_params_)
                    else:
                        filter_conditions.append(
                            "json_extract(value, concat('$.', %s)) = CAST(%s AS JSON)"
                        )
                        filter_params.extend([key, json.dumps(value)])

            base_query = """
                SELECT prefix, `key`, value, created_at, updated_at
                FROM store
                WHERE prefix LIKE %s
            """
            params: list = [f"{_namespace_to_text(op.namespace_prefix)}%"]

            if filter_conditions:
                params.extend(filter_params)
                base_query += " AND " + " AND ".join(filter_conditions)

            base_query += " ORDER BY updated_at DESC"
            base_query += " LIMIT %s OFFSET %s"
            params.extend([op.limit, op.offset])

            queries.append((base_query, params))

        return queries

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, Sequence]]:
        queries: list[tuple[str, Sequence]] = []
        for _, op in list_ops:
            query = """
                SELECT truncated_prefix, MIN(prefix)
                FROM (
                    SELECT
                        prefix,
                        CASE
                            WHEN %s IS NOT NULL THEN
                                SUBSTRING_INDEX(prefix, '.', %s)
                            ELSE prefix
                        END AS truncated_prefix
                    FROM store
            """
            params: list[Any] = [op.max_depth, op.max_depth]

            conditions = []
            if op.match_conditions:
                for condition in op.match_conditions:
                    if condition.match_type == "prefix":
                        conditions.append("prefix LIKE %s")
                        params.append(
                            f"{_namespace_to_text(condition.path, handle_wildcards=True)}%"
                        )
                    elif condition.match_type == "suffix":
                        conditions.append("prefix LIKE %s")
                        params.append(
                            f"%{_namespace_to_text(condition.path, handle_wildcards=True)}"
                        )
                    else:
                        logger.warning(
                            f"Unknown match_type in list_namespaces: {condition.match_type}"
                        )

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += ") AS subquery "

            query += " GROUP BY truncated_prefix"
            query += " ORDER BY truncated_prefix LIMIT %s OFFSET %s"
            params.extend([op.limit, op.offset])
            queries.append((query, tuple(params)))

        return queries

    def _get_filter_condition(self, key: str, op: str, value: Any) -> tuple[str, list]:
        """Helper to generate filter conditions."""
        if op == "$eq":
            return "json_extract(value, concat('$.', %s)) = CAST(%s AS JSON)", [
                key,
                json.dumps(value),
            ]
        elif op == "$gt":
            return "CAST(json_extract(value, concat('$.', %s)) AS CHAR) > %s", [
                key,
                str(value),
            ]
        elif op == "$gte":
            return "CAST(json_extract(value, concat('$.', %s)) AS CHAR) >= %s", [
                key,
                str(value),
            ]
        elif op == "$lt":
            return "CAST(json_extract(value, concat('$.', %s)) AS CHAR) < %s", [
                key,
                str(value),
            ]
        elif op == "$lte":
            return "CAST(json_extract(value, concat('$.', %s)) AS CHAR) <= %s", [
                key,
                str(value),
            ]
        elif op == "$ne":
            return "json_extract(value, concat('$.', %s)) != CAST(%s AS JSON)", [
                key,
                json.dumps(value),
            ]
        else:
            raise ValueError(f"Unsupported operator: {op}")


class BaseSyncMySQLStore(
    BaseStore, BaseMySQLStore[_internal.Conn[_internal.C]], Generic[_internal.C, R]
):
    __slots__ = ("_deserializer", "lock")

    def __init__(
        self,
        conn: _internal.Conn[_internal.C],
        *,
        deserializer: Optional[
            Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]
        ] = None,
    ) -> None:
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.lock = threading.Lock()

    @staticmethod
    def _get_cursor_from_connection(conn: _internal.C) -> R:
        raise NotImplementedError

    @contextmanager
    def _cursor(self, *, pipeline: bool = False) -> Iterator[R]:
        """Create a database cursor as a context manager.

        Args:
            pipeline (bool): whether to use transaction context manager and handle concurrency
        """
        with _internal.get_connection(self.conn) as conn:
            if pipeline:
                # a connection can only be used by one
                # thread/coroutine at a time, so we acquire a lock
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

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        with self._cursor(pipeline=True) as cur:
            if GetOp in grouped_ops:
                self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results, cur
                )

            if SearchOp in grouped_ops:
                self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )
            if PutOp in grouped_ops:
                self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), cur
                )

        return results

    def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: R,
    ) -> None:
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            cur.execute(query, params)
            rows = cast(list[Row], cur.fetchall())
            key_to_row = {row["key"]: row for row in rows}
            for idx, key in items:
                row = key_to_row.get(key)
                if row:
                    results[idx] = _row_to_item(
                        namespace, row, loader=self._deserializer
                    )
                else:
                    results[idx] = None

    def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cur: R,
    ) -> None:
        queries = self._prepare_batch_PUT_queries(put_ops)
        for query, params in queries:
            cur.execute(query, params)

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: R,
    ) -> None:
        for (query, params), (idx, _) in zip(
            self._prepare_batch_search_queries(search_ops), search_ops
        ):
            cur.execute(query, params)
            rows = cast(list[Row], cur.fetchall())
            results[idx] = [
                _row_to_search_item(
                    _decode_ns_bytes(row["prefix"]), row, loader=self._deserializer
                )
                for row in rows
            ]

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: R,
    ) -> None:
        for (query, params), (idx, _) in zip(
            self._get_batch_list_namespaces_queries(list_ops), list_ops
        ):
            cur.execute(query, params)
            rows = cast(list[dict], cur.fetchall())
            results[idx] = [_decode_ns_bytes(row["truncated_prefix"]) for row in rows]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)

    def setup(self) -> None:
        """Set up the store database.

        This method creates the necessary tables in the MySQL database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """

        def _get_version(cur: R, table: str) -> int:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    v INTEGER PRIMARY KEY
                )
            """
            )
            cur.execute(f"SELECT v FROM {table} ORDER BY v DESC LIMIT 1")
            row = cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            return version

        with self._cursor() as cur:
            version = _get_version(cur, table="store_migrations")
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                cur.execute(sql)
                cur.execute("INSERT INTO store_migrations (v) VALUES (%s)", (v,))


class Row(TypedDict):
    key: str
    value: Any
    prefix: str
    created_at: datetime
    updated_at: datetime


def _namespace_to_text(
    namespace: tuple[str, ...], handle_wildcards: bool = False
) -> str:
    """Convert namespace tuple to text string."""
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)
    return ".".join(namespace)


def _row_to_item(
    namespace: tuple[str, ...],
    row: Row,
    *,
    loader: Optional[Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]] = None,
) -> Item:
    """Convert a row from the database into an Item."""
    loader = loader or _json_loads
    val = row["value"]
    return Item(
        value=val if isinstance(val, dict) else loader(val),
        key=row["key"],
        namespace=namespace,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_search_item(
    namespace: tuple[str, ...],
    row: Row,
    *,
    loader: Optional[Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]] = None,
) -> SearchItem:
    """Convert a row from the database into an Item."""
    loader = loader or _json_loads
    val = row["value"]
    score = row.get("score")
    if score is not None:
        try:
            score = float(score)  # type: ignore[arg-type]
        except ValueError:
            logger.warning("Invalid score: %s", score)
            score = None
    return SearchItem(
        value=val if isinstance(val, dict) else loader(val),
        key=row["key"],
        namespace=namespace,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        score=score,
    )


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    tot = 0
    for idx, op in enumerate(ops):
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot


def _json_loads(content: Union[bytes, orjson.Fragment]) -> Any:
    if isinstance(content, orjson.Fragment):
        if hasattr(content, "buf"):
            content = content.buf
        else:
            if isinstance(content.contents, bytes):
                content = content.contents
            else:
                content = content.contents.encode()
    return orjson.loads(cast(bytes, content))


def _decode_ns_bytes(namespace: Union[str, bytes, list]) -> tuple[str, ...]:
    if isinstance(namespace, list):
        return tuple(namespace)
    if isinstance(namespace, bytes):
        namespace = namespace.decode()[1:]
    return tuple(namespace.split("."))
