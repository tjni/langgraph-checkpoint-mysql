import asyncio
import logging
import urllib.parse
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional, Union, cast

import aiomysql  # type: ignore
import orjson
import pymysql

from langgraph.checkpoint.mysql import _ainternal
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.mysql.base import (
    BaseMySQLStore,
    Row,
    _decode_ns_bytes,
    _group_ops,
    _row_to_item,
    _row_to_search_item,
)

logger = logging.getLogger(__name__)


class AIOMySQLStore(AsyncBatchedBaseStore, BaseMySQLStore[_ainternal.Conn]):
    __slots__ = ("_deserializer", "lock")

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        deserializer: Optional[
            Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]
        ] = None,
    ) -> None:
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        async with _ainternal.get_connection(self.conn) as conn:
            await self._execute_batch(grouped_ops, results, conn)

        return results

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
    ) -> AsyncIterator["AIOMySQLStore"]:
        """Create a new AIOMySQLStore instance from a connection string.

        Args:
            conn_string (str): The MySQL connection info string.

        Returns:
            AIOMySQLStore: A new AIOMySQLStore instance.
        """
        async with aiomysql.connect(
            **cls.parse_conn_string(conn_string),
            autocommit=True,
        ) as conn:
            # This seems necessary until https://github.com/PyMySQL/PyMySQL/pull/1119
            # is merged into aiomysql.
            await conn.set_charset(pymysql.connections.DEFAULT_CHARSET)

            yield cls(conn=conn)

    async def setup(self) -> None:
        """Set up the store database asynchronously.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """

        async def _get_version(cur: aiomysql.DictCursor, table: str) -> int:
            await cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    v INTEGER PRIMARY KEY
                )
            """
            )
            await cur.execute(f"SELECT v FROM {table} ORDER BY v DESC LIMIT 1")
            row = await cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            return version

        async with _ainternal.get_connection(self.conn) as conn:
            async with self._cursor(conn) as cur:
                version = await _get_version(cur, table="store_migrations")
                for v, sql in enumerate(
                    self.MIGRATIONS[version + 1 :], start=version + 1
                ):
                    await cur.execute(sql)
                    await cur.execute(
                        "INSERT INTO store_migrations (v) VALUES (%s)", (v,)
                    )

    async def _execute_batch(
        self,
        grouped_ops: dict,
        results: list[Result],
        conn: aiomysql.Connection,
    ) -> None:
        async with self._cursor(conn, pipeline=True) as cur:
            if GetOp in grouped_ops:
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                    results,
                    cur,
                )

            if SearchOp in grouped_ops:
                await self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                await self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )

            if PutOp in grouped_ops:
                await self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]),
                    cur,
                )

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: aiomysql.DictCursor,
    ) -> None:
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            await cur.execute(query, params)
            rows = cast(list[Row], await cur.fetchall())
            key_to_row = {row["key"]: row for row in rows}
            for idx, key in items:
                row = key_to_row.get(key)
                if row:
                    results[idx] = _row_to_item(
                        namespace, row, loader=self._deserializer
                    )
                else:
                    results[idx] = None

    async def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cur: aiomysql.DictCursor,
    ) -> None:
        queries = self._prepare_batch_PUT_queries(put_ops)
        for query, params in queries:
            await cur.execute(query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: aiomysql.DictCursor,
    ) -> None:
        queries = self._prepare_batch_search_queries(search_ops)
        for (idx, _), (query, params) in zip(search_ops, queries):
            await cur.execute(query, params)
            rows = cast(list[Row], await cur.fetchall())
            items = [
                _row_to_search_item(
                    _decode_ns_bytes(row["prefix"]), row, loader=self._deserializer
                )
                for row in rows
            ]
            results[idx] = items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: aiomysql.DictCursor,
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for (query, params), (idx, _) in zip(queries, list_ops):
            await cur.execute(query, params)
            rows = cast(list[dict], await cur.fetchall())
            namespaces = [_decode_ns_bytes(row["truncated_prefix"]) for row in rows]
            results[idx] = namespaces

    @asynccontextmanager
    async def _cursor(
        self, conn: aiomysql.Connection, *, pipeline: bool = False
    ) -> AsyncIterator[aiomysql.DictCursor]:
        """Create a database cursor as a context manager.
        Args:
            conn: The database connection to use
            pipeline: whether to use transaction context manager and handle concurrency
        """
        if pipeline:
            # a connection can only be used by one
            # thread/coroutine at a time, so we acquire a lock
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
            async with self.lock, conn.cursor(aiomysql.DictCursor) as cur:
                yield cur
