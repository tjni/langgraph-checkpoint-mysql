import asyncio
import logging
import urllib.parse
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    Optional,
    Sequence,
    Union,
    cast,
)

import aiomysql  # type: ignore
import orjson
import pymysql
import pymysql.constants.ER

from langgraph.checkpoint.mysql import _ainternal

from langgraph.store.base import GetOp, ListNamespacesOp, Op, PutOp, Result, SearchOp
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.mysql.base import (
    BaseMySQLStore,
    PoolConfig,
    Row,
    _decode_ns_bytes,
    _group_ops,
    _row_to_item,
)

logger = logging.getLogger(__name__)


class AIOMySQLStore(AsyncBatchedBaseStore, BaseMySQLStore[_ainternal.Conn]):
    __slots__ = ("_deserializer",)

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
        self.loop = asyncio.get_running_loop()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        # tasks = []

        async with _ainternal.get_connection(self.conn) as conn:
            await self._execute_batch(grouped_ops, results, conn)

    async def _execute_batch(
        self,
        grouped_ops: dict,
        results: list[Result],
        conn: aiomysql.Connection
    ) -> None:
        async with self._cursor(conn) as cur:
            if GetOp in grouped_ops:
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                    results,
                    cur
                )

            if PutOp in grouped_ops:
                await self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]),
                    cur
                )

            if SearchOp in grouped_ops:
                await self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur
                )

            if ListNamespacesOp in grouped_ops:
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur
                )

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return asyncio.run_coroutine_threadsafe(self.abatch(ops), self.loop).result()

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
        queries = self._get_batch_PUT_queries(put_ops)
        for query, params in queries:
            await cur.execute(query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: aiomysql.DictCursor,
    ) -> None:
        queries = self._get_batch_search_queries(search_ops)

        for (query, params), (idx, _) in zip(queries, search_ops):
            await cur.execute(query, params)

            rows = cast(list[Row], await cur.fetchall())
            items = [
                _row_to_item(
                    _decode_ns_bytes(row["prefix"]), row, loader=self._deserializer
                )
                for row in rows
            ]
            results[idx] = items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        cursors: list[tuple[aiomysql.DictCursor, int]] = []
        for (query, params), (idx, _) in zip(queries, list_ops):
            cur = await self._cursor()
            await cur.execute(query, params)
            cursors.append((cur, idx))

        for cur, idx in cursors:
            rows = cast(list[dict], await cur.fetchall())
            namespaces = [_decode_ns_bytes(row["truncated_prefix"]) for row in rows]
            results[idx] = namespaces


    @asynccontextmanager
    async def _cursor(
        self, conn: aiomysql.Connection
    ) -> AsyncIterator[aiomysql.DictCursor]:
        """Create a database cursor as a context manager.
        Args:
            conn: The database connection to use
        """
        async with conn.cursor(binary=True) as cur:
            yield cur

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pool_config: Optional[PoolConfig] = None,
    ) -> AsyncIterator["AIOMySQLStore"]:
        """Create a new AIOMySQLStore instance from a connection string.

        Args:
            conn_string (str): The MySQL connection info string.
            pool_config (Optional[PoolConfig]): Configuration for the connection pool.
                If provided, will create a connection pool and use it instead of a single connection.
        Returns:
            AIOMySQLStore: A new AIOMySQLStore instance.
        """
        logger.info(f"Creating AIOMySQLStore from connection string: {conn_string}")
        parsed = urllib.parse.urlparse(conn_string)

        # In order to provide additional params via the connection string,
        # we convert the parsed.query to a dict so we can access the values.
        # This is necessary when using a unix socket, for example.
        params_as_dict = dict(urllib.parse.parse_qsl(parsed.query))

        if pool_config is not None:
            pc = pool_config.copy()
            async with aiomysql.create_pool(
                host=parsed.hostname or "localhost",
                user=parsed.username,
                password=parsed.password or "",
                db=parsed.path[1:],
                port=parsed.port or 3306,
                unix_socket=params_as_dict.get("unix_socket"),
                autocommit=True
                **cast(dict, pc),
            ) as pool:
                pool.set_charset(pymysql.connections.DEFAULT_CHARSET)
                yield cls(conn=pool)
        else:
            async with aiomysql.connect(
                host=parsed.hostname or "localhost",
                user=parsed.username,
                password=parsed.password or "",
                db=parsed.path[1:],
                port=parsed.port or 3306,
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
        async with _ainternal.get_connection(self.conn) as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                try:
                    await cur.execute(
                        "SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1"
                    )
                    row = cast(dict, await cur.fetchone())
                    if row is None:
                        version = -1
                    else:
                        version = row["v"]
                except pymysql.ProgrammingError as e:
                    if e.args[0] != pymysql.constants.ER.NO_SUCH_TABLE:
                        raise
                    version = -1
                    # Create store_migrations table if it doesn't exist
                    await cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS store_migrations (
                            v INTEGER PRIMARY KEY
                        )
                        """
                    )
                for v, migration in enumerate(
                    self.MIGRATIONS[version + 1 :], start=version + 1
                ):
                    await cur.execute(migration)
                    await cur.execute("INSERT INTO store_migrations (v) VALUES (%s)", (v,))
