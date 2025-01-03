"""Shared async utility functions for the MySQL checkpoint & storage classes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncContextManager,
    Generic,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    cast,
)


class AsyncDictCursor(AsyncContextManager, Protocol):
    """
    Protocol that a cursor should implement.

    Modeled after DBAPICursor from Typeshed.
    """

    async def execute(
        self,
        operation: str,
        parameters: Union[Sequence[Any], Mapping[str, Any]] = ...,
        /,
    ) -> object: ...
    async def executemany(
        self, operation: str, seq_of_parameters: Sequence[Sequence[Any]], /
    ) -> object: ...
    async def fetchone(self) -> Optional[dict[str, Any]]: ...
    async def fetchall(self) -> Sequence[dict[str, Any]]: ...

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]: ...


R = TypeVar("R", bound=AsyncDictCursor)  # cursor type


class AIOMySQLConnection(AsyncContextManager, Protocol):
    """From aiomysql package."""

    async def begin(self) -> None:
        """Begin transaction."""
        ...

    async def commit(self) -> None:
        """Commit changes to stable storage."""
        ...

    async def rollback(self) -> None:
        """Roll back the current transaction."""
        ...

    async def set_charset(self, charset: str) -> None:
        """Sets the character set for the current connection"""
        ...


C = TypeVar("C", bound=AIOMySQLConnection)  # connection type
COut = TypeVar("COut", bound=AIOMySQLConnection, covariant=True)  # connection type


class AIOMySQLPool(Protocol, Generic[COut]):
    """From aiomysql package."""

    def acquire(self) -> COut:
        """Gets a connection from the connection pool."""
        ...


Conn = Union[C, AIOMySQLPool[C]]


@asynccontextmanager
async def get_connection(
    conn: Conn[C],
) -> AsyncIterator[C]:
    if hasattr(conn, "cursor"):
        yield cast(C, conn)
    elif hasattr(conn, "acquire"):
        async with cast(AIOMySQLPool[C], conn).acquire() as _conn:
            # This seems necessary until https://github.com/PyMySQL/PyMySQL/pull/1119
            # is merged into aiomysql.
            await _conn.set_charset("utf8mb4")
            yield _conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
