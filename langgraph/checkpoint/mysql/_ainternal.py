"""Shared async utility functions for the MySQL checkpoint & storage classes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    Union,
    cast,
)


class AsyncDictCursor(AbstractAsyncContextManager, Protocol):
    """
    Protocol that a cursor should implement.

    Modeled after DBAPICursor from Typeshed.
    """

    async def execute(
        self,
        operation: str,
        parameters: Sequence[Any] | Mapping[str, Any] = ...,
        /,
    ) -> object: ...
    async def executemany(
        self, operation: str, seq_of_parameters: Sequence[Sequence[Any]], /
    ) -> object: ...
    async def fetchone(self) -> dict[str, Any] | None: ...
    async def fetchall(self) -> Sequence[dict[str, Any]]: ...

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]: ...


R = TypeVar("R", bound=AsyncDictCursor)  # cursor type


class AsyncConnection(AbstractAsyncContextManager, Protocol):
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


C = TypeVar("C", bound=AsyncConnection)  # connection type
COut = TypeVar("COut", bound=AsyncConnection, covariant=True)  # connection type


class AsyncPool(Protocol, Generic[COut]):
    def acquire(self) -> COut:
        """Gets a connection from the connection pool."""
        ...


Conn = Union[C, AsyncPool[C]]


@asynccontextmanager
async def get_connection(
    conn: Conn[C],
) -> AsyncIterator[C]:
    if hasattr(conn, "cursor"):
        yield cast(C, conn)
    elif hasattr(conn, "acquire"):
        async with cast(AsyncPool[C], conn).acquire() as _conn:
            yield _conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
