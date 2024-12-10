"""Shared utility functions for the MySQL checkpoint & storage classes."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import ContextManager, Generic, Protocol, TypeVar, Union, cast


class Connection(ContextManager, Protocol):
    """Protocol that a MySQL connection should implement."""

    def begin(self) -> None:
        """Begin transaction."""
        ...

    def commit(self) -> None:
        """Commit changes to stable storage."""
        ...

    def rollback(self) -> None:
        """Roll back the current transaction."""
        ...


COut = TypeVar("COut", bound=Connection, covariant=True)  # connection type
C = TypeVar("C", bound=Connection)  # connection type


class ConnectionPool(Protocol, Generic[COut]):
    """Protocol that a MySQL connection pool should implement."""

    def get_connection(self) -> COut:
        """Gets a connection from the connection pool."""
        ...


Conn = Union[C, ConnectionPool[C]]


@contextmanager
def get_connection(conn: Conn[C]) -> Iterator[C]:
    if hasattr(conn, "cursor"):
        yield cast(C, conn)
    elif hasattr(conn, "get_connection"):
        with cast(ConnectionPool[C], conn).get_connection() as _conn:
            yield _conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
