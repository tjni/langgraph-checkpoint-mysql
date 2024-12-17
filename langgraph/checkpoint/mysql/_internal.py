"""Shared utility functions for the MySQL checkpoint & storage classes."""

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
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


class MySQLConnectionPool(Protocol, Generic[COut]):
    """From mysql-connector-python package."""

    def get_connection(self) -> COut:
        """Gets a connection from the connection pool."""
        ...


class SQLAlchemyPoolProxiedConnection(Protocol):
    def close(self) -> None: ...


class SQLAlchemyConnectionPool(Protocol):
    """From sqlalchemy package."""

    def connect(self) -> SQLAlchemyPoolProxiedConnection:
        """Gets a connection from the connection pool."""
        ...


ConnectionFactory = Callable[[], C]
Conn = Union[C, ConnectionFactory[C], MySQLConnectionPool[C], SQLAlchemyConnectionPool]


@contextmanager
def get_connection(conn: Conn[C]) -> Iterator[C]:
    if hasattr(conn, "cursor"):
        yield cast(C, conn)
    elif hasattr(conn, "get_connection"):
        with cast(MySQLConnectionPool[C], conn).get_connection() as _conn:
            yield _conn
    elif hasattr(conn, "connect"):
        proxy_conn = cast(SQLAlchemyConnectionPool, conn).connect()
        with closing(proxy_conn) as _conn:
            yield cast(C, _conn)
    elif callable(conn):
        with conn() as _conn:
            yield _conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
