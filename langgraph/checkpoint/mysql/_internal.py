"""Shared utility functions for the MySQL checkpoint & storage classes."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, closing, contextmanager
from typing import (
    Any,
    Protocol,
    TypeVar,
    Union,
    cast,
)


class DictCursor(AbstractContextManager, Protocol):
    """
    Protocol that a cursor should implement.

    Modeled after DBAPICursor from Typeshed.
    """

    def execute(
        self,
        operation: str,
        parameters: Sequence[Any] | Mapping[str, Any] = ...,
        /,
    ) -> object: ...
    def executemany(
        self, operation: str, seq_of_parameters: Sequence[Sequence[Any]], /
    ) -> object: ...
    def fetchone(self) -> dict[str, Any] | None: ...
    def fetchall(self) -> Sequence[dict[str, Any]]: ...


R = TypeVar("R", bound=DictCursor)  # cursor type


class Connection(AbstractContextManager, Protocol):
    """Protocol that a synchronous MySQL connection should implement."""

    def begin(self) -> None:
        """Begin transaction."""
        ...

    def commit(self) -> None:
        """Commit changes to stable storage."""
        ...

    def rollback(self) -> None:
        """Roll back the current transaction."""
        ...


C = TypeVar("C", bound=Connection)  # connection type


ConnectionFactory = Callable[[], Any]
Conn = Union[C, ConnectionFactory]


@contextmanager
def get_connection(conn: Conn[C]) -> Iterator[C]:
    if hasattr(conn, "cursor"):
        yield cast(C, conn)
    elif callable(conn):
        _conn = conn()
        if isinstance(_conn, AbstractContextManager):
            yield cast(C, _conn)
        else:
            with closing(_conn) as __conn:
                yield __conn
    # This is kept for backwards incompatibility and should be removed when we
    # can make a breaking change in favor of just passing a Callable.
    elif hasattr(conn, "connect"):
        # sqlalchemy pool
        factory: ConnectionFactory = getattr(conn, "connect")  # noqa: B009
        with get_connection(factory) as _conn:
            yield _conn
    else:
        raise TypeError(f"Invalid connection or pool type: {type(conn)}")
