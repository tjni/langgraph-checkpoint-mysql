import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pymysql
import pymysql.constants.ER
from pymysql.cursors import DictCursor
from typing_extensions import Self, override

from langgraph.checkpoint.mysql import BaseSyncMySQLSaver
from langgraph.checkpoint.mysql import Conn as BaseConn
from langgraph.checkpoint.mysql.shallow import BaseShallowSyncMySQLSaver

Conn = BaseConn[pymysql.Connection]  # type: ignore


class PyMySQLSaver(BaseSyncMySQLSaver[pymysql.Connection, DictCursor]):
    @staticmethod
    def parse_conn_string(conn_string: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(conn_string)

        # In order to provide additional params via the connection string,
        # we convert the parsed.query to a dict so we can access the values.
        # This is necessary when using a unix socket, for example.
        params_as_dict = dict(urllib.parse.parse_qsl(parsed.query))

        return {
            "host": parsed.hostname,
            "user": parsed.username,
            "password": parsed.password or "",
            "database": parsed.path[1:] or None,
            "port": parsed.port or 3306,
            "unix_socket": params_as_dict.get("unix_socket"),
        }

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
    ) -> Iterator[Self]:
        """Create a new PyMySQLSaver instance from a connection string.

        Args:
            conn_string (str): The MySQL connection info string.

        Returns:
            PyMySQLSaver: A new PyMySQLSaver instance.

        Example:
            conn_string=mysql://user:password@localhost/db?unix_socket=/path/to/socket
        """
        with pymysql.connect(
            **cls.parse_conn_string(conn_string),
            autocommit=True,
        ) as conn:
            yield cls(conn)

    @override
    @staticmethod
    def _get_cursor_from_connection(conn: pymysql.Connection) -> DictCursor:
        return conn.cursor(DictCursor)


class ShallowPyMySQLSaver(BaseShallowSyncMySQLSaver):
    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
    ) -> Iterator[Self]:
        """Create a new ShallowPyMySQLSaver instance from a connection string.

        Args:
            conn_string (str): The MySQL connection info string.

        Returns:
            ShallowPyMySQLSaver: A new ShallowPyMySQLSaver instance.

        Example:
            conn_string=mysql://user:password@localhost/db?unix_socket=/path/to/socket
        """
        with pymysql.connect(
            **PyMySQLSaver.parse_conn_string(conn_string),
            autocommit=True,
        ) as conn:
            yield cls(conn)

    @override
    @staticmethod
    def _get_cursor_from_connection(conn: pymysql.Connection) -> DictCursor:
        return conn.cursor(DictCursor)


__all__ = ["PyMySQLSaver", "ShallowPyMySQLSaver", "Conn"]
