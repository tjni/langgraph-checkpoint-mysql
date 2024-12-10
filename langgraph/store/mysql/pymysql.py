import urllib.parse
from contextlib import contextmanager
from typing import Any, Iterator

import pymysql
from pymysql.cursors import DictCursor
from typing_extensions import Self, override

from langgraph.store.mysql.base import BaseSyncMySQLStore


class PyMySQLStore(BaseSyncMySQLStore[pymysql.Connection, DictCursor]):
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
        """Create a new PyMySQLStore instance from a connection string.

        Args:
            conn_string (str): The MySQL connection info string.

        Returns:
            PyMySQLStore: A new PyMySQLStore instance.
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
