import urllib.parse
from contextlib import contextmanager
from typing import Iterator

import pymysql
import pymysql.constants.ER
from pymysql.cursors import DictCursor
from typing_extensions import Self, override

from langgraph.store.mysql.base import BaseSyncMySQLStore


class PyMySQLStore(BaseSyncMySQLStore[pymysql.Connection, DictCursor]):
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
        parsed = urllib.parse.urlparse(conn_string)

        with pymysql.connect(
            host=parsed.hostname,
            user=parsed.username,
            password=parsed.password or "",
            database=parsed.path[1:],
            port=parsed.port or 3306,
            autocommit=True,
        ) as conn:
            yield cls(conn)

    @override
    @staticmethod
    def _is_no_such_table_error(e: Exception) -> bool:
        return (
            isinstance(e, pymysql.ProgrammingError)
            and e.args[0] == pymysql.constants.ER.NO_SUCH_TABLE
        )

    @override
    def _cursor(self) -> DictCursor:
        return self.conn.cursor(DictCursor)
