from __future__ import annotations

import base64
import json
from typing import NamedTuple

Base64Blob = str


def decode_base64_blob(base64_blob: Base64Blob) -> bytes:
    # When MySQL returns a blob in a JSON array, it is base64 encoded and a prefix
    # of "base64:type251:" attached to it.
    parts = base64_blob.rsplit(":", 1)
    return base64.b64decode(parts[-1])


class MySQLPendingWrite(NamedTuple):
    """
    The pending write tuple we receive from our DB query.
    """

    task_id: str
    channel: str
    type_: str
    blob: Base64Blob
    idx: int


def deserialize_pending_writes(value: str) -> list[tuple[str, str, str, bytes]]:
    if not value:
        return []

    values = (MySQLPendingWrite(*write) for write in json.loads(value))

    return [
        (db.task_id, db.channel, db.type_, decode_base64_blob(db.blob))
        for db in sorted(values, key=lambda db: (db.task_id, db.idx))
    ]


class MySQLPendingSend(NamedTuple):
    task_path: str
    task_id: str
    type_: str
    blob: Base64Blob
    idx: int


def deserialize_pending_sends(value: str) -> list[tuple[str, bytes]]:
    if not value:
        return []

    values = (MySQLPendingSend(*send) for send in json.loads(value))

    return [
        (db.type_, decode_base64_blob(db.blob))
        for db in sorted(values, key=lambda db: (db.task_path, db.task_id, db.idx))
    ]


class MySQLChannelValue(NamedTuple):
    channel: str
    type_: str
    blob: Base64Blob | None


def deserialize_channel_values(value: str) -> list[tuple[str, str, bytes | None]]:
    if not value:
        return []

    values = (MySQLChannelValue(*channel_value) for channel_value in json.loads(value))

    return [
        (
            db.channel,
            db.type_,
            decode_base64_blob(db.blob) if db.blob is not None else None,
        )
        for db in values
    ]


def mysql_mariadb_branch(mysql_fragment: str, mariadb_fragment: str) -> str:
    # MariaDB ignores MySQL conditional comments with version numbers between
    # 500700 and 999999. We can use this to our advantage.
    return f"/*!50700 {mysql_fragment}*//*M! {mariadb_fragment}*/"
