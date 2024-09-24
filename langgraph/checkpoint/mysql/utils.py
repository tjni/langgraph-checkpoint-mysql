import base64
import json
from typing import NamedTuple

# When MySQL returns a blob in a JSON array, it is base64 encoded and a prefix
# of "base64:type251:" attached to it.
MySQLBase64Blob = str


def decode_base64_blob(base64_blob: MySQLBase64Blob) -> bytes:
    _, data = base64_blob.rsplit(":", 1)
    return base64.b64decode(data)


class MySQLPendingWrite(NamedTuple):
    """
    The pending write tuple we receive from our DB query.
    """

    task_id: str
    channel: str
    type_: str
    blob: MySQLBase64Blob
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
    type_: str
    blob: MySQLBase64Blob
    idx: int


def deserialize_pending_sends(value: str) -> list[tuple[str, bytes]]:
    if not value:
        return []

    values = (MySQLPendingSend(*send) for send in json.loads(value))

    return [
        (db.type_, decode_base64_blob(db.blob))
        for db in sorted(values, key=lambda db: db.idx)
    ]


class MySQLChannelValue(NamedTuple):
    channel: str
    type_: str
    blob: MySQLBase64Blob


def deserialize_channel_values(value: str) -> list[tuple[str, str, bytes]]:
    if not value:
        return []

    values = (MySQLChannelValue(*channel_value) for channel_value in json.loads(value))

    return [(db.channel, db.type_, decode_base64_blob(db.blob)) for db in values]
