import json
import random
from typing import Any, List, Optional, Sequence, Tuple, cast

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import TASKS, ChannelProtocol

MetadataInput = Optional[dict[str, Any]]

"""
To add a new migration, add a new string to the MIGRATIONS list.
The position of the migration in the list is the version number.
"""
MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);""",
    """CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id VARCHAR(150) NOT NULL,
    checkpoint_ns VARCHAR(150) NOT NULL DEFAULT '',
    checkpoint_id VARCHAR(150) NOT NULL,
    parent_checkpoint_id VARCHAR(150),
    type VARCHAR(150),
    checkpoint JSON NOT NULL,
    metadata JSON NOT NULL DEFAULT ('{}'),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id VARCHAR(150) NOT NULL,
    checkpoint_ns VARCHAR(150) NOT NULL DEFAULT '',
    channel VARCHAR(150) NOT NULL,
    version VARCHAR(150) NOT NULL,
    type VARCHAR(150) NOT NULL,
    `blob` LONGBLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id VARCHAR(150) NOT NULL,
    checkpoint_ns VARCHAR(150) NOT NULL DEFAULT '',
    checkpoint_id VARCHAR(150) NOT NULL,
    task_id VARCHAR(150) NOT NULL,
    idx INTEGER NOT NULL,
    channel VARCHAR(150) NOT NULL,
    type VARCHAR(150),
    `blob` LONGBLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);""",
    "ALTER TABLE checkpoint_blobs MODIFY COLUMN `blob` LONGBLOB;",
]

SELECT_SQL = f"""
select
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        select json_arrayagg(json_array(bl.channel, bl.type, bl.blob))
        from
        (
            select channel, json_unquote(
                json_extract(checkpoint, concat('$.channel_versions.', channel))
            ) as version
            from json_table(
                json_keys(checkpoint, '$.channel_versions'),
                '$[*]' columns (channel VARCHAR(150) PATH '$')
            ) as channels
        ) as channel_versions
        inner join checkpoint_blobs bl
            on bl.thread_id = checkpoints.thread_id
            and bl.checkpoint_ns = checkpoints.checkpoint_ns
            and bl.channel = channel_versions.channel
            and bl.version = channel_versions.version
    ) as channel_values,
    (
        select
        json_arrayagg(json_array(cw.task_id, cw.channel, cw.type, cw.blob, cw.idx))
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns = checkpoints.checkpoint_ns
            and cw.checkpoint_id = checkpoints.checkpoint_id
    ) as pending_writes,
    (
        select json_arrayagg(json_array(cw.type, cw.blob, cw.idx))
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns = checkpoints.checkpoint_ns
            and cw.checkpoint_id = checkpoints.parent_checkpoint_id
            and cw.channel = '{TASKS}'
    ) as pending_sends
from checkpoints """

UPSERT_CHECKPOINT_BLOBS_SQL = """
    INSERT IGNORE INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, `blob`)
    VALUES (%s, %s, %s, %s, %s, %s)
"""

UPSERT_CHECKPOINTS_SQL = """
    INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES (%s, %s, %s, %s, %s, %s) AS new
    ON DUPLICATE KEY UPDATE
        checkpoint = new.checkpoint,
        metadata = new.metadata;
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, `blob`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) AS new
    ON DUPLICATE KEY UPDATE
        channel = new.channel,
        type = new.type,
        `blob` = new.blob;
"""

INSERT_CHECKPOINT_WRITES_SQL = """
    INSERT IGNORE INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, `blob`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


class BaseMySQLSaver(BaseCheckpointSaver[str]):
    SELECT_SQL = SELECT_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    jsonplus_serde = JsonPlusSerializer()

    def _load_checkpoint(
        self,
        checkpoint: dict[str, Any],
        channel_values: list[tuple[str, str, bytes]],
        pending_sends: list[tuple[str, bytes]],
    ) -> Checkpoint:
        return {
            **checkpoint,
            "pending_sends": [
                self.serde.loads_typed((c, b)) for c, b in pending_sends or []
            ],
            "channel_values": self._load_blobs(channel_values),
        }

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
        return {**checkpoint, "pending_sends": []}

    def _load_blobs(self, blob_values: list[tuple[str, str, bytes]]) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k: self.serde.loads_typed((t, v)) for k, t, v in blob_values if t != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, Optional[bytes]]]:
        if not versions:
            return []

        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),
                *(
                    self.serde.dumps_typed(values[k])
                    if k in values
                    else ("empty", None)
                ),
            )
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, writes: list[tuple[str, str, str, bytes]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid,
                    channel,
                    self.serde.loads_typed((t, v)),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def _load_metadata(self, metadata: str) -> CheckpointMetadata:
        return self.jsonplus_serde.loads(metadata.encode())

    def _dump_metadata(self, metadata: CheckpointMetadata) -> str:
        serialized_metadata = self.jsonplus_serde.dumps(metadata)
        # NOTE: we're using JSON serializer (not msgpack), so we need to remove null characters before writing
        return serialized_metadata.decode().replace("\\u0000", "")

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _search_where(
        self,
        config: Optional[RunnableConfig],
        filter: MetadataInput,
        before: Optional[RunnableConfig] = None,
    ) -> Tuple[str, List[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before.

        This method returns a tuple of a string and a tuple of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = %s")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = %s ")
                param_values.append(checkpoint_id)

        # construct predicate for metadata filter
        if filter:
            wheres.append("json_contains(metadata, %s) ")
            param_values.append(json.dumps(filter))

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )
