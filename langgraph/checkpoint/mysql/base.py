import json
import random
from collections.abc import Sequence
from typing import Any, Optional, cast

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    CheckpointMetadata,
    get_checkpoint_id,
)
from langgraph.checkpoint.mysql.utils import mysql_mariadb_branch
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import TASKS

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
    """
    CREATE INDEX checkpoints_thread_id_idx ON checkpoints (thread_id);
    """,
    """
    CREATE INDEX checkpoint_blobs_thread_id_idx ON checkpoint_blobs (thread_id);
    """,
    """
    CREATE INDEX checkpoint_writes_thread_id_idx ON checkpoint_writes (thread_id);
    """,
    """
    CREATE INDEX checkpoints_checkpoint_id_idx ON checkpoints (checkpoint_id);
    """,
    # The following three migrations were contributed to buy more room for
    # nested subgraphs, since that contributes to checkpoint_ns length.
    "ALTER TABLE checkpoints MODIFY COLUMN `checkpoint_ns` VARCHAR(255) NOT NULL DEFAULT '';",
    "ALTER TABLE checkpoint_blobs MODIFY COLUMN `checkpoint_ns` VARCHAR(255) NOT NULL DEFAULT '';",
    "ALTER TABLE checkpoint_writes MODIFY COLUMN `checkpoint_ns` VARCHAR(255) NOT NULL DEFAULT '';",
    # The following three migrations drastically increase the size of the
    # checkpoint_ns field to support deeply nested subgraphs.
    """
    ALTER TABLE checkpoints
    DROP PRIMARY KEY,
    ADD PRIMARY KEY (thread_id, checkpoint_id),
    MODIFY COLUMN `checkpoint_ns` VARCHAR(2000) NOT NULL DEFAULT '';
    """,
    """
    ALTER TABLE checkpoint_blobs
    DROP PRIMARY KEY,
    ADD PRIMARY KEY (thread_id, channel, version),
    MODIFY COLUMN `checkpoint_ns` VARCHAR(2000) NOT NULL DEFAULT '';
    """,
    """
    ALTER TABLE checkpoint_writes
    DROP PRIMARY KEY,
    ADD PRIMARY KEY (thread_id, checkpoint_id, task_id, idx),
    MODIFY COLUMN `checkpoint_ns` VARCHAR(2000) NOT NULL DEFAULT '';
    """,
    # The following three migrations restore checkpoint_ns as part of the
    # primary key, but hashed to fit into the primary key size limit.
    f"""
    ALTER TABLE checkpoints
    {
        mysql_mariadb_branch(
            "ADD COLUMN checkpoint_ns_hash BINARY(16) AS (UNHEX(MD5(checkpoint_ns))) STORED,",
            "ADD COLUMN checkpoint_ns_hash BINARY(16),",
        )
    }
    DROP PRIMARY KEY,
    ADD PRIMARY KEY (thread_id, checkpoint_ns_hash, checkpoint_id);
    """,
    f"""
    ALTER TABLE checkpoint_blobs
    {
        mysql_mariadb_branch(
            "ADD COLUMN checkpoint_ns_hash BINARY(16) AS (UNHEX(MD5(checkpoint_ns))) STORED,",
            "ADD COLUMN checkpoint_ns_hash BINARY(16),",
        )
    }
    DROP PRIMARY KEY,
    ADD PRIMARY KEY (thread_id, checkpoint_ns_hash, channel, version);
    """,
    f"""
    ALTER TABLE checkpoint_writes
    {
        mysql_mariadb_branch(
            "ADD COLUMN checkpoint_ns_hash BINARY(16) AS (UNHEX(MD5(checkpoint_ns))) STORED,",
            "ADD COLUMN checkpoint_ns_hash BINARY(16),",
        )
    }
    DROP PRIMARY KEY,
    ADD PRIMARY KEY (thread_id, checkpoint_ns_hash, checkpoint_id, task_id, idx);
    """,
    """
    ALTER TABLE checkpoint_writes ADD COLUMN task_path VARCHAR(2000) NOT NULL DEFAULT '';
    """,
    # No longer use STORED generated columns, because MariaDB does not support
    # using them in primary keys.
    #
    #  https://github.com/tjni/langgraph-checkpoint-mysql/issues/51
    #
    """
    ALTER TABLE checkpoints MODIFY COLUMN checkpoint_ns_hash BINARY(16);
    """,
    """
    ALTER TABLE checkpoint_blobs MODIFY COLUMN checkpoint_ns_hash BINARY(16);
    """,
    """
    ALTER TABLE checkpoint_writes MODIFY COLUMN checkpoint_ns_hash BINARY(16);
    """,
]

SELECT_SQL = f"""
with channel_versions as (
    select thread_id, checkpoint_ns_hash, checkpoint_id, channel, json_unquote(
        json_extract(checkpoint, concat('$.channel_versions.', '"', channel, '"'))
    ) as version
    from checkpoints, json_table(
        json_keys(checkpoint, '$.channel_versions'),
        '$[*]' columns (channel VARCHAR(150) CHARACTER SET utf8mb4 PATH '$')
    ) as channels
    {{WHERE}}
)
select
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        select json_arrayagg(json_array(
            bl.channel,
            bl.type,
            {mysql_mariadb_branch("bl.blob", "to_base64(bl.blob)")}
        ))
        from channel_versions
        inner join checkpoint_blobs bl
            on bl.channel = channel_versions.channel
            and bl.version = channel_versions.version
        where bl.thread_id = checkpoints.thread_id
            and bl.checkpoint_ns_hash = checkpoints.checkpoint_ns_hash
            and channel_versions.thread_id = checkpoints.thread_id
            and channel_versions.checkpoint_ns_hash = checkpoints.checkpoint_ns_hash
            and channel_versions.checkpoint_id = checkpoints.checkpoint_id
    ) as channel_values,
    (
        select
        json_arrayagg(json_array(
            cw.task_id,
            cw.channel,
            cw.type,
            {mysql_mariadb_branch("cw.blob", "to_base64(cw.blob)")},
            cw.idx
        ))
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns_hash = checkpoints.checkpoint_ns_hash
            and cw.checkpoint_id = checkpoints.checkpoint_id
    ) as pending_writes
from checkpoints {{WHERE}} """

SELECT_PENDING_SENDS_SQL = f"""
select
    checkpoint_id,
    json_arrayagg(json_array(
        task_path,
        task_id,
        type,
        {mysql_mariadb_branch("`blob`", "to_base64(`blob`)")},
        idx
    )) as sends
from checkpoint_writes
where thread_id = %s
    and checkpoint_id in ({{CHECKPOINT_ID_PLACEHOLDERS}})
    and channel = '{TASKS}'
group by checkpoint_id
"""

UPSERT_CHECKPOINT_BLOBS_SQL = """
    INSERT IGNORE INTO checkpoint_blobs (thread_id, checkpoint_ns, checkpoint_ns_hash, channel, version, type, `blob`)
    VALUES (%s, %s, UNHEX(MD5(%s)), %s, %s, %s, %s)
"""

UPSERT_CHECKPOINTS_SQL = f"""
    INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_ns_hash, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES (%s, %s, UNHEX(MD5(%s)), %s, %s, %s, %s) {mysql_mariadb_branch("AS new", "")}
    ON DUPLICATE KEY UPDATE
        checkpoint = {mysql_mariadb_branch("new.checkpoint", "VALUE(checkpoint)")},
        metadata = {mysql_mariadb_branch("new.metadata", "VALUE(metadata)")}
"""

UPSERT_CHECKPOINT_WRITES_SQL = f"""
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_ns_hash, checkpoint_id, task_id, task_path, idx, channel, type, `blob`)
    VALUES (%s, %s, UNHEX(MD5(%s)), %s, %s, %s, %s, %s, %s, %s) {mysql_mariadb_branch("AS new", "")}
    ON DUPLICATE KEY UPDATE
        channel = {mysql_mariadb_branch("new.channel", "VALUE(channel)")},
        type = {mysql_mariadb_branch("new.type", "VALUE(type)")},
        `blob` = {mysql_mariadb_branch("new.blob", "VALUE(`blob`)")};
"""

INSERT_CHECKPOINT_WRITES_SQL = """
    INSERT IGNORE INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_ns_hash, checkpoint_id, task_id, task_path, idx, channel, type, `blob`)
    VALUES (%s, %s, UNHEX(MD5(%s)), %s, %s, %s, %s, %s, %s, %s)
"""


class BaseMySQLSaver(BaseCheckpointSaver[str]):
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    jsonplus_serde = JsonPlusSerializer()

    def _migrate_pending_sends(
        self,
        pending_sends: list[tuple[str, bytes]],
        checkpoint: dict[str, Any],
        channel_values: list[tuple[str, str, Optional[bytes]]],
    ) -> None:
        if not pending_sends:
            return
        # add to values
        enc, blob = self.serde.dumps_typed(
            [self.serde.loads_typed((c, b)) for c, b in pending_sends],
        )
        channel_values.append((TASKS, enc, blob))
        # add to versions
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None)
        )

    def _load_blobs(
        self, blob_values: list[tuple[str, str, Optional[bytes]]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k: self.serde.loads_typed((t, v))
            for k, t, v in blob_values
            if t != "empty" and v is not None
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, str, Optional[bytes]]]:
        if not versions:
            return []

        return [
            (
                thread_id,
                checkpoint_ns,
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
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def _load_metadata(self, metadata: str) -> CheckpointMetadata:
        try:
            return json.loads(metadata)
        except (TypeError, json.JSONDecodeError):
            # This is a best effort fallback for backwards compatibility with
            # old checkpoints and old versions of LangGraph prior to "writes"
            # being removed from metadata in
            #
            #   https://github.com/langchain-ai/langgraph/pull/4822
            #
            # It's a little unclear to me if this catches all issues due to theThis is to address issues such as
            # complexity of the changes, but I hope it addresses issues like
            #
            #   https://github.com/langchain-ai/langgraph/issues/5769
            #
            return self.jsonplus_serde.loads(metadata.encode())

    def _dump_metadata(self, metadata: CheckpointMetadata) -> str:
        try:
            return json.dumps(metadata)
        except TypeError:
            # This is a best effort fallback for backwards compatibility with
            # old checkpoints and old versions of LangGraph prior to "writes"
            # being removed from metadata in
            #
            #   https://github.com/langchain-ai/langgraph/pull/4822
            #
            # It's a little unclear to me if this catches all issues due to theThis is to address issues such as
            # complexity of the changes, but I hope it addresses issues like
            #
            #   https://github.com/langchain-ai/langgraph/issues/5769
            #
            serialized_metadata = self.jsonplus_serde.dumps(metadata)
            # NOTE: we're using JSON serializer (not msgpack), so we need to remove null characters before writing
            return serialized_metadata.decode().replace("\\u0000", "")

    def get_next_version(self, current: Optional[str]) -> str:
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
    ) -> tuple[str, dict[str, Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before.

        This method returns a tuple of a string and a dict of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The dict of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = {}

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = %(thread_id)s ")
            param_values["thread_id"] = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns_hash = UNHEX(MD5(%(checkpoint_ns)s))")
                param_values["checkpoint_ns"] = checkpoint_ns

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = %(checkpoint_id)s ")
                param_values["checkpoint_id"] = checkpoint_id

        # construct predicate for metadata filter
        if filter:
            wheres.append("json_contains(metadata, %(filter)s) ")
            param_values["filter"] = json.dumps(filter)

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %(before)s ")
            param_values["before"] = get_checkpoint_id(before)

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )

    @staticmethod
    def _select_sql(where: str) -> str:
        return SELECT_SQL.replace("{WHERE}", where)

    @staticmethod
    def _select_pending_sends_sql(num_ids: int) -> str:
        placeholders = ",".join(["%s"] * num_ids)
        return SELECT_PENDING_SENDS_SQL.replace(
            "{CHECKPOINT_ID_PLACEHOLDERS}", placeholders
        )
