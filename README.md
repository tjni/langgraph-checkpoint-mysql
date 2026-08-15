# LangGraph Checkpoint MySQL

Implementation of LangGraph CheckpointSaver that uses MySQL.

> [!TIP]
> The code in this repository tries to mimic the code in [langgraph-checkpoint-postgres](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres) as much as possible to enable keeping in sync with the official checkpointer implementation.

> [!NOTE]
> In order to keep the queries close to the Postgres queries, we use features that require MySQL >= 8.0.19 or MariaDB >= 10.7.1.

> [!NOTE]
> MySQL >= 9.6.0 discontinued using the MD5 function in generated columns, but we have not created a migration path for this yet.

## Dependencies

- To use synchronous `PyMySQLSaver`, install `langgraph-checkpoint-mysql[pymysql]`.
- To use asynchronous `AIOMySQLSaver`, install `langgraph-checkpoint-mysql[aiomysql]`.
- To use asynchronous `AsyncMySaver`, install `langgraph-checkpoint-mysql[asyncmy]`.

There is currently no support for other drivers.

## Usage

> [!IMPORTANT]
> When using MySQL checkpointers for the first time, make sure to call `.setup()` method on them to create required tables. See example below.

> [!IMPORTANT]
> When manually creating MySQL connections and passing them to `PyMySQLSaver` or `AIOMySQLSaver`, make sure to include `autocommit=True`.
>
> **Why this parameter is required:**
> - `autocommit=True`: Required for the `.setup()` method to properly commit the checkpoint tables to the database. Without this, table creation may not be persisted.

```python
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

DB_URI = "mysql://mysql:mysql@localhost:3306/mysql"
with PyMySQLSaver.from_conn_string(DB_URI) as checkpointer:
    # call .setup() the first time you're using the checkpointer
    checkpointer.setup()
    checkpoint = {
        "v": 4,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
            "__start__": 1
            },
            "node": {
            "start:node": 2
            }
        },
    }

    # store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # load checkpoint
    checkpointer.get(read_config)

    # list checkpoints
    list(checkpointer.list(read_config))
```

### Custom table names

Pass `CheckpointTableConfig` to namespace the checkpoint tables with a common
prefix, override individual table names, or combine both approaches. Omitting
`table_config` preserves the default table names and behavior.

```python
from langgraph.checkpoint.mysql import CheckpointTableConfig
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

table_config = CheckpointTableConfig(
    prefix="my_app_",
    checkpoints="graph_history",
)

with PyMySQLSaver.from_conn_string(
    DB_URI,
    table_config=table_config,
) as checkpointer:
    checkpointer.setup()
```

Explicit names are complete names and do not receive the prefix. The example
above therefore uses `graph_history`, `my_app_checkpoint_migrations`,
`my_app_checkpoint_blobs`, and `my_app_checkpoint_writes`.

The available checkpoint fields and their defaults are:

| Field | Default table |
| --- | --- |
| `migrations` | `checkpoint_migrations` |
| `checkpoints` | `checkpoints` |
| `blobs` | `checkpoint_blobs` |
| `writes` | `checkpoint_writes` |

For example, `CheckpointTableConfig(checkpoints="graph_history")` renames only
the `checkpoints` table, while all unspecified tables keep their default names.
The same configuration API is available for synchronous, asynchronous, and
shallow savers.

MySQL stores accept `StoreTableConfig`, with `migrations` and `store` as the
individually configurable fields:

```python
from langgraph.store.mysql import PyMySQLStore, StoreTableConfig

with PyMySQLStore.from_conn_string(
    DB_URI,
    table_config=StoreTableConfig(prefix="my_app_"),
) as store:
    store.setup()
```

> [!IMPORTANT]
> A different configuration selects or creates a different set of tables. It
> does not rename existing tables or migrate their data. Keep the same
> configuration for every process that needs to share the same checkpoint or
> store data.

### Async

```python
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver

async with AIOMySQLSaver.from_conn_string(DB_URI) as checkpointer:
    checkpoint = {
        "v": 4,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
            "__start__": 1
            },
            "node": {
            "start:node": 2
            }
        },
    }

    # store checkpoint
    await checkpointer.aput(write_config, checkpoint, {}, {})

    # load checkpoint
    await checkpointer.aget(read_config)

    # list checkpoints
    [c async for c in checkpointer.alist(read_config)]
```
