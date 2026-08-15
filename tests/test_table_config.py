import pytest

from langgraph._mysql import (
    CheckpointTableConfig,
    StoreTableConfig,
    quote_identifier,
    render_sql,
)
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
from langgraph.checkpoint.mysql.asyncmy import AsyncMySaver
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.store.mysql import PyMySQLStore
from langgraph.store.mysql.aio import AIOMySQLStore
from langgraph.store.mysql.asyncmy import AsyncMyStore


@pytest.fixture(autouse=True)
def clear_test_db() -> None:
    """These configuration tests do not require a database."""


def test_checkpoint_table_config_defaults() -> None:
    assert CheckpointTableConfig().resolved() == {
        "checkpoint_migrations": "checkpoint_migrations",
        "checkpoints": "checkpoints",
        "checkpoint_blobs": "checkpoint_blobs",
        "checkpoint_writes": "checkpoint_writes",
    }


def test_checkpoint_table_prefix_and_override() -> None:
    assert CheckpointTableConfig(
        prefix="app_", checkpoints="graph-history"
    ).resolved() == {
        "checkpoint_migrations": "app_checkpoint_migrations",
        "checkpoints": "graph-history",
        "checkpoint_blobs": "app_checkpoint_blobs",
        "checkpoint_writes": "app_checkpoint_writes",
    }


def test_store_table_prefix_and_override() -> None:
    assert StoreTableConfig(prefix="app_", store="documents").resolved() == {
        "store_migrations": "app_store_migrations",
        "store": "documents",
    }


@pytest.mark.parametrize(
    "config",
    [
        CheckpointTableConfig(checkpoints=""),
        CheckpointTableConfig(prefix="x" * 65),
        CheckpointTableConfig(checkpoints="same", blobs="SAME"),
        StoreTableConfig(store=""),
        StoreTableConfig(store="same", migrations="SAME"),
    ],
)
def test_invalid_table_config(config: object) -> None:
    with pytest.raises(ValueError):
        config.resolved()  # type: ignore[attr-defined]


def test_render_sql_quotes_configured_identifiers() -> None:
    names = CheckpointTableConfig(
        prefix="tenant-`one`_", checkpoints="custom-checkpoints"
    ).resolved()

    assert render_sql(
        "SELECT checkpoints.id FROM checkpoints JOIN checkpoint_blobs",
        names,
    ) == (
        "SELECT `custom-checkpoints`.id FROM `custom-checkpoints` "
        "JOIN `tenant-``one``_checkpoint_blobs`"
    )
    assert quote_identifier("a`b") == "`a``b`"
    assert render_sql("SELECT * FROM `checkpoints`", names) == (
        "SELECT * FROM `checkpoints`"
    )


def test_sync_saver_and_store_render_configured_tables() -> None:
    saver = PyMySQLSaver(
        object(),  # type: ignore[arg-type]
        table_config=CheckpointTableConfig(prefix="app_", checkpoints="graph-history"),
    )
    store = PyMySQLStore(
        object(),  # type: ignore[arg-type]
        table_config=StoreTableConfig(prefix="app_", store="documents"),
    )

    saver_sql = "\n".join(saver.MIGRATIONS) + saver.SELECT_SQL
    assert "`graph-history`" in saver_sql
    assert "`app_checkpoint_migrations`" in saver_sql
    assert "`app_checkpoint_blobs`" in saver.UPSERT_CHECKPOINT_BLOBS_SQL
    assert "`app_checkpoint_writes`" in saver.UPSERT_CHECKPOINT_WRITES_SQL
    assert "`documents`" in "\n".join(store.MIGRATIONS)
    assert "`app_store_migrations`" in store._render_sql(
        "SELECT v FROM store_migrations"
    )


@pytest.mark.anyio
async def test_async_savers_and_stores_render_configured_tables() -> None:
    checkpoint_config = CheckpointTableConfig(prefix="async_")
    store_config = StoreTableConfig(prefix="async_")

    savers = [
        AIOMySQLSaver(object(), table_config=checkpoint_config),  # type: ignore[arg-type]
        AsyncMySaver(object(), table_config=checkpoint_config),  # type: ignore[arg-type]
    ]
    stores = [
        AIOMySQLStore(object(), table_config=store_config),  # type: ignore[arg-type]
        AsyncMyStore(object(), table_config=store_config),  # type: ignore[arg-type]
    ]

    assert all("`async_checkpoints`" in saver.SELECT_SQL for saver in savers)
    assert all("`async_store`" in "\n".join(store.MIGRATIONS) for store in stores)
