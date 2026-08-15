from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


def _validate_identifier(name: str, field: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{field} must be a string")
    if not name:
        raise ValueError(f"{field} must not be empty")
    if "\x00" in name:
        raise ValueError(f"{field} must not contain NUL characters")
    if len(name) > 64:
        raise ValueError(f"{field} must not exceed 64 characters")
    return name


def quote_identifier(name: str) -> str:
    """Quote a validated MySQL identifier."""
    return f"`{name.replace('`', '``')}`"


def render_sql(sql: str, table_names: Mapping[str, str]) -> str:
    """Replace trusted default table identifiers with configured identifiers."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9_$`])("
        + "|".join(map(re.escape, table_names))
        + r")(?![A-Za-z0-9_$`])"
    )
    return pattern.sub(lambda match: quote_identifier(table_names[match.group(1)]), sql)


@dataclass(frozen=True)
class CheckpointTableConfig:
    """Names of the MySQL tables used by checkpoint savers.

    ``prefix`` is applied only to table names that are not explicitly supplied.
    Explicit table names are treated as complete names.
    """

    prefix: str = ""
    migrations: str | None = None
    checkpoints: str | None = None
    blobs: str | None = None
    writes: str | None = None

    def resolved(self) -> dict[str, str]:
        if not isinstance(self.prefix, str):
            raise TypeError("prefix must be a string")
        if "\x00" in self.prefix:
            raise ValueError("prefix must not contain NUL characters")

        names = {
            "checkpoint_migrations": (
                self.migrations
                if self.migrations is not None
                else f"{self.prefix}checkpoint_migrations"
            ),
            "checkpoints": (
                self.checkpoints
                if self.checkpoints is not None
                else f"{self.prefix}checkpoints"
            ),
            "checkpoint_blobs": (
                self.blobs
                if self.blobs is not None
                else f"{self.prefix}checkpoint_blobs"
            ),
            "checkpoint_writes": (
                self.writes
                if self.writes is not None
                else f"{self.prefix}checkpoint_writes"
            ),
        }
        for default_name, name in names.items():
            _validate_identifier(name, default_name)
        if len({name.casefold() for name in names.values()}) != len(names):
            raise ValueError("checkpoint table names must be distinct")
        return names


@dataclass(frozen=True)
class StoreTableConfig:
    """Names of the MySQL tables used by stores."""

    prefix: str = ""
    migrations: str | None = None
    store: str | None = None

    def resolved(self) -> dict[str, str]:
        if not isinstance(self.prefix, str):
            raise TypeError("prefix must be a string")
        if "\x00" in self.prefix:
            raise ValueError("prefix must not contain NUL characters")

        names = {
            "store_migrations": (
                self.migrations
                if self.migrations is not None
                else f"{self.prefix}store_migrations"
            ),
            "store": (self.store if self.store is not None else f"{self.prefix}store"),
        }
        for default_name, name in names.items():
            _validate_identifier(name, default_name)
        if len({name.casefold() for name in names.values()}) != len(names):
            raise ValueError("store table names must be distinct")
        return names
