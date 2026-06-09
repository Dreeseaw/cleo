"""Connection-agnostic, read-only SQL execution + live schema introspection.

Cleo runs against *your* database connection — psycopg2 / sqlite3 / duckdb / a SQLAlchemy
`raw_connection()`, anything that speaks DB-API 2.0 — so there's no "pull it into DuckDB first" step.
Every query Cleo issues is guarded read-only and the transaction is rolled back, never committed.
"""
from __future__ import annotations

from typing import Any, Callable

from .contract import is_readonly


def _normalize_cell(v: Any) -> Any:
    if v is None or isinstance(v, (int, str, bool)):
        return v
    if isinstance(v, float):
        return round(v, 8)
    return str(v)


# An executor is a callable: sql -> (columns, rows, truncated). Build one from any DB-API connection.
Executor = Callable[[str, int], "tuple[list, list, bool]"]


def make_executor(source: Any) -> Executor:
    """Accept a DB-API 2.0 connection OR a ready callable, return a read-only executor.

    A custom callable must have signature (sql, limit) -> (columns, rows, truncated).
    """
    if callable(source) and not hasattr(source, "cursor"):
        return source
    if not hasattr(source, "cursor"):
        raise TypeError("source must be a DB-API 2.0 connection (has .cursor()) or an executor callable")

    def _execute(sql: str, limit: int) -> tuple[list, list, bool]:
        cur = source.cursor()
        try:
            try:  # best-effort read-only transaction (Postgres/CockroachDB honour it; others ignore)
                cur.execute("SET TRANSACTION READ ONLY")
            except Exception:
                try:
                    source.rollback()
                except Exception:
                    pass
            wrapped = f"SELECT * FROM ({sql.strip().rstrip(';')}) AS _cleo LIMIT {limit + 1}"
            cur.execute(wrapped)
            fetched = cur.fetchall()
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = [[_normalize_cell(c) for c in row] for row in fetched]
            return rows[:limit], columns, len(rows) > limit
        finally:
            try:
                source.rollback()  # never commit — Cleo is read-only
            except Exception:
                pass
            try:
                cur.close()
            except Exception:
                pass

    # signature is (sql, limit) -> (rows, columns, truncated); align tuple order below
    def _executor(sql: str, limit: int = 20) -> tuple[list, list, bool]:
        rows, columns, truncated = _execute(sql, limit)
        return columns, rows, truncated

    return _executor


def run_readonly(executor: Executor, sql: str, limit: int = 20,
                 dialect: str | None = "duckdb") -> tuple[list, list, bool, str | None]:
    """Validate `sql` is read-only, run it via `executor`, return (columns, rows, truncated, error)."""
    ok, why = is_readonly(sql, dialect=dialect)
    if not ok:
        return [], [], False, f"rejected_non_readonly:{why}"
    try:
        columns, rows, truncated = executor(sql, limit)
        return columns, rows, truncated, None
    except Exception as exc:  # surface DB errors back to the model as an observation
        return [], [], False, str(exc)[:160]


_SYS_SCHEMAS = ("information_schema", "pg_catalog", "pg_toast", "sys", "mysql", "performance_schema")


def introspect_schema(source: Any, tables: list[str] | None = None,
                      db_schema: str | None = None, max_tables: int = 60) -> str:
    """Build CREATE TABLE DDL from a live connection. Scope big DBs with `tables=[...]` / `db_schema=`.

    Tries ANSI `information_schema.columns` (Postgres/MySQL/DuckDB); falls back to SQLite `PRAGMA`.
    """
    if not hasattr(source, "cursor"):
        raise TypeError("schema introspection needs a DB-API connection; pass schema=... instead")
    want = {t.lower() for t in tables} if tables else None
    cur = source.cursor()
    try:
        try:
            q = ("SELECT table_name, column_name, data_type FROM information_schema.columns "
                 "WHERE lower(table_schema) NOT IN ({}) "
                 "ORDER BY table_name, ordinal_position").format(
                ",".join("'%s'" % s for s in _SYS_SCHEMAS))
            cur.execute(q)
            fetched = cur.fetchall()
            cols_by_table: dict[str, list[str]] = {}
            for table_name, column_name, data_type in fetched:
                if want is not None and str(table_name).lower() not in want:
                    continue
                cols_by_table.setdefault(str(table_name), []).append(f"{column_name} {data_type}")
        except Exception:
            cols_by_table = _introspect_sqlite(cur, want)
    finally:
        try:
            source.rollback()
        except Exception:
            pass
        try:
            cur.close()
        except Exception:
            pass
    if not cols_by_table:
        raise RuntimeError("no tables found to introspect; pass schema=<DDL string> or tables=[...]")
    items = list(cols_by_table.items())[:max_tables]
    return "\n".join(f"CREATE TABLE {t} ({', '.join(cols)});" for t, cols in items)


def _introspect_sqlite(cur: Any, want: set[str] | None) -> dict[str, list[str]]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    names = [r[0] for r in cur.fetchall()]
    out: dict[str, list[str]] = {}
    for name in names:
        if want is not None and name.lower() not in want:
            continue
        cur.execute(f'PRAGMA table_info("{name}")')
        out[name] = [f"{r[1]} {r[2]}" for r in cur.fetchall()]
    return out
