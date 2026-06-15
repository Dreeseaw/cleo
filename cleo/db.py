"""Read-only SQL execution and live schema introspection.

Cleo runs against your DB-API 2.0 connection: psycopg2, sqlite3, duckdb, or a SQLAlchemy
`raw_connection()`. Supported dialects: Postgres, MySQL, SQLite, and DuckDB.

The SQL guard blocks writes and common side-effect functions. For production, still use a
least-privilege, read-only DB role with a statement timeout.
"""
from __future__ import annotations

import datetime
import decimal
import re
import time
from typing import Any, Callable

from .contract import is_readonly

# An executor maps (sql, limit) -> (columns, rows, truncated).
Executor = Callable[[str, int], "tuple[list, list, bool]"]

# Cleo writes DuckDB SQL. Transpile it before execution when the target DB differs.
MODEL_DIALECT = "duckdb"
_DRIVER_DIALECT = {"sqlite3": "sqlite", "psycopg2": "postgres", "psycopg": "postgres",
                   "duckdb": "duckdb", "pymysql": "mysql", "mysql": "mysql",
                   "mariadb": "mysql", "MySQLdb": "mysql"}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_SYS_SCHEMAS = ("information_schema", "pg_catalog", "pg_toast", "sys", "mysql", "performance_schema")
_STMT_TIMEOUT_MS = 15000


def detect_dialect(conn: Any) -> str:
    """Map a DB-API connection to a sqlglot dialect, falling back to DuckDB."""
    mod = type(conn).__module__.split(".")[0]
    return _DRIVER_DIALECT.get(mod, MODEL_DIALECT)


def transpile_sql(sql: str, target: str | None, source: str = MODEL_DIALECT) -> str:
    """Best-effort SQL translation from the model dialect to the target dialect."""
    if not target or target == source:
        return sql
    try:
        import sqlglot
        out = sqlglot.transpile(sql, read=source, write=target)
        return out[0] if out else sql
    except Exception:
        return sql  # let the DB report the error if translation fails


def _quiet(fn) -> None:
    try:
        fn()
    except Exception:
        pass


def _sqlite_timeout_guard(conn: Any):
    if type(conn).__module__.split(".")[0] != "sqlite3" or not hasattr(conn, "set_progress_handler"):
        return lambda: None
    deadline = time.monotonic() + (_STMT_TIMEOUT_MS / 1000)

    def progress() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(progress, 10000)
    return lambda: conn.set_progress_handler(None, 0)


def _normalize_cell(v: Any) -> Any:
    if v is None or isinstance(v, (int, str, bool, float)):
        return round(v, 8) if isinstance(v, float) else v
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(v))} bytes>"
    return str(v)


def _is_autocommit(conn: Any) -> bool:
    ac = getattr(conn, "autocommit", None)
    if ac is True:          # psycopg2 / sqlite3(3.12+) explicit autocommit
        return True
    if ac is False:         # explicitly transactional
        return False
    # ac is None (attr absent) or sqlite3 LEGACY_TRANSACTION_CONTROL (-1) -> defer to isolation_level
    return getattr(conn, "isolation_level", "x") is None


def make_executor(source: Any) -> Executor:
    """Create a read-only executor from a DB-API connection or executor callable.

    Use a non-autocommit connection dedicated to Cleo. SQLAlchemy users can pass
    `engine.raw_connection()`.
    """
    if callable(source) and not hasattr(source, "cursor"):
        return source
    if not hasattr(source, "cursor"):
        raise TypeError("source must be a DB-API 2.0 connection (has .cursor()) or an executor callable")
    if _is_autocommit(source):
        raise ValueError("connection is in autocommit mode; Cleo needs a non-autocommit, read-only "
                         "connection so it can roll back and leave no trace")

    def execute(sql: str, limit: int = 20) -> tuple[list, list, bool]:
        cur = source.cursor()
        clear_timeout = _sqlite_timeout_guard(source)
        try:
            _quiet(lambda: cur.execute(f"SET statement_timeout = {_STMT_TIMEOUT_MS}"))  # PG; ignored elsewhere
            wrapped = f"SELECT * FROM ({sql.strip().rstrip(';')}) AS _cleo LIMIT {int(limit) + 1}"
            cur.execute(wrapped)
            fetched = cur.fetchmany(int(limit) + 1)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = [[_normalize_cell(c) for c in row] for row in fetched]
            return columns, rows[:limit], len(rows) > limit
        finally:
            _quiet(clear_timeout)
            _quiet(source.rollback)  # never commit
            _quiet(cur.close)

    return execute


def run_readonly(executor: Executor, sql: str, limit: int = 20,
                 dialect: str | None = MODEL_DIALECT) -> tuple[list, list, bool, str | None]:
    """Validate, transpile, and run read-only SQL. Returns columns, rows, truncated, error."""
    ok, why = is_readonly(sql, dialect=MODEL_DIALECT)
    if not ok:
        return [], [], False, f"rejected ({why})"
    exec_sql = transpile_sql(sql, dialect)
    try:
        columns, rows, truncated = executor(exec_sql, limit)
        return columns, rows, truncated, None
    except Exception as exc:  # surface DB errors back to the model as an observation
        return [], [], False, str(exc)[:120]


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _ddl_ident(name: str) -> str:
    """Render an identifier for SQL, quoting only when needed."""
    s = str(name)
    return s if _IDENT_RE.match(s) else _quote_ident(s)


def introspect_schema(source: Any, tables: list[str] | None = None, db_schema: str | None = None,
                      max_tables: int = 60, fks: bool = False, samples: int = 0) -> str:
    """Build CREATE TABLE DDL from a live connection.

    Uses `information_schema` when available, then SQLite PRAGMA. Scope large DBs with
    `tables=` or `db_schema=`. `fks=True` and `samples=N` add optional grounding hints.
    """
    if not hasattr(source, "cursor"):
        raise TypeError("schema introspection needs a DB-API connection; pass schema=... instead")
    want = {t.lower() for t in tables} if tables else None
    cur = source.cursor()
    try:
        cols_by_table = _introspect_ansi(cur, want, db_schema)
        if cols_by_table is None:
            cols_by_table = _introspect_sqlite(cur, want)
    finally:
        _quiet(source.rollback)
        _quiet(cur.close)
    if not cols_by_table:
        raise RuntimeError("no tables found to introspect; pass schema=<DDL string> or tables=[...]")
    if len(cols_by_table) > max_tables and not (tables or db_schema):
        raise RuntimeError(
            f"{len(cols_by_table)} tables found (> max_tables={max_tables}); scope with "
            f"tables=[...] or db_schema='...' so the prompt stays focused")
    lines = []
    for t, cols in list(cols_by_table.items())[:max_tables]:
        lines.append(f"CREATE TABLE {_ddl_ident(t)} ({', '.join(cols)});")
        if fks:
            for frm, rt, rc in _table_fks(source, t):
                lines.append(f"  -- {_ddl_ident(t)}.{_ddl_ident(frm)} -> {_ddl_ident(rt)}.{_ddl_ident(rc)}")
        if samples > 0:
            srows = _table_samples(source, t, samples)
            if srows:
                lines.append(f"  -- examples: {srows}")
    return "\n".join(lines)


def _table_fks(source: Any, table: str) -> list:
    """Foreign keys for `table` (SQLite PRAGMA; empty for engines where it doesn't apply)."""
    cur = source.cursor()
    try:
        cur.execute(f"PRAGMA foreign_key_list({_quote_ident(table)})")
        return [(r[3], r[2], r[4]) for r in cur.fetchall()]  # (from_col, ref_table, ref_col)
    except Exception:
        return []
    finally:
        _quiet(source.rollback)
        _quiet(cur.close)


def _table_samples(source: Any, table: str, n: int) -> str:
    cur = source.cursor()
    try:
        cur.execute(f"SELECT * FROM {_quote_ident(table)} LIMIT {int(n)}")
        rows = [[_normalize_cell(c) for c in row] for row in cur.fetchmany(n)]
        body = ", ".join(str(r) for r in rows)
        return body[:240] + (" ..." if len(body) > 240 else "")
    except Exception:
        return ""
    finally:
        _quiet(source.rollback)
        _quiet(cur.close)


def _introspect_ansi(cur: Any, want: set[str] | None, db_schema: str | None) -> dict | None:
    """Use information_schema, or return None so SQLite PRAGMA can try next."""
    where = "lower(table_schema) NOT IN ({})".format(", ".join("'%s'" % s for s in _SYS_SCHEMAS))
    if db_schema:
        if not _IDENT_RE.match(db_schema):  # validated -> safe to inline (no driver-paramstyle guessing)
            raise ValueError(f"invalid db_schema: {db_schema!r}")
        where += f" AND lower(table_schema) = lower('{db_schema}')"
    try:
        cur.execute("SELECT table_schema, table_name, column_name, data_type FROM information_schema.columns "
                    f"WHERE {where} ORDER BY table_schema, table_name, ordinal_position")
        fetched = cur.fetchall()
    except Exception:
        return None
    multi_schema = len({r[0] for r in fetched}) > 1
    out: dict[str, list[str]] = {}
    for table_schema, table_name, column_name, data_type in fetched:
        if want is not None and str(table_name).lower() not in want:
            continue
        key = f"{table_schema}.{table_name}" if multi_schema else str(table_name)
        out.setdefault(key, []).append(f"{_ddl_ident(column_name)} {data_type}")
    return out


def _introspect_sqlite(cur: Any, want: set[str] | None) -> dict[str, list[str]]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    names = [r[0] for r in cur.fetchall()]
    out: dict[str, list[str]] = {}
    for name in names:
        if want is not None and str(name).lower() not in want:
            continue
        cur.execute(f"PRAGMA table_info({_quote_ident(name)})")  # identifier quoted -> injection-safe
        out[name] = [f"{_ddl_ident(r[1])} {r[2]}" for r in cur.fetchall()]
    return out
