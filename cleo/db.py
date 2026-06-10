"""Connection-agnostic, read-only SQL execution + live schema introspection.

Cleo runs against *your* DB-API 2.0 connection — psycopg2 / sqlite3 / duckdb / a SQLAlchemy
`raw_connection()`. Supported dialects: Postgres, MySQL, SQLite, DuckDB (the `LIMIT` row-cap and
`information_schema`/`PRAGMA` introspection do not cover Oracle/SQL Server).

Read-only is enforced primarily by the SQL guard in `contract.is_readonly` (statement + AST + side-effect
function denylist). For production, ALSO run Cleo under a least-privilege, read-only DB role with a
statement timeout — the in-process guard is defense-in-depth, not a substitute for DB permissions.
"""
from __future__ import annotations

import datetime
import decimal
import re
from typing import Any, Callable

from .contract import is_readonly

# An executor maps (sql, limit) -> (columns, rows, truncated).
Executor = Callable[[str, int], "tuple[list, list, bool]"]

# Cleo is trained to write DuckDB-flavored SQL. The harness TRANSPILES it to the connection's dialect
# before executing, so the model's native output runs correctly against SQLite/Postgres/MySQL/etc.
MODEL_DIALECT = "duckdb"
_DRIVER_DIALECT = {"sqlite3": "sqlite", "psycopg2": "postgres", "psycopg": "postgres",
                   "duckdb": "duckdb", "pymysql": "mysql", "mysql": "mysql",
                   "mariadb": "mysql", "MySQLdb": "mysql"}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_SYS_SCHEMAS = ("information_schema", "pg_catalog", "pg_toast", "sys", "mysql", "performance_schema")
_STMT_TIMEOUT_MS = 15000


def detect_dialect(conn: Any) -> str:
    """Best-effort: map a DB-API connection to a sqlglot dialect (falls back to the model's own)."""
    mod = type(conn).__module__.split(".")[0]
    return _DRIVER_DIALECT.get(mod, MODEL_DIALECT)


def transpile_sql(sql: str, target: str | None, source: str = MODEL_DIALECT) -> str:
    """Translate the model's `source`-dialect SQL to the connection's `target` dialect (best-effort)."""
    if not target or target == source:
        return sql
    try:
        import sqlglot
        out = sqlglot.transpile(sql, read=source, write=target)
        return out[0] if out else sql
    except Exception:
        return sql  # if it won't transpile, run as-is and let the DB report the error


def _quiet(fn) -> None:
    try:
        fn()
    except Exception:
        pass


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
    """A DB-API 2.0 connection OR a callable `(sql, limit) -> (columns, rows, truncated)` -> read-only executor.

    The connection should be dedicated to Cleo and NOT in autocommit (Cleo relies on rollback to leave no
    trace); a query is also wrapped read-only and capped. SQLAlchemy: pass `engine.raw_connection()`.
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
        try:
            _quiet(lambda: cur.execute(f"SET statement_timeout = {_STMT_TIMEOUT_MS}"))  # PG; ignored elsewhere
            wrapped = f"SELECT * FROM ({sql.strip().rstrip(';')}) AS _cleo LIMIT {int(limit) + 1}"
            cur.execute(wrapped)
            fetched = cur.fetchmany(int(limit) + 1)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = [[_normalize_cell(c) for c in row] for row in fetched]
            return columns, rows[:limit], len(rows) > limit
        finally:
            _quiet(source.rollback)  # never commit — Cleo is read-only
            _quiet(cur.close)

    return execute


def run_readonly(executor: Executor, sql: str, limit: int = 20,
                 dialect: str | None = MODEL_DIALECT) -> tuple[list, list, bool, str | None]:
    """Guard the model's SQL (read-only), TRANSPILE it from DuckDB to the connection's `dialect`, run it.

    Returns (columns, rows, truncated, error). The guard parses the model's native DuckDB; the transpiled
    SQL is what executes — so a DuckDB-trained model runs correctly against SQLite/Postgres/etc.
    """
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
    """Render an identifier as it must be written in SQL: quoted iff it needs quoting.

    Schema text is what the model copies identifiers from — rendering `Examination Date` or `T-CHO`
    unquoted teaches it to write unparseable SQL against real-world schemas.
    """
    s = str(name)
    return s if _IDENT_RE.match(s) else _quote_ident(s)


def introspect_schema(source: Any, tables: list[str] | None = None, db_schema: str | None = None,
                      max_tables: int = 60, fks: bool = False, samples: int = 0) -> str:
    """Build CREATE TABLE DDL from a live connection. Scope big DBs with `tables=[...]` / `db_schema=`.

    Tries ANSI `information_schema.columns` (Postgres/MySQL/DuckDB); falls back to SQLite `PRAGMA`. Raises
    if there are more than `max_tables` tables and no scoping was given (rather than silently truncating).
    `fks=True` appends foreign-key relationships (SQLite); `samples=N` appends N example rows per table —
    both help the model ground columns to the right table.
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
    """information_schema path. Returns None (to trigger the SQLite fallback) if it isn't available."""
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
