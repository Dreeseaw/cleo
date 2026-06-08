"""Schema-agnostic analytic tools for DuckDB-backed SQL workflows."""

from __future__ import annotations

import datetime as _dt
import decimal
import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb

from .sql_parse import validate_safe_sql


DEFAULT_SAMPLE_LIMIT = 5
DEFAULT_MAX_JOIN_HOPS = 3
DEFAULT_MAX_JOIN_PATHS = 20
DEFAULT_MAX_PROFILE_DISTINCT = 10_000
DEFAULT_CHART_MAX_ROWS = 200
AGGREGATE_FUNCTIONS = {"sum", "avg", "min", "max", "count", "count_distinct"}
FILTER_OPERATORS = {"=", "!=", "<>", "<", "<=", ">", ">=", "in", "not in", "is null", "is not null"}


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _quote_column_ref(ref: str) -> str:
    parts = [part for part in str(ref).split(".") if part]
    if not parts:
        raise ValueError("empty column reference")
    if len(parts) > 2:
        raise ValueError(f"unsupported column reference: {ref}")
    return ".".join(quote_ident(part) for part in parts)


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, decimal.Decimal)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (_dt.date, _dt.datetime, _dt.time)):
        value = value.isoformat()
    return "'" + str(value).replace("'", "''") + "'"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (_dt.date, _dt.datetime, _dt.time)):
        return value.isoformat()
    return str(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _connect_readonly(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(
        str(db_path),
        config={"access_mode": "READ_ONLY", "enable_external_access": "false"},
    )


def _safe_limit(limit: int, *, default: int = DEFAULT_SAMPLE_LIMIT, hard_max: int = 100) -> int:
    try:
        value = int(limit)
    except Exception:
        value = default
    return max(0, min(value, hard_max))


def _table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table],
    ).fetchone()
    return bool(row and row[0])


def _table_names(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY table_name
        """
    ).fetchall()
    return [(str(name), str(table_type)) for name, table_type in rows]


def _column_rows(conn: duckdb.DuckDBPyConnection, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT column_name, data_type, is_nullable, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table],
    ).fetchall()
    return [
        {
            "name": str(name),
            "data_type": str(data_type),
            "nullable": str(is_nullable).upper() == "YES",
            "ordinal_position": int(ordinal),
        }
        for name, data_type, is_nullable, ordinal in rows
    ]


def _row_count(conn: duckdb.DuckDBPyConnection, table: str) -> int | None:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}").fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def _constraint_rows(conn: duckdb.DuckDBPyConnection) -> tuple[dict[tuple[str, str], str], list[dict[str, Any]]]:
    primary_keys: dict[tuple[str, str], str] = {}
    foreign_keys: list[dict[str, Any]] = []
    try:
        pk_rows = conn.execute(
            """
            SELECT tc.table_name, kcu.column_name, tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_schema = kcu.constraint_schema
             AND tc.constraint_name = kcu.constraint_name
             AND tc.table_name = kcu.table_name
            WHERE tc.table_schema = 'main'
              AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY tc.table_name, kcu.ordinal_position
            """
        ).fetchall()
        primary_keys = {(str(table), str(column)): str(name) for table, column, name in pk_rows}
    except Exception:
        primary_keys = {}

    try:
        fk_rows = conn.execute(
            """
            SELECT
              tc.table_name,
              kcu.column_name,
              ccu.table_name AS referenced_table,
              ccu.column_name AS referenced_column,
              tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_schema = kcu.constraint_schema
             AND tc.constraint_name = kcu.constraint_name
             AND tc.table_name = kcu.table_name
            JOIN information_schema.referential_constraints rc
              ON tc.constraint_schema = rc.constraint_schema
             AND tc.constraint_name = rc.constraint_name
            JOIN information_schema.constraint_column_usage ccu
              ON rc.unique_constraint_schema = ccu.constraint_schema
             AND rc.unique_constraint_name = ccu.constraint_name
            WHERE tc.table_schema = 'main'
              AND tc.constraint_type = 'FOREIGN KEY'
            ORDER BY tc.table_name, kcu.column_name
            """
        ).fetchall()
        for table, column, ref_table, ref_column, name in fk_rows:
            foreign_keys.append(
                {
                    "left_table": str(table),
                    "left_column": str(column),
                    "right_table": str(ref_table),
                    "right_column": str(ref_column),
                    "source": "foreign_key",
                    "confidence": 1.0,
                    "constraint_name": str(name),
                }
            )
    except Exception:
        foreign_keys = []
    return primary_keys, foreign_keys


def schema_catalog(db_path: str | Path, *, include_columns: bool = True) -> dict[str, Any]:
    """Return tables, columns, row counts, and constraint-derived relationships."""

    conn = _connect_readonly(db_path)
    try:
        primary_keys, foreign_keys = _constraint_rows(conn)
        tables: list[dict[str, Any]] = []
        for table, table_type in _table_names(conn):
            entry: dict[str, Any] = {
                "name": table,
                "type": table_type,
                "row_count": _row_count(conn, table),
            }
            if include_columns:
                columns = _column_rows(conn, table)
                for column in columns:
                    column["primary_key"] = (table, column["name"]) in primary_keys
                entry["columns"] = columns
            tables.append(entry)
        return {
            "ok": True,
            "tool": "schema_catalog",
            "db_path": str(db_path),
            "table_count": len(tables),
            "tables": tables,
            "relationships": foreign_keys,
            "catalog_hash": _stable_hash({"tables": tables, "relationships": foreign_keys}),
        }
    except Exception as exc:
        return {"ok": False, "tool": "schema_catalog", "error_class": type(exc).__name__, "errors": [str(exc)[:300]]}
    finally:
        conn.close()


def _is_numeric_type(data_type: str) -> bool:
    upper = data_type.upper()
    return any(token in upper for token in ("INT", "REAL", "DOUBLE", "DECIMAL", "NUMERIC", "FLOAT", "HUGEINT", "UBIGINT"))


def _is_temporal_type(data_type: str) -> bool:
    upper = data_type.upper()
    return any(token in upper for token in ("DATE", "TIME", "TIMESTAMP"))


def _profile_column(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    column: dict[str, Any],
    *,
    row_count: int,
    max_exact_distinct: int,
) -> dict[str, Any]:
    col = str(column["name"])
    ident = quote_ident(col)
    table_ident = quote_ident(table)
    null_row = conn.execute(f"SELECT SUM(CASE WHEN {ident} IS NULL THEN 1 ELSE 0 END) FROM {table_ident}").fetchone()
    null_count = int(null_row[0] or 0) if null_row else 0
    distinct_count: int | None
    if row_count <= max_exact_distinct:
        distinct_row = conn.execute(f"SELECT COUNT(DISTINCT {ident}) FROM {table_ident}").fetchone()
        distinct_count = int(distinct_row[0] or 0) if distinct_row else 0
    else:
        distinct_count = None
    stats: dict[str, Any] = {
        "name": col,
        "data_type": column["data_type"],
        "nullable": bool(column["nullable"]),
        "null_count": null_count,
        "null_pct": round(null_count / row_count, 6) if row_count else 0.0,
        "distinct_count": distinct_count,
        "cardinality_ratio": round(distinct_count / row_count, 6) if row_count and distinct_count is not None else None,
    }
    if row_count:
        try:
            minmax = conn.execute(f"SELECT MIN({ident}), MAX({ident}) FROM {table_ident}").fetchone()
            if minmax:
                stats["min"] = _jsonable(minmax[0])
                stats["max"] = _jsonable(minmax[1])
        except Exception:
            pass
        if distinct_count is not None and 0 < distinct_count <= 12 and not _is_numeric_type(str(column["data_type"])):
            try:
                rows = conn.execute(
                    f"SELECT {ident} AS value, COUNT(*) AS n FROM {table_ident} "
                    f"WHERE {ident} IS NOT NULL GROUP BY 1 ORDER BY n DESC, 1 LIMIT 12"
                ).fetchall()
                stats["top_values"] = [{"value": _jsonable(value), "count": int(count)} for value, count in rows]
            except Exception:
                pass
    return stats


def table_profile(
    db_path: str | Path,
    table: str,
    *,
    max_exact_distinct: int = DEFAULT_MAX_PROFILE_DISTINCT,
) -> dict[str, Any]:
    """Return row count, dtypes, null rates, cardinality, and compact value stats."""

    conn = _connect_readonly(db_path)
    try:
        if not _table_exists(conn, table):
            return {"ok": False, "tool": "table_profile", "error_class": "not_found", "errors": [f"unknown table: {table}"]}
        columns = _column_rows(conn, table)
        count = _row_count(conn, table)
        row_count = int(count or 0)
        profiled = [
            _profile_column(
                conn,
                table,
                column,
                row_count=row_count,
                max_exact_distinct=max(0, int(max_exact_distinct)),
            )
            for column in columns
        ]
        return {
            "ok": True,
            "tool": "table_profile",
            "table": table,
            "row_count": row_count,
            "column_count": len(profiled),
            "columns": profiled,
            "profile_hash": _stable_hash({"table": table, "row_count": row_count, "columns": profiled}),
        }
    except Exception as exc:
        return {"ok": False, "tool": "table_profile", "table": table, "error_class": type(exc).__name__, "errors": [str(exc)[:300]]}
    finally:
        conn.close()


def sample_rows(
    db_path: str | Path,
    table: str,
    *,
    limit: int = DEFAULT_SAMPLE_LIMIT,
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return bounded raw rows for a table."""

    conn = _connect_readonly(db_path)
    try:
        if not _table_exists(conn, table):
            return {"ok": False, "tool": "sample_rows", "error_class": "not_found", "errors": [f"unknown table: {table}"]}
        available = {column["name"] for column in _column_rows(conn, table)}
        if columns:
            missing = [column for column in columns if column not in available]
            if missing:
                return {"ok": False, "tool": "sample_rows", "error_class": "not_found", "errors": [f"unknown columns: {missing}"]}
            projection = ", ".join(quote_ident(column) for column in columns)
        else:
            projection = "*"
        visible_limit = _safe_limit(limit)
        rows = conn.execute(
            f"SELECT {projection} FROM {quote_ident(table)} LIMIT {visible_limit + 1}"
        ).fetchall()
        cursor = conn.execute(f"SELECT {projection} FROM {quote_ident(table)} LIMIT 0")
        output_columns = [str(desc[0]) for desc in cursor.description or []]
        visible = rows[:visible_limit]
        normalized_rows = [[_jsonable(cell) for cell in row] for row in visible]
        return {
            "ok": True,
            "tool": "sample_rows",
            "table": table,
            "limit": visible_limit,
            "columns": output_columns,
            "rows": normalized_rows,
            "truncated": len(rows) > visible_limit,
            "sample_hash": _stable_hash({"columns": output_columns, "rows": normalized_rows}),
        }
    except Exception as exc:
        return {"ok": False, "tool": "sample_rows", "table": table, "error_class": type(exc).__name__, "errors": [str(exc)[:300]]}
    finally:
        conn.close()


def _norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _singular(value: str) -> str:
    lowered = value.lower()
    if lowered.endswith("ies") and len(lowered) > 3:
        return lowered[:-3] + "y"
    if lowered.endswith("es") and len(lowered) > 2:
        return lowered[:-2]
    if lowered.endswith("s") and len(lowered) > 1:
        return lowered[:-1]
    return lowered


def _column_join_score(left_table: str, left_column: str, right_table: str, right_column: str) -> tuple[float, str] | None:
    lcol = left_column.lower()
    rcol = right_column.lower()
    left_prefix = lcol[:-3] if lcol.endswith("_id") else ""
    right_prefix = rcol[:-3] if rcol.endswith("_id") else ""
    if lcol == rcol and (lcol == "id" or lcol.endswith("_id") or lcol.endswith("id")):
        return 0.78, "same_name_id_column"
    right_names = {_norm_name(right_table), _norm_name(_singular(right_table))}
    left_names = {_norm_name(left_table), _norm_name(_singular(left_table))}
    if left_prefix and rcol in {"id", f"{left_prefix}_id"} and _norm_name(_singular(right_table)) == _norm_name(left_prefix):
        return 0.9, "name_based_fk_to_pk"
    if right_prefix and lcol in {"id", f"{right_prefix}_id"} and _norm_name(_singular(left_table)) == _norm_name(right_prefix):
        return 0.9, "name_based_fk_to_pk"
    if left_prefix and _norm_name(left_prefix) in right_names and rcol in {"id", f"{left_prefix}_id"}:
        return 0.85, "table_prefix_fk"
    if right_prefix and _norm_name(right_prefix) in left_names and lcol in {"id", f"{right_prefix}_id"}:
        return 0.85, "table_prefix_fk"
    return None


@dataclass(frozen=True)
class JoinEdge:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    source: str
    confidence: float

    def key(self) -> tuple[str, str, str, str]:
        forward = (self.left_table, self.left_column, self.right_table, self.right_column)
        reverse = (self.right_table, self.right_column, self.left_table, self.left_column)
        return min(forward, reverse)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_table": self.left_table,
            "left_column": self.left_column,
            "right_table": self.right_table,
            "right_column": self.right_column,
            "source": self.source,
            "confidence": self.confidence,
            "condition": (
                f"{quote_ident(self.left_table)}.{quote_ident(self.left_column)} = "
                f"{quote_ident(self.right_table)}.{quote_ident(self.right_column)}"
            ),
        }


def _join_edges(conn: duckdb.DuckDBPyConnection) -> list[JoinEdge]:
    explicit = [
        JoinEdge(
            row["left_table"],
            row["left_column"],
            row["right_table"],
            row["right_column"],
            str(row.get("source") or "foreign_key"),
            float(row.get("confidence") or 1.0),
        )
        for row in _constraint_rows(conn)[1]
    ]
    tables = [name for name, _table_type in _table_names(conn)]
    columns_by_table = {table: _column_rows(conn, table) for table in tables}
    edges_by_key: dict[tuple[str, str, str, str], JoinEdge] = {edge.key(): edge for edge in explicit}
    for i, left_table in enumerate(tables):
        for right_table in tables[i + 1 :]:
            for left_col in columns_by_table[left_table]:
                for right_col in columns_by_table[right_table]:
                    scored = _column_join_score(left_table, left_col["name"], right_table, right_col["name"])
                    if scored is None:
                        continue
                    confidence, source = scored
                    edge = JoinEdge(left_table, left_col["name"], right_table, right_col["name"], source, confidence)
                    old = edges_by_key.get(edge.key())
                    if old is None or edge.confidence > old.confidence:
                        edges_by_key[edge.key()] = edge
    edges = list(edges_by_key.values())
    edges.sort(key=lambda e: (-e.confidence, e.left_table, e.right_table, e.left_column, e.right_column))
    return edges


def _oriented(edge: JoinEdge, from_table: str) -> dict[str, Any]:
    if edge.left_table == from_table:
        return edge.to_dict()
    return JoinEdge(
        edge.right_table,
        edge.right_column,
        edge.left_table,
        edge.left_column,
        edge.source,
        edge.confidence,
    ).to_dict()


def _path_tables(path: Sequence[dict[str, Any]], start: str) -> list[str]:
    tables = [start]
    current = start
    for edge in path:
        nxt = str(edge["right_table"]) if edge["left_table"] == current else str(edge["left_table"])
        tables.append(nxt)
        current = nxt
    return tables


def _find_paths(
    edges: Sequence[JoinEdge],
    *,
    start_table: str | None,
    end_table: str | None,
    max_hops: int,
    max_paths: int,
) -> list[dict[str, Any]]:
    graph: dict[str, list[JoinEdge]] = {}
    for edge in edges:
        graph.setdefault(edge.left_table, []).append(edge)
        graph.setdefault(edge.right_table, []).append(edge)
    starts = [start_table] if start_table else sorted(graph)
    found: list[dict[str, Any]] = []
    for start in starts:
        if start not in graph:
            continue
        queue: deque[tuple[str, list[dict[str, Any]], set[str], float]] = deque([(start, [], {start}, 1.0)])
        while queue and len(found) < max_paths:
            table, path, seen, confidence = queue.popleft()
            if path and (end_table is None or table == end_table):
                found.append(
                    {
                        "start_table": start,
                        "end_table": table,
                        "hop_count": len(path),
                        "tables": _path_tables(path, start),
                        "confidence": round(confidence, 6),
                        "joins": path,
                    }
                )
                if end_table is not None:
                    continue
            if len(path) >= max_hops:
                continue
            for edge in graph.get(table, []):
                next_table = edge.right_table if edge.left_table == table else edge.left_table
                if next_table in seen:
                    continue
                queue.append(
                    (
                        next_table,
                        [*path, _oriented(edge, table)],
                        {*seen, next_table},
                        confidence * edge.confidence,
                    )
                )
    found.sort(key=lambda p: (int(p["hop_count"]), -float(p["confidence"]), p["start_table"], p["end_table"]))
    return found[:max_paths]


def find_join_paths(
    db_path: str | Path,
    *,
    start_table: str | None = None,
    end_table: str | None = None,
    max_hops: int = DEFAULT_MAX_JOIN_HOPS,
    max_paths: int = DEFAULT_MAX_JOIN_PATHS,
) -> dict[str, Any]:
    """Find explicit-FK and name-based join paths between tables."""

    conn = _connect_readonly(db_path)
    try:
        tables = {name for name, _table_type in _table_names(conn)}
        if start_table is not None and start_table not in tables:
            return {"ok": False, "tool": "find_join_paths", "error_class": "not_found", "errors": [f"unknown start_table: {start_table}"]}
        if end_table is not None and end_table not in tables:
            return {"ok": False, "tool": "find_join_paths", "error_class": "not_found", "errors": [f"unknown end_table: {end_table}"]}
        edges = _join_edges(conn)
        paths = _find_paths(
            edges,
            start_table=start_table,
            end_table=end_table,
            max_hops=max(1, int(max_hops)),
            max_paths=max(1, int(max_paths)),
        )
        return {
            "ok": True,
            "tool": "find_join_paths",
            "start_table": start_table,
            "end_table": end_table,
            "max_hops": max_hops,
            "edges": [edge.to_dict() for edge in edges],
            "paths": paths,
            "path_count": len(paths),
            "join_graph_hash": _stable_hash({"edges": [edge.to_dict() for edge in edges], "paths": paths}),
        }
    except Exception as exc:
        return {"ok": False, "tool": "find_join_paths", "error_class": type(exc).__name__, "errors": [str(exc)[:300]]}
    finally:
        conn.close()


def _table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {column["name"] for column in _column_rows(conn, table)}


def _validate_column_ref(conn: duckdb.DuckDBPyConnection, ref: str, *, base_table: str) -> None:
    parts = [part for part in str(ref).split(".") if part]
    if len(parts) == 1:
        if parts[0] not in _table_columns(conn, base_table):
            raise ValueError(f"unknown column on {base_table}: {ref}")
        return
    if len(parts) == 2:
        table, column = parts
        if not _table_exists(conn, table):
            raise ValueError(f"unknown table in column reference: {table}")
        if column not in _table_columns(conn, table):
            raise ValueError(f"unknown column in reference {ref}")
        return
    raise ValueError(f"unsupported column reference: {ref}")


def _metric_expr(metric_column: str | None, metric_function: str) -> str:
    function = metric_function.lower()
    if function not in AGGREGATE_FUNCTIONS:
        raise ValueError(f"unsupported metric_function: {metric_function}")
    if function == "count" and not metric_column:
        return "COUNT(*)"
    if not metric_column:
        raise ValueError(f"metric_column is required for {metric_function}")
    column = _quote_column_ref(metric_column)
    if function == "count_distinct":
        return f"COUNT(DISTINCT {column})"
    return f"{function.upper()}({column})"


def _build_from_clause(base_table: str, joins: Sequence[dict[str, Any]] | None) -> str:
    clause = f"FROM {quote_ident(base_table)}"
    seen = {base_table}
    for join in joins or []:
        if not isinstance(join, dict):
            raise ValueError("joins must contain objects")
        left_table = str(join.get("left_table") or "")
        left_column = str(join.get("left_column") or "")
        right_table = str(join.get("right_table") or "")
        right_column = str(join.get("right_column") or "")
        if not all([left_table, left_column, right_table, right_column]):
            raise ValueError("join entries require left_table, left_column, right_table, right_column")
        next_table = right_table if left_table in seen else left_table
        if next_table in seen:
            continue
        condition = (
            f"{quote_ident(left_table)}.{quote_ident(left_column)} = "
            f"{quote_ident(right_table)}.{quote_ident(right_column)}"
        )
        clause += f"\nJOIN {quote_ident(next_table)} ON {condition}"
        seen.add(next_table)
    return clause


def _build_filters(filters: Sequence[dict[str, Any]] | None) -> tuple[str, list[Any], str]:
    clauses: list[str] = []
    params: list[Any] = []
    rendered: list[str] = []
    for item in filters or []:
        if not isinstance(item, dict):
            raise ValueError("filters must contain objects")
        column = str(item.get("column") or "")
        operator = str(item.get("op") or item.get("operator") or "=").lower()
        if not column:
            raise ValueError("filter column is required")
        if operator not in FILTER_OPERATORS:
            raise ValueError(f"unsupported filter operator: {operator}")
        ident = _quote_column_ref(column)
        if operator in {"is null", "is not null"}:
            clauses.append(f"{ident} {operator.upper()}")
            rendered.append(f"{ident} {operator.upper()}")
            continue
        value = item.get("value")
        if operator in {"in", "not in"}:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"{operator.upper()} filter value must be an array")
            values = list(value)
            if not values:
                raise ValueError(f"{operator.upper()} filter value must not be empty")
            placeholders = ", ".join("?" for _ in values)
            literals = ", ".join(_sql_literal(value) for value in values)
            clauses.append(f"{ident} {operator.upper()} ({placeholders})")
            rendered.append(f"{ident} {operator.upper()} ({literals})")
            params.extend(values)
        else:
            clauses.append(f"{ident} {operator.upper()} ?")
            rendered.append(f"{ident} {operator.upper()} {_sql_literal(value)}")
            params.append(value)
    if not clauses:
        return "", [], ""
    return "WHERE " + " AND ".join(clauses), params, "WHERE " + " AND ".join(rendered)


def _render_sql(parameterized_sql: str, params: Sequence[Any]) -> str:
    rendered = parameterized_sql
    for value in params:
        rendered = rendered.replace("?", _sql_literal(value), 1)
    return rendered


def aggregate_window_template(
    db_path: str | Path,
    *,
    pattern: str,
    table: str,
    metric_column: str | None = None,
    metric_function: str = "sum",
    grain_columns: Sequence[str] | None = None,
    filters: Sequence[dict[str, Any]] | None = None,
    joins: Sequence[dict[str, Any]] | None = None,
    date_column: str | None = None,
    order_by: str | None = None,
    partition_by: Sequence[str] | None = None,
    n: int = 10,
    period: str = "month",
    rank_function: str = "rank",
    metric_alias: str = "metric_value",
) -> dict[str, Any]:
    """Build deterministic DuckDB SQL for common aggregate/window patterns."""

    conn = _connect_readonly(db_path)
    try:
        if not _table_exists(conn, table):
            return {"ok": False, "tool": "aggregate_window_template", "error_class": "not_found", "errors": [f"unknown table: {table}"]}
        pattern = pattern.lower()
        metric_function = metric_function.lower()
        grain_columns = list(grain_columns or [])
        partition_by = list(partition_by or [])
        for ref in [*grain_columns, *partition_by]:
            _validate_column_ref(conn, ref, base_table=table)
        if metric_column:
            _validate_column_ref(conn, metric_column, base_table=table)
        if date_column:
            _validate_column_ref(conn, date_column, base_table=table)
        if order_by:
            _validate_column_ref(conn, order_by, base_table=table)
        for item in filters or []:
            _validate_column_ref(conn, str(item.get("column") or ""), base_table=table)
        for join in joins or []:
            for ref_table, ref_column in (
                (str(join.get("left_table") or ""), str(join.get("left_column") or "")),
                (str(join.get("right_table") or ""), str(join.get("right_column") or "")),
            ):
                if not _table_exists(conn, ref_table):
                    raise ValueError(f"unknown join table: {ref_table}")
                if ref_column not in _table_columns(conn, ref_table):
                    raise ValueError(f"unknown join column: {ref_table}.{ref_column}")

        metric = _metric_expr(metric_column, metric_function)
        from_clause = _build_from_clause(table, joins)
        where_param, params, where_rendered = _build_filters(filters)
        group_exprs = [_quote_column_ref(column) for column in grain_columns]
        select_grains = ", ".join(group_exprs)
        group_by = f"GROUP BY {', '.join(group_exprs)}" if group_exprs else ""
        order_grains = ", ".join(group_exprs)
        warnings: list[str] = []

        if pattern == "group_by_agg":
            projection = f"{select_grains}, {metric} AS {quote_ident(metric_alias)}" if select_grains else f"{metric} AS {quote_ident(metric_alias)}"
            order_clause = f"ORDER BY {order_grains}" if order_grains else f"ORDER BY {quote_ident(metric_alias)} DESC"
            parameterized_sql = "\n".join(part for part in [f"SELECT {projection}", from_clause, where_param, group_by, order_clause] if part)
        elif pattern == "top_n":
            limit = max(1, min(int(n), 100))
            projection = f"{select_grains}, {metric} AS {quote_ident(metric_alias)}" if select_grains else f"{metric} AS {quote_ident(metric_alias)}"
            parameterized_sql = "\n".join(
                part
                for part in [
                    f"SELECT {projection}",
                    from_clause,
                    where_param,
                    group_by,
                    f"ORDER BY {quote_ident(metric_alias)} DESC",
                    f"LIMIT {limit}",
                ]
                if part
            )
        elif pattern == "running_total":
            if not date_column:
                raise ValueError("date_column is required for running_total")
            date_expr = _quote_column_ref(date_column)
            partition_exprs = [_quote_column_ref(column) for column in partition_by or grain_columns]
            grouped_columns = [*partition_exprs, date_expr]
            partition_clause = f"PARTITION BY {', '.join(partition_exprs)} " if partition_exprs else ""
            projection_columns = ", ".join(grouped_columns)
            parameterized_sql = "\n".join(
                part
                for part in [
                    f"SELECT {projection_columns},",
                    f"       {metric} AS {quote_ident(metric_alias)},",
                    f"       SUM({metric}) OVER ({partition_clause}ORDER BY {date_expr} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS \"running_total\"",
                    from_clause,
                    where_param,
                    f"GROUP BY {', '.join(grouped_columns)}",
                    f"ORDER BY {', '.join(grouped_columns)}",
                ]
                if part
            )
        elif pattern == "period_over_period":
            if not date_column:
                raise ValueError("date_column is required for period_over_period")
            if period.lower() not in {"day", "week", "month", "quarter", "year"}:
                raise ValueError(f"unsupported period: {period}")
            bucket = f"date_trunc('{period.lower()}', {_quote_column_ref(date_column)})"
            partition_exprs = [_quote_column_ref(column) for column in grain_columns]
            inner_columns = [*partition_exprs, f"{bucket} AS \"period_start\""]
            inner_group = [*partition_exprs, bucket]
            outer_partition = f"PARTITION BY {', '.join(partition_exprs)} " if partition_exprs else ""
            outer_grains = ", ".join(partition_exprs)
            outer_projection = f"{outer_grains}, \"period_start\"" if outer_grains else "\"period_start\""
            parameterized_sql = "\n".join(
                part
                for part in [
                    "WITH period_metrics AS (",
                    f"  SELECT {', '.join(inner_columns)}, {metric} AS {quote_ident(metric_alias)}",
                    f"  {from_clause}",
                    f"  {where_param}" if where_param else "",
                    f"  GROUP BY {', '.join(inner_group)}",
                    ")",
                    f"SELECT {outer_projection},",
                    f"       {quote_ident(metric_alias)},",
                    f"       LAG({quote_ident(metric_alias)}) OVER ({outer_partition}ORDER BY \"period_start\") AS \"previous_value\",",
                    f"       {quote_ident(metric_alias)} - LAG({quote_ident(metric_alias)}) OVER ({outer_partition}ORDER BY \"period_start\") AS \"delta_value\",",
                    f"       CASE WHEN LAG({quote_ident(metric_alias)}) OVER ({outer_partition}ORDER BY \"period_start\") = 0 THEN NULL",
                    f"            ELSE ({quote_ident(metric_alias)} - LAG({quote_ident(metric_alias)}) OVER ({outer_partition}ORDER BY \"period_start\"))",
                    f"                 / LAG({quote_ident(metric_alias)}) OVER ({outer_partition}ORDER BY \"period_start\") END AS \"delta_pct\"",
                    "FROM period_metrics",
                    f"ORDER BY {outer_projection}",
                ]
                if part
            )
        elif pattern in {"rank", "dense_rank"}:
            function = "DENSE_RANK" if pattern == "dense_rank" or rank_function.lower() == "dense_rank" else "RANK"
            rank_partitions = [_quote_column_ref(column) for column in partition_by]
            grouped_columns = [*rank_partitions, *group_exprs]
            if not group_exprs:
                raise ValueError("grain_columns are required for rank templates")
            partition_clause = f"PARTITION BY {', '.join(rank_partitions)} " if rank_partitions else ""
            projection_columns = ", ".join(grouped_columns)
            parameterized_sql = "\n".join(
                part
                for part in [
                    f"SELECT {projection_columns},",
                    f"       {metric} AS {quote_ident(metric_alias)},",
                    f"       {function}() OVER ({partition_clause}ORDER BY {metric} DESC) AS \"metric_rank\"",
                    from_clause,
                    where_param,
                    f"GROUP BY {', '.join(grouped_columns)}",
                    f"ORDER BY {', '.join([*rank_partitions, '\"metric_rank\"']) if rank_partitions else '\"metric_rank\"'}",
                ]
                if part
            )
        else:
            raise ValueError(f"unsupported pattern: {pattern}")

        if joins:
            warnings.append("joins are applied as inner joins; inspect fanout with find_join_paths/table_profile first")
        if where_rendered:
            sql = _render_sql(parameterized_sql, params)
        else:
            sql = parameterized_sql
        validate_safe_sql(sql)
        return {
            "ok": True,
            "tool": "aggregate_window_template",
            "pattern": pattern,
            "sql": sql,
            "parameterized_sql": parameterized_sql,
            "parameters": [_jsonable(value) for value in params],
            "warnings": warnings,
            "template_hash": _stable_hash({"pattern": pattern, "sql": sql, "parameters": params}),
        }
    except Exception as exc:
        return {"ok": False, "tool": "aggregate_window_template", "error_class": type(exc).__name__, "errors": [str(exc)[:300]]}
    finally:
        conn.close()


def _explain_text(rows: Sequence[Sequence[Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        if len(row) >= 2 and str(row[0]).lower() in {"physical_plan", "logical_plan", "logical_opt"}:
            parts.append(str(row[1]))
        elif row:
            parts.append(" ".join(str(cell) for cell in row if cell is not None))
    return "\n".join(parts)


def explain_plan(db_path: str | Path, *, sql: str) -> dict[str, Any]:
    """Return a normalized DuckDB EXPLAIN summary for safe read-only SQL."""

    conn = _connect_readonly(db_path)
    try:
        validate_safe_sql(sql)
        rows = conn.execute("EXPLAIN " + sql.strip().rstrip(";")).fetchall()
        plan_text = _explain_text(rows)
        known_tables = [name for name, _table_type in _table_names(conn)]
        referenced_tables = sorted(
            table for table in known_tables if re.search(rf"\b{re.escape(table)}\b", sql, flags=re.IGNORECASE) or f"Table: {table}" in plan_text
        )
        scans = []
        for table in referenced_tables:
            pattern = rf"Table:\s*{re.escape(table)}(?P<body>.*?)(?:~(?P<rows>[\d,]+)\s+rows)?"
            match = re.search(pattern, plan_text, flags=re.DOTALL)
            scans.append(
                {
                    "table": table,
                    "operator": "scan",
                    "estimated_rows": int(match.group("rows").replace(",", "")) if match and match.group("rows") else None,
                }
            )
        join_ops = re.findall(r"\b([A-Z_]*JOIN)\b", plan_text)
        conditions = re.findall(r"Conditions:\s*([^\n│]+)", plan_text)
        est_rows = [int(value.replace(",", "")) for value in re.findall(r"~([\d,]+)\s+rows", plan_text)]
        joins = [
            {
                "operator": op,
                "condition": conditions[index].strip() if index < len(conditions) else None,
                "estimated_rows": est_rows[index] if index < len(est_rows) else None,
            }
            for index, op in enumerate(join_ops)
        ]
        return {
            "ok": True,
            "tool": "explain_plan",
            "referenced_tables": referenced_tables,
            "scans": scans,
            "joins": joins,
            "join_count": len(joins),
            "estimated_rows": est_rows,
            "plan_text": plan_text,
            "plan_hash": _stable_hash({"tables": referenced_tables, "scans": scans, "joins": joins, "estimated_rows": est_rows}),
        }
    except Exception as exc:
        return {"ok": False, "tool": "explain_plan", "error_class": type(exc).__name__, "errors": [str(exc)[:300]]}
    finally:
        conn.close()


def _normalize_result_schema(result_schema: Sequence[Any]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for index, item in enumerate(result_schema):
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("column") or item.get("field") or f"col_{index}")
            data_type = str(item.get("data_type") or item.get("type") or "")
        else:
            name = str(item)
            data_type = ""
        fields.append({"name": name, "data_type": data_type})
    return fields


def _rows_as_dicts(fields: Sequence[dict[str, Any]], rows: Sequence[Any], *, max_rows: int) -> list[dict[str, Any]]:
    names = [field["name"] for field in fields]
    out: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        if isinstance(row, dict):
            out.append({name: _jsonable(row.get(name)) for name in names})
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            out.append({name: _jsonable(row[index]) if index < len(row) else None for index, name in enumerate(names)})
    return out


def _field_role(field: dict[str, Any], values: Sequence[Any]) -> str:
    dtype = str(field.get("data_type") or "")
    if _is_temporal_type(dtype):
        return "temporal"
    if _is_numeric_type(dtype):
        return "quantitative"
    observed = [value for value in values if value is not None]
    if observed and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in observed):
        return "quantitative"
    if observed and all(isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", value) for value in observed):
        return "temporal"
    return "nominal"


def result_to_chart_spec(
    *,
    result_schema: Sequence[Any],
    rows: Sequence[Any],
    intent: str | None = None,
    max_rows: int = DEFAULT_CHART_MAX_ROWS,
) -> dict[str, Any]:
    """Build a deterministic Vega-Lite spec from result shape and sample rows."""

    try:
        fields = _normalize_result_schema(result_schema)
        if not fields:
            return {"ok": False, "tool": "result_to_chart_spec", "error_class": "not_chartable", "errors": ["result_schema is empty"]}
        visible_limit = _safe_limit(max_rows, default=DEFAULT_CHART_MAX_ROWS, hard_max=500)
        values = _rows_as_dicts(fields, list(rows), max_rows=visible_limit)
        columns = {field["name"]: [row.get(field["name"]) for row in values] for field in fields}
        roles = {field["name"]: _field_role(field, columns[field["name"]]) for field in fields}
        temporal = [field["name"] for field in fields if roles[field["name"]] == "temporal"]
        numeric = [field["name"] for field in fields if roles[field["name"]] == "quantitative"]
        nominal = [field["name"] for field in fields if roles[field["name"]] == "nominal"]
        warnings: list[str] = []
        if len(rows) > visible_limit:
            warnings.append(f"input rows truncated to {visible_limit} for chart spec values")
        if len(fields) > 6:
            warnings.append("wide result: only selected fields are encoded")

        if temporal and numeric:
            chart_type = "line"
            mark: str | dict[str, Any] = {"type": "line", "point": True}
            x, y = temporal[0], numeric[0]
        elif len(numeric) >= 2 and not temporal:
            chart_type = "scatter"
            mark = "point"
            x, y = numeric[0], numeric[1]
        elif nominal and numeric:
            chart_type = "bar"
            mark = "bar"
            x, y = nominal[0], numeric[0]
        elif nominal:
            chart_type = "bar"
            mark = "bar"
            x, y = nominal[0], "__count"
            warnings.append("no quantitative measure found; chart uses row count aggregation")
        else:
            return {
                "ok": False,
                "tool": "result_to_chart_spec",
                "error_class": "not_chartable",
                "errors": ["could not infer a useful chart from result shape"],
                "fields": fields,
                "roles": roles,
            }

        encoding: dict[str, Any] = {
            "x": {"field": x, "type": roles.get(x, "nominal")},
            "y": {"field": y, "type": roles.get(y, "quantitative")},
        }
        if y == "__count":
            encoding["y"] = {"aggregate": "count", "type": "quantitative", "title": "Records"}
        if chart_type == "bar" and y != "__count":
            encoding["x"]["sort"] = "-y"
        color_field = next((field for field in nominal if field != x), None)
        if color_field and chart_type in {"line", "scatter"}:
            encoding["color"] = {"field": color_field, "type": "nominal"}

        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": mark,
            "data": {"values": values},
            "encoding": encoding,
        }
        if intent:
            spec["description"] = intent
        return {
            "ok": True,
            "tool": "result_to_chart_spec",
            "chartable": True,
            "chart_type": chart_type,
            "fields": fields,
            "roles": roles,
            "warnings": warnings,
            "spec": spec,
            "spec_hash": _stable_hash(spec),
        }
    except Exception as exc:
        return {"ok": False, "tool": "result_to_chart_spec", "error_class": type(exc).__name__, "errors": [str(exc)[:300]]}


def run_tool(db_path: str | Path, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch a named analytic tool with JSON-like arguments."""

    payload = args or {}
    if tool == "schema_catalog":
        return schema_catalog(db_path, include_columns=bool(payload.get("include_columns", True)))
    if tool == "table_profile":
        return table_profile(db_path, str(payload["table"]))
    if tool == "sample_rows":
        return sample_rows(
            db_path,
            str(payload["table"]),
            limit=int(payload.get("limit", DEFAULT_SAMPLE_LIMIT)),
            columns=payload.get("columns"),
        )
    if tool == "find_join_paths":
        return find_join_paths(
            db_path,
            start_table=payload.get("start_table"),
            end_table=payload.get("end_table"),
            max_hops=int(payload.get("max_hops", DEFAULT_MAX_JOIN_HOPS)),
            max_paths=int(payload.get("max_paths", DEFAULT_MAX_JOIN_PATHS)),
        )
    if tool == "aggregate_window_template":
        template_payload = dict(payload.get("params") or {}) if isinstance(payload.get("params"), dict) else {}
        template_payload.update({key: value for key, value in payload.items() if key != "params"})
        return aggregate_window_template(
            db_path,
            pattern=str(template_payload["pattern"]),
            table=str(template_payload["table"]),
            metric_column=template_payload.get("metric_column"),
            metric_function=str(template_payload.get("metric_function", "sum")),
            grain_columns=template_payload.get("grain_columns"),
            filters=template_payload.get("filters"),
            joins=template_payload.get("joins"),
            date_column=template_payload.get("date_column"),
            order_by=template_payload.get("order_by"),
            partition_by=template_payload.get("partition_by"),
            n=int(template_payload.get("n", 10)),
            period=str(template_payload.get("period", "month")),
            rank_function=str(template_payload.get("rank_function", "rank")),
            metric_alias=str(template_payload.get("metric_alias", "metric_value")),
        )
    if tool == "explain_plan":
        return explain_plan(db_path, sql=str(payload["sql"]))
    if tool == "result_to_chart_spec":
        return result_to_chart_spec(
            result_schema=payload.get("result_schema") or payload.get("schema") or [],
            rows=payload.get("rows") or [],
            intent=payload.get("intent"),
            max_rows=int(payload.get("max_rows", DEFAULT_CHART_MAX_ROWS)),
        )
    return {"ok": False, "tool": tool, "error_class": "unknown_tool", "errors": [f"unknown tool: {tool}"]}


def run_tool_trace(db_path: str | Path, calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Execute a compact list of tool calls and return call/observation pairs."""

    trace: list[dict[str, Any]] = []
    for call in calls:
        tool = str(call.get("tool") or "")
        args = dict(call.get("args") or {})
        trace.append({"type": "tool_call", "tool": tool, "args": args})
        trace.append({"type": "tool_observation", "tool": tool, "observation": run_tool(db_path, tool, args)})
    return trace
