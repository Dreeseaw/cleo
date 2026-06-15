"""Label-free runtime evidence selection for Cleo pass@N candidates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceResult:
    selected_index: int
    summaries: list[dict[str, Any]] = field(default_factory=list)
    override: bool = False
    override_reasons: list[str] = field(default_factory=list)


def select_candidate(candidates: list[dict[str, Any]], conn: Any = None,
                     dialect: str | None = None) -> EvidenceResult:
    """Choose a candidate using only product-visible evidence.

    The selector intentionally avoids gold SQL, expected rows, or benchmark labels.
    It scores execution success, result-cluster agreement, observed literal reuse,
    and DB-visible literal support.
    """
    if not candidates:
        return EvidenceResult(selected_index=0)
    summaries = [_summarize_candidate(i, cand, conn, dialect) for i, cand in enumerate(candidates)]
    _attach_cluster_sizes(summaries)
    for summary in summaries:
        summary["selector_score"] = _selector_score(summary)
    selected = max(range(len(summaries)), key=lambda i: (summaries[i]["selector_score"], -i))
    greedy = summaries[0]
    chosen = summaries[selected]
    override = selected != 0 and chosen["positive_evidence"] > greedy["positive_evidence"]
    reasons = list(chosen.get("evidence_reasons") or [])
    if override and chosen["result_cluster_size"] > greedy.get("result_cluster_size", 0):
        reasons.append(f"larger result cluster ({chosen['result_cluster_size']} vs {greedy.get('result_cluster_size', 0)})")
    return EvidenceResult(
        selected_index=selected,
        summaries=summaries,
        override=override,
        override_reasons=reasons[:6],
    )


def _summarize_candidate(index: int, cand: dict[str, Any], conn: Any, dialect: str | None) -> dict[str, Any]:
    sql = cand.get("sql")
    rows = cand.get("rows")
    error = cand.get("error")
    gathers = cand.get("gathers") or []
    literals = _literal_predicates(sql)
    discovered = _discovered_values(gathers)
    literal_support = []
    positive_evidence = 0
    reasons: list[str] = []
    for pred in literals:
        support = {"predicate": pred, "gather": False, "db": False}
        lit = str(pred["literal"])
        if lit in discovered:
            support["gather"] = True
            positive_evidence += 1
            reasons.append(f"literal {lit!r} was observed in a gather")
        if conn is not None and pred.get("table") and pred.get("column"):
            if _db_literal_supported(conn, pred["table"], pred["column"], lit):
                support["db"] = True
                positive_evidence += 1
                reasons.append(f"literal {lit!r} exists at {pred['table']}.{pred['column']}")
        literal_support.append(support)

    final_kind = "sql" if sql else ("error" if error else "empty")
    result_digest = _rows_digest(rows) if error is None and rows is not None else None
    return {
        "candidate_id": cand.get("candidate_id") or f"candidate_{index}",
        "sample_index": cand.get("sample_index", index),
        "greedy": index == 0,
        "final_kind": final_kind,
        "sql": sql,
        "sql_len_chars": len(sql or ""),
        "execution_error": error,
        "execution_ok": bool(sql and error is None),
        "row_count": len(rows) if rows is not None else None,
        "column_count": len(cand.get("columns") or []),
        "gather_count": len(gathers),
        "result_digest": result_digest,
        "result_cluster_size": 0,
        "literal_predicates": literals,
        "literal_support": literal_support,
        "positive_evidence": positive_evidence,
        "unsupported_literal_count": sum(1 for s in literal_support if not (s["gather"] or s["db"])),
        "evidence_reasons": reasons,
    }


def _selector_score(summary: dict[str, Any]) -> float:
    score = 0.0
    if summary["execution_ok"]:
        score += 60.0
    elif summary["sql"]:
        score += 8.0
    else:
        score -= 25.0
    if summary["execution_error"]:
        score -= 18.0
    score += min(int(summary.get("result_cluster_size") or 0), 5) * 5.0
    score += int(summary.get("positive_evidence") or 0) * 12.0
    score += min(int(summary.get("gather_count") or 0), 3) * 1.5
    score -= int(summary.get("unsupported_literal_count") or 0) * 2.0
    score -= min(int(summary.get("sql_len_chars") or 0), 2000) * 0.002
    return score


def _attach_cluster_sizes(summaries: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for summary in summaries:
        digest = summary.get("result_digest")
        if digest:
            counts[digest] = counts.get(digest, 0) + 1
    for summary in summaries:
        digest = summary.get("result_digest")
        summary["result_cluster_size"] = counts.get(digest, 0) if digest else 0


def _rows_digest(rows: Any) -> str:
    body = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _discovered_values(gathers: list) -> set[str]:
    values: set[str] = set()
    for _sql, _cols, rows in gathers:
        for row in rows or []:
            for cell in row:
                if isinstance(cell, str):
                    values.add(cell)
    return values


def _literal_predicates(sql: str | None) -> list[dict[str, str]]:
    if not sql:
        return []
    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return []
    aliases = {}
    try:
        from sqlglot import exp

        for table in tree.find_all(exp.Table):
            if table.name:
                aliases[table.alias_or_name or table.name] = table.name
    except Exception:
        aliases = {}

    out = []
    for eq in tree.find_all(exp.EQ):
        left, right = eq.left, eq.right
        col, lit = _column_literal(left, right)
        if col is None:
            col, lit = _column_literal(right, left)
        if col is None or lit is None:
            continue
        table = aliases.get(col.table or "", col.table or "")
        out.append({"table": table, "column": col.name, "literal": str(lit.this)})
    return out


def _column_literal(left: Any, right: Any):
    try:
        from sqlglot import exp
    except Exception:
        return None, None
    if isinstance(left, exp.Column) and isinstance(right, exp.Literal) and right.is_string:
        return left, right
    return None, None


def _db_literal_supported(conn: Any, table: str, column: str, literal: str) -> bool:
    if not hasattr(conn, "cursor"):
        return False
    cur = conn.cursor()
    try:
        query = (
            f'SELECT COUNT(*) FROM {_quote_ident(table)} '
            f'WHERE CAST({_quote_ident(column)} AS TEXT) = ?'
        )
        cur.execute(query, (literal,))
        row = cur.fetchone()
        return bool(row and int(row[0]) > 0)
    except Exception:
        return False
    finally:
        _quiet(lambda: conn.rollback())
        _quiet(cur.close)


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _quiet(fn) -> None:
    try:
        fn()
    except Exception:
        pass
