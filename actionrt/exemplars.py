"""Leak-safe structural exemplar retrieval for primitive runtime."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).replace("_", " ").lower())


def _token_counter(text: str) -> Counter[str]:
    tokens = _tokens(text)
    counter: Counter[str] = Counter(tokens)
    for left, right in zip(tokens, tokens[1:]):
        counter[f"{left}_{right}"] += 1
    return counter


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _jaccard(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    return len(left & right) / len(union) if union else None


def _sql_feature_summary(sql: str | None) -> dict[str, Any]:
    if not sql or not str(sql).strip():
        return {"tables": [], "columns": [], "shape": "parse_error", "template": ""}
    try:
        import sqlglot
        from sqlglot import exp

        parsed = sqlglot.parse_one(str(sql), read="duckdb")
        tables = sorted({str(table.name).lower() for table in parsed.find_all(exp.Table) if table.name})
        columns = sorted({str(column.name).lower() for column in parsed.find_all(exp.Column) if column.name})
        aggregates = sorted({node.key.lower() for node in parsed.find_all(exp.AggFunc)})
        group = parsed.args.get("group") or exp.Group()
        shape = "|".join(
            [
                f"select={len(parsed.expressions or [])}",
                f"tables={len(tables)}",
                f"joins={len(list(parsed.find_all(exp.Join)))}",
                f"where={int(parsed.args.get('where') is not None)}",
                f"group={len(group.expressions or [])}",
                f"order={int(parsed.args.get('order') is not None)}",
                f"limit={int(parsed.args.get('limit') is not None)}",
                "agg=" + ",".join(aggregates),
            ]
        )
        template = parsed.sql(dialect="duckdb", normalize=True, pretty=False).lower()
    except Exception:
        lowered = re.sub(r"\s+", " ", str(sql).strip().lower())
        tables = sorted(set(re.findall(r"\bfrom\s+([a-zA-Z_][\w]*)|\bjoin\s+([a-zA-Z_][\w]*)", lowered)))
        flat_tables = sorted({item for pair in tables for item in pair if item})
        columns = sorted(set(re.findall(r"\bselect\s+(.*?)\s+from\b", lowered)))
        shape = "parse_error"
        template = lowered
        tables = flat_tables
    template = re.sub(r"'[^']*'", "?", template)
    template = re.sub(r"\b\d+(?:\.\d+)?\b", "?", template)
    template = re.sub(r"\s+", " ", template).strip()
    return {"tables": tables, "columns": columns, "shape": shape, "template": template}


def _question_shape_tokens(question: str) -> set[str]:
    tokens = set(_tokens(question))
    features: set[str] = set()
    if tokens & {"average", "avg", "mean", "sum", "total", "count", "number", "minimum", "maximum", "min", "max"}:
        features.add("aggregate")
    if tokens & {"average", "avg", "mean"}:
        features.add("avg")
    if tokens & {"sum", "total"}:
        features.add("sum")
    if tokens & {"count", "number", "many"}:
        features.add("count")
    if tokens & {"by", "per", "each", "group", "grouped"}:
        features.add("group")
    if tokens & {"top", "highest", "lowest", "largest", "smallest", "most", "least", "rank"}:
        features.update({"order", "limit"})
    if tokens & {"where", "with", "without", "over", "under", "between", "before", "after", "during", "only"}:
        features.add("filter")
    if tokens & {"date", "year", "month", "day", "daily", "monthly", "yearly", "recent"}:
        features.add("time")
    if tokens & {"distinct", "unique"}:
        features.add("distinct")
    return features


def _sql_shape_tokens(sql: str | None) -> set[str]:
    summary = _sql_feature_summary(sql)
    shape = str(summary.get("shape") or "")
    features: set[str] = set()
    for item in shape.split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key == "joins" and value != "0":
            features.add("join")
        elif key == "where" and value == "1":
            features.add("filter")
        elif key == "group" and value != "0":
            features.add("group")
        elif key == "order" and value == "1":
            features.add("order")
        elif key == "limit" and value == "1":
            features.add("limit")
        elif key == "agg" and value:
            features.add("aggregate")
            features.update(token for token in value.split(",") if token)
    template = str(summary.get("template") or "")
    if "distinct" in template:
        features.add("distinct")
    if any(token in template for token in ("date", "year", "month", "day")):
        features.add("time")
    return features


def _schema_terms_from_question(question: str, catalog: dict[str, list[str]] | None) -> set[str]:
    if not catalog:
        return set()
    question_tokens = set(_tokens(question))
    question_text = " ".join(question_tokens)
    matched: set[str] = set()
    for table, columns in catalog.items():
        table_tokens = set(_tokens(table))
        if table_tokens and (table_tokens <= question_tokens or table_tokens & question_tokens):
            matched.add(str(table).lower())
        for column in columns:
            column_tokens = set(_tokens(column))
            if column_tokens and (column_tokens <= question_tokens or " ".join(column_tokens) in question_text):
                matched.add(str(column).lower())
    return matched


def load_exemplar_bank(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    bank_path = Path(path)
    if not bank_path.exists():
        raise FileNotFoundError(f"exemplar bank not found: {bank_path}")
    if bank_path.suffix == ".jsonl":
        by_db: dict[str, list[dict[str, Any]]] = {}
        for line in bank_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            db_id = row.get("db_id")
            sql = row.get("sql") or row.get("gold_sql")
            if isinstance(db_id, str) and isinstance(sql, str) and sql.strip():
                by_db.setdefault(db_id, []).append(dict(row))
        return by_db
    obj = json.loads(bank_path.read_text(encoding="utf-8"))
    raw = obj.get("databases", obj) if isinstance(obj, dict) else {}
    if not isinstance(raw, dict):
        raise ValueError(f"exemplar bank must be a JSON object: {bank_path}")
    by_db = {}
    for db_id, rows in raw.items():
        if not isinstance(rows, list):
            continue
        clean = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            sql = row.get("sql") or row.get("gold_sql")
            if isinstance(sql, str) and sql.strip():
                item = dict(row)
                item.setdefault("db_id", str(db_id))
                clean.append(item)
        by_db[str(db_id)] = clean
    return by_db


def _leakage_stats(eval_sql: str | None, exemplar_sql: str | None) -> dict[str, Any]:
    eval_features = _sql_feature_summary(eval_sql)
    exemplar_features = _sql_feature_summary(exemplar_sql)
    eval_tables = set(eval_features["tables"])
    exemplar_tables = set(exemplar_features["tables"])
    eval_columns = set(eval_features["columns"])
    exemplar_columns = set(exemplar_features["columns"])
    template_similarity = SequenceMatcher(
        None, str(eval_features.get("template") or ""), str(exemplar_features.get("template") or "")
    ).ratio()
    string_similarity = SequenceMatcher(None, str(eval_sql or "").lower(), str(exemplar_sql or "").lower()).ratio()
    return {
        "table_jaccard": _jaccard(eval_tables, exemplar_tables) or 0.0,
        "column_jaccard": _jaccard(eval_columns, exemplar_columns) or 0.0,
        "eval_shape": eval_features.get("shape"),
        "exemplar_shape": exemplar_features.get("shape"),
        "shape_match": eval_features.get("shape") == exemplar_features.get("shape"),
        "template_similarity": template_similarity,
        "sql_string_similarity": string_similarity,
    }


def _is_high_risk(leakage: dict[str, Any], similarity_cap: float) -> bool:
    table_jaccard = float(leakage.get("table_jaccard") or 0.0)
    template_similarity = float(leakage.get("template_similarity") or 0.0)
    string_similarity = float(leakage.get("sql_string_similarity") or 0.0)
    return (
        bool(leakage.get("shape_match"))
        and table_jaccard >= 0.80
        and (template_similarity >= similarity_cap or string_similarity >= similarity_cap)
    )


def _structural_score(
    *,
    question: str,
    catalog: dict[str, list[str]] | None,
    exemplar_question: str,
    exemplar_sql: str,
    leakage: dict[str, Any],
) -> dict[str, float]:
    eval_schema_terms = _schema_terms_from_question(question, catalog)
    exemplar_features = _sql_feature_summary(exemplar_sql)
    exemplar_schema_terms = {
        *[str(item).lower() for item in exemplar_features.get("tables", [])],
        *[str(item).lower() for item in exemplar_features.get("columns", [])],
    }
    schema_score = _jaccard(eval_schema_terms, exemplar_schema_terms)
    if schema_score is None:
        schema_score = float(leakage.get("table_jaccard") or 0.0)
    eval_shape = _question_shape_tokens(question)
    exemplar_shape = _sql_shape_tokens(exemplar_sql)
    shape_score = _jaccard(eval_shape, exemplar_shape) or 0.0
    eval_embed = _token_counter(" ".join([question, " ".join(sorted(eval_schema_terms)), " ".join(sorted(eval_shape))]))
    exemplar_embed = _token_counter(
        " ".join(
            [
                exemplar_question,
                str(exemplar_features.get("template") or ""),
                str(exemplar_features.get("shape") or ""),
                " ".join(sorted(exemplar_schema_terms)),
            ]
        )
    )
    embedding_score = _cosine(eval_embed, exemplar_embed)
    combined = (0.45 * float(schema_score)) + (0.25 * float(shape_score)) + (0.30 * embedding_score)
    return {
        "combined": combined,
        "schema_overlap": float(schema_score),
        "shape_similarity": float(shape_score),
        "embedding_similarity": embedding_score,
    }


def retrieve_structural_exemplars(
    *,
    db_id: str,
    question: str,
    catalog: dict[str, list[str]] | None,
    exemplar_bank: dict[str, list[dict[str, Any]]],
    k: int,
    similarity_cap: float = 0.75,
    current_gold_sql: str | None = None,
) -> dict[str, Any]:
    scored: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for idx, row in enumerate(exemplar_bank.get(db_id, [])):
        sql = row.get("sql") or row.get("gold_sql")
        if not isinstance(sql, str) or not sql.strip():
            continue
        leakage = _leakage_stats(current_gold_sql, sql)
        if current_gold_sql and _is_high_risk(leakage, similarity_cap):
            continue
        retrieval = _structural_score(
            question=question,
            catalog=catalog,
            exemplar_question=str(row.get("question") or ""),
            exemplar_sql=sql,
            leakage=leakage,
        )
        selected = {
            "id": row.get("id") or f"{db_id}_deployment_exemplar_{idx:02d}",
            "question": row.get("question"),
            "gold_sql": sql,
            "source": row.get("source", "approved_sql"),
            "similarity_score": retrieval["combined"],
            "retrieval": retrieval,
            "leakage": leakage,
        }
        scored.append(
            (
                (
                    -retrieval["combined"],
                    -retrieval["schema_overlap"],
                    -retrieval["shape_similarity"],
                    -retrieval["embedding_similarity"],
                    float(leakage.get("template_similarity") or 0.0),
                    float(leakage.get("sql_string_similarity") or 0.0),
                ),
                selected,
            )
        )
    scored.sort(key=lambda item: item[0])
    return {
        "ok": True,
        "type": "exemplar_observation",
        "tool": "retrieve_exemplars",
        "db_id": db_id,
        "selection": "structural",
        "similarity_cap": similarity_cap,
        "exemplars": [item[1] for item in scored[: max(0, k)]],
        "candidate_count": len(scored),
    }
