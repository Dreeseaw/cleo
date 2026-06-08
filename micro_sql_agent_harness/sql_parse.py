"""Minimal SQL extraction and validation for SQLite SELECT/WITH queries."""

from __future__ import annotations

import re
from typing import Optional


DESTRUCTIVE_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def extract_sql(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = strip_code_fence(text)
    match = re.search(r"\b(with|select)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    sql = text[match.start() :].strip()
    # Keep a single trailing semicolon; validation rejects internal multi-statements.
    return sql


def _without_string_literals(sql: str) -> str:
    out = []
    quote = None
    i = 0
    while i < len(sql):
        ch = sql[i]
        if quote:
            if ch == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(" ")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def has_multiple_statements(sql: str) -> bool:
    stripped = _without_string_literals(sql).strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1]
    return ";" in stripped


def validate_safe_sql(sql: str) -> None:
    candidate = sql.strip()
    if not candidate:
        raise ValueError("empty SQL")
    if has_multiple_statements(candidate):
        raise ValueError("multiple SQL statements are not allowed")
    first = re.match(r"^\s*(\w+)", candidate)
    if not first or first.group(1).lower() not in {"select", "with"}:
        raise ValueError("only SELECT/WITH queries are allowed")
    scrubbed = _without_string_literals(candidate)
    if DESTRUCTIVE_RE.search(scrubbed):
        raise ValueError("destructive or unsafe SQL keyword is not allowed")


def extract_and_validate_sql(text: Optional[str]) -> Optional[str]:
    sql = extract_sql(text)
    if sql is not None:
        validate_safe_sql(sql)
    return sql
