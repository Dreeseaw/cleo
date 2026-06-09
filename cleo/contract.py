"""The Cleo action contract — prompt format, action parsing, read-only guard, observation rendering.

These are byte-faithful to how Cleo v1.0 was trained/evaluated. Do not "improve" the strings or the
observation format: the model's accuracy depends on seeing exactly this at inference. Pure module
(no DB / no model deps) so it can be reused and tested in isolation.
"""
from __future__ import annotations

import json
import re

_ACTION_RE = re.compile(r"\{.*\}", re.S)
_SELECT_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.I)
_DESTRUCTIVE_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|VACUUM|ATTACH|DETACH|PRAGMA|COPY|GRANT|MERGE)\b",
    re.I,
)

INSTRUCTION = (
    'You are a SQL analyst agent with tools. Before answering you MAY inspect the data to discover real '
    'column values, codes, or domain conventions you are unsure about (e.g. how '
    '"current"/"active"/"completed" is actually encoded).\n\n'
    'Respond with EXACTLY one JSON object, nothing else:\n'
    '- {"tool":"gather","sql":"SELECT ..."}  run a read-only query to inspect values/schema (returns up '
    'to 20 rows). Use it when the answer depends on a literal/code you must verify.\n'
    '- {"tool":"final","sql":"SELECT ..."}   give the final read-only SELECT answer.\n'
    '- {"tool":"final","clarify":"..."}      if the request is ambiguous, underspecified, or out-of-schema.\n\n'
    'SQL must be a single read-only SELECT/WITH. Use only tables/columns in the schema.'
)


def parse_action(text: str) -> dict | None:
    """Pull the model's JSON action out of `text` (robust to trailing tokens)."""
    m = _ACTION_RE.search(text or "")
    if not m:
        return None
    blob = m.group(0)
    for end in range(len(blob), 0, -1):
        if blob[end - 1] != "}":
            continue
        try:
            obj = json.loads(blob[:end])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalize_action(obj: dict | None) -> dict | None:
    """-> {'kind':'gather'|'final','sql':..} | {'kind':'final','clarify':..} | None."""
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool")
    if tool == "gather" and isinstance(obj.get("sql"), str):
        return {"kind": "gather", "sql": obj["sql"]}
    if tool == "final":
        if isinstance(obj.get("sql"), str):
            return {"kind": "final", "sql": obj["sql"]}
        clar = obj.get("clarify") or obj.get("clarification")
        if isinstance(clar, str):
            return {"kind": "final", "clarify": clar}
        return None
    # bare single-shot contract (no "tool" key)
    if isinstance(obj.get("sql"), str):
        return {"kind": "final", "sql": obj["sql"]}
    clar = obj.get("clarify") or obj.get("clarification")
    if isinstance(clar, str):
        return {"kind": "final", "clarify": clar}
    return None


def is_readonly(sql: str, dialect: str | None = "duckdb") -> tuple[bool, str | None]:
    """True iff `sql` is a single read-only SELECT/WITH. AST-checked via sqlglot when available."""
    body = (sql or "").strip()
    if not body:
        return False, "empty"
    if ";" in body.rstrip(";"):
        return False, "multi_statement"
    if not _SELECT_RE.search(body) or _DESTRUCTIVE_RE.search(body):
        return False, "not_readonly"
    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        return True, None  # regex guard already passed; sqlglot just deepens it
    for read in (dialect, None):
        try:
            parsed = sqlglot.parse(body, read=read)
            break
        except Exception:
            parsed = None
    if not parsed or len(parsed) != 1:
        return False, "parse_error"
    for bad in (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command):
        if list(parsed[0].find_all(bad)):
            return False, "unsafe"
    return True, None


def render_prompt(schema: str, question: str, observations: list[tuple[str, str]],
                  gathers_left: int, steps_left: int) -> str:
    """Build the exact prompt the model was trained on. `observations` = [(gather_sql, obs_text), ...]."""
    parts = [f"Instruction:\n{INSTRUCTION}", f"Schema:\n{schema.strip()}", f"Question:\n{question.strip()}"]
    if observations:
        parts.append("Observations:\n" + "\n".join(
            f"[gather {i}] {gsql}\n-> {otext}" for i, (gsql, otext) in enumerate(observations, 1)))
    if gathers_left <= 0 or steps_left <= 1:
        parts.append('You must now return a {"tool":"final",...} action.')
    parts.append("Action:")
    return "\n\n".join(parts) + "\n"


def format_observation(columns: list, rows: list, truncated: bool) -> str:
    """Render a gather result exactly as the training env did (the model keys on this shape)."""
    body = json.dumps(rows, ensure_ascii=False)
    if len(body) > 360:
        body = body[:360] + " ...(truncated)"
    return f"cols={columns} rows({len(rows)}{'+' if truncated else ''})={body}"
