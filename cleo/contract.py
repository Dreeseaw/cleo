"""The Cleo action contract — prompt format, action parsing, read-only guard, observation rendering.

These are byte-faithful to how Cleo v1.0 was trained/evaluated. Do not "improve" the prompt strings or
the observation format: the model's accuracy depends on seeing exactly this at inference. (A test asserts
INSTRUCTION matches the training harness.) Pure module — no DB, no model deps.
"""
from __future__ import annotations

import json
import re

# INSTRUCTION is byte-identical to the training/eval harness (tooluse/env.py). The "When a gather
# reveals..." sentence is load-bearing — it is what tells the model to bind the EXACT discovered value.
INSTRUCTION = """You are a SQL analyst agent with tools. Before answering you MAY inspect the data to discover real column values, codes, or domain conventions you are unsure about (e.g. how "current"/"active"/"completed" is actually encoded).

Respond with EXACTLY one JSON object, nothing else:
- {"tool":"gather","sql":"SELECT ..."}  run a read-only query to inspect values/schema (returns up to 20 rows). Use it when the answer depends on a literal/code you must verify.
- {"tool":"final","sql":"SELECT ..."}   give the final read-only SELECT answer.
- {"tool":"final","clarify":"..."}      if the request is ambiguous, underspecified, or out-of-schema.

When a gather reveals the real stored value differs from the wording in the question (e.g. a code, abbreviation, casing, or sentinel), use the EXACT stored value you observed in your final SQL.
SQL must be a single read-only SELECT/WITH. Use only tables/columns in the schema."""

_ACTION_RE = re.compile(r"\{.*\}", re.S)
_SELECT_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.I)
# Statement-level destructive keywords sqlglot may not model as nodes. REPLACE/MERGE/GRANT are NOT here
# (they false-positive on SELECT REPLACE(...) etc.); destructive *statements* are caught by the AST.
_DESTRUCTIVE_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|VACUUM|ATTACH|DETACH|PRAGMA|COPY|CALL|GRANT|REVOKE)\b",
    re.I,
)
# Functions whose effect is a write / file read / network / DoS even inside a SELECT — denied by name.
_SIDE_EFFECT_FUNCS = frozenset({
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "pg_read_server_files",
    "lo_import", "lo_export", "lo_get", "lo_put",
    "dblink", "dblink_exec", "dblink_connect",
    "pg_sleep", "sleep", "waitfor", "pg_terminate_backend", "pg_cancel_backend",
    "nextval", "setval",
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto", "read_text",
    "read_blob", "glob", "install", "load", "query", "sniff_csv",
})


def parse_action(text: str) -> dict | None:
    """Pull the model's JSON action out of `text` (robust to trailing tokens)."""
    s = text or ""
    i = s.find("{")
    if i < 0:
        return None
    try:  # JSONDecoder parses one object and ignores trailing tokens — no scan loop needed
        obj, _ = json.JSONDecoder().raw_decode(s, i)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def normalize_action(obj: dict | None) -> dict | None:
    """-> {'kind':'gather'|'final','sql':..} | {'kind':'final','clarify':..} | None.

    A `tool` key that isn't gather/final is invalid (returns None) — matches the training harness, which
    only falls back to the bare single-shot contract when there is NO `tool` key at all.
    """
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool", "final")  # no tool key => the bare single-shot contract, treated as final
    sql = obj.get("sql")
    if tool == "gather" and isinstance(sql, str):
        return {"kind": "gather", "sql": sql}
    if tool == "final":
        if isinstance(sql, str):
            return {"kind": "final", "sql": sql}
        clar = obj.get("clarify") or obj.get("clarification")
        if isinstance(clar, str):
            return {"kind": "final", "clarify": clar}
    return None  # unknown tool value (e.g. {"tool":"foo",...}) is invalid, like the training harness


def is_readonly(sql: str, dialect: str | None = "duckdb") -> tuple[bool, str | None]:
    """True iff `sql` is a single read-only SELECT/WITH with no write/side-effecting parts.

    Fails CLOSED: if sqlglot can't parse it, it is rejected. sqlglot is a hard dependency.
    """
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
        return False, "no_sql_parser"  # fail closed — never run unparsed SQL against a real DB
    parsed = None
    for read in (dialect, None):
        try:
            parsed = sqlglot.parse(body, read=read)
            break
        except Exception:
            parsed = None
    if not parsed or len(parsed) != 1 or parsed[0] is None:
        return False, "parse_error"
    tree = parsed[0]
    for bad in (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
                exp.Command, exp.Into, exp.Merge, exp.Set):
        if list(tree.find_all(bad)):
            return False, "unsafe"
    for fn in tree.find_all(exp.Anonymous, exp.Func):
        name = (fn.name or "").lower()
        if not name:
            try:
                name = fn.sql_name().lower()
            except Exception:
                name = ""
        if name in _SIDE_EFFECT_FUNCS:
            return False, f"unsafe_function:{name}"
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
