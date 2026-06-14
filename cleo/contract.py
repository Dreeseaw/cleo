"""Cleo's action contract: prompts, parsing, SQL guard, and observations.

Keep these strings byte-faithful to training. Model accuracy depends on the exact prompt
and observation format. This module has no DB or model dependencies.
"""
from __future__ import annotations

import json
import re
from difflib import get_close_matches

# INSTRUCTION is byte-identical to the training/eval harness. The "When a gather reveals..."
# sentence teaches the model to bind the exact discovered value.
INSTRUCTION = """You are a SQL analyst agent with tools. Before answering you MAY inspect the data to discover real column values, codes, or domain conventions you are unsure about (e.g. how "current"/"active"/"completed" is actually encoded).

Respond with EXACTLY one JSON object, nothing else:
- {"tool":"gather","sql":"SELECT ..."}  run a read-only query to inspect values/schema (returns up to 20 rows). Use it when the answer depends on a literal/code you must verify.
- {"tool":"final","sql":"SELECT ..."}   give the final read-only SELECT answer.
- {"tool":"final","clarify":"..."}      if the request is ambiguous, underspecified, or out-of-schema.

When a gather reveals the real stored value differs from the wording in the question (e.g. a code, abbreviation, casing, or sentinel), use the EXACT stored value you observed in your final SQL.
SQL must be a single read-only SELECT/WITH. Use only tables/columns in the schema."""

GATHER_MANY_LINE = (
    '- {"tool":"gather_many","queries":[{"sql":"SELECT ..."},{"sql":"SELECT ..."}]}  '
    "run up to 3 independent read-only probes in one turn (returns up to 20 rows each)."
)
TERMINAL_CONTRACT_LABEL = "<terminal_contract_sentinel>"
TERMINAL_CONTRACT_OBSERVATION = (
    'FINAL_CONTRACT_REQUIRED: return exactly one {"tool":"final",...} action; '
    "no further gather calls are allowed."
)

_ACTION_RE = re.compile(r"\{.*\}", re.S)
_SELECT_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.I)
# Statement-level destructive keywords sqlglot may not model as nodes. REPLACE/MERGE/GRANT are NOT here
# (they false-positive on SELECT REPLACE(...) etc.); destructive *statements* are caught by the AST.
_DESTRUCTIVE_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|VACUUM|ATTACH|DETACH|PRAGMA|COPY|CALL|GRANT|REVOKE)\b",
    re.I,
)
# Functions with write, file, network, or sleep side effects are denied by name.
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
    try:  # JSONDecoder parses one object and ignores trailing tokens.
        obj, _ = json.JSONDecoder().raw_decode(s, i)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def normalize_action(obj: dict | None) -> dict | None:
    """-> {'kind':'gather'|'gather_many'|'final',...} | None.

    Unknown `tool` values are invalid. Missing `tool` means the bare single-shot final contract.
    """
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool", "final")  # no tool key => the bare single-shot contract, treated as final
    sql = obj.get("sql")
    if tool == "gather" and isinstance(sql, str):
        return {"kind": "gather", "sql": sql}
    if tool == "gather_many" and isinstance(obj.get("queries"), list):
        queries = []
        for q in obj["queries"]:
            if not isinstance(q, dict) or not isinstance(q.get("sql"), str):
                return None
            queries.append({"sql": q["sql"]})
        return {"kind": "gather_many", "queries": queries}
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
        return False, "no_sql_parser"  # fail closed
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


def instruction_text(enable_gather_many: bool = False) -> str:
    if not enable_gather_many:
        return INSTRUCTION
    return INSTRUCTION.replace(
        '- {"tool":"final","sql":"SELECT ..."}   give the final read-only SELECT answer.',
        GATHER_MANY_LINE + '\n- {"tool":"final","sql":"SELECT ..."}   give the final read-only SELECT answer.',
    )


def render_prompt(schema: str, question: str, observations: list[tuple[str, str]],
                  gathers_left: int, steps_left: int, enable_gather_many: bool = False) -> str:
    """Build the exact prompt the model was trained on. `observations` = [(gather_sql, obs_text), ...]."""
    parts = [f"Instruction:\n{instruction_text(enable_gather_many)}",
             f"Schema:\n{schema.strip()}", f"Question:\n{question.strip()}"]
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


def format_many_observation(results: list[dict]) -> str:
    """Compact typed observation for gather_many, preserving per-query errors."""
    compact = []
    for res in results:
        item = {"i": res.get("index")}
        if res.get("error") is not None:
            item["error"] = str(res["error"])[:120]
        else:
            item["cols"] = res.get("cols", [])
            rows = res.get("rows", [])
            body = json.dumps(rows, ensure_ascii=False)
            if len(body) > 240:
                rows = body[:240] + " ...(truncated)"
            item["rows"] = rows
            item["truncated"] = bool(res.get("truncated"))
        compact.append(item)
    return "results=" + json.dumps(compact, ensure_ascii=False)


def _parse_schema_catalog(schema: str) -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = {}
    for m in re.finditer(r"CREATE\s+TABLE\s+([^\s(]+)\s*\((.*?)\)\s*;", schema or "", re.I | re.S):
        table = m.group(1).strip().strip('"')
        cols = []
        for part in m.group(2).split(","):
            token = part.strip()
            if not token or token.upper().startswith(("FOREIGN KEY", "PRIMARY KEY", "UNIQUE", "CHECK")):
                continue
            col = token.split()[0].strip().strip('"')
            if col:
                cols.append(col)
        if table and cols:
            catalog[table] = cols
    return catalog


def _alias_map(sql: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql or "", read="duckdb")
    except Exception:
        return aliases
    for table in tree.find_all(exp.Table):
        name = table.name
        alias = table.alias_or_name
        if name:
            aliases[alias or name] = name
    return aliases


def _extract_missing_column(error_text: str) -> tuple[str | None, str | None]:
    err = error_text or ""
    m = re.search(r'Table "([^"]+)" does not have a column named "([^"]+)"', err, re.I)
    if m:
        return m.group(2), m.group(1)
    m = re.search(r"no such column:\s*([A-Za-z_][\w$]*)(?:\.([A-Za-z_][\w$]*))?", err, re.I)
    if m:
        return (m.group(2) or m.group(1)), (m.group(1) if m.group(2) else None)
    m = re.search(r'Referenced column "([^"]+)" not found', err, re.I)
    if m:
        return m.group(1), None
    return None, None


def _quote_ident(name: str) -> str:
    s = str(name)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_$]*$", s):
        return s
    return '"' + s.replace('"', '""') + '"'


def _qualified_ref_pattern(alias: str, column: str) -> re.Pattern[str]:
    alias_re = re.escape(alias)
    column_re = re.escape(column)
    return re.compile(
        rf'(?<![\w"])("{alias_re}"|{alias_re})\s*\.\s*("{column_re}"|{column_re})(?![\w"])',
        re.I,
    )


def _projection_region(sql: str) -> str:
    m = re.search(r"\bFROM\b", sql or "", re.I)
    return (sql or "")[: m.start()] if m else sql or ""


def _split_projection(sql: str) -> tuple[str, str]:
    m = re.search(r"\bFROM\b", sql or "", re.I)
    if not m:
        return sql or "", ""
    return (sql or "")[: m.start()], (sql or "")[m.start():]


def _column_case(catalog: dict[str, list[str]], table: str, column: str) -> str | None:
    wanted = column.lower()
    for candidate in catalog.get(table, []):
        if candidate.lower() == wanted:
            return candidate
    return None


def _age_from_birthday_context(question: str) -> bool:
    text = (question or "").lower()
    if "age" not in text or "birthday" not in text:
        return False
    return not re.search(
        r"\b(older|oldest|younger|youngest|under|over|between|at least|less than|greater than)\b|\bage\s*(=|>|<)",
        text,
    )


def _typed_age_from_birthday_candidate(
    schema: str,
    sql: str,
    missing_owner: str,
    question: str,
) -> dict[str, str] | None:
    catalog = _parse_schema_catalog(schema)
    aliases = _alias_map(sql)
    owner_table = aliases.get(missing_owner)
    if not owner_table:
        return None
    birthday = _column_case(catalog, owner_table, "Birthday")
    if birthday is None or not _age_from_birthday_context(question):
        return None

    pattern = _qualified_ref_pattern(missing_owner, "age")
    projection, rest = _split_projection(sql)
    if not pattern.search(projection):
        return None

    expr = (
        "CAST(STRFTIME(CURRENT_DATE, '%Y') AS INTEGER) - "
        f"CAST(STRFTIME({_quote_ident(missing_owner)}.{_quote_ident(birthday)}, '%Y') AS INTEGER)"
    )
    repaired = pattern.sub(f"({expr})", projection) + rest
    drop_pat = re.compile(
        rf"\s+AND\s+(?:\"{re.escape(missing_owner)}\"|{re.escape(missing_owner)})\s*\.\s*(?:\"age\"|age)\s*="
        rf"\s*\(SELECT\b.*Birthday.*\)\s*;?\s*$",
        re.I | re.S,
    )
    repaired = drop_pat.sub("", repaired)
    if repaired == sql or pattern.search(repaired):
        return None
    return {
        "sql": repaired,
        "provenance": "typed_repair_controller:typed_age_from_birthday",
        "missing_column": "age",
        "from_alias": missing_owner,
        "birthday_column": birthday,
    }


def typed_repair_candidate(schema: str, sql: str, error_text: str, question: str = "") -> dict[str, str] | None:
    """Return one conservative deterministic SQL repair, or None."""
    missing_col, missing_owner = _extract_missing_column(error_text)
    if not missing_col or not missing_owner:
        return None
    if missing_col.lower() == "age":
        repaired_age = _typed_age_from_birthday_candidate(schema, sql, missing_owner, question)
        if repaired_age is not None:
            return repaired_age
    catalog = _parse_schema_catalog(schema)
    aliases = _alias_map(sql)
    owner_table = aliases.get(missing_owner)
    if not owner_table or missing_col in catalog.get(owner_table, []):
        return None

    pattern = _qualified_ref_pattern(missing_owner, missing_col)
    matches = list(pattern.finditer(sql or ""))
    if not matches:
        return None
    if pattern.search(_projection_region(sql)):
        return None

    candidate_aliases = [
        alias
        for alias, table in sorted(aliases.items())
        if alias != missing_owner and missing_col in catalog.get(table, [])
    ]
    if len(candidate_aliases) != 1:
        return None

    target_alias = candidate_aliases[0]
    repaired = pattern.sub(f"{_quote_ident(target_alias)}.{_quote_ident(missing_col)}", sql)
    if repaired == sql:
        return None
    return {
        "sql": repaired,
        "provenance": "typed_repair_controller:missing_column_owner_where_only",
        "missing_column": missing_col,
        "from_alias": missing_owner,
        "to_alias": target_alias,
    }


def format_repair_observation(schema: str, sql: str, error_text: str) -> str:
    """Structured, product-visible context for a failed final SQL repair turn."""
    catalog = _parse_schema_catalog(schema)
    aliases = _alias_map(sql)
    missing_col, missing_owner = _extract_missing_column(error_text)
    owner_table = aliases.get(missing_owner or "") if missing_owner else None
    exact = []
    nearest = []
    if missing_col:
        all_cols = sorted({c for cols in catalog.values() for c in cols})
        for table, cols in sorted(catalog.items()):
            if missing_col in cols:
                exact.append({"table": table, "column": missing_col})
        for col in get_close_matches(missing_col, all_cols, n=5, cutoff=0.55):
            nearest.append({"column": col, "tables": [t for t, cols in sorted(catalog.items()) if col in cols][:4]})
    relevant_tables = []
    for alias, table in aliases.items():
        if table in catalog:
            relevant_tables.append({"alias": alias, "table": table, "columns": catalog[table][:24]})
    quote_hints = sorted(set(re.findall(r'(?<!["\w])([A-Za-z_][\w]*-[A-Za-z0-9_-]+)(?!["\w])', sql or "")))
    context = {
        "error": str(error_text)[:220],
        "missing_column": missing_col,
        "missing_owner": missing_owner,
        "owner_table": owner_table,
        "aliases": aliases,
        "relevant_tables": relevant_tables[:8],
        "exact_column_matches": exact[:8],
        "nearest_columns": nearest[:5],
        "quote_identifiers_if_needed": quote_hints[:5],
    }
    return (
        f"ERROR: {str(error_text)[:220]}\n"
        f"REPAIR_CONTEXT: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
        'Return only a corrected {"tool":"final","sql":"..."} action.'
    )
