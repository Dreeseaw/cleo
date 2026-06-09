"""Tool-use rollout environment for the Cleo SQL analyst.

The model drives a multi-step loop against a REAL DuckDB:
  - {"tool":"gather","sql":"SELECT DISTINCT ..."}  -> read-only probe, returns <=20 rows (discovery)
  - {"tool":"final","sql":"SELECT ..."}            -> terminal answer (scored vs gold denotation)
  - {"tool":"final","clarify":"..."}               -> terminal clarification

Bootstrap: a BARE {"sql":...} / {"clarification":...} (the single-shot v0.9 contract) is treated as a
final action, so a warm-started v0.9 is immediately functional and RL only has to *discover* that
gathering first raises reward on value-discovery questions.

Reward (terminal): denotation-correct +R, wrong -, invalid/unsafe -, no-final -, minus a small per-gather
cost so gathering is reinforced only when it actually helps get denotation right.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

ACTION_RE = re.compile(r"\{.*\}", re.S)
SELECT_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.I)
DESTRUCTIVE_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|VACUUM|ATTACH|DETACH|PRAGMA|COPY)\b", re.I)

INSTRUCTION = """You are a SQL analyst agent with tools. Before answering you MAY inspect the data to discover real column values, codes, or domain conventions you are unsure about (e.g. how "current"/"active"/"completed" is actually encoded).

Respond with EXACTLY one JSON object, nothing else:
- {"tool":"gather","sql":"SELECT ..."}  run a read-only query to inspect values/schema (returns up to 20 rows). Use it when the answer depends on a literal/code you must verify.
- {"tool":"final","sql":"SELECT ..."}   give the final read-only SELECT answer.
- {"tool":"final","clarify":"..."}      if the request is ambiguous, underspecified, or out-of-schema.

When a gather reveals the real stored value differs from the wording in the question (e.g. a code, abbreviation, casing, or sentinel), use the EXACT stored value you observed in your final SQL.
SQL must be a single read-only SELECT/WITH. Use only tables/columns in the schema."""


def _parse_action(text: str) -> dict | None:
    m = ACTION_RE.search(text or "")
    if not m:
        return None
    # try progressively shorter prefixes ending in '}' for robustness
    blob = m.group(0)
    for end in range(len(blob), 0, -1):
        if blob[end - 1] != "}":
            continue
        try:
            obj = json.loads(blob[:end])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def normalize_action(obj: dict | None) -> dict | None:
    """-> {'kind':'gather'|'final', 'sql':..} or {'kind':'final','clarify':..} or None (invalid)."""
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool")
    if tool == "gather" and isinstance(obj.get("sql"), str):
        return {"kind": "gather", "sql": obj["sql"]}
    if tool == "final":
        if isinstance(obj.get("sql"), str):
            return {"kind": "final", "sql": obj["sql"]}
        if isinstance(obj.get("clarify"), str) or isinstance(obj.get("clarification"), str):
            return {"kind": "final", "clarify": obj.get("clarify") or obj.get("clarification")}
        return None
    # bare single-shot v0.9 contract
    if "sql" in obj and isinstance(obj["sql"], str) and "tool" not in obj:
        return {"kind": "final", "sql": obj["sql"]}
    if ("clarification" in obj or "clarify" in obj) and "tool" not in obj:
        return {"kind": "final", "clarify": obj.get("clarification") or obj.get("clarify")}
    return None


def normalize_cell(v: Any) -> Any:
    if v is None or isinstance(v, (int, str, bool)):
        return v
    if isinstance(v, float):
        return round(v, 8)
    return str(v)


def _canon(rows):
    return sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, separators=(",", ":")))


def rows_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    return _canon(a) == _canon(b)


def validate_readonly(sql: str):
    body = (sql or "").strip()
    if not body:
        return False, "empty"
    if ";" in body.rstrip(";"):
        return False, "multi_statement"
    if not SELECT_RE.search(body) or DESTRUCTIVE_RE.search(body):
        return False, "not_readonly"
    try:
        parsed = sqlglot.parse(body, read="duckdb")
    except Exception as exc:
        return False, f"parse:{str(exc)[:60]}"
    if len(parsed) != 1:
        return False, "multi_statement"
    for bad in (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command):
        if list(parsed[0].find_all(bad)):
            return False, "unsafe"
    return True, None


def execute(db_path: str, sql: str, limit: int = 200):
    try:
        con = duckdb.connect(str(db_path), config={"access_mode": "READ_ONLY", "enable_external_access": "false"})
        try:
            cur = con.execute(f"SELECT * FROM ({sql.strip().rstrip(';')}) AS r LIMIT {limit+1}")
            rows = [[normalize_cell(c) for c in row] for row in cur.fetchall()]
            cols = [d[0] for d in cur.description] if cur.description else []
            return rows[:limit], cols, (len(rows) > limit), None
        finally:
            con.close()
    except Exception as exc:
        return None, [], False, str(exc)[:120]


@dataclass
class Task:
    id: str
    db_path: str
    schema_ddl: str
    question: str
    gold_sql: str = ""
    gold_type: str = "sql"   # "sql" | "clarification"
    gold_rows: list | None = None


@dataclass
class Rollout:
    task: Task
    obs: list = field(default_factory=list)   # list of (gather_sql, obs_text)
    observed_values: set = field(default_factory=set)  # string values seen in gather rows
    n_gather: int = 0
    n_bad: int = 0
    done: bool = False
    reward: float = 0.0
    final_kind: str = ""      # "sql" | "clarify" | "none"
    final_sql: str = ""
    final_clarify: str = ""
    denotation: bool = False
    steps_text: list = field(default_factory=list)  # list of (prompt, action_text) for credit assignment


class ToolUseEnv:
    def __init__(self, max_steps: int = 4, max_gather: int = 3, gather_cost: float = 0.0,
                 r_correct: float = 2.0, obs_row_limit: int = 20):
        self.max_steps = max_steps
        self.max_gather = max_gather
        self.gather_cost = gather_cost
        self.r_correct = r_correct
        self.obs_row_limit = obs_row_limit

    def render_prompt(self, r: Rollout) -> str:
        parts = [f"Instruction:\n{INSTRUCTION}", f"Schema:\n{r.task.schema_ddl.strip()}",
                 f"Question:\n{r.task.question.strip()}"]
        if r.obs:
            ob = []
            for i, (gsql, otext) in enumerate(r.obs, 1):
                ob.append(f"[gather {i}] {gsql}\n-> {otext}")
            parts.append("Observations:\n" + "\n".join(ob))
        budget_left = self.max_gather - r.n_gather
        steps_left = self.max_steps - len(r.obs)
        if budget_left <= 0 or steps_left <= 1:
            parts.append("You must now return a {\"tool\":\"final\",...} action.")
        parts.append("Action:")
        return "\n\n".join(parts) + "\n"

    def _gold_rows(self, task: Task):
        if task.gold_rows is not None:
            return task.gold_rows
        rows, _, _, err = execute(task.db_path, task.gold_sql)
        return rows if err is None else None

    def step(self, r: Rollout, action_text: str) -> None:
        """Apply one model action to rollout r (mutates)."""
        prompt = self.render_prompt(r)
        r.steps_text.append((prompt, action_text))
        act = normalize_action(_parse_action(action_text))
        if act is None:
            r.n_bad += 1
            r.obs.append(("<invalid>", "ERROR: could not parse a valid JSON action."))
            if r.n_bad >= 2 or (len(r.obs) >= self.max_steps):
                self._finalize_noanswer(r)
            return
        if act["kind"] == "gather":
            ok, why = validate_readonly(act["sql"])
            if not ok or r.n_gather >= self.max_gather:
                r.n_bad += 1
                msg = f"ERROR: gather rejected ({why or 'gather budget exhausted'})."
                r.obs.append((act["sql"], msg))
            else:
                rows, cols, trunc, err = execute(r.task.db_path, act["sql"], self.obs_row_limit)
                r.n_gather += 1
                if err is not None:
                    r.obs.append((act["sql"], f"ERROR: {err}"))
                else:
                    for row in rows:
                        for c in row:
                            if isinstance(c, str) and 0 < len(c) <= 64:
                                r.observed_values.add(c)
                    body = json.dumps(rows, ensure_ascii=False)
                    if len(body) > 360:
                        body = body[:360] + " ...(truncated)"
                    r.obs.append((act["sql"], f"cols={cols} rows({len(rows)}{'+'if trunc else ''})={body}"))
            if len(r.obs) >= self.max_steps:
                self._finalize_noanswer(r)
            return
        # final
        self._finalize(r, act)

    def _finalize(self, r: Rollout, act: dict) -> None:
        r.done = True
        cost = self.gather_cost * r.n_gather
        if "clarify" in act:
            r.final_kind = "clarify"
            r.final_clarify = act.get("clarify", "")
            r.reward = (2.0 if r.task.gold_type == "clarification" else -0.5) - cost
            return
        r.final_kind = "sql"
        r.final_sql = act["sql"]
        if r.task.gold_type == "clarification":
            r.reward = -0.5 - cost
            return
        ok, why = validate_readonly(act["sql"])
        if not ok:
            r.reward = -1.0 - cost
            return
        bonus = self._literal_hit_bonus(r, act["sql"])  # reward binding a DISCOVERED literal
        pred_rows, _, trunc, err = execute(r.task.db_path, act["sql"])
        if err is not None:
            r.reward = -0.6 - cost + bonus
            return
        gold = self._gold_rows(r.task)
        if gold is not None and pred_rows is not None and not trunc and rows_equal(pred_rows, gold) and pred_rows != []:
            r.denotation = True
            r.reward = self.r_correct - cost
        elif pred_rows == [] and gold not in (None, []):
            r.reward = -0.5 - cost + bonus
        else:
            r.reward = -0.2 - cost + bonus

    def _literal_hit_bonus(self, r: Rollout, sql: str) -> float:
        if not r.observed_values:
            return 0.0
        q = r.task.question.lower()
        for lit in set(re.findall(r"'([^']{1,64})'", sql)):
            if lit in r.observed_values and lit.lower() not in q:
                return 0.4  # used a value discovered via gather, not stated in the question
        return 0.0

    def _finalize_noanswer(self, r: Rollout) -> None:
        r.done = True
        r.final_kind = "none"
        r.reward = -1.0 - self.gather_cost * r.n_gather
