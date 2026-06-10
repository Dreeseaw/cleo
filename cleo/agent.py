"""Cleo — a tool-using SQL analyst you point at your own database connection.

    from cleo import Cleo
    cleo = Cleo.from_gguf("cleo_v1_0-no_mtp-Q8_0.gguf")
    ans = cleo.ask("How many employees are currently in each department?", conn)
    if ans.ok:
        print(ans.sql, ans.rows)
    elif ans.status == "clarify":
        print(ans.clarification)
    else:
        print("error:", ans.error)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from . import contract
from .db import MODEL_DIALECT, detect_dialect, introspect_schema, make_executor, run_readonly


@dataclass
class Answer:
    """Result of `Cleo.ask`. Exactly one of (sql, clarification, error) is the outcome.

    `bool(answer)` and `.ok` are True only on a successful SQL answer. A clarification is not an "answer":
    handle it via `status == "clarify"`. `.status` is one of "ok" | "clarify" | "error" | "empty".
    `rows`/`columns` are populated only for a successful sql answer with execute_final=True (and
    `rows == []` means the query executed and returned no rows).
    """
    sql: str | None = None
    rows: list | None = None
    columns: list | None = None
    clarification: str | None = None
    gathers: list = field(default_factory=list)   # [(gather_sql, columns|None, rows|None), ...]
    error: str | None = None
    raw: str | None = None                         # last raw model output (debugging)

    @property
    def ok(self) -> bool:
        return self.error is None and self.sql is not None

    @property
    def status(self) -> str:
        if self.error is not None:
            return "error"
        if self.clarification is not None:
            return "clarify"
        return "ok" if self.sql is not None else "empty"

    @property
    def discovered(self) -> list:
        """Distinct short string values Cleo saw while probing (derived from `gathers`)."""
        seen: dict = {}
        for _sql, _cols, rows in self.gathers:
            for row in rows or []:
                for c in row:
                    if isinstance(c, str) and 0 < len(c) <= 64:
                        seen[c] = None
        return list(seen)

    def __bool__(self) -> bool:
        return self.ok


class Cleo:
    def __init__(self, backend: Any, default_max_gather: int = 3):
        self.backend = backend
        self.default_max_gather = default_max_gather
        self._lock = threading.Lock()  # backends (llama.cpp / HF) are not thread-safe

    @classmethod
    def from_gguf(cls, model_path: str, *, n_ctx: int = 4096, n_threads: int = 8,
                  n_gpu_layers: int = 0, **kw) -> "Cleo":
        from .backends import GGUFBackend
        return cls(GGUFBackend(model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers), **kw)

    @classmethod
    def from_hf(cls, model: str = "dreeseaw/cleo", *, device: str | None = None, **kw) -> "Cleo":
        from .backends import HFBackend
        return cls(HFBackend(model, device=device), **kw)

    def ask(self, question: str, conn: Any = None, *, schema: str | None = None,
            tables: list[str] | None = None, db_schema: str | None = None,
            max_gather: int | None = None, max_repair: int = 2, execute_final: bool = True,
            row_limit: int = 1000, dialect: str | None = None,
            schema_fks: bool = False, schema_samples: int = 0,
            max_new_tokens: int = 256) -> Answer:
        """Answer `question` against `conn` (a DB-API 2.0 connection or an executor callable).

        Cleo probes the data read-only to discover real values, then returns final SQL (and runs it,
        capped at `row_limit`, unless execute_final=False). Pass `schema=` to skip introspection, or
        `tables=[...]` / `db_schema=` to scope a large database.

        Cleo writes DuckDB-flavored SQL; the harness transpiles it to the connection's dialect before
        executing. `dialect` is auto-detected from the connection (sqlite/postgres/mysql/duckdb); pass it
        explicitly to override, or when using a bare executor callable.

        Raises ValueError/TypeError for *setup* problems (no conn, autocommit conn, too many tables with
        no scoping). Model/DB *runtime* outcomes are returned on `Answer.error` — check `ans.ok`.
        """
        if conn is None:
            raise ValueError("ask() needs a connection or executor as the second argument")
        max_gather = self.default_max_gather if max_gather is None else max_gather
        executor = make_executor(conn)
        if dialect is None:
            dialect = detect_dialect(conn) if hasattr(conn, "cursor") else MODEL_DIALECT
        if schema is None:
            schema = introspect_schema(conn, tables=tables, db_schema=db_schema,
                                       fks=schema_fks, samples=schema_samples)

        observations: list[tuple[str, str]] = []   # (sql, obs_text) -> prompt
        gather_log: list[tuple] = []                # (sql, columns|None, rows|None) -> Answer.gathers
        n_gather = 0
        n_bad = 0
        n_repair = 0
        max_steps = max_gather + 1 + max_repair
        last = None
        for step in range(max_steps):
            prompt = contract.render_prompt(schema, question, observations,
                                            gathers_left=max_gather - n_gather,
                                            steps_left=max_steps - step)
            last = self._generate(prompt, max_new_tokens)
            act = contract.normalize_action(contract.parse_action(last))

            if act is None:
                n_bad += 1
                observations.append(("<invalid>", "ERROR: could not parse a valid JSON action."))
                if n_bad >= 2 or len(observations) >= max_steps:
                    return Answer(error="no_answer", gathers=gather_log, raw=last)
                continue

            if act["kind"] == "gather":
                ok, why = contract.is_readonly(act["sql"], dialect=MODEL_DIALECT)
                if not ok or n_gather >= max_gather:
                    n_bad += 1
                    reason = why if not ok else "gather budget exhausted"
                    observations.append((act["sql"], f"ERROR: gather rejected ({reason})."))
                    gather_log.append((act["sql"], None, None))
                    if len(observations) >= max_steps:
                        return Answer(error="no_answer", gathers=gather_log, raw=last)
                    continue
                cols, rows, trunc, err = run_readonly(executor, act["sql"], limit=20, dialect=dialect)
                n_gather += 1
                if err:
                    observations.append((act["sql"], f"ERROR: {err}"))
                    gather_log.append((act["sql"], None, None))
                else:
                    observations.append((act["sql"], contract.format_observation(cols, rows, trunc)))
                    gather_log.append((act["sql"], cols, rows))
                if len(observations) >= max_steps:
                    return Answer(error="no_answer", gathers=gather_log, raw=last)
                continue

            # final
            if "clarify" in act:
                return Answer(clarification=act["clarify"], gathers=gather_log, raw=last)
            ans = Answer(sql=act["sql"], gathers=gather_log, raw=last)
            if execute_final:
                cols, rows, trunc, err = run_readonly(executor, act["sql"], limit=row_limit, dialect=dialect)
                if err:
                    # self-repair: surface the DB error (as a familiar error observation) and let it retry
                    if n_repair < max_repair:
                        n_repair += 1
                        observations.append((act["sql"], f"ERROR: {err}"))
                        gather_log.append((act["sql"], None, None))
                        continue
                    ans.error = err
                else:
                    ans.columns, ans.rows = cols, rows
            return ans
        return Answer(error="no_answer", gathers=gather_log, raw=last)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        with self._lock:
            return self.backend.generate(prompt, max_new_tokens=max_new_tokens)
