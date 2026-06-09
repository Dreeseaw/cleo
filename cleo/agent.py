"""Cleo — a tool-using SQL analyst you point at your own database connection.

    from cleo import Cleo
    cleo = Cleo.from_gguf("cleo_v1_0-no_mtp-Q8_0.gguf")
    ans = cleo.ask("How many employees are currently in each department?", conn)
    print(ans.sql, ans.rows)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import contract
from .db import introspect_schema, make_executor, run_readonly


@dataclass
class Answer:
    sql: str | None = None
    rows: list | None = None
    columns: list | None = None
    clarification: str | None = None
    gathers: list = field(default_factory=list)      # [(sql, columns, rows), ...]
    discovered: list = field(default_factory=list)   # distinct string values Cleo saw while probing
    error: str | None = None
    raw: str | None = None                            # last model output (for debugging)

    def __bool__(self) -> bool:
        return self.sql is not None or self.clarification is not None


class Cleo:
    def __init__(self, backend: Any, default_max_gather: int = 3):
        self.backend = backend
        self.default_max_gather = default_max_gather

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_gguf(cls, model_path: str, *, n_ctx: int = 4096, n_threads: int = 8,
                  n_gpu_layers: int = 0, **kw) -> "Cleo":
        from .backends import GGUFBackend
        return cls(GGUFBackend(model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers), **kw)

    @classmethod
    def from_hf(cls, model: str = "dreeseaw/cleo", *, device: str | None = None, **kw) -> "Cleo":
        from .backends import HFBackend
        return cls(HFBackend(model, device=device), **kw)

    # -- inference ----------------------------------------------------------
    def ask(self, question: str, conn: Any = None, *, schema: str | None = None,
            tables: list[str] | None = None, db_schema: str | None = None,
            max_gather: int | None = None, execute_final: bool = True,
            row_limit: int = 1000, dialect: str | None = "duckdb",
            max_new_tokens: int = 160) -> Answer:
        """Answer `question` against `conn` (any DB-API 2.0 connection or executor callable).

        Cleo probes the data read-only to discover real values/codes, then returns final SQL (and runs
        it, capped at `row_limit`, unless execute_final=False). Pass `schema=` to skip introspection,
        or `tables=[...]` to scope a large database.
        """
        if conn is None:
            raise ValueError("ask() needs a connection or executor as the second argument")
        max_gather = self.default_max_gather if max_gather is None else max_gather
        executor = make_executor(conn)
        if schema is None:
            schema = introspect_schema(conn, tables=tables, db_schema=db_schema)

        observations: list[tuple[str, str]] = []
        gather_log: list[tuple] = []   # (sql, columns, rows) exposed on Answer.gathers
        discovered: list[str] = []
        max_steps = max_gather + 1
        last_text = None
        for step in range(max_steps):
            prompt = contract.render_prompt(schema, question, observations,
                                            gathers_left=max_gather - len(observations),
                                            steps_left=max_steps - step)
            last_text = self.backend.generate(prompt, max_new_tokens=max_new_tokens)
            act = contract.normalize_action(contract.parse_action(last_text))
            if act is None:
                if step == max_steps - 1:
                    return Answer(error="unparsed_action", raw=last_text)
                observations.append(("<invalid>", "ERROR: could not parse a valid JSON action."))
                continue
            if act["kind"] == "gather" and len(observations) < max_gather:
                cols, rows, trunc, err = run_readonly(executor, act["sql"], limit=20, dialect=dialect)
                obs_text = f"ERROR: {err}" if err else contract.format_observation(cols, rows, trunc)
                if not err:
                    gather_log.append((act["sql"], cols, rows))
                    for row in rows:
                        for c in row:
                            if isinstance(c, str) and 0 < len(c) <= 64 and c not in discovered:
                                discovered.append(c)
                else:
                    gather_log.append((act["sql"], None, None))
                observations.append((act["sql"], obs_text))
                continue
            # final
            if "clarify" in act:
                return Answer(clarification=act["clarify"], gathers=gather_log,
                              discovered=discovered, raw=last_text)
            return self._finalize(act["sql"], executor, gather_log, discovered, execute_final,
                                  row_limit, dialect, last_text)
        return Answer(error="no_final_action", raw=last_text)

    def _finalize(self, sql, executor, gather_log, discovered, execute_final, row_limit, dialect, raw):
        ans = Answer(sql=sql, gathers=gather_log, discovered=discovered, raw=raw)
        if execute_final:
            cols, rows, trunc, err = run_readonly(executor, sql, limit=row_limit, dialect=dialect)
            if err:
                ans.error = err
            else:
                ans.columns, ans.rows = cols, rows
        return ans
