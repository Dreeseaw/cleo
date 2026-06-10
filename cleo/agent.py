"""A tool-using SQL analyst for your own database connection.

    from cleo import Cleo
    cleo = Cleo.from_gguf()   # downloads and caches the current champion
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
    """Result from `Cleo.ask`.

    `.ok` and `bool(answer)` are true only for a SQL answer. Clarifications use
    `status == "clarify"`. Errors use `status == "error"`.
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
        """Short string values Cleo saw while probing."""
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
    def from_gguf(cls, model_path: str | None = None, *, n_ctx: int = 4096, n_threads: int = 8,
                  n_gpu_layers: int = 0, **kw) -> "Cleo":
        """Create a GGUF-backed Cleo. No path means download or reuse the cached default."""
        from . import backends
        if model_path is None:
            model_path = backends.download_gguf()
        return cls(backends.GGUFBackend(model_path, n_ctx=n_ctx, n_threads=n_threads,
                                        n_gpu_layers=n_gpu_layers), **kw)

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

        Cleo may probe the data read-only, then returns final SQL and optionally runs it.
        Pass `schema=` to skip introspection. Use `tables=` or `db_schema=` to scope large DBs.

        Model SQL starts as DuckDB and is transpiled to the target dialect before execution.
        Pass `dialect=` to override auto-detection or when using a bare executor callable.

        Setup problems raise. Model or DB runtime failures are returned on `Answer.error`.
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
                    # Surface the DB error as an observation so the model can repair it.
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
