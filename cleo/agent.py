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

import json
import threading
from dataclasses import dataclass, field
from typing import Any

from . import contract
from .db import MODEL_DIALECT, detect_dialect, introspect_schema, make_executor, run_readonly
from .evidence import select_candidate


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
    terminal_contract_sentinel_enabled: bool = False
    terminal_contract_sentinel_fired: bool = False
    terminal_contract_sentinel_observation_appended: bool = False
    post_sentinel_generation_count: int = 0
    post_sentinel_gather_rejected: bool = False
    post_sentinel_sql_blocked: bool = False
    selector: str | None = None
    candidate_id: str | None = None
    evidence_override: bool = False
    evidence_override_reasons: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)

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
            max_new_tokens: int = 256, enable_gather_many: bool = False,
            verifier_repair_context: bool = False, typed_repair_controller: bool = False,
            terminal_contract_sentinel: bool = False,
            sample: bool = False, temperature: float = 0.0, top_p: float = 0.95,
            seed: int | None = None) -> Answer:
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
        sentinel_fired = False
        sentinel_observation_appended = False
        post_sentinel_generation_count = 0
        post_sentinel_gather_rejected = False
        post_sentinel_sql_blocked = False

        def _answer(**kwargs) -> Answer:
            return Answer(
                **kwargs,
                terminal_contract_sentinel_enabled=terminal_contract_sentinel,
                terminal_contract_sentinel_fired=sentinel_fired,
                terminal_contract_sentinel_observation_appended=sentinel_observation_appended,
                post_sentinel_generation_count=post_sentinel_generation_count,
                post_sentinel_gather_rejected=post_sentinel_gather_rejected,
                post_sentinel_sql_blocked=post_sentinel_sql_blocked,
            )

        def _fire_sentinel() -> bool:
            nonlocal sentinel_fired, sentinel_observation_appended
            if not terminal_contract_sentinel or sentinel_fired:
                return False
            sentinel_fired = True
            sentinel_observation_appended = True
            observations.append((contract.TERMINAL_CONTRACT_LABEL, contract.TERMINAL_CONTRACT_OBSERVATION))
            return True

        def _no_answer(**kwargs) -> Answer | None:
            if _fire_sentinel():
                return None
            return _answer(error="no_answer", gathers=gather_log, **kwargs)

        step = 0
        while step < max_steps or (
            terminal_contract_sentinel and sentinel_fired and post_sentinel_generation_count == 0
        ):
            prompt = contract.render_prompt(schema, question, observations,
                                            gathers_left=max_gather - n_gather,
                                            steps_left=max_steps - step,
                                            enable_gather_many=enable_gather_many)
            step += 1
            gen_seed = None if seed is None else seed + step
            last = self._generate(
                prompt,
                max_new_tokens,
                sample=sample,
                temperature=temperature,
                top_p=top_p,
                seed=gen_seed,
            )
            act = contract.normalize_action(contract.parse_action(last))
            post_sentinel = sentinel_fired and post_sentinel_generation_count == 0
            if post_sentinel:
                post_sentinel_generation_count += 1

            if act is None:
                if post_sentinel:
                    return _answer(error="no_answer", gathers=gather_log, raw=last)
                n_bad += 1
                observations.append(("<invalid>", "ERROR: could not parse a valid JSON action."))
                if n_bad >= 2 or len(observations) >= max_steps:
                    ans = _no_answer(raw=last)
                    if ans is not None:
                        return ans
                continue

            if act["kind"] == "gather":
                if post_sentinel:
                    post_sentinel_gather_rejected = True
                    return _answer(error="no_answer", gathers=gather_log, raw=last)
                ok, why = contract.is_readonly(act["sql"], dialect=MODEL_DIALECT)
                if not ok or n_gather >= max_gather:
                    n_bad += 1
                    reason = why if not ok else "gather budget exhausted"
                    observations.append((act["sql"], f"ERROR: gather rejected ({reason})."))
                    gather_log.append((act["sql"], None, None))
                    if len(observations) >= max_steps:
                        ans = _no_answer(raw=last)
                        if ans is not None:
                            return ans
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
                    ans = _no_answer(raw=last)
                    if ans is not None:
                        return ans
                continue

            if act["kind"] == "gather_many":
                if post_sentinel:
                    post_sentinel_gather_rejected = True
                    return _answer(error="no_answer", gathers=gather_log, raw=last)
                queries = act.get("queries") or []
                label = "gather_many " + json.dumps([q.get("sql", "") for q in queries], ensure_ascii=False)
                if not enable_gather_many:
                    n_bad += 1
                    observations.append((label, "ERROR: gather_many rejected (disabled)."))
                elif not (1 <= len(queries) <= 3):
                    n_bad += 1
                    observations.append((label, "ERROR: gather_many rejected (query_count)."))
                elif n_gather + len(queries) > max_gather:
                    n_bad += 1
                    observations.append((label, "ERROR: gather_many rejected (gather budget exhausted)."))
                else:
                    results = []
                    for i, q in enumerate(queries, 1):
                        sql = q["sql"]
                        ok, why = contract.is_readonly(sql, dialect=MODEL_DIALECT)
                        if not ok:
                            n_bad += 1
                            results.append({"index": i, "error": f"rejected ({why})"})
                            gather_log.append((sql, None, None))
                            continue
                        cols, rows, trunc, err = run_readonly(executor, sql, limit=20, dialect=dialect)
                        n_gather += 1
                        if err:
                            results.append({"index": i, "error": err})
                            gather_log.append((sql, None, None))
                        else:
                            results.append({"index": i, "cols": cols, "rows": rows, "truncated": trunc})
                            gather_log.append((sql, cols, rows))
                    observations.append((label, contract.format_many_observation(results)))
                if len(observations) >= max_steps:
                    ans = _no_answer(raw=last)
                    if ans is not None:
                        return ans
                continue

            # final
            if "clarify" in act:
                if post_sentinel:
                    return _answer(error="no_answer", gathers=gather_log, raw=last)
                return _answer(clarification=act["clarify"], gathers=gather_log, raw=last)
            if post_sentinel:
                ok, _why = contract.is_readonly(act["sql"], dialect=MODEL_DIALECT)
                if not ok:
                    post_sentinel_sql_blocked = True
                    return _answer(error="no_answer", gathers=gather_log, raw=last)
            ans = _answer(sql=act["sql"], gathers=gather_log, raw=last)
            if execute_final:
                cols, rows, trunc, err = run_readonly(executor, act["sql"], limit=row_limit, dialect=dialect)
                if err:
                    # Surface the DB error as an observation so the model can repair it.
                    if n_repair < max_repair:
                        if typed_repair_controller:
                            repaired = contract.typed_repair_candidate(schema, act["sql"], err, question)
                            if repaired:
                                rcols, rrows, _rtrunc, rerr = run_readonly(
                                    executor, repaired["sql"], limit=row_limit, dialect=dialect
                                )
                                if rerr is None:
                                    return _answer(
                                        sql=repaired["sql"], columns=rcols, rows=rrows,
                                        gathers=gather_log, raw=last,
                                    )
                        n_repair += 1
                        obs = (
                            contract.format_repair_observation(schema, act["sql"], err)
                            if verifier_repair_context
                            else f"ERROR: {err}"
                        )
                        observations.append((act["sql"], obs))
                        gather_log.append((act["sql"], None, None))
                        continue
                    ans.error = err
                else:
                    ans.columns, ans.rows = cols, rows
            return ans
        ans = _no_answer(raw=last)
        if ans is not None:
            return ans
        return _answer(error="no_answer", gathers=gather_log, raw=last)

    def ask_hardel(self, question: str, conn: Any = None, *, k: int = 8,
                   temperature: float = 0.7, top_p: float = 0.95, seed: int = 6151,
                   return_candidates: bool = True, **kwargs) -> Answer:
        """Run greedy + sampled candidates, then select with product-visible evidence.

        This is the hardel runtime path: it keeps the same model/harness contract as
        `ask()`, but uses execution traces, result clusters, observed literals, and
        live DB literal support to pick among candidates without labels.
        """
        if conn is None:
            raise ValueError("ask_hardel() needs a connection or executor as the second argument")
        if k < 0:
            raise ValueError("k must be >= 0")
        run_kwargs = dict(kwargs)
        run_kwargs.setdefault("verifier_repair_context", True)
        run_kwargs.setdefault("typed_repair_controller", True)
        run_kwargs.setdefault("terminal_contract_sentinel", True)
        if run_kwargs.get("schema") is None and hasattr(conn, "cursor"):
            run_kwargs["schema"] = introspect_schema(
                conn,
                tables=run_kwargs.get("tables"),
                db_schema=run_kwargs.get("db_schema"),
                fks=run_kwargs.get("schema_fks", False),
                samples=run_kwargs.get("schema_samples", 0),
            )
        dialect = run_kwargs.get("dialect")
        if dialect is None:
            dialect = detect_dialect(conn) if hasattr(conn, "cursor") else MODEL_DIALECT
            run_kwargs["dialect"] = dialect

        answers: list[Answer] = []
        for i in range(k + 1):
            if i == 0:
                ans = self.ask(
                    question,
                    conn,
                    sample=False,
                    temperature=0.0,
                    top_p=top_p,
                    seed=None,
                    **run_kwargs,
                )
            else:
                ans = self.ask(
                    question,
                    conn,
                    sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed + i * 1009,
                    **run_kwargs,
                )
            ans.candidate_id = "greedy" if i == 0 else f"sample_{i}"
            answers.append(ans)

        candidate_rows = [
            {
                "candidate_id": ans.candidate_id,
                "sample_index": i,
                "sql": ans.sql,
                "rows": ans.rows,
                "columns": ans.columns,
                "error": ans.error,
                "gathers": ans.gathers,
            }
            for i, ans in enumerate(answers)
        ]
        result = select_candidate(candidate_rows, conn=conn if hasattr(conn, "cursor") else None, dialect=dialect)
        selected = answers[result.selected_index]
        selected.selector = "evidence_runtime"
        selected.evidence_override = result.override
        selected.evidence_override_reasons = result.override_reasons
        selected.candidate_id = result.summaries[result.selected_index]["candidate_id"]
        selected.candidates = result.summaries if return_candidates else []
        return selected

    def _generate(self, prompt: str, max_new_tokens: int, *, sample: bool = False,
                  temperature: float = 0.0, top_p: float = 0.95,
                  seed: int | None = None) -> str:
        with self._lock:
            try:
                return self.backend.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    sample=sample,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
            except TypeError:
                return self.backend.generate(prompt, max_new_tokens=max_new_tokens)
