# cleo harness — review swarm findings (2026-06-09)

Five flavored review agents criticized the `cleo` package independently. This is the synthesis; the
"fixed" column reflects the follow-up pass. Severity is the reviewer's.

## CRITICAL — fixed

| id | finding | fix |
|---|---|---|
| FID-C1 | `INSTRUCTION` dropped a whole sentence vs training (`env.py:39`): *"When a gather reveals the real stored value differs… use the EXACT stored value you observed in your final SQL."* — the binding instruction, the model's whole point. Missing on every call → silent value-discovery regression. | Restored byte-identical; test asserts equality. |
| SEC-1 | Side-effecting functions pass the read-only guard: `pg_read_file`, `lo_export`, `pg_ls_dir`, `nextval/setval`, DuckDB `read_csv_auto`/`read_parquet` (SSRF/file-read/file-write), `dblink`, `pg_sleep`. | AST function **denylist** added; guard walks `exp.Anonymous`/`exp.Func`. |
| SEC-2 | `SELECT … INTO newtab` is a write and passes (not an `exp.Create` node). | Reject `exp.Into` in the AST check. |

## HIGH — fixed

| id | finding | fix |
|---|---|---|
| SEC-4 / XDR-C1/C2 | `is_readonly` failed **open** if sqlglot missing/raised; rollback is no guarantee under autocommit; `SET TRANSACTION READ ONLY` *raises* (not "ignored") on sqlite/MySQL/DuckDB. | Guard fails **closed**; sqlglot is a hard dep; `SET TRANSACTION READ ONLY` is Postgres-gated; autocommit is detected and refused; README safety claim rewritten (parser is the boundary + recommend a least-privilege DB role). |
| SEC-5 | PRAGMA identifier injection in the SQLite introspection fallback (`table_info("{name}")`). | Identifier quoting (double `"`), name validation. |
| FID-H3 | Over-budget `gather` was **hijacked as the final answer** (a `SELECT DISTINCT` probe returned as the user's answer). | Mirror env: reject with an observation + continue. |
| FID-H2 | Gather budget counted *all* observations (invalid/errored too), shifting the "finalize now" nudge vs training. | Track successful gathers separately (`n_gather`). |
| FID-H4 | `normalize_action` accepted malformed actions env rejects (e.g. `{"tool":"foo","sql":…}` → final). | Added the `"tool" not in obj` guard. |
| XDR-H1 | Hard-coded `LIMIT n+1` wrapper breaks on SQL Server/Oracle. | Documented supported dialects (Postgres/MySQL/SQLite/DuckDB); honest boundary. |
| XDR-H2/H4 | Introspection `fetchall` is unbounded (OOM on a warehouse); 60-table cap was silent. | Scoping pushed into SQL; over-cap without scoping now raises a helpful error. |
| XDR-H3 | `db_schema` accepted, advertised, **never used**; table names not schema-qualified (collisions). | Implemented schema scoping + schema-qualified keys. |
| DX-H1 | `Answer(error=…)` was **falsy**, so `if ans:` swallowed errors. | `__bool__` excludes errors; added `.ok` and `.status`. |
| DX-H2/H3 | Raise-vs-return contract undocumented; README API didn't match signature. | Documented in `ask()` docstring; README corrected. |

## MED / LOW — fixed or documented

- MIN-H1 / DX-M2: collapsed the `_execute`/`_executor` tuple-shuffle into one columns-first function.
- MIN-M3: `_quiet(fn)` helper replaces five `try/except: pass`.
- MIN-M4: dropped the leaky `max_new_tokens`? (kept, bumped default — long final SQL was truncating at 160).
- MIN-L11: `Answer.discovered` is now a derived property (no threading, no O(n²)).
- XDR-M1 / FID-L7: removed `REPLACE`/`MERGE`/`GRANT` from the keyword regex (false-positives on `SELECT REPLACE(...)`); destructive detection is AST-based.
- SEC-3 / XDR: best-effort `statement_timeout`; `fetchmany` instead of `fetchall`.
- FID-M5/M6: `n_bad>=2` early no-answer finalize; rejected-gather error text matches env; error truncation `[:120]`.
- DX-M1/M4: example gets `from __future__ import annotations` + a backend lock (llama.cpp/HF generate isn't thread-safe).
- DX-L1/L2/L5: `py.typed`, `LICENSE`, dependency upper bounds.
- DX-L3: import name `cleo` collides with Poetry's `cleo` — noted in README.
- DX-L4: added a `tests/` suite (the injectable `Cleo(backend=…)` path + sqlite/duckdb integration + readonly-guard + INSTRUCTION-parity).

## Explicitly NOT supported (documented honestly)
- **Oracle / SQL Server**: `LIMIT` wrapper + `information_schema`/PRAGMA introspection don't cover them.
- Read-only is enforced by the **SQL guard**; for production, run Cleo under a **least-privilege, read-only DB role** with a statement timeout. The guard is defense-in-depth, not a substitute for DB permissions.
