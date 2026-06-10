# cleo

A small (~2B) **tool-using SQL analyst** you point at your own database connection. Cleo probes the data
read-only to **discover real values, codes, and conventions** (e.g. that *"current"* means
`to_date = '9999-01-01'`, or status `'O'` not `'open'`) **before** writing its answer — the thing a
one-shot text-to-SQL model can't do because it has to guess the literal.

No server, no pre-staging. Hand it a live DB-API connection and ask:

```python
from cleo import Cleo
import psycopg2

cleo = Cleo.from_gguf("cleo_v1_0-no_mtp-Q8_0.gguf")     # CPU-friendly; or Cleo.from_hf("dreeseaw/cleo")
conn = psycopg2.connect("postgresql://...")            # your existing connection — Postgres, SQLite, DuckDB, ...

ans = cleo.ask("How many employees are currently in each department?", conn)
print(ans.sql)            # the final read-only SELECT
print(ans.rows)           # executed result (rows), or None if Cleo asked to clarify
print(ans.clarification)  # set when the question is ambiguous / out-of-schema
print(ans.discovered)     # real values Cleo found while probing
```

`conn` is **any DB-API 2.0 connection** (`psycopg2`, `sqlite3`, `duckdb`, a SQLAlchemy
`engine.raw_connection()`), or a callable `executor(sql, limit) -> (columns, rows, truncated)`.

### Safety
Every statement Cleo issues is **validated read-only** (single `SELECT`/`WITH`, AST-checked) and run in a
rolled-back transaction — it never writes. Point it at production safely.

### Big databases
Schema is introspected from the connection. Scope it so the prompt stays focused:

```python
cleo.ask("...", conn, tables=["employees", "departments"])   # only these tables
cleo.ask("...", conn, schema=my_ddl_string)                  # or hand it the DDL yourself
```

### As an MCP tool
`cleo.ask(...)` is a single call with no setup — drop it straight into an MCP server (see
`examples/mcp_tool.py`).

## Install
```bash
pip install "cleo-sql[gguf]"     # llama-cpp-python backend (CPU/Mac/CUDA)
pip install "cleo-sql[hf]"       # transformers backend (GPU)
# HF weights (private): Cleo.from_hf("dreeseaw/cleo") pulls the current champion automatically
# GGUF (still v1.0; v1.2 not yet quantized): hf download dreeseaw/cleo cleo_v1_0-no_mtp-Q8_0.gguf --local-dir .
```

## Model versions (HF `dreeseaw/cleo`)
- **main = v1.2-bird** (2026-06-10): BIRD-repair distillation champion — BIRD-minidev 30.65%
  (434, same-harness), VD 57.6%, exec-error rate 12.7%. Best with `ask(..., max_repair=2,
  schema_fks=True)` (needs this package version for the repair loop + quoted-DDL introspection).
- `revision="v0.9"`: original single-shot SFT model.

## API
- `Cleo.from_gguf(path, *, n_ctx=4096, n_threads=8, n_gpu_layers=0)`
- `Cleo.from_hf(model="dreeseaw/cleo", *, device=None)`
- `cleo.ask(question, conn, *, schema=None, tables=None, max_gather=3, max_repair=2, execute_final=True, row_limit=1000, dialect=None, schema_fks=False, schema_samples=0) -> Answer` — `dialect` auto-detects from the connection; model SQL is transpiled from DuckDB; failed finals self-repair from the DB error up to `max_repair` times.
- `Answer(sql, rows, columns, clarification, gathers, discovered, error)` — truthy when answered.

Trained by behavioral cloning on denotation-verified teacher trajectories. v1.0 beats the one-shot
baseline on value-discovery (13.6% → 51.5%) and on out-of-distribution databases (59.3% → 64.2%).
