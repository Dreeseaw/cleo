# cleo

A unified micro-harness + fine-tuned Qwen3.5-2B (a "hardel") for SQL-based analytical workflows 
that you point at your own database connections. Cleo's been trained on probing the data read-only to 
**discover real values, codes, and conventions** deep in the data, **repair broken queries** in-flight, 
and treat clarity & observability as first-class features.

No server, no pre-staging. Hand it a live DB-API connection and ask:

```python
from cleo import Cleo
import psycopg2

cleo = Cleo.from_gguf()   # downloads + caches model (CPU-friendly); or Cleo.from_hf(...)
conn = psycopg2.connect("postgresql://...") # your existing connection — Postgres, SQLite, DuckDB, ...

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

### CLI
```bash
cleo "total revenue by region" --db warehouse.duckdb
cleo "active users this week?" --db postgresql://me@host/db --tables users,sessions --json
```

### As an MCP tool
`cleo mcp --db "$DATABASE_URL"` serves Cleo as an MCP server (`pip install "cleo-sql[gguf,mcp]"`), so
any agent gets a natural-language `query_database` tool. Claude Code / Claude Desktop config:

```json
"cleo": { "command": "cleo", "args": ["mcp", "--db", "postgresql://..."] }
```

## Install
Not on PyPI yet — install from the repo:

```bash
git clone https://github.com/Dreeseaw/cleo && cd cleo
pip install -e ".[gguf]"         # llama-cpp-python backend (CPU/Mac/CUDA)
pip install -e ".[hf]"           # transformers backend (GPU)
# Weights download + cache themselves on first use: Cleo.from_gguf() pulls the current champion
# Q8_0 GGUF (bf16-parity) from HF (private — `hf auth login` first); same for Cleo.from_hf().
```

Run the tests (no model/GPU needed): `pip install -e ".[test]" && pytest`

## API
- `Cleo.from_gguf(path=None, *, n_ctx=4096, n_threads=8, n_gpu_layers=0)` — no path downloads/caches the current champion
- `Cleo.from_hf(model="dreeseaw/cleo", *, device=None)`
- `cleo.ask(question, conn, *, schema=None, tables=None, max_gather=3, max_repair=2, execute_final=True, row_limit=1000, dialect=None, schema_fks=False, schema_samples=0) -> Answer` — `dialect` auto-detects from the connection; model SQL is transpiled from DuckDB; failed finals self-repair from the DB error up to `max_repair` times.
- `Answer(sql, rows, columns, clarification, gathers, discovered, error)` — truthy when answered.
