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
conn = psycopg2.connect("postgresql://...") # your existing connection: Postgres, SQLite, DuckDB, ...

ans = cleo.ask("How many employees are currently in each department?", conn)
print(ans.sql)            # the final read-only SELECT
print(ans.rows)           # executed result (rows), or None if Cleo asked to clarify
print(ans.clarification)  # set when the question is ambiguous / out-of-schema
print(ans.discovered)     # real values Cleo found while probing
```

For harder value-binding questions, use the hardel runtime. It runs greedy plus
sampled candidates through the same harness, executes them read-only, then picks
with label-free evidence such as result clusters, observed literal reuse, and live
DB literal support:

```python
ans = cleo.ask_hardel(
    "How many employees are currently in each department?",
    conn,
    k=8,              # sampled candidates after the greedy candidate
    temperature=0.7,
)
print(ans.sql, ans.rows)
print(ans.evidence_override, ans.evidence_override_reasons)
```

`conn` is **any DB-API 2.0 connection** (`psycopg2`, `sqlite3`, `duckdb`, a SQLAlchemy
`engine.raw_connection()`), or a callable `executor(sql, limit) -> (columns, rows, truncated)`.

### Safety
Every statement Cleo issues is **validated read-only** (single `SELECT`/`WITH`, AST-checked) and run in a
rolled-back transaction. It never writes.

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
cleo "current customers by segment" --db warehouse.duckdb --hardel --k 8 --json
```

### As an MCP tool
`cleo mcp --db "$DATABASE_URL"` serves a natural-language `query_database` tool
(`pip install "cleo-sql[gguf,mcp]"`). Claude Code / Claude Desktop config:

```json
"cleo": { "command": "cleo", "args": ["mcp", "--db", "postgresql://..."] }
```

## Install

```bash
pip install "cleo-sql[gguf]"       # llama-cpp-python backend (CPU/Mac/CUDA)
pip install "cleo-sql[hf]"         # transformers backend (GPU)
pip install "cleo-sql[gguf,mcp]"   # MCP server extras
# Weights download + cache themselves on first use: Cleo.from_gguf() pulls the current champion
# Q8_0 GGUF (bf16 parity) from HF.
```

Run the tests (no model/GPU needed): `pip install -e ".[test]" && pytest`

## API
- `Cleo.from_gguf(path=None, *, n_ctx=4096, n_threads=8, n_gpu_layers=0)`: no path downloads/caches the current champion
- `Cleo.from_hf(model="dreeseaw/cleo", *, device=None)`
- `cleo.ask(question, conn, *, schema=None, tables=None, max_gather=3, max_repair=2, execute_final=True, row_limit=1000, dialect=None, schema_fks=False, schema_samples=0) -> Answer`: auto-detects dialect, transpiles from DuckDB SQL, and retries failed finals up to `max_repair`.
- `cleo.ask_hardel(question, conn, *, k=8, temperature=0.7, top_p=0.95, seed=6151, **ask_kwargs) -> Answer`: runs pass@N through the live harness and selects with product-visible evidence.
- `Answer(sql, rows, columns, clarification, gathers, discovered, error, selector, evidence_override, evidence_override_reasons)`: truthy when answered.
