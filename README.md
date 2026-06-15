# cleo

A unified micro-harness + fine-tuned Qwen3.5-2B (a "hardel") for SQL-based analytical workflows 
that you point at your own database connections. Cleo's been trained on probing the data read-only to 
**discover real values, codes, and conventions** deep in the data, **repair broken queries** in-flight, 
and treat clarity & observability as first-class features.

### The "hardel" thesis
By marrying the harness & model together with different training, inference, and tooling techniques,
you can extract more intelligence-per-parameter than traditional model/harness combos. This is particularly
useful in constrained workflow scenarios, such as SQL-based analytical workflows!

Currently, some features of cleo that are only possible/useful in a unified hardel are:
- Training on the exact same gather, repair, and answer contract it uses at inference time
- Searching over candidate queries with live execution evidence, not just model likelihood
- Co-designing the model contract, SQL safety layer, dialect handling, timeouts, and clarification behavior as one system

## Usage

No server, no pre-staging. Hand it a live DB-API connection and ask:

```python
from cleo import Cleo
import psycopg2

cleo = Cleo.from_hf("dreeseaw/cleo")  # auto-selects the best available HF device
conn = psycopg2.connect("postgresql://...") # your existing connection: Postgres, SQLite, DuckDB, ...

ans = cleo("How many employees are currently in each department?", conn)
print(ans.sql)            # the final read-only SELECT
print(ans.rows)           # executed result (rows), or None if Cleo asked to clarify
print(ans.clarification)  # set when the question is ambiguous / out-of-schema
print(ans.discovered)     # real values Cleo found while probing
```

Cleo's default runtime is the hardel: a greedy candidate plus sampled candidates
run through the same read-only harness, then a label-free selector chooses using
execution evidence such as result clusters, observed literal reuse, and live DB
literal support:

```python
ans = cleo(
    "How many employees are currently in each department?",
    conn,
    k=8,              # sampled candidates after the greedy candidate; default is 4
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
cleo("...", conn, tables=["employees", "departments"])   # only these tables
cleo("...", conn, schema=my_ddl_string)                  # or hand it the DDL yourself
```

### CLI
```bash
cleo "total revenue by region" --db warehouse.duckdb
cleo "active users this week?" --db postgresql://me@host/db --tables users,sessions --json
cleo "current customers by segment" --db warehouse.duckdb --k 8 --json
```

### As an MCP tool
`cleo mcp --db "$DATABASE_URL"` serves a natural-language `query_database` tool
(`pip install "cleo-sql[hf,mcp]"` for the current hardel release). Claude Code / Claude Desktop config:

```json
"cleo": { "command": "cleo", "args": ["mcp", "--db", "postgresql://..."] }
```

## Install

```bash
pip install "cleo-sql[hf] @ git+https://github.com/Dreeseaw/cleo.git@master"      # current hardel release via transformers (GPU)
pip install "cleo-sql[hf,int8] @ git+https://github.com/Dreeseaw/cleo.git@master" # optional CUDA bitsandbytes int8 load path
pip install "cleo-sql[hf,mcp] @ git+https://github.com/Dreeseaw/cleo.git@master"  # MCP server extras
pip install "cleo-sql[gguf] @ git+https://github.com/Dreeseaw/cleo.git@master"    # legacy/experimental llama-cpp-python backend
# HF weights download + cache themselves on first use: Cleo.from_hf("dreeseaw/cleo").
# GGUF loading depends on llama.cpp Python binding support for the model architecture.
```

PyPI currently trails this GitHub release; install from GitHub for v1.4 hardel until the next PyPI publish.
The current release ships bf16 Transformers weights. `Cleo.from_hf()` auto-selects CUDA, XPU, MPS, then CPU
when PyTorch exposes them; CUDA is the tested fast path for the current Qwen3.5 hardel. For CUDA machines
that need lower VRAM, use `Cleo.from_hf("dreeseaw/cleo", quantization="int8")` after installing the `int8`
extra. The GGUF files on Hugging Face are legacy artifacts from earlier releases, not the current v1.4
hardel.

Run the tests (no model/GPU needed): `pip install -e ".[test]" && pytest`

## API
- `Cleo.from_hf(model="dreeseaw/cleo", *, device=None, quantization=None)`: current HF hardel release with automatic device and dtype selection.
- `Cleo.from_gguf(path=None, *, n_ctx=4096, n_threads=8, n_gpu_layers=0)`: legacy llama-cpp-python backend when the local binding supports the target GGUF.
- `cleo(question, conn, *, k=4, temperature=0.7, top_p=0.95, seed=6151, **ask_kwargs) -> Answer`: default hardel runtime; runs candidate search through the live harness and selects with product-visible evidence.
- `cleo.ask(...)`: same as `cleo(...)`.
- `cleo.ask_once(question, conn, *, schema=None, tables=None, max_gather=3, max_repair=2, execute_final=True, row_limit=1000, dialect=None, schema_fks=False, schema_samples=0) -> Answer`: explicit single-candidate engine for low-latency or debugging use.
- `Answer(sql, rows, columns, clarification, gathers, discovered, error, selector, evidence_override, evidence_override_reasons)`: truthy when answered.
