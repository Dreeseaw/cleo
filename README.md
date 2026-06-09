# Cleo

**A small (~2B) SQL analyst that discovers values in your data before it answers.**

Most text-to-SQL models map `question → SQL` in one shot — so when the right query depends on a literal
that only lives in the *data* (a status code `'O'` not `'open'`, `"current"` meaning the sentinel
`to_date = '9999-01-01'`, `GB` not `UK`), they guess, and guess wrong. Cleo instead issues read-only
`gather` probes to **look first**, then writes its answer:

```python
from cleo import Cleo
import psycopg2

cleo = Cleo.from_gguf("cleo_v1_0-no_mtp-Q8_0.gguf")          # CPU-friendly; or Cleo.from_hf("dreeseaw/cleo")
ans  = cleo.ask("employees currently in each department?", psycopg2.connect(DSN))

ans.sql          # SELECT d.dept_name, COUNT(*) ... WHERE de.to_date = '9999-01-01' ...
ans.rows         # executed result
ans.discovered   # ["9999-01-01", ...] — the convention it found by probing
```

Point it at a DB-API 2.0 connection (Postgres, MySQL, SQLite, DuckDB; SQLAlchemy via `raw_connection()`).
No server, no copying data into a local engine first. Every query is **validated read-only** (statement +
AST + a side-effecting-function denylist) and rolled back. For production, also run Cleo under a
**least-privilege, read-only DB role** — the in-process guard is defense-in-depth, not a substitute for
database permissions.

## Results

Held-out, denotation-scored (execute predicted vs gold SQL, compare row-sets; schemas disjoint from all
training data):

| benchmark | v0.9 (one-shot) | **Cleo v1.0 (tool-use)** |
|---|---|---|
| **value-discovery** — answer needs a discovered literal | 13.6% | **51.5%** |
| general SQL, **out-of-distribution** databases | 59.3% | **64.2%** |
| general SQL, in-distribution | 57.5% | 47.5% |

The tool pays for itself on value-discovery (+38 points, where a one-shot model is structurally capped)
and on *new* databases — the case that matters when you point it at a schema it has never seen.

## How it was trained

v1.0 was produced by **behavioral cloning on denotation-verified teacher trajectories**, for **~$1.30 of
teacher inference** — no reinforcement learning, no stored logits:

1. A cheap teacher drives the gather→final loop on ~700 curated questions across 472 schemas.
2. A trajectory is kept **only if its final answer is denotation-correct** against gold.
3. The 2B student is supervised on the kept `(state → action)` pairs, then calibrated with a slice of
   "answer-directly" examples so it doesn't over-probe simple questions.

The full method — and an honest account of the approaches that *failed* (tool-use RL hitting a
gather-but-ignore wall, the clean-base-vs-warm-start surprise, the calibration⊥discovery tension at 2B,
on-policy DAgger) — is in **[TECH_REPORT.md](TECH_REPORT.md)**.

## Install

```bash
pip install "cleo-sql[gguf]"        # llama-cpp-python backend (CPU / Mac / CUDA)
pip install "cleo-sql[hf]"          # transformers backend (GPU)
hf download dreeseaw/cleo cleo_v1_0-no_mtp-Q8_0.gguf --local-dir .
```

Scope a large database, or hand Cleo the DDL yourself:

```python
cleo.ask("...", conn, tables=["orders", "customers"])   # only introspect these
cleo.ask("...", conn, schema=my_ddl_string)             # skip introspection
```

Drop it into an MCP server in a few lines — see [`examples/mcp_tool.py`](examples/mcp_tool.py).

> Notes: the distribution is `cleo-sql` but the import is `cleo` (heads-up: the Poetry CLI framework also
> uses `import cleo`). Supported dialects are Postgres / MySQL / SQLite / DuckDB; Oracle and SQL Server
> aren't (the row-cap and introspection assume `LIMIT` + `information_schema`/PRAGMA). Use a non-autocommit
> connection so Cleo can roll back.

## What's here

| path | what |
|---|---|
| [`cleo/`](cleo/) | the package: `contract` (the trained action protocol), `db` (connection-agnostic read-only execution + schema introspection), `backends` (GGUF / HF), `agent` (the `Cleo` loop) |
| [`TECH_REPORT.md`](TECH_REPORT.md) | training method + failure analysis |
| [`examples/`](examples/) | MCP tool, quickstart |
| [`tooluse/`](tooluse/) | the research training/eval harness (reference) |

## Links

- **Model**: [`dreeseaw/cleo`](https://huggingface.co/dreeseaw/cleo) — Q8_0 GGUF + bf16
- **Benchmark**: [`dreeseaw/cleo-value-discovery`](https://huggingface.co/datasets/dreeseaw/cleo-value-discovery) — the value-discovery suite (open)
- **Report**: [TECH_REPORT.md](TECH_REPORT.md)

---

*A personal research project. Cleo is a 2B model: it sits on the value-discovery / general-SQL trade-off
its size allows, and its residual errors are wrong-value bindings (it probes, then occasionally binds the
wrong literal). It is meant as a small, honest, useful tool — and a study in getting real behavior into a
small model cheaply.*
