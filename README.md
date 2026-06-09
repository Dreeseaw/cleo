# Cleo v1.0 — tool-use SQL analyst (gather → discover values → answer)

**v1.0 is here (2026-06-09).** Cleo now *uses tools*: it issues read-only `gather` probes against the
live DB to **discover real values/codes/conventions** before answering, instead of one-shot guessing.
This is the lever that broke the value-discovery ceiling of v0.9.

**Use it as a Python package** — point it at your own DB connection, no server, no pre-staging:

```python
from cleo import Cleo
cleo = Cleo.from_gguf("cleo_v1_0-no_mtp-Q8_0.gguf")   # or Cleo.from_hf("dreeseaw/cleo")
ans  = cleo.ask("employees currently in each department?", conn)   # conn = any DB-API 2.0 connection
ans.sql, ans.rows, ans.clarification, ans.discovered
```

`conn` is any `psycopg2` / `sqlite3` / `duckdb` / SQLAlchemy connection; every query is validated
read-only and rolled back. Drop it into an MCP server (`examples/mcp_tool.py`) or a REPL. Package docs:
[`cleo/`](cleo/) · [`pyproject.toml`](pyproject.toml). Weights on HF `dreeseaw/cleo`:
`cleo_v1_0-no_mtp-Q8_0.gguf` (recommended) and the bf16 model under `v1.0/`.

| suite | v0.9 (single-shot) | **v1.0 (tool-use)** |
|---|---|---|
| value-discovery (gather-required) | 13.6% | **51.5%** |
| general OOD (new DBs) | 59.3% | **64.2%** |

Trained by **behavioral cloning on denotation-verified teacher trajectories** (no stored logits, ~$1.4
teacher cost); Q8_0 GGUF preserves the deltas. The `tooluse/` training harness, legacy one-shot
`cleo_infer.py`, and the `actionrt/` GBNF loop are kept for reference but superseded by the `cleo` package.

---

# Cleo

A small (~2B) **analyst-SQL** model + a thin **harness** — turns a database **schema + question**
into a read-only SQL query, or an honest **clarification** when the request is
ambiguous/unsafe/out-of-schema. Personal research project.

- Output is strict JSON, exactly one key: `{"sql": "..."}` or `{"clarification": "..."}`.
- The model is *not* trained on tool-call trajectories — only (schema + question → final answer).
  Grounding/execution is the **harness's** job (`actionrt/`).

**Code** lives here (GitHub). **Weights** (GGUF) live on Hugging Face: `dreeseaw/cleo` (private).

## Two ways to run

1. **One-shot** (`cleo_infer.py`) — single generation, schema+question → SQL/clarify. Simple, fast.
2. **Full harness** (`run_harness_demo.py`) — the `actionrt` GBNF-constrained **agentic loop**
   (gather/read/write primitives over DuckDB, grounding, fixed-mode fallback). This is the real
   product. **Heads-up for v0.9:** the model was SFT'd on final answers, *not* on harness
   rollouts, so it drives the multi-step loop clumsily (takes valid actions but often doesn't
   converge to a final answer). That's expected for a pre-release — **RLVR-in-harness (v0.95) is
   what teaches it to use the loop.** Use this to dogfood the plumbing.

## Lineage (v0.9)

`Qwen3.5-2B-Base` → MSH3 SFT (LoRA→merged) → OPD on-policy distillation from a Qwen3.6-27B
teacher (LoRA→merged) → **v0.9 amalgamation SFT** (LoRA→merged → GGUF): a LLaVA-style ~53k-row
mixture (capability packs + a 40k diverse leak-free SynSQL-2.5M slice + multi-turn/recovery +
abstain/clarify + grounding traces).

### Results vs the prior champion (one-shot, answerable-denotation accuracy)

| suite | prior champion | **Cleo v0.9** |
|---|---|---|
| real_db_ood_v2 | 15/43 | 17/43 |
| **real_db_ood_v3 (OOD)** | 32/86 | **61/86** |
| **in-dist canonical** | 19/99 | **59/99** |

## Setup (Mac — fast via Metal)

```bash
git clone https://github.com/Dreeseaw/cleo.git && cd cleo
pip install -r requirements.txt          # llama-cpp-python builds with Metal on Mac
huggingface-cli login                    # access the private dreeseaw/cleo weights
hf download dreeseaw/cleo cleo_v0_9-no_mtp-Q4_K_M.gguf --local-dir ./weights

# one-shot:
python cleo_infer.py --model weights/cleo_v0_9-no_mtp-Q4_K_M.gguf \
  --schema "CREATE TABLE orders (id INT, customer_id INT, amount REAL, status TEXT); CREATE TABLE customers (id INT, name TEXT, country TEXT);" \
  --question "Total order amount for US customers, by status."

# full agentic harness (demo DB; rough on v0.9 pre-RLVR — that's the point):
PYTHONPATH=. python run_harness_demo.py --backend llama-cpp \
  --model-path weights/cleo_v0_9-no_mtp-Q4_K_M.gguf --n-gpu-layers -1 --max-steps 6
```

One-shot expected:
```json
{"sql": "SELECT status, SUM(amount) AS total_amount FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE country = 'US') GROUP BY status ORDER BY status;"}
```

## Notes

- `Q4_K_M` GGUF ≈ 1.27 GB. Runs on Mac (Metal, `n_gpu_layers=-1`), CUDA, or CPU (`0`).
- Architecture: `qwen3_5` (hybrid linear-attention); the GGUF was converted with `--no-mtp`
  (the MTP head isn't supported by llama.cpp's loader).
- v0.9 is SFT only. **v0.95 = RLVR-in-harness** (teaches the agentic loop) — planned next.
