# Cleo v1.0 tool-use harness (gather → final)

Cleo v1.0 is a **tool-using** SQL analyst: given a schema + question + a live DuckDB, it issues read-only
`gather` probes to **discover real values/codes/conventions** before answering — the lever that a
single-shot model (v0.9) structurally cannot pull (e.g. *"current employees"* → the sentinel
`to_date='9999-01-01'`, or status `'O'` not `'open'`, which the model cannot guess but **can discover**).

## Action contract
The model emits exactly one JSON action per turn:
- `{"tool":"gather","sql":"SELECT ..."}` — read-only probe, returns ≤20 rows (discovery).
- `{"tool":"final","sql":"SELECT ..."}` — terminal read-only SELECT answer.
- `{"tool":"final","clarify":"..."}` — terminal clarification if ambiguous / out-of-schema.

`env.py` is the authoritative environment (prompt rendering, action parsing, gather execution +
observation formatting, read-only validation). `cleo_infer_tooluse.py` drives that **same** env, so
inference is byte-faithful to how the model was trained/evaluated.

## Run
```bash
pip install -r ../requirements.txt
huggingface-cli login                      # private dreeseaw/cleo weights
hf download dreeseaw/cleo cleo_v1_0-no_mtp-Q8_0.gguf --local-dir ./weights

# GGUF (Q8 — recommended; Q4 erodes the trained tool-use deltas)
python tooluse/cleo_infer_tooluse.py --backend gguf \
    --model weights/cleo_v1_0-no_mtp-Q8_0.gguf \
    --db your.duckdb --question "How many employees are currently in each department?"

# transformers (bf16, base precision) — the v1.0/ subfolder on HF
python tooluse/cleo_infer_tooluse.py --backend hf \
    --model dreeseaw/cleo --db your.duckdb --question "..."
```

## Evaluation (held-out, denotation-scored)
| suite | v0.9 (single-shot) | **v1.0 (tool-use)** |
|---|---|---|
| value-discovery (66 q, gather-required) | 13.6% | **51.5%** |
| general OOD (new DBs, 81 q) | 59.3% | **64.2%** |
| general in-distribution (80 q) | 57.5% | 47.5% |

v1.0 wins on the use-case-relevant axes — value-discovery and **new** databases (OOD ≈ a fresh
RDS/Snowflake schema). In-distribution is v0.9's overfit home turf; v1.0's one-shot accuracy there is at
parity (the gap is residual over-gathering, not lost SQL competence).

## How v1.0 was trained (brief)
v0.9 (55k-SFT single-shot) → **behavioral cloning on denotation-verified teacher trajectories**: a
teacher drives this harness, only correct trajectories' (prompt → action) pairs are kept (no logits ever
stored), and the student is BC-distilled on them. ~$1.4 of teacher inference. Tool-use **RL** stalled
(gather-but-ignore: exploration can't overcome the single-shot prior); direct supervision broke through.
Quantize → **Q8_0 GGUF** (preserves the deltas; verified VD parity 51.5%).

Architecture: `qwen3_5`; GGUF converted with `--no-mtp` (MTP head unused by llama.cpp).
Notes: `gather` is **read-only** (validated); the env caps gather budget and surfaces observations back
to the model each turn. The legacy `actionrt/` GBNF loop is superseded by this harness.
