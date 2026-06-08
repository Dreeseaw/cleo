# Cleo

A small (~2B) **analyst-SQL** model — turns a database **schema + question** into a read-only
SQL query, or an honest **clarification** when the request is ambiguous/unsafe/out-of-schema.
Personal research project.

- Output is strict JSON, exactly one key: `{"sql": "..."}` or `{"clarification": "..."}`.
- Inference-only. The model is *not* trained on tool-call trajectories — only
  (schema + question → final answer). Grounding/execution is the harness's job.

**Code** lives here (GitHub). **Weights** live on Hugging Face: `dreeseaw/cleo` (private).

## Lineage (v0.9)

`Qwen3.5-2B-Base` → MSH3 SFT (LoRA→merged) → OPD on-policy distillation from a Qwen3.6-27B
teacher (LoRA→merged) → **v0.9 amalgamation SFT** (LoRA→merged): a LLaVA-style ~53k-row mixture
(capability packs + a 40k diverse leak-free SynSQL-2.5M slice + multi-turn/recovery +
abstain/clarify + grounding traces).

### Results vs the prior champion (answerable-denotation accuracy)

| suite | prior champion | **Cleo v0.9** |
|---|---|---|
| real_db_ood_v2 | 15/43 | 17/43 |
| **real_db_ood_v3 (OOD)** | 32/86 | **61/86** |
| **in-dist canonical** | 19/99 | **59/99** |

## Setup (Mac, inference-only — perf is not a priority)

```bash
git clone https://github.com/Dreeseaw/cleo.git && cd cleo
pip install "transformers>=5.3" torch duckdb sentencepiece huggingface_hub
huggingface-cli login                         # access the private dreeseaw/cleo weights
hf download dreeseaw/cleo --local-dir ./weights
python cleo_infer.py --model ./weights \
  --schema "CREATE TABLE orders (id INT, customer_id INT, amount REAL, status TEXT); CREATE TABLE customers (id INT, name TEXT, country TEXT);" \
  --question "Total order amount for US customers, by status."
```

Expected:
```json
{"sql": "SELECT status, SUM(amount) AS total_amount FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE country = 'US') GROUP BY status ORDER BY status;"}
```

Execute the SQL too (read-only) by passing a DuckDB file:
```bash
python cleo_infer.py --model ./weights --schema-file schema.sql --question "..." --db mydata.duckdb
```

## Notes

- Runs on CUDA / Apple **MPS** / CPU. On Mac the linear-attention fast kernels
  (`flash-linear-attention`) are CUDA-only → torch fallback (functional, just slower).
- Architecture: `qwen3_5` (hybrid linear-attention); requires `transformers>=5.3`.
- v0.9 is SFT only; an RLVR-tuned **v0.95** is planned.
