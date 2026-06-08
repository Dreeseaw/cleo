#!/usr/bin/env python3
"""Cleo v0.9 — minimal inference.

(schema + question) -> {"sql": "..."} for an answerable read-only query,
or {"clarification": "..."} when the request is ambiguous/unsafe/out-of-schema.

Self-contained: needs only `transformers>=5.3` + `torch` (+ optional `duckdb` to execute).
Runs on CUDA, Apple MPS, or CPU (slow on Mac — the linear-attention fast kernels are
CUDA-only, so it falls back to the torch implementation; fine for light use).

Examples:
  python cleo_infer.py --schema "CREATE TABLE t (a INT, b TEXT)" --question "rows where b='x'"
  python cleo_infer.py --schema-file schema.sql --question "..." --db mydata.duckdb
  python cleo_infer.py --model dreeseaw/cleo --schema-file s.sql --question "..."
"""
import argparse, json, sys
import torch

INSTRUCTION = """You are a toy SQLite SQL agent.

Output contract (mandatory): Return strict JSON.
- Return exactly one JSON object and nothing else: no Markdown, no code fences, no prose before or after JSON.
- The object must contain exactly one key:
  - {"sql": "..."} for an answerable read-only SQLite query.
  - {"clarification": "..."} when the request is ambiguous, underspecified, unsafe, or asks for data outside the schema.
- Never include both keys. Never include extra keys.

SQL contract:
- SQL must be a single SQLite statement beginning with SELECT or WITH.
- Do not write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH, VACUUM, or multiple statements.
- Use only tables and columns present in the supplied schema.
- Choose stable, human-readable output aliases with AS when computing, grouping, or renaming columns.
- Prefer deterministic ordering when a question implies ranked, top/bottom, first/last, or otherwise comparable results."""


def load(model_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return tok, model, dev


def infer(tok, model, dev, schema, question, max_new_tokens=256):
    user = (f"Schema:\n{schema.strip()}\n\nQuestion:\n{question.strip()}\n\n"
            "Return exactly one JSON object that follows the output contract.")
    prompt = f"Instruction:\n{INSTRUCTION}\n\nInput:\n{user}\n\nOutput:\n"
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        return json.loads(text)
    except Exception:
        return {"_unparsed": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".", help="HF id (dreeseaw/cleo) or local model dir")
    ap.add_argument("--schema", help="CREATE TABLE ... DDL string")
    ap.add_argument("--schema-file", help="path to a .sql file with the schema DDL")
    ap.add_argument("--question", required=True)
    ap.add_argument("--db", help="optional DuckDB file; if set and result is SQL, executes it read-only")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    schema = a.schema or (open(a.schema_file).read() if a.schema_file else "")
    if not schema:
        sys.exit("provide --schema or --schema-file")
    tok, model, dev = load(a.model)
    res = infer(tok, model, dev, schema, a.question, a.max_new_tokens)
    print(json.dumps(res, indent=2))
    if a.db and "sql" in res:
        import duckdb
        try:
            rows = duckdb.connect(a.db, read_only=True).execute(res["sql"]).fetchall()
            print("\n-- result (first 20 rows) --")
            for r in rows[:20]:
                print(r)
        except Exception as e:
            print("exec error:", e)


if __name__ == "__main__":
    main()
