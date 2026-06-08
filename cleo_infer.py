#!/usr/bin/env python3
"""Cleo — one-shot inference via llama.cpp / GGUF.

(schema + question) -> {"sql": "..."} (read-only) or {"clarification": "..."}.
Single generation, no agentic loop. For the full harness loop see run_harness_demo.py.

Needs: llama-cpp-python. Model: a Cleo GGUF, e.g. ./weights/cleo_v0_9-no_mtp-Q4_K_M.gguf
(download from the private HF repo dreeseaw/cleo). On Mac, n_gpu_layers=-1 uses Metal.

  python cleo_infer.py --model weights/cleo_v0_9-no_mtp-Q4_K_M.gguf \
    --schema "CREATE TABLE t (a INT, b TEXT)" --question "rows where b='x'"
"""
import argparse, json, sys

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a Cleo GGUF")
    ap.add_argument("--schema")
    ap.add_argument("--schema-file")
    ap.add_argument("--question", required=True)
    ap.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = offload all (Mac Metal/CUDA); 0 = CPU")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--db", help="optional DuckDB file; if result is SQL, executes it read-only")
    a = ap.parse_args()
    schema = a.schema or (open(a.schema_file).read() if a.schema_file else "")
    if not schema:
        sys.exit("provide --schema or --schema-file")

    from llama_cpp import Llama
    llm = Llama(model_path=a.model, n_ctx=a.n_ctx, n_gpu_layers=a.n_gpu_layers, verbose=False)
    user = (f"Schema:\n{schema.strip()}\n\nQuestion:\n{a.question.strip()}\n\n"
            "Return exactly one JSON object that follows the output contract.")
    prompt = f"Instruction:\n{INSTRUCTION}\n\nInput:\n{user}\n\nOutput:\n"
    out = llm(prompt, max_tokens=a.max_tokens, temperature=0.0)
    text = out["choices"][0]["text"].strip()
    try:
        res = json.loads(text)
    except Exception:
        res = {"_unparsed": text}
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
