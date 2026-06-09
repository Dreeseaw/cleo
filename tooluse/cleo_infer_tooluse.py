#!/usr/bin/env python3
"""Cleo v1.0 tool-use inference — (schema + question + DuckDB) -> gather* -> final SQL.

The model inspects the data with read-only `gather` queries to discover real values/codes/conventions
(e.g. "current" -> the sentinel `to_date='9999-01-01'`, status `'O'` not `'open'`) before its final
answer. Drives the SAME ToolUseEnv used in training/eval, so inference is format-faithful to the model.

  # transformers (bf16):
  python cleo_infer_tooluse.py --backend hf  --model dreeseaw/cleo --db sales.duckdb \
      --question "Revenue from completed orders by region."
  # GGUF (Q8, CPU/GPU via llama-cpp-python):
  python cleo_infer_tooluse.py --backend gguf --model cleo_v1_0-no_mtp-Q8_0.gguf --db sales.duckdb \
      --question "Revenue from completed orders by region."
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import ToolUseEnv, Task, Rollout  # noqa: E402


def schema_ddl(con):
    out = []
    for (t,) in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema NOT IN ('information_schema','pg_catalog')").fetchall():
        cols = con.execute(f'PRAGMA table_info("{t}")').fetchall()
        out.append(f"CREATE TABLE {t} (" + ", ".join(f"{c[1]} {c[2]}" for c in cols) + ");")
    return "\n".join(out)


class HFRunner:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.dev).eval()

    def gen(self, prompt, max_new=160):
        enc = self.tok(prompt, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.m.generate(**enc, do_sample=False, max_new_tokens=max_new,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


class GGUFRunner:
    def __init__(self, model_path, n_ctx=4096):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=8, verbose=False)

    def gen(self, prompt, max_new=160):
        return self.llm(prompt, max_tokens=max_new, temperature=0.0, stop=["\n\n\n"])["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "gguf"], default="hf")
    ap.add_argument("--model", required=True, help="HF dir/id or path to .gguf")
    ap.add_argument("--db", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--max-gather", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()
    import duckdb
    con = duckdb.connect(a.db, read_only=True)
    schema = schema_ddl(con)
    con.close()
    runner = HFRunner(a.model) if a.backend == "hf" else GGUFRunner(a.model)
    env = ToolUseEnv(max_steps=a.max_gather + 1, max_gather=a.max_gather)
    task = Task(id="live", db_path=a.db, schema_ddl=schema, question=a.question,
                gold_sql="", gold_type="sql")
    r = Rollout(task=task)
    while not r.done:
        text = runner.gen(env.render_prompt(r), a.max_new)
        env.step(r, text)
    if r.final_kind == "clarify":
        print(json.dumps({"clarification": r.final_clarify, "gathers": r.n_gather}, indent=2))
    else:
        print(json.dumps({"sql": r.final_sql, "gathers": r.n_gather,
                          "discovered": sorted(r.observed_values)[:12]}, indent=2))


if __name__ == "__main__":
    main()
