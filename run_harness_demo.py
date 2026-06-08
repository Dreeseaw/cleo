#!/usr/bin/env python3
"""Run a v1 primitive-engine smoke demo on a tiny DuckDB database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actionrt.runtime import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    ActionRuntime,
    LlamaCppActionBackend,
    RuntimeConfig,
    ScriptedActionBackend,
    summarize_results,
)


DEFAULT_OUT_DIR = ROOT / "results" / "primitive_engine_demo_20260607"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build_demo_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE TABLE customers(customer_id INTEGER, region VARCHAR)")
        con.execute("CREATE TABLE orders(order_id INTEGER, customer_id INTEGER, status VARCHAR, revenue DOUBLE)")
        con.execute("INSERT INTO customers VALUES (1, 'North'), (2, 'South')")
        con.execute(
            """
            INSERT INTO orders VALUES
              (10, 1, 'complete', 100.0),
              (11, 1, 'cancelled', 50.0),
              (12, 2, 'complete', 75.0),
              (13, 2, 'complete', 25.0)
            """
        )
        con.commit()
    finally:
        con.close()
    return path


def build_exemplar_bank(path: Path) -> Path:
    payload = {
        "databases": {
            "primitive_demo": [
                {
                    "id": "demo_revenue_by_region",
                    "question": "What is total completed revenue by region?",
                    "sql": (
                        "SELECT c.region, SUM(o.revenue) AS total_revenue "
                        "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
                        "WHERE o.status = 'complete' GROUP BY c.region ORDER BY c.region"
                    ),
                    "source": "approved_sql",
                },
                {
                    "id": "demo_orders_by_status",
                    "question": "How many orders exist for each status?",
                    "sql": "SELECT status, COUNT(*) AS n FROM orders GROUP BY status ORDER BY status",
                    "source": "approved_sql",
                },
            ]
        }
    }
    write_json(path, payload)
    return path


def scripted_actions(question: str) -> list[dict[str, Any]]:  # noqa: ARG001
    return [
        {"type": "tool_call", "tool": "inspect_schema", "args": {"include_columns": True, "include_relationships": True}},
        {"type": "tool_call", "tool": "retrieve_exemplars", "args": {"question": "total completed revenue by region", "k": 1, "schema_scope": "same_db"}},
        {"type": "tool_call", "tool": "profile_columns", "args": {"table": "orders", "columns": ["status"], "top_values_limit": 4, "allow_value_examples": True}},
        {
            "type": "tool_call",
            "tool": "exec_sql",
            "args": {
                "sql": (
                    "SELECT c.region, SUM(o.revenue) AS total_revenue "
                    "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
                    "WHERE o.status = 'complete' GROUP BY c.region ORDER BY c.region"
                ),
                "purpose": "answer",
                "row_limit": 20,
            },
        },
        {
            "type": "final_answer",
            "answer": "Completed revenue is 100.0 for North and 100.0 for South.",
            "sql_id": "last_sql",
            "chart_id": None,
            "confidence": 1.0,
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["scripted", "llama-cpp"], default="scripted")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--question", default="Show completed order revenue by customer region.")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=6)
    parser.add_argument("--n-batch", type=int, default=64)
    parser.add_argument("--n-gpu-layers", type=int, default=16)
    parser.add_argument("--fixed-mode-fallback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    db_path = build_demo_db(args.out_dir / "primitive_demo.duckdb")
    bank_path = build_exemplar_bank(args.out_dir / "exemplar_bank.json")

    if args.backend == "scripted":
        backend = ScriptedActionBackend(scripted_actions(args.question))
        continue_after_sql = True
    else:
        backend = LlamaCppActionBackend(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_batch=args.n_batch,
            n_gpu_layers=args.n_gpu_layers,
            seed=0,
            verbose=False,
        )
        continue_after_sql = False

    runtime = ActionRuntime(
        backend,
        config=RuntimeConfig(
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            row_limit=args.row_limit,
            continue_after_sql=continue_after_sql,
            out_dir=args.out_dir,
            exemplar_bank_path=bank_path,
            db_id="primitive_demo",
            fixed_mode_fallback=args.fixed_mode_fallback,
        ),
    )
    result = runtime.run(question=args.question, db_path=db_path)
    summary = summarize_results([result])
    summary.update({"backend": args.backend, "db_path": str(db_path), "exemplar_bank": str(bank_path)})
    write_json(args.out_dir / f"{args.backend}_record.json", result.to_dict())
    write_json(args.out_dir / f"{args.backend}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
