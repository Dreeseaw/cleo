"""Ask one question from the shell, or serve Cleo as an MCP tool.

    cleo "total revenue by region" --db warehouse.duckdb
    cleo "active users this week?" --db postgresql://me@host/db --tables users,sessions --json
    cleo mcp --db "$DATABASE_URL"        # MCP server (pip install "cleo-sql[hf,mcp]")

By default, Cleo uses the current HF hardel release with automatic device selection.
Use --backend gguf for a local or legacy GGUF file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def connect(db: str):
    """Open `db`: a postgres:// URL, a .duckdb file, or any other path as SQLite."""
    if db.startswith(("postgres://", "postgresql://")):
        import psycopg2
        return psycopg2.connect(db)
    if not Path(db).exists():
        sys.exit(f"cleo: no such database file: {db}")  # sqlite3.connect would silently create it
    if db.endswith(".duckdb"):
        import duckdb
        return duckdb.connect(db, read_only=True)
    import sqlite3
    return sqlite3.connect(db)


def _make_cleo(args):
    from .agent import Cleo
    if args.backend == "gguf":
        return Cleo.from_gguf(args.model, n_gpu_layers=args.n_gpu_layers)
    return Cleo.from_hf(args.model or "dreeseaw/cleo", device=args.device,
                        quantization=args.quantization)


def _answer(cleo, db: str, question: str, tables: list[str] | None, *, single_pass: bool = False,
            k: int = 4, temperature: float = 0.7, top_p: float = 0.95, seed: int = 6151) -> "dict":
    conn = connect(db)
    try:
        ans = cleo.ask(
            question, conn, tables=tables, schema_fks=True,
            runtime=not single_pass, k=k, temperature=temperature, top_p=top_p, seed=seed,
        )
    finally:
        conn.close()
    return {"status": ans.status, "sql": ans.sql, "columns": ans.columns, "rows": ans.rows,
            "clarification": ans.clarification, "discovered_values": ans.discovered,
            "error": ans.error, "selector": ans.selector,
            "evidence_override": ans.evidence_override,
            "evidence_override_reasons": ans.evidence_override_reasons,
            "candidate_count": len(ans.candidates)}


def _serve_mcp(args) -> None:
    from mcp.server.fastmcp import FastMCP
    cleo = _make_cleo(args)
    server = FastMCP("cleo")

    @server.tool()
    def query_database(question: str, tables: list[str] | None = None,
                       hardel: bool = True, k: int = 4) -> dict:
        """Answer a database question with read-only SQL. Use `tables` to scope large schemas."""
        return _answer(cleo, args.db, question, tables, single_pass=not hardel, k=k)

    server.run()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    mcp_mode = bool(argv) and argv[0] == "mcp"
    ap = argparse.ArgumentParser(prog="cleo mcp" if mcp_mode else "cleo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    if not mcp_mode:
        ap.add_argument("question")
        ap.add_argument("--json", action="store_true", help="print the full Answer as JSON")
    ap.add_argument("--db", required=True, help="database file (.sqlite/.duckdb/...) or postgres:// URL")
    ap.add_argument("--backend", choices=["hf", "gguf"], default="hf",
                    help="generation backend (default: hf)")
    ap.add_argument("--model", help="HF repo id for --backend hf, or local GGUF path for --backend gguf")
    ap.add_argument("--device", help="HF device: auto, cuda, cuda:0, xpu, mps, or cpu (default: auto)")
    ap.add_argument("--quantization", choices=["none", "int8"],
                    help="HF quantization mode (default: none; int8 requires CUDA bitsandbytes)")
    ap.add_argument("--n-gpu-layers", type=int, default=0,
                    help="number of GGUF layers to offload when --backend gguf")
    ap.add_argument("--tables", help="comma-separated table scope for big schemas")
    ap.add_argument("--single-pass", action="store_true", help="disable the default evidence-selected runtime")
    ap.add_argument("--hardel", action="store_true", help="deprecated: the hardel runtime is now the default")
    ap.add_argument("--k", type=int, default=4, help="number of sampled candidates for the runtime")
    ap.add_argument("--temperature", type=float, default=0.7, help="sampling temperature for runtime candidates")
    ap.add_argument("--top-p", type=float, default=0.95, help="sampling top-p for runtime candidates")
    ap.add_argument("--seed", type=int, default=6151, help="sampling seed for runtime candidates")
    args = ap.parse_args(argv[1:] if mcp_mode else argv)
    args.tables = args.tables.split(",") if args.tables else None

    if mcp_mode:
        return _serve_mcp(args)

    out = _answer(
        _make_cleo(args), args.db, args.question, args.tables,
        single_pass=args.single_pass, k=args.k, temperature=args.temperature,
        top_p=args.top_p, seed=args.seed,
    )
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    elif out["status"] == "clarify":
        print(f"cleo asks: {out['clarification']}")
    elif out["status"] != "ok":
        sys.exit(f"cleo: {out['error']}")
    else:
        print(out["sql"].strip())
        if out["columns"] is not None:
            print("\t".join(str(c) for c in out["columns"]))
            for row in out["rows"]:
                print("\t".join(str(c) for c in row))


if __name__ == "__main__":
    main()
