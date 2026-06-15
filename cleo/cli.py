"""Ask one question from the shell, or serve Cleo as an MCP tool.

    cleo "total revenue by region" --db warehouse.duckdb
    cleo "active users this week?" --db postgresql://me@host/db --tables users,sessions --json
    cleo mcp --db "$DATABASE_URL"        # MCP server (pip install "cleo-sql[gguf,mcp]")

By default, Cleo downloads and caches the current GGUF from HF. Pass --model for a local file.
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
    return Cleo.from_gguf(args.model)


def _answer(cleo, db: str, question: str, tables: list[str] | None, *, hardel: bool = False,
            k: int = 8, temperature: float = 0.7, top_p: float = 0.95, seed: int = 6151) -> "dict":
    conn = connect(db)
    try:
        if hardel:
            ans = cleo.ask_hardel(
                question, conn, tables=tables, schema_fks=True, k=k,
                temperature=temperature, top_p=top_p, seed=seed,
            )
        else:
            ans = cleo.ask(question, conn, tables=tables, schema_fks=True)
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
                       hardel: bool = False, k: int = 8) -> dict:
        """Answer a database question with read-only SQL. Use `tables` to scope large schemas."""
        return _answer(cleo, args.db, question, tables, hardel=hardel, k=k)

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
    ap.add_argument("--model", help="local GGUF path (default: download the current champion from HF)")
    ap.add_argument("--tables", help="comma-separated table scope for big schemas")
    ap.add_argument("--hardel", action="store_true", help="run greedy + sampled candidates and select by evidence")
    ap.add_argument("--k", type=int, default=8, help="number of sampled candidates for --hardel")
    ap.add_argument("--temperature", type=float, default=0.7, help="sampling temperature for --hardel")
    ap.add_argument("--top-p", type=float, default=0.95, help="sampling top-p for --hardel")
    ap.add_argument("--seed", type=int, default=6151, help="sampling seed for --hardel")
    args = ap.parse_args(argv[1:] if mcp_mode else argv)
    args.tables = args.tables.split(",") if args.tables else None

    if mcp_mode:
        return _serve_mcp(args)

    out = _answer(
        _make_cleo(args), args.db, args.question, args.tables,
        hardel=args.hardel, k=args.k, temperature=args.temperature,
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
