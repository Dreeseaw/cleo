"""Expose Cleo as an MCP tool — an LLM agent can ask your database questions in natural language.

    pip install "cleo-sql[gguf]" mcp psycopg2-binary
    python examples/mcp_tool.py

The whole integration is one `cleo.ask(...)` call; Cleo handles the discover-then-answer loop.
"""
import os

import psycopg2
from mcp.server.fastmcp import FastMCP

from cleo import Cleo

cleo = Cleo.from_gguf(os.environ.get("CLEO_GGUF", "cleo_v1_0-no_mtp-Q8_0.gguf"))
DSN = os.environ["DATABASE_URL"]

mcp = FastMCP("cleo")


@mcp.tool()
def query_database(question: str, tables: list[str] | None = None) -> dict:
    """Answer a natural-language question about the database with read-only SQL.

    Optionally pass `tables` to scope a large schema. Returns the generated SQL and its result rows.
    """
    conn = psycopg2.connect(DSN)
    try:
        ans = cleo.ask(question, conn, tables=tables)
    finally:
        conn.close()
    return {
        "sql": ans.sql,
        "columns": ans.columns,
        "rows": ans.rows,
        "clarification": ans.clarification,
        "discovered_values": ans.discovered,
        "error": ans.error,
    }


if __name__ == "__main__":
    mcp.run()
