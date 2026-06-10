"""Cleo — a small tool-using SQL analyst you point at your own database connection.

    from cleo import Cleo
    cleo = Cleo.from_gguf()   # downloads + caches the current champion; or pass a local GGUF path
    ans  = cleo.ask("revenue from completed orders by region", conn)   # conn = any DB-API 2.0 connection
    print(ans.sql, ans.rows)

Or from the shell: `cleo "revenue by region" --db warehouse.duckdb`, `cleo mcp --db ...`.
"""
from .agent import Answer, Cleo

__all__ = ["Cleo", "Answer"]
__version__ = "1.3.0"
