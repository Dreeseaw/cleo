"""A small tool-using SQL analyst for your own database connection.

    from cleo import Cleo
    cleo = Cleo.from_hf()     # downloads and caches the current hardel release
    ans  = cleo.ask("revenue from completed orders by region", conn)   # conn = any DB-API 2.0 connection
    print(ans.sql, ans.rows)

Shell: `cleo "revenue by region" --db warehouse.duckdb`, or `cleo mcp --db ...`.
"""
from .agent import Answer, Cleo

__all__ = ["Cleo", "Answer"]
__version__ = "1.4.1"
