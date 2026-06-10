"""Cleo — a small tool-using SQL analyst you point at your own database connection.

    from cleo import Cleo
    cleo = Cleo.from_gguf("cleo_v1_2_bird-no_mtp-Q8_0.gguf")   # or Cleo.from_hf("dreeseaw/cleo")
    ans  = cleo.ask("revenue from completed orders by region", conn)   # conn = any DB-API 2.0 connection
    print(ans.sql, ans.rows)
"""
from .agent import Answer, Cleo

__all__ = ["Cleo", "Answer"]
__version__ = "1.2.0"
