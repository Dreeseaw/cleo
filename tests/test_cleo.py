"""Tests for the cleo package. Run: `pytest` (or `python tests/test_cleo.py`).

No model/GPU needed — `Cleo(backend=FakeBackend(...))` injects scripted model output.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleo import Answer, Cleo
from cleo.contract import (INSTRUCTION, format_observation, is_readonly, normalize_action,
                           parse_action, render_prompt)
from cleo.db import detect_dialect, introspect_schema, make_executor, run_readonly, transpile_sql


# ---------------------------------------------------------------- contract / fidelity
def test_instruction_has_binding_sentence():
    # the load-bearing sentence dropped in an earlier draft (must match the training harness)
    assert "use the EXACT stored value you observed in your final SQL." in INSTRUCTION
    assert INSTRUCTION.endswith("Use only tables/columns in the schema.")
    assert len(INSTRUCTION) == 918


def test_parse_action():
    assert parse_action('{"tool":"gather","sql":"SELECT 1"}') == {"tool": "gather", "sql": "SELECT 1"}
    assert parse_action('junk {"tool":"final","sql":"SELECT 1"} trailing') == {"tool": "final", "sql": "SELECT 1"}
    assert parse_action("no json here") is None


def test_normalize_action_rejects_unknown_tool():
    assert normalize_action({"tool": "gather", "sql": "SELECT 1"}) == {"kind": "gather", "sql": "SELECT 1"}
    assert normalize_action({"sql": "SELECT 1"}) == {"kind": "final", "sql": "SELECT 1"}  # bare contract
    assert normalize_action({"tool": "foo", "sql": "SELECT 1"}) is None                   # fidelity guard
    assert normalize_action({"tool": "gather", "clarify": "x"}) is None


def test_render_prompt_budget_nudge():
    p = render_prompt("CREATE TABLE t(x int);", "q?", [], gathers_left=0, steps_left=1)
    assert 'You must now return a {"tool":"final",...} action.' in p
    assert render_prompt("s", "q", [], gathers_left=3, steps_left=4).endswith("Action:\n")


def test_format_observation():
    assert format_observation(["status"], [["O"], ["C"]], False) == 'cols=[\'status\'] rows(2)=[["O"], ["C"]]'


# ---------------------------------------------------------------- read-only guard (security)
def test_readonly_accepts_plain_selects():
    assert is_readonly("SELECT * FROM t")[0]
    assert is_readonly("WITH x AS (SELECT 1) SELECT * FROM x")[0]
    assert is_readonly("SELECT REPLACE(name,'a','b') FROM t")[0]   # REPLACE() is NOT a write (no false-positive)


def test_readonly_rejects_writes_and_side_effects():
    for bad in [
        "DELETE FROM t",
        "SELECT 1; DROP TABLE t",
        "SELECT * INTO t2 FROM t",                       # SELECT INTO is a write
        "SELECT pg_read_file('/etc/passwd')",            # file read
        "SELECT lo_export(1,'/tmp/x')",                  # file write
        "SELECT read_csv_auto('http://attacker/x')",     # SSRF / external read
        "SELECT pg_sleep(60)",                           # DoS
        "SELECT nextval('s')",                           # sequence mutation
        "UPDATE t SET x=1",
    ]:
        ok, why = is_readonly(bad)
        assert not ok, f"should reject: {bad} (got {why})"


# ---------------------------------------------------------------- db layer
def _orders_db(isolation_level=""):
    con = sqlite3.connect(":memory:", isolation_level=isolation_level)
    con.executescript("CREATE TABLE orders (id INTEGER, region TEXT, status TEXT);"
                      "INSERT INTO orders VALUES (1,'US','C'),(2,'US','O'),(3,'EU','C');")
    return con


def test_executor_runs_and_rolls_back():
    con = _orders_db()
    ex = make_executor(con)
    cols, rows, trunc, err = run_readonly(ex, "SELECT DISTINCT status FROM orders")
    assert err is None and sorted(r[0] for r in rows) == ["C", "O"]
    # a write is rejected by the guard, never reaches the DB
    _, _, _, err2 = run_readonly(ex, "DELETE FROM orders")
    assert err2 and "rejected" in err2
    assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3


def test_executor_refuses_autocommit():
    con = _orders_db(isolation_level=None)  # autocommit -> rollback can't undo, so Cleo must refuse
    try:
        make_executor(con)
        assert False, "should have refused autocommit connection"
    except ValueError:
        pass


def test_introspect_sqlite():
    schema = introspect_schema(_orders_db())
    assert "CREATE TABLE orders" in schema and "status" in schema


def test_introspect_ansi_information_schema():
    # DuckDB exercises the ANSI information_schema path (SQLite above covers the PRAGMA fallback)
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE emp (id INTEGER, \"Hire Date\" DATE)")
    schema = introspect_schema(con)
    assert "CREATE TABLE emp" in schema
    assert '"Hire Date"' in schema      # needs-quoting identifiers are rendered quoted in the DDL
    assert detect_dialect(con) == "duckdb"


def test_dialect_detect_and_transpile():
    assert detect_dialect(_orders_db()) == "sqlite"
    # DuckDB STRFTIME(date, fmt) has the args in the OPPOSITE order from SQLite STRFTIME(fmt, date)
    duck = "SELECT STRFTIME(d, '%Y') FROM t"
    assert transpile_sql(duck, "duckdb") == duck                  # no-op when target == model dialect
    assert transpile_sql(duck, "sqlite") == "SELECT STRFTIME('%Y', d) FROM t"


def test_transpile_makes_duckdb_sql_run_on_sqlite():
    con = sqlite3.connect(":memory:")
    con.executescript("CREATE TABLE t (d TEXT); INSERT INTO t VALUES ('2021-05-01'),('2022-06-01');")
    ex = make_executor(con)
    # DuckDB arg order would give wrong results on SQLite; transpiled, it's correct
    cols, rows, trunc, err = run_readonly(ex, "SELECT STRFTIME(d, '%Y') AS y FROM t ORDER BY y", dialect="sqlite")
    assert err is None and [r[0] for r in rows] == ["2021", "2022"], (rows, err)


# ---------------------------------------------------------------- agent loop (fake backend)
class FakeBackend:
    def __init__(self, scripted):
        self.scripted, self.i = scripted, 0

    def generate(self, prompt, max_new_tokens=256):
        out = self.scripted[min(self.i, len(self.scripted) - 1)]
        self.i += 1
        return out


def test_full_loop_gather_then_final():
    cleo = Cleo(FakeBackend([
        '{"tool":"gather","sql":"SELECT DISTINCT status FROM orders"}',
        '{"tool":"final","sql":"SELECT region, COUNT(*) AS n FROM orders WHERE status=\'C\' GROUP BY region"}',
    ]))
    ans = cleo.ask("completed orders by region", _orders_db())
    assert ans.ok and ans.status == "ok"
    assert sorted(ans.rows) == [["EU", 1], ["US", 1]]
    assert len(ans.gathers) == 1 and "C" in ans.discovered


def test_over_budget_gather_is_not_hijacked_as_final():
    # a model that ONLY ever gathers must end in no_answer, never have its probe returned as the answer
    cleo = Cleo(FakeBackend(['{"tool":"gather","sql":"SELECT DISTINCT status FROM orders"}']),
                default_max_gather=2)
    ans = cleo.ask("q", _orders_db())
    assert ans.status == "error" and ans.error == "no_answer"
    assert ans.sql is None


def test_clarify():
    cleo = Cleo(FakeBackend(['{"tool":"final","clarify":"which metric?"}']))
    ans = cleo.ask("ambiguous", _orders_db())
    assert ans.status == "clarify" and ans.clarification == "which metric?" and not ans.ok


def test_self_repair_recovers_a_failed_final():
    # first final references a non-existent column (exec error); after the error is surfaced, it fixes it
    cleo = Cleo(FakeBackend([
        '{"tool":"final","sql":"SELECT nope FROM orders"}',
        '{"tool":"final","sql":"SELECT region FROM orders"}',
    ]))
    ans = cleo.ask("q", _orders_db(), max_gather=0, max_repair=2)
    assert ans.ok and ans.sql == "SELECT region FROM orders", (ans.sql, ans.error)


def test_self_repair_gives_up_after_budget():
    cleo = Cleo(FakeBackend(['{"tool":"final","sql":"SELECT nope FROM orders"}']))
    ans = cleo.ask("q", _orders_db(), max_gather=0, max_repair=1)
    assert ans.status == "error" and not ans.ok


def test_schema_enrichment_fks_and_samples():
    con = sqlite3.connect(":memory:")
    con.executescript(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);"
        "CREATE TABLE orders2 (id INTEGER, customer_id INTEGER, FOREIGN KEY(customer_id) REFERENCES customers(id));"
        "INSERT INTO customers VALUES (1,'A'); INSERT INTO orders2 VALUES (10,1);")
    schema = introspect_schema(con, fks=True, samples=2)
    assert "orders2.customer_id -> customers.id" in schema
    assert "examples:" in schema


# ---------------------------------------------------------------- weights resolution / CLI
def test_from_gguf_defaults_to_hub_download():
    from cleo import backends
    orig_dl, orig_be = backends.download_gguf, backends.GGUFBackend
    backends.download_gguf = lambda: "/fake/champion.gguf"
    backends.GGUFBackend = lambda path, **kw: ("backend", path)
    try:
        cleo = Cleo.from_gguf()                       # no path -> resolved from the hub
        assert cleo.backend == ("backend", "/fake/champion.gguf")
        cleo = Cleo.from_gguf("/local/m.gguf")        # explicit path -> no download
        assert cleo.backend == ("backend", "/local/m.gguf")
    finally:
        backends.download_gguf, backends.GGUFBackend = orig_dl, orig_be


def test_cli_connect():
    import tempfile, os
    from cleo.cli import connect
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.sqlite")
        sqlite3.connect(p).close()
        con = connect(p)
        assert hasattr(con, "cursor")
        con.close()
        try:                                          # missing file must NOT be silently created
            connect(os.path.join(d, "nope.sqlite"))
            assert False, "should exit on missing db file"
        except SystemExit:
            pass


# ---------------------------------------------------------------- Answer semantics (DX)
def test_answer_truthiness():
    assert bool(Answer(sql="SELECT 1"))
    assert not bool(Answer(error="boom"))          # errors are falsy AND .status=="error"
    assert Answer(error="boom").status == "error"
    assert not bool(Answer())                       # empty


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
