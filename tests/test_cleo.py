"""Tests for the cleo package. Run: `pytest` (or `python tests/test_cleo.py`).

No model or GPU needed. `Cleo(backend=FakeBackend(...))` injects scripted model output.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleo import Answer, Cleo
from cleo.contract import (INSTRUCTION, TERMINAL_CONTRACT_OBSERVATION, format_many_observation,
                           format_observation, is_readonly, normalize_action, parse_action, render_prompt,
                           typed_repair_candidate)
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
    assert normalize_action({"tool": "gather_many", "queries": [{"sql": "SELECT 1"}]}) == {
        "kind": "gather_many", "queries": [{"sql": "SELECT 1"}]}
    assert normalize_action({"sql": "SELECT 1"}) == {"kind": "final", "sql": "SELECT 1"}  # bare contract
    assert normalize_action({"tool": "foo", "sql": "SELECT 1"}) is None                   # fidelity guard
    assert normalize_action({"tool": "gather", "clarify": "x"}) is None
    assert normalize_action({"tool": "gather_many", "queries": [{"not_sql": "SELECT 1"}]}) is None


def test_render_prompt_budget_nudge():
    p = render_prompt("CREATE TABLE t(x int);", "q?", [], gathers_left=0, steps_left=1)
    assert 'You must now return a {"tool":"final",...} action.' in p
    assert render_prompt("s", "q", [], gathers_left=3, steps_left=4).endswith("Action:\n")


def test_format_observation():
    assert format_observation(["status"], [["O"], ["C"]], False) == 'cols=[\'status\'] rows(2)=[["O"], ["C"]]'
    obs = format_many_observation([
        {"index": 1, "cols": ["status"], "rows": [["O"]], "truncated": False},
        {"index": 2, "error": "boom"},
    ])
    assert '"i": 1' in obs and '"cols": ["status"]' in obs and '"error": "boom"' in obs


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


def _patients_db():
    con = sqlite3.connect(":memory:")
    con.executescript(
        "CREATE TABLE Patient (ID INTEGER, Birthday TEXT);"
        "CREATE TABLE Examination (ID INTEGER, Symptoms TEXT);"
        "INSERT INTO Patient VALUES (1,'2000-01-01'),(2,'2001-01-01');"
        "INSERT INTO Examination VALUES (1,'pain'),(2,NULL);"
    )
    return con


def _thrombosis_db():
    con = sqlite3.connect(":memory:")
    con.executescript(
        "CREATE TABLE Patient (ID INTEGER, Birthday TEXT);"
        "CREATE TABLE Examination (ID INTEGER, RVVT TEXT);"
        "INSERT INTO Patient VALUES (1,'2000-01-01'),(2,'2010-01-01');"
        "INSERT INTO Examination VALUES (1,'+'),(2,'-');"
    )
    return con


def _products_db():
    con = sqlite3.connect(":memory:")
    con.executescript(
        "CREATE TABLE products (id INTEGER, current_flag TEXT, sku TEXT);"
        "INSERT INTO products VALUES (1,'is_current','A'),(2,'is_current','B'),(3,'archived','C');"
    )
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
        self.prompts = []
        self.kwargs = []

    def generate(self, prompt, max_new_tokens=256, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
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


def test_full_loop_gather_many_then_final():
    cleo = Cleo(FakeBackend([
        '{"tool":"gather_many","queries":[{"sql":"SELECT DISTINCT status FROM orders"},{"sql":"SELECT DISTINCT region FROM orders"}]}',
        '{"tool":"final","sql":"SELECT region, COUNT(*) AS n FROM orders WHERE status=\'C\' GROUP BY region"}',
    ]))
    ans = cleo.ask("completed orders by region", _orders_db(), enable_gather_many=True)
    assert ans.ok and sorted(ans.rows) == [["EU", 1], ["US", 1]]
    assert len(ans.gathers) == 2 and {"C", "O", "EU", "US"}.issubset(set(ans.discovered))


def test_gather_many_disabled_by_default():
    cleo = Cleo(FakeBackend([
        '{"tool":"gather_many","queries":[{"sql":"SELECT DISTINCT status FROM orders"}]}',
        '{"tool":"final","sql":"SELECT COUNT(*) FROM orders"}',
    ]))
    ans = cleo.ask("q", _orders_db())
    assert ans.ok and ans.rows == [[3]]
    assert ans.gathers == []


def test_over_budget_gather_is_not_hijacked_as_final():
    # a model that ONLY ever gathers must end in no_answer, never have its probe returned as the answer
    cleo = Cleo(FakeBackend(['{"tool":"gather","sql":"SELECT DISTINCT status FROM orders"}']),
                default_max_gather=2)
    ans = cleo.ask("q", _orders_db())
    assert ans.status == "error" and ans.error == "no_answer"
    assert ans.sql is None


def test_terminal_contract_sentinel_default_off_no_answer_unchanged():
    cleo = Cleo(FakeBackend(['{"tool":"gather","sql":"SELECT DISTINCT status FROM orders"}']),
                default_max_gather=0)
    ans = cleo.ask("q", _orders_db(), max_repair=0)
    assert ans.status == "error"
    assert ans.error == "no_answer"
    assert ans.sql is None
    assert not ans.terminal_contract_sentinel_fired


def test_terminal_contract_sentinel_rescues_no_answer_to_valid_sql():
    backend = FakeBackend([
        '{"tool":"gather","sql":"SELECT DISTINCT status FROM orders"}',
        '{"tool":"final","sql":"SELECT COUNT(*) FROM orders"}',
    ])
    cleo = Cleo(backend, default_max_gather=0)
    ans = cleo.ask("q", _orders_db(), max_repair=0, terminal_contract_sentinel=True)
    assert ans.ok
    assert ans.rows == [[3]]
    assert ans.terminal_contract_sentinel_enabled
    assert ans.terminal_contract_sentinel_fired
    assert ans.terminal_contract_sentinel_observation_appended
    assert ans.post_sentinel_generation_count == 1
    assert TERMINAL_CONTRACT_OBSERVATION in backend.prompts[1]


def test_terminal_contract_sentinel_does_not_fire_on_normal_final():
    cleo = Cleo(FakeBackend(['{"tool":"final","sql":"SELECT COUNT(*) FROM orders"}']))
    ans = cleo.ask("q", _orders_db(), max_gather=0, max_repair=0, terminal_contract_sentinel=True)
    assert ans.ok
    assert ans.rows == [[3]]
    assert ans.terminal_contract_sentinel_enabled
    assert not ans.terminal_contract_sentinel_fired


def test_terminal_contract_sentinel_rejects_post_sentinel_gather_without_execution():
    cleo = Cleo(FakeBackend([
        '{"tool":"gather","sql":"SELECT DISTINCT status FROM orders"}',
        '{"tool":"gather","sql":"SELECT DISTINCT region FROM orders"}',
    ]), default_max_gather=0)
    ans = cleo.ask("q", _orders_db(), max_repair=0, terminal_contract_sentinel=True)
    assert ans.status == "error"
    assert ans.error == "no_answer"
    assert ans.gathers == [("SELECT DISTINCT status FROM orders", None, None)]
    assert ans.post_sentinel_gather_rejected


def test_terminal_contract_sentinel_blocks_post_sentinel_unsafe_sql():
    cleo = Cleo(FakeBackend([
        '{"tool":"gather","sql":"SELECT DISTINCT status FROM orders"}',
        '{"tool":"final","sql":"DELETE FROM orders"}',
    ]), default_max_gather=0)
    con = _orders_db()
    ans = cleo.ask("q", con, max_repair=0, terminal_contract_sentinel=True)
    assert ans.status == "error"
    assert ans.error == "no_answer"
    assert ans.sql is None
    assert ans.post_sentinel_sql_blocked
    assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3


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


def test_typed_repair_candidate_rewrites_only_where_owner():
    schema = "CREATE TABLE Patient (ID INTEGER, Birthday TEXT); CREATE TABLE Examination (ID INTEGER, Symptoms TEXT);"
    sql = (
        "SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID "
        "WHERE T1.Symptoms IS NOT NULL"
    )
    repaired = typed_repair_candidate(schema, sql, "no such column: T1.Symptoms")
    assert repaired is not None
    assert "T2.Symptoms IS NOT NULL" in repaired["sql"]


def test_typed_repair_candidate_refuses_projection_rewrite():
    schema = "CREATE TABLE Patient (ID INTEGER, Birthday TEXT); CREATE TABLE Examination (ID INTEGER, Symptoms TEXT);"
    sql = "SELECT T1.Symptoms FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID"
    assert typed_repair_candidate(schema, sql, "no such column: T1.Symptoms") is None


def test_typed_repair_candidate_derives_projection_age_from_birthday():
    schema = "CREATE TABLE Patient (ID INTEGER, Birthday TEXT); CREATE TABLE Examination (ID INTEGER, RVVT TEXT);"
    sql = (
        "SELECT T1.ID, T2.age FROM Examination AS T1 INNER JOIN Patient AS T2 ON T1.ID = T2.ID "
        "WHERE T1.RVVT = '+' AND T2.age = (SELECT CAST(SUBSTR(T2.Birthday, 1, 4) AS INTEGER) "
        "- CAST(SUBSTR(SUBSTR(T2.Birthday, 5, 2), 1, 4) AS INTEGER) FROM Patient AS T2)"
    )
    repaired = typed_repair_candidate(
        schema,
        sql,
        "no such column: T2.age",
        "State the ID and age of patient with positive degree of coagulation. "
        "Hint: age refers to SUBTRACT(year(current_timestamp), year(Birthday));",
    )

    assert repaired is not None
    assert repaired["provenance"] == "typed_repair_controller:typed_age_from_birthday"
    assert "T2.age" not in repaired["sql"]
    assert "STRFTIME(CURRENT_DATE, '%Y')" in repaired["sql"]
    assert "AND" not in repaired["sql"].split("WHERE", 1)[1]


def test_typed_repair_candidate_refuses_age_without_birthday_context():
    schema = "CREATE TABLE Patient (ID INTEGER, Birthday TEXT); CREATE TABLE Examination (ID INTEGER, RVVT TEXT);"
    sql = "SELECT T1.ID, T2.age FROM Examination AS T1 INNER JOIN Patient AS T2 ON T1.ID = T2.ID"
    assert typed_repair_candidate(schema, sql, "no such column: T2.age", "State the patient age.") is None


def test_typed_repair_controller_returns_safe_rewrite_without_model_retry():
    backend = FakeBackend([
        '{"tool":"final","sql":"SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Examination AS T2 '
        'ON T1.ID = T2.ID WHERE T1.Symptoms IS NOT NULL"}',
        '{"tool":"final","sql":"SELECT 999"}',
    ])
    cleo = Cleo(backend)
    schema = "CREATE TABLE Patient (ID INTEGER, Birthday TEXT); CREATE TABLE Examination (ID INTEGER, Symptoms TEXT);"
    ans = cleo.ask(
        "How many patients have symptoms?",
        _patients_db(),
        schema=schema,
        max_gather=0,
        max_repair=1,
        typed_repair_controller=True,
    )
    assert ans.ok
    assert ans.rows == [[1]]
    assert "T2.Symptoms IS NOT NULL" in ans.sql
    assert len(backend.prompts) == 1


def test_typed_repair_controller_returns_age_from_birthday_without_model_retry():
    backend = FakeBackend([
        '{"tool":"final","sql":"SELECT T1.ID, T2.age FROM Examination AS T1 INNER JOIN Patient AS T2 '
        "ON T1.ID = T2.ID WHERE T1.RVVT = '+' AND T2.age = (SELECT CAST(SUBSTR(T2.Birthday, 1, 4) AS INTEGER) "
        "- CAST(SUBSTR(SUBSTR(T2.Birthday, 5, 2), 1, 4) AS INTEGER) FROM Patient AS T2)\"}",
        '{"tool":"final","sql":"SELECT 999"}',
    ])
    cleo = Cleo(backend)
    schema = "CREATE TABLE Patient (ID INTEGER, Birthday TEXT); CREATE TABLE Examination (ID INTEGER, RVVT TEXT);"
    ans = cleo.ask(
        "State the ID and age of patient with positive degree of coagulation. "
        "Hint: age refers to SUBTRACT(year(current_timestamp), year(Birthday));",
        _thrombosis_db(),
        schema=schema,
        max_gather=0,
        max_repair=1,
        typed_repair_controller=True,
    )
    expected_age = sqlite3.connect(":memory:").execute("SELECT CAST(STRFTIME('%Y', 'now') AS INTEGER) - 2000").fetchone()[0]
    assert ans.ok
    assert ans.rows == [[1, expected_age]]
    assert "T2.age" not in ans.sql
    assert len(backend.prompts) == 1


def test_verifier_repair_context_is_opt_in():
    generic_backend = FakeBackend([
        '{"tool":"final","sql":"SELECT nope FROM orders"}',
        '{"tool":"final","sql":"SELECT region FROM orders"}',
    ])
    generic = Cleo(generic_backend)
    generic.ask("q", _orders_db(), max_gather=0, max_repair=1)
    assert "REPAIR_CONTEXT" not in generic_backend.prompts[1]

    enriched_backend = FakeBackend([
        '{"tool":"final","sql":"SELECT nope FROM orders"}',
        '{"tool":"final","sql":"SELECT region FROM orders"}',
    ])
    enriched = Cleo(enriched_backend)
    enriched.ask("q", _orders_db(), max_gather=0, max_repair=1, verifier_repair_context=True)
    assert "REPAIR_CONTEXT" in enriched_backend.prompts[1]
    assert '"missing_column": "nope"' in enriched_backend.prompts[1]


def test_hardel_selects_evidence_backed_sample():
    backend = FakeBackend([
        '{"tool":"final","sql":"SELECT COUNT(*) FROM products WHERE current_flag=\'current\'"}',
        '{"tool":"gather","sql":"SELECT DISTINCT current_flag FROM products"}',
        '{"tool":"final","sql":"SELECT COUNT(*) FROM products WHERE current_flag=\'is_current\'"}',
    ])
    cleo = Cleo(backend)
    ans = cleo.ask_hardel("How many products are current?", _products_db(), k=1, seed=12)

    assert ans.ok
    assert ans.sql == "SELECT COUNT(*) FROM products WHERE current_flag='is_current'"
    assert ans.rows == [[2]]
    assert ans.selector == "evidence_runtime"
    assert ans.evidence_override
    assert ans.candidate_id == "sample_1"
    assert len(ans.candidates) == 2
    assert any("is_current" in reason for reason in ans.evidence_override_reasons)
    assert backend.kwargs[0]["sample"] is False
    assert backend.kwargs[1]["sample"] is True


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
