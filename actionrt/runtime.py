"""Executable AMALGAM action runtime."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import duckdb

from .contract import ACTION_CONTRACT, ActionContract, ActionValidationError, action_contract_gbnf, parse_and_validate_action
from .exemplars import load_exemplar_bank, retrieve_structural_exemplars
from .fallback import build_fixed_mode_fallback


ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = ROOT / "tasks" / "micro_sql_agent_harness"
if HARNESS_ROOT.exists() and str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from micro_sql_agent_harness.analytic_tools import quote_ident, run_tool, sample_rows, schema_catalog, table_profile  # noqa: E402
from micro_sql_agent_harness.sql_parse import validate_safe_sql  # noqa: E402


DEFAULT_MODEL_PATH = (
    "/home/wdree/percy/cothogonal/local_gguf/amalgam_opd2b_20260605/"
    "amalgam_opd2b_20260605-no_mtp-Q4_K_M.gguf"
)


@dataclass(frozen=True)
class Generation:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: Any = None


class ActionBackend(Protocol):
    name: str

    def generate(self, prompt: str, *, grammar: str, max_tokens: int, temperature: float) -> Generation:
        ...


@dataclass
class RuntimeConfig:
    max_steps: int = 6
    max_tokens: int = 256
    temperature: float = 0.0
    row_limit: int = 100
    continue_after_sql: bool = False
    out_dir: str | Path | None = None
    exemplar_bank_path: str | Path | None = None
    db_id: str | None = None
    fixed_mode_fallback: bool = False
    fixed_mode_max_chars: int = 3600


@dataclass
class RuntimeStep:
    index: int
    prompt_chars: int
    raw_text: str
    latency_s: float
    valid_action: bool
    action: dict[str, Any] | None = None
    validation_error: str | None = None
    observation: dict[str, Any] | None = None
    terminal: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    trace_ref: str | None = None
    fallback_policy: str | None = None


@dataclass
class RuntimeResult:
    question: str
    db_path: str
    backend: str
    contract_version: str
    terminal_action: dict[str, Any] | None
    steps: list[RuntimeStep]
    parser_crash: bool = False
    fixed_mode_fallback_used: bool = False
    trace_ledger: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: _dt.datetime.now(tz=_dt.timezone.utc).isoformat())

    @property
    def action_count(self) -> int:
        return len(self.steps)

    @property
    def valid_action_count(self) -> int:
        return sum(1 for step in self.steps if step.valid_action)

    @property
    def action_validity_rate(self) -> float:
        return self.valid_action_count / self.action_count if self.action_count else 0.0

    @property
    def total_latency_s(self) -> float:
        return sum(step.latency_s for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "db_path": self.db_path,
            "backend": self.backend,
            "contract_version": self.contract_version,
            "started_at": self.started_at,
            "terminal_action": self.terminal_action,
            "parser_crash": self.parser_crash,
            "fixed_mode_fallback_used": self.fixed_mode_fallback_used,
            "action_count": self.action_count,
            "valid_action_count": self.valid_action_count,
            "action_validity_rate": self.action_validity_rate,
            "total_latency_s": round(self.total_latency_s, 6),
            "steps": [
                {
                    "index": step.index,
                    "prompt_chars": step.prompt_chars,
                    "raw_text": step.raw_text,
                    "latency_s": round(step.latency_s, 6),
                    "valid_action": step.valid_action,
                    "action": step.action,
                    "validation_error": step.validation_error,
                    "observation": step.observation,
                    "terminal": step.terminal,
                    "prompt_tokens": step.prompt_tokens,
                    "completion_tokens": step.completion_tokens,
                    "trace_ref": step.trace_ref,
                    "fallback_policy": step.fallback_policy,
                }
                for step in self.steps
            ],
            "trace_ledger": self.trace_ledger,
        }


class ScriptedActionBackend:
    """Deterministic backend that emits a fixed action sequence."""

    name = "scripted"

    def __init__(self, actions: list[dict[str, Any] | str]) -> None:
        self.actions = list(actions)
        self.index = 0

    def generate(self, prompt: str, *, grammar: str, max_tokens: int, temperature: float) -> Generation:  # noqa: ARG002
        started = time.perf_counter()
        if self.index >= len(self.actions):
            action: dict[str, Any] | str = {
                "type": "final_answer",
                "answer": "No additional scripted action is available.",
                "confidence": 0.0,
            }
        else:
            action = self.actions[self.index]
        self.index += 1
        text = action if isinstance(action, str) else json.dumps(action, separators=(",", ":"))
        return Generation(text=text, latency_s=time.perf_counter() - started)


class LlamaCppActionBackend:
    """CPU-only llama.cpp/GGUF backend using llama-cpp-python grammars."""

    name = "llama-cpp"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        n_ctx: int = 8192,
        n_threads: int = 8,
        n_batch: int = 128,
        n_gpu_layers: int = 0,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.verbose = verbose
        self._llm: Any | None = None
        self._grammar_cache: dict[str, Any] = {}

    def _load(self) -> Any:
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except Exception as exc:
                raise RuntimeError(
                    "llama-cpp-python is required for LlamaCppActionBackend. "
                    "Install it or run with a Python environment that exposes llama_cpp."
                ) from exc
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                n_gpu_layers=self.n_gpu_layers,
                seed=self.seed,
                verbose=self.verbose,
            )
        return self._llm

    def _grammar(self, grammar: str) -> Any:
        cached = self._grammar_cache.get(grammar)
        if cached is not None:
            return cached
        try:
            from llama_cpp import LlamaGrammar
        except Exception as exc:
            raise RuntimeError("llama_cpp.LlamaGrammar is required for grammar-constrained decoding") from exc
        compiled = LlamaGrammar.from_string(grammar, verbose=self.verbose)
        self._grammar_cache[grammar] = compiled
        return compiled

    def generate(self, prompt: str, *, grammar: str, max_tokens: int, temperature: float) -> Generation:
        llm = self._load()
        compiled_grammar = self._grammar(grammar)
        started = time.perf_counter()
        raw = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            grammar=compiled_grammar,
            echo=False,
        )
        latency_s = time.perf_counter() - started
        choices = raw.get("choices") if isinstance(raw, dict) else None
        text = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            text = str(choices[0].get("text") or "")
        usage = raw.get("usage") if isinstance(raw, dict) else {}
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        return Generation(
            text=text,
            latency_s=latency_s,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            raw=raw,
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return round(value, 8)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _db_fingerprint(db_path: str | Path) -> str:
    path = Path(db_path)
    try:
        stat = path.stat()
        payload = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        payload = {"path": str(path)}
    return _stable_hash(payload)[:16]


def _safe_key(key: str) -> Path:
    parts = [part for part in str(key).split("/") if part not in {"", ".", ".."}]
    cleaned = [re.sub(r"[^A-Za-z0-9_.-]+", "_", part)[:120] or "_" for part in parts]
    return Path(*cleaned) if cleaned else Path("_")


class TraceLedger:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) / "ledger" if root is not None else None
        self.entries: list[dict[str, Any]] = []
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def add(self, *, kind: str, action: dict[str, Any] | None, payload: dict[str, Any]) -> str:
        ref = f"ledger/{len(self.entries):04d}"
        entry = {
            "ref": ref,
            "kind": kind,
            "action": action,
            "payload_hash": _stable_hash(payload),
            "created_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        }
        self.entries.append(entry)
        if self.root is not None:
            target = self.root / f"{len(self.entries) - 1:04d}.json"
            target.write_text(
                json.dumps({"entry": entry, "payload": payload}, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        return ref


class ScratchStore:
    SCOPES = {"turn", "session", "db", "global_eval"}

    def __init__(self, root: str | Path | None, *, db_path: str | Path, catalog_hash: str | None) -> None:
        base = Path(root) if root is not None else Path(os.environ.get("ACTIONRT_SCRATCH", "/tmp/actionrt_scratch"))
        self.root = base / "scratch"
        self.db_path = str(db_path)
        self.db_fingerprint = _db_fingerprint(db_path)
        self.catalog_hash = catalog_hash
        self.root.mkdir(parents=True, exist_ok=True)

    def _scope_root(self, scope: str) -> Path:
        normalized = scope if scope in self.SCOPES else "turn"
        if normalized == "db":
            return self.root / "db" / self.db_fingerprint / str(self.catalog_hash or "no_catalog")
        return self.root / normalized

    def _path(self, key: str, scope: str) -> Path:
        return self._scope_root(scope) / Path(str(_safe_key(key)) + ".json")

    def store(
        self,
        *,
        key: str,
        value: Any,
        scope: str = "turn",
        ttl_turns: int | None = None,
        visibility: str = "model_readable",
        overwrite: bool = True,
        catalog_hash: str | None = None,
        profile_hash: str | None = None,
        allow_long_lived_raw: bool = False,
    ) -> dict[str, Any]:
        normalized_scope = scope if scope in self.SCOPES else "turn"
        if normalized_scope in {"db", "global_eval"} and not allow_long_lived_raw:
            text = json.dumps(value, ensure_ascii=False, default=str)
            if "rows" in text.lower() and len(text) > 2048:
                return {
                    "ok": False,
                    "type": "scratch_observation",
                    "tool": "file_store",
                    "error_class": "long_lived_raw_forbidden",
                    "errors": ["raw-looking large values require allow_long_lived_raw=true outside turn/session scope"],
                }
        path = self._path(key, normalized_scope)
        if path.exists() and not overwrite:
            return {
                "ok": False,
                "type": "scratch_observation",
                "tool": "file_store",
                "error_class": "already_exists",
                "errors": [f"scratch key already exists: {key}"],
            }
        version = 1
        if path.exists():
            try:
                version = int(json.loads(path.read_text(encoding="utf-8")).get("metadata", {}).get("version", 0)) + 1
            except Exception:
                version = 1
        metadata = {
            "key": key,
            "scope": normalized_scope,
            "version": version,
            "visibility": visibility,
            "ttl_turns": ttl_turns,
            "catalog_hash": catalog_hash if catalog_hash is not None else self.catalog_hash,
            "profile_hash": profile_hash,
            "db_fingerprint": self.db_fingerprint if normalized_scope == "db" else None,
            "value_hash": _stable_hash(value),
            "size_bytes": len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")),
            "created_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"metadata": metadata, "value": value}, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return {"ok": True, "type": "scratch_observation", "tool": "file_store", **metadata}

    def read(
        self,
        *,
        key: str,
        scope: str = "turn",
        max_bytes: int = 4096,
        version: str | None = None,
        catalog_hash: str | None = None,
        profile_hash: str | None = None,
    ) -> dict[str, Any]:
        normalized_scope = scope if scope in self.SCOPES else "turn"
        path = self._path(key, normalized_scope)
        if not path.exists():
            return {"ok": True, "type": "scratch_observation", "tool": "file_read", "key": key, "scope": normalized_scope, "miss": True}
        obj = json.loads(path.read_text(encoding="utf-8"))
        metadata = dict(obj.get("metadata") or {})
        stale_reasons: list[str] = []
        expected_catalog = catalog_hash if catalog_hash is not None else (self.catalog_hash if normalized_scope == "db" else None)
        if expected_catalog and metadata.get("catalog_hash") and metadata.get("catalog_hash") != expected_catalog:
            stale_reasons.append("catalog_hash_mismatch")
        if profile_hash and metadata.get("profile_hash") and metadata.get("profile_hash") != profile_hash:
            stale_reasons.append("profile_hash_mismatch")
        if version is not None and str(metadata.get("version")) != str(version):
            stale_reasons.append("version_mismatch")
        value = obj.get("value")
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        truncated = len(encoded.encode("utf-8")) > max_bytes
        if truncated:
            value = encoded[:max(0, max_bytes)]
        return {
            "ok": True,
            "type": "scratch_observation",
            "tool": "file_read",
            "key": key,
            "scope": normalized_scope,
            "miss": False,
            "stale": bool(stale_reasons),
            "stale_reasons": stale_reasons,
            "metadata": metadata,
            "value": value,
            "truncated": truncated,
        }

    def list(self, *, prefix: str = "", scope: str = "turn", include_values: bool = False, limit: int = 50) -> dict[str, Any]:
        normalized_scope = scope if scope in self.SCOPES else "turn"
        root = self._scope_root(normalized_scope)
        rows: list[dict[str, Any]] = []
        if root.exists():
            for path in sorted(root.rglob("*.json")):
                obj = json.loads(path.read_text(encoding="utf-8"))
                metadata = dict(obj.get("metadata") or {})
                key = str(metadata.get("key") or "")
                if prefix and not key.startswith(prefix):
                    continue
                row = {"key": key, "metadata": metadata}
                if include_values:
                    row["value"] = obj.get("value")
                rows.append(row)
                if len(rows) >= limit:
                    break
        return {
            "ok": True,
            "type": "scratch_observation",
            "tool": "file_list",
            "scope": normalized_scope,
            "prefix": prefix,
            "items": rows,
            "truncated": len(rows) >= limit,
        }


def execute_readonly_sql(db_path: str | Path, sql: str, *, row_limit: int, timeout_ms: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    validate_safe_sql(sql)
    con = duckdb.connect(str(db_path), config={"access_mode": "READ_ONLY", "enable_external_access": "false"})
    try:
        cur = con.execute(f"SELECT * FROM ({sql.strip().rstrip(';')}) AS actionrt_sql LIMIT {int(row_limit) + 1}")
        columns = [str(desc[0]) for desc in cur.description or []]
        rows = cur.fetchall()
        visible = rows[:row_limit]
        return {
            "ok": True,
            "type": "sql_observation",
            "columns": columns,
            "rows": [[_jsonable(cell) for cell in row] for row in visible],
            "row_count": len(visible),
            "truncated": len(rows) > row_limit,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "timeout_ms": timeout_ms,
            "sql_hash": _stable_hash(sql)[:16],
        }
    except Exception as exc:
        return {
            "ok": False,
            "type": "sql_observation",
            "error_class": type(exc).__name__,
            "errors": [str(exc)[:300]],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "timeout_ms": timeout_ms,
            "sql_hash": _stable_hash(sql)[:16],
        }
    finally:
        con.close()


def compact_schema_context(db_path: str | Path) -> str:
    catalog = schema_catalog(db_path, include_columns=True)
    if not catalog.get("ok"):
        return json.dumps({"schema_catalog_error": catalog}, ensure_ascii=False, sort_keys=True)
    tables = []
    for table in catalog.get("tables") or []:
        columns = [
            {"name": column.get("name"), "type": column.get("data_type")}
            for column in table.get("columns") or []
        ]
        tables.append({"name": table.get("name"), "row_count": table.get("row_count"), "columns": columns})
    return json.dumps(
        {
            "tables": tables,
            "relationships": catalog.get("relationships") or [],
            "catalog_hash": catalog.get("catalog_hash"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def compact_runtime_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Trim observations before feeding them back to the model."""

    if observation.get("type") == "sql_observation":
        return {
            "ok": observation.get("ok"),
            "type": "sql_observation",
            "columns": observation.get("columns") or [],
            "rows": (observation.get("rows") or [])[:5],
            "row_count": observation.get("row_count"),
            "truncated": observation.get("truncated"),
            "error_class": observation.get("error_class"),
            "errors": observation.get("errors"),
            "elapsed_ms": observation.get("elapsed_ms"),
            "sql_hash": observation.get("sql_hash"),
            "trace_ref": observation.get("trace_ref"),
        }
    if observation.get("type") == "exemplar_observation":
        return {
            "ok": observation.get("ok"),
            "type": "exemplar_observation",
            "tool": "retrieve_exemplars",
            "db_id": observation.get("db_id"),
            "selection": observation.get("selection"),
            "exemplars": [
                {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "gold_sql": item.get("gold_sql"),
                    "similarity_score": item.get("similarity_score"),
                    "retrieval": item.get("retrieval"),
                }
                for item in (observation.get("exemplars") or [])[:5]
            ],
            "candidate_count": observation.get("candidate_count"),
            "trace_ref": observation.get("trace_ref"),
        }
    if observation.get("type") == "scratch_observation":
        compact = {
            "ok": observation.get("ok"),
            "type": "scratch_observation",
            "tool": observation.get("tool"),
            "key": observation.get("key"),
            "scope": observation.get("scope"),
            "miss": observation.get("miss"),
            "stale": observation.get("stale"),
            "stale_reasons": observation.get("stale_reasons"),
            "version": observation.get("version") or (observation.get("metadata") or {}).get("version"),
            "hash": observation.get("value_hash") or (observation.get("metadata") or {}).get("value_hash"),
            "items": [
                {"key": item.get("key"), "metadata": item.get("metadata")}
                for item in (observation.get("items") or [])[:20]
            ],
            "value": observation.get("value"),
            "truncated": observation.get("truncated"),
            "trace_ref": observation.get("trace_ref"),
            "error_class": observation.get("error_class"),
            "errors": observation.get("errors"),
        }
        if compact["value"] is not None and len(json.dumps(compact["value"], default=str)) > 1000:
            compact["value"] = json.dumps(compact["value"], ensure_ascii=False, default=str)[:1000]
            compact["truncated"] = True
        return compact
    if observation.get("type") == "answerability_observation":
        return {
            "ok": observation.get("ok"),
            "type": "answerability_observation",
            "answerable": observation.get("answerable"),
            "score": observation.get("score"),
            "missing_slots": observation.get("missing_slots"),
            "recommendation": observation.get("recommendation"),
            "trace_ref": observation.get("trace_ref"),
        }
    if observation.get("type") == "clarification_observation":
        return {
            "ok": observation.get("ok"),
            "type": "clarification_observation",
            "question": observation.get("question"),
            "missing_slots": observation.get("missing_slots"),
            "trace_ref": observation.get("trace_ref"),
        }
    if observation.get("type") == "fixed_mode_fallback":
        return {
            "ok": observation.get("ok"),
            "type": "fixed_mode_fallback",
            "fallback_policy": observation.get("fallback_policy"),
            "digest_chars": observation.get("digest_chars"),
            "truncated": observation.get("truncated"),
            "catalog_hash": observation.get("catalog_hash"),
            "exemplars": [
                {"id": item.get("id"), "question": item.get("question"), "gold_sql": item.get("gold_sql")}
                for item in (observation.get("exemplars") or [])[:2]
            ],
            "trace_ref": observation.get("trace_ref"),
        }
    if observation.get("type") != "tool_observation":
        return observation
    payload = observation.get("observation")
    if not isinstance(payload, dict):
        return observation
    tool = observation.get("tool") or payload.get("tool")
    compact: dict[str, Any] = {"ok": observation.get("ok"), "type": "tool_observation", "tool": tool}
    if payload.get("ok") is False:
        compact.update({"error_class": payload.get("error_class"), "errors": payload.get("errors")})
        return compact
    if tool in {"schema_catalog", "inspect_schema"}:
        compact["catalog_hash"] = payload.get("catalog_hash")
        compact["tables"] = [
            {
                "name": table.get("name"),
                "columns": [column.get("name") for column in table.get("columns") or []],
            }
            for table in payload.get("tables") or []
        ]
        compact["relationships"] = payload.get("relationships") or []
        compact["trace_ref"] = observation.get("trace_ref")
        return compact
    if tool in {"table_profile", "profile_columns"}:
        compact["table"] = payload.get("table")
        compact["row_count"] = payload.get("row_count")
        compact["profile_hash"] = payload.get("profile_hash")
        compact["columns"] = [
            {
                "name": column.get("name"),
                "data_type": column.get("data_type"),
                "null_pct": column.get("null_pct"),
                "distinct_count": column.get("distinct_count"),
                "top_values": column.get("top_values"),
            }
            for column in payload.get("columns") or []
        ]
        compact["trace_ref"] = observation.get("trace_ref")
        return compact
    if tool in {"sample_rows", "sample"}:
        compact["table"] = payload.get("table")
        compact["columns"] = payload.get("columns") or []
        compact["rows"] = (payload.get("rows") or [])[:5]
        compact["truncated"] = payload.get("truncated")
        compact["sample_hash"] = payload.get("sample_hash")
        compact["trace_ref"] = observation.get("trace_ref")
        return compact
    if tool == "find_join_paths":
        compact["start_table"] = payload.get("start_table")
        compact["end_table"] = payload.get("end_table")
        compact["paths"] = (payload.get("paths") or [])[:3]
        compact["path_count"] = payload.get("path_count")
        compact["trace_ref"] = observation.get("trace_ref")
        return compact
    compact["trace_ref"] = observation.get("trace_ref")
    return compact


class ActionRuntime:
    def __init__(
        self,
        backend: ActionBackend,
        *,
        contract: ActionContract = ACTION_CONTRACT,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.backend = backend
        self.contract = contract
        self.config = config or RuntimeConfig()
        self.grammar = action_contract_gbnf(contract)
        self.out_dir = Path(self.config.out_dir) if self.config.out_dir is not None else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.exemplar_bank = load_exemplar_bank(self.config.exemplar_bank_path)

    def _db_id(self, db_path: str | Path) -> str:
        return self.config.db_id or Path(db_path).stem

    def _catalog_hash(self, db_path: str | Path) -> str | None:
        catalog = schema_catalog(db_path, include_columns=True)
        value = catalog.get("catalog_hash")
        return str(value) if value else None

    def _scratch(self, db_path: str | Path) -> ScratchStore:
        return ScratchStore(self.out_dir, db_path=db_path, catalog_hash=self._catalog_hash(db_path))

    def _catalog_map(self, db_path: str | Path) -> dict[str, list[str]]:
        catalog = schema_catalog(db_path, include_columns=True)
        mapped: dict[str, list[str]] = {}
        for table in catalog.get("tables") or []:
            name = table.get("name")
            if not isinstance(name, str):
                continue
            mapped[name] = [str(column.get("name")) for column in table.get("columns") or [] if column.get("name")]
        return mapped

    def render_prompt(
        self,
        *,
        question: str,
        db_path: str | Path,
        schema_context: str | None,
        observations: list[dict[str, Any]],
    ) -> str:
        schema_block = schema_context if schema_context is not None else compact_schema_context(db_path)
        tool_names = ", ".join(tool.name for tool in self.contract.tools)
        lines = [
            "You are the AMALGAM in-process analytic action runtime.",
            "Return exactly one JSON object matching the canonical action contract. No markdown, no prose.",
            "Allowed action types: tool_call, sql, clarification, final_answer.",
            f"Available tools: {tool_names}.",
            "Use v1 primitives when grounding is needed: inspect_schema, retrieve_exemplars, profile_columns, sample, exec_sql, file_store/file_read/file_list, answerability_probe, ask_clarification.",
            "Do not repeat a tool_call when the same tool observation is already present; duplicate tool calls are rejected.",
            "If the schema context already lists the needed tables and columns, prefer exec_sql or sql over more inspection.",
            "Use sql only for a read-only DuckDB SELECT or WITH query.",
            "Use clarification when the request is ambiguous or unsafe.",
            "Use final_answer after observations when the user-facing answer is ready.",
            "Canonical examples:",
            '{"type":"tool_call","tool":"inspect_schema","args":{"include_columns":true,"include_relationships":true}}',
            '{"type":"tool_call","tool":"retrieve_exemplars","args":{"question":"total revenue by region","k":2,"schema_scope":"same_db"}}',
            '{"type":"tool_call","tool":"exec_sql","args":{"sql":"SELECT COUNT(*) AS rows FROM orders","purpose":"answer","row_limit":20}}',
            '{"type":"sql","sql":"SELECT COUNT(*) AS rows FROM orders"}',
            '{"type":"clarification","question":"What date range should recent cover?"}',
            '{"type":"final_answer","answer":"Returned 3 rows.","sql_id":"last_sql","chart_id":null,"confidence":0.8}',
            "",
            "Schema context:",
            schema_block,
            "",
            "User question:",
            question,
        ]
        if observations:
            lines.extend(
                [
                    "",
                    "Prior observations:",
                    json.dumps(observations, ensure_ascii=False, sort_keys=True),
                    "",
                    "Return the next action. Use these observations; do not repeat an already observed tool call.",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def _inspect_schema(self, db_path: str | Path, args: dict[str, Any]) -> dict[str, Any]:
        include_columns = bool(args.get("include_columns", True))
        catalog = schema_catalog(db_path, include_columns=include_columns)
        if not catalog.get("ok"):
            catalog["tool"] = "inspect_schema"
            return catalog
        requested_tables = set(args.get("tables") or [])
        max_tables = int(args.get("max_tables", 200))
        max_columns = int(args.get("max_columns_per_table", 200))
        tables = []
        for table in catalog.get("tables") or []:
            if requested_tables and table.get("name") not in requested_tables:
                continue
            item = dict(table)
            if isinstance(item.get("columns"), list):
                item["columns"] = item["columns"][:max_columns]
            tables.append(item)
            if len(tables) >= max_tables:
                break
        catalog.update(
            {
                "tool": "inspect_schema",
                "tables": tables,
                "table_count": len(tables),
                "relationships": catalog.get("relationships") if bool(args.get("include_relationships", True)) else [],
            }
        )
        return catalog

    def _profile_columns(self, db_path: str | Path, args: dict[str, Any]) -> dict[str, Any]:
        payload = table_profile(
            db_path,
            str(args["table"]),
            max_exact_distinct=int(args.get("max_exact_distinct", 10_000)),
        )
        payload["tool"] = "profile_columns"
        if not payload.get("ok"):
            return payload
        requested = set(args.get("columns") or [])
        top_limit = int(args.get("top_values_limit", 12))
        allow_values = bool(args.get("allow_value_examples", True))
        columns = []
        for column in payload.get("columns") or []:
            if requested and column.get("name") not in requested:
                continue
            item = dict(column)
            if not allow_values:
                item.pop("top_values", None)
            elif isinstance(item.get("top_values"), list):
                item["top_values"] = item["top_values"][:top_limit]
            columns.append(item)
        payload["columns"] = columns
        payload["column_count"] = len(columns)
        payload["catalog_hash"] = self._catalog_hash(db_path)
        payload["profile_hash"] = _stable_hash({"table": args["table"], "columns": columns})[:32]
        return payload

    def _sample(self, db_path: str | Path, args: dict[str, Any]) -> dict[str, Any]:
        method = str(args.get("method", "head"))
        if method != "head":
            return {"ok": False, "tool": "sample", "error_class": "unsupported_method", "errors": [f"unsupported sample method: {method}"]}
        where_sql = args.get("where_sql")
        if not where_sql:
            payload = sample_rows(
                db_path,
                str(args["table"]),
                limit=int(args.get("limit", 5)),
                columns=args.get("columns"),
            )
            payload["tool"] = "sample"
            return payload
        table = str(args["table"])
        columns = args.get("columns") or []
        projection = ", ".join(quote_ident(column) for column in columns) if columns else "*"
        sql = f"SELECT {projection} FROM {quote_ident(table)} WHERE {where_sql}"
        try:
            validate_safe_sql(sql)
            obs = execute_readonly_sql(db_path, sql, row_limit=int(args.get("limit", 5)))
            obs.update({"tool": "sample", "table": table, "columns": obs.get("columns") or columns, "sample_hash": _stable_hash(obs.get("rows") or [])[:32]})
            return obs
        except Exception as exc:
            return {"ok": False, "tool": "sample", "error_class": type(exc).__name__, "errors": [str(exc)[:300]]}

    def _answerability_probe(self, db_path: str | Path, args: dict[str, Any]) -> dict[str, Any]:
        catalog = self._catalog_map(db_path)
        known_tables = {table.lower() for table in catalog}
        known_columns = {column.lower() for columns in catalog.values() for column in columns}
        missing: list[str] = []
        for table in args.get("required_tables") or []:
            if str(table).lower() not in known_tables:
                missing.append(f"table:{table}")
        for column in args.get("required_columns") or []:
            if str(column).lower() not in known_columns:
                missing.append(f"column:{column}")
        question = str(args.get("question") or "")
        ambiguous_terms = sorted(set(re.findall(r"\b(recent|latest|current|top|best|active)\b", question.lower())))
        missing.extend(f"ambiguous:{term}" for term in ambiguous_terms if term in {"recent", "current", "best", "active"})
        candidate_sql = args.get("candidate_sql")
        sql_safe = True
        if isinstance(candidate_sql, str) and candidate_sql.strip():
            try:
                validate_safe_sql(candidate_sql)
            except Exception:
                sql_safe = False
                missing.append("unsafe_sql")
        score = max(0.0, 1.0 - (0.25 * len(missing)))
        answerable = not missing and score >= 0.5 and sql_safe
        return {
            "ok": True,
            "type": "answerability_observation",
            "tool": "answerability_probe",
            "answerable": answerable,
            "score": round(score, 3),
            "missing_slots": missing,
            "recommendation": "exec_sql" if answerable else "ask_clarification",
        }

    def _dispatch_tool(self, db_path: str | Path, action: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        tool = action["tool"]
        args = action.get("args") or {}
        if tool == "inspect_schema":
            return {"ok": True, "type": "tool_observation", "tool": tool, "observation": self._inspect_schema(db_path, args)}, False
        if tool == "profile_columns":
            return {"ok": True, "type": "tool_observation", "tool": tool, "observation": self._profile_columns(db_path, args)}, False
        if tool == "sample":
            return {"ok": True, "type": "tool_observation", "tool": tool, "observation": self._sample(db_path, args)}, False
        if tool == "retrieve_exemplars":
            db_id = str(args.get("db_id") or self._db_id(db_path))
            observation = retrieve_structural_exemplars(
                db_id=db_id,
                question=str(args["question"]),
                catalog=self._catalog_map(db_path),
                exemplar_bank=self.exemplar_bank,
                k=int(args.get("k", 2)),
                similarity_cap=float(args.get("similarity_cap", 0.75)),
                current_gold_sql=args.get("current_gold_sql"),
            )
            return observation, False
        if tool == "exec_sql":
            observation = execute_readonly_sql(
                db_path,
                str(args["sql"]),
                row_limit=int(args.get("row_limit", self.config.row_limit)),
                timeout_ms=int(args.get("timeout_ms", 5000)),
            )
            observation["purpose"] = args.get("purpose", "probe")
            return observation, bool(args.get("purpose") == "answer" and not self.config.continue_after_sql)
        if tool == "file_store":
            observation = self._scratch(db_path).store(
                key=str(args["key"]),
                value=args.get("value"),
                scope=str(args.get("scope", "turn")),
                ttl_turns=args.get("ttl_turns"),
                visibility=str(args.get("visibility", "model_readable")),
                overwrite=bool(args.get("overwrite", True)),
                catalog_hash=args.get("catalog_hash"),
                profile_hash=args.get("profile_hash"),
                allow_long_lived_raw=bool(args.get("allow_long_lived_raw", False)),
            )
            return observation, False
        if tool == "file_read":
            observation = self._scratch(db_path).read(
                key=str(args["key"]),
                scope=str(args.get("scope", "turn")),
                max_bytes=int(args.get("max_bytes", 4096)),
                version=args.get("version"),
                catalog_hash=args.get("catalog_hash"),
                profile_hash=args.get("profile_hash"),
            )
            return observation, False
        if tool == "file_list":
            observation = self._scratch(db_path).list(
                prefix=str(args.get("prefix", "")),
                scope=str(args.get("scope", "turn")),
                include_values=bool(args.get("include_values", False)),
                limit=int(args.get("limit", 50)),
            )
            return observation, False
        if tool == "ask_clarification":
            return {
                "ok": True,
                "type": "clarification_observation",
                "tool": "ask_clarification",
                "question": args["question"],
                "missing_slots": args.get("missing_slots") or [],
            }, True
        if tool == "answerability_probe":
            return self._answerability_probe(db_path, args), False
        return {
            "ok": True,
            "type": "tool_observation",
            "tool": tool,
            "observation": run_tool(db_path, tool, args),
        }, False

    def _dispatch(
        self,
        db_path: str | Path,
        action: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, bool]:
        action_type = action["type"]
        if action_type == "tool_call":
            if any(item.get("action") == action for item in observations):
                return (
                    {
                        "ok": False,
                        "type": "tool_observation",
                        "tool": action["tool"],
                        "error_class": "duplicate_tool_call",
                        "errors": ["duplicate tool call skipped before dispatch"],
                    },
                    False,
                )
            return self._dispatch_tool(db_path, action)
        if action_type == "sql":
            try:
                observation = execute_readonly_sql(db_path, action["sql"], row_limit=self.config.row_limit)
            except Exception as exc:
                observation = {
                    "ok": False,
                    "type": "sql_observation",
                    "error_class": type(exc).__name__,
                    "errors": [str(exc)[:300]],
                }
            return observation, not self.config.continue_after_sql
        if action_type in {"clarification", "final_answer"}:
            return None, True
        return {"ok": False, "type": "runtime_error", "errors": [f"unsupported action type: {action_type}"]}, True

    def _fallback_prompt(
        self,
        *,
        question: str,
        db_path: str | Path,
        fallback_observation: dict[str, Any],
    ) -> str:
        schema_block = compact_schema_context(db_path)
        digest = str(fallback_observation.get("digest") or "")
        return "\n".join(
            [
                "You are the AMALGAM primitive runtime in fixed-mode fallback.",
                "Return exactly one JSON object matching the canonical action contract. No markdown, no prose.",
                "The digest below is labeled fallback policy, not primitive-composition evidence.",
                "Use sql for a safe read-only DuckDB SELECT/WITH answer, or clarification if the request is ambiguous.",
                "",
                "Schema context:",
                schema_block,
                "",
                digest,
                "",
                "User question:",
                question,
                "",
                "Return one terminal sql or clarification action.",
            ]
        ).strip() + "\n"

    def _run_fixed_mode_fallback(
        self,
        *,
        question: str,
        db_path: str | Path,
        next_index: int,
        steps: list[RuntimeStep],
        ledger: TraceLedger,
    ) -> tuple[dict[str, Any] | None, bool]:
        fallback_observation = build_fixed_mode_fallback(
            db_path=db_path,
            question=question,
            db_id=self._db_id(db_path),
            exemplar_bank=self.exemplar_bank,
            max_chars=self.config.fixed_mode_max_chars,
        )
        fallback_ref = ledger.add(kind="fixed_mode_fallback", action=None, payload=fallback_observation)
        fallback_observation["trace_ref"] = fallback_ref
        prompt = self._fallback_prompt(question=question, db_path=db_path, fallback_observation=fallback_observation)
        try:
            generation = self.backend.generate(
                prompt,
                grammar=self.grammar,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            action = parse_and_validate_action(generation.text, self.contract)
            observation, terminal = self._dispatch(db_path, action, [])
            trace_ref = None
            if observation is not None:
                trace_ref = ledger.add(kind="fallback_action_observation", action=action, payload=observation)
                observation["trace_ref"] = trace_ref
            steps.append(
                RuntimeStep(
                    index=next_index,
                    prompt_chars=len(prompt),
                    raw_text=generation.text,
                    latency_s=generation.latency_s,
                    valid_action=True,
                    action=action,
                    observation=observation,
                    terminal=terminal,
                    prompt_tokens=generation.prompt_tokens,
                    completion_tokens=generation.completion_tokens,
                    trace_ref=trace_ref or fallback_ref,
                    fallback_policy="fixed_static_grounding_digest",
                )
            )
            return action if terminal else None, True
        except Exception as exc:
            steps.append(
                RuntimeStep(
                    index=next_index,
                    prompt_chars=len(prompt),
                    raw_text=locals().get("generation", Generation("", 0.0)).text,
                    latency_s=locals().get("generation", Generation("", 0.0)).latency_s,
                    valid_action=False,
                    validation_error=f"fixed_mode_fallback_failed: {type(exc).__name__}: {str(exc)[:300]}",
                    terminal=True,
                    trace_ref=fallback_ref,
                    fallback_policy="fixed_static_grounding_digest",
                )
            )
            return None, True

    def run(
        self,
        *,
        question: str,
        db_path: str | Path,
        schema_context: str | None = None,
    ) -> RuntimeResult:
        prompt_observations: list[dict[str, Any]] = []
        steps: list[RuntimeStep] = []
        terminal_action: dict[str, Any] | None = None
        parser_crash = False
        fallback_used = False
        ledger = TraceLedger(self.out_dir)

        for index in range(self.config.max_steps):
            prompt = self.render_prompt(
                question=question,
                db_path=db_path,
                schema_context=schema_context,
                observations=prompt_observations,
            )
            try:
                generation = self.backend.generate(
                    prompt,
                    grammar=self.grammar,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                action = parse_and_validate_action(generation.text, self.contract)
                observation, terminal = self._dispatch(db_path, action, prompt_observations)
                trace_ref = None
                if observation is not None:
                    trace_ref = ledger.add(kind="action_observation", action=action, payload=observation)
                    observation["trace_ref"] = trace_ref
                    prompt_observations.append(
                        {"action": action, "observation": compact_runtime_observation(observation)}
                    )
                if terminal:
                    terminal_action = action
                steps.append(
                    RuntimeStep(
                        index=index,
                        prompt_chars=len(prompt),
                        raw_text=generation.text,
                        latency_s=generation.latency_s,
                        valid_action=True,
                        action=action,
                        observation=observation,
                        terminal=terminal,
                        prompt_tokens=generation.prompt_tokens,
                        completion_tokens=generation.completion_tokens,
                        trace_ref=trace_ref,
                    )
                )
                if terminal:
                    break
            except ActionValidationError as exc:
                parser_crash = False
                steps.append(
                    RuntimeStep(
                        index=index,
                        prompt_chars=len(prompt),
                        raw_text=locals().get("generation", Generation("", 0.0)).text,
                        latency_s=locals().get("generation", Generation("", 0.0)).latency_s,
                        valid_action=False,
                        validation_error=str(exc),
                        terminal=not self.config.fixed_mode_fallback,
                    )
                )
                if self.config.fixed_mode_fallback:
                    terminal_action, fallback_used = self._run_fixed_mode_fallback(
                        question=question,
                        db_path=db_path,
                        next_index=index + 1,
                        steps=steps,
                        ledger=ledger,
                    )
                break
            except Exception as exc:
                parser_crash = True
                steps.append(
                    RuntimeStep(
                        index=index,
                        prompt_chars=len(prompt),
                        raw_text="",
                        latency_s=0.0,
                        valid_action=False,
                        validation_error=f"{type(exc).__name__}: {str(exc)[:300]}",
                        terminal=not self.config.fixed_mode_fallback,
                    )
                )
                if self.config.fixed_mode_fallback:
                    terminal_action, fallback_used = self._run_fixed_mode_fallback(
                        question=question,
                        db_path=db_path,
                        next_index=index + 1,
                        steps=steps,
                        ledger=ledger,
                    )
                break
        else:
            if self.config.fixed_mode_fallback and terminal_action is None:
                terminal_action, fallback_used = self._run_fixed_mode_fallback(
                    question=question,
                    db_path=db_path,
                    next_index=len(steps),
                    steps=steps,
                    ledger=ledger,
                )

        return RuntimeResult(
            question=question,
            db_path=str(db_path),
            backend=self.backend.name,
            contract_version=self.contract.version,
            terminal_action=terminal_action,
            steps=steps,
            parser_crash=parser_crash,
            fixed_mode_fallback_used=fallback_used,
            trace_ledger=ledger.entries,
        )


def summarize_results(results: list[RuntimeResult]) -> dict[str, Any]:
    action_count = sum(result.action_count for result in results)
    valid_action_count = sum(result.valid_action_count for result in results)
    total_latency = sum(result.total_latency_s for result in results)
    return {
        "sessions": len(results),
        "actions": action_count,
        "valid_actions": valid_action_count,
        "invalid_actions": action_count - valid_action_count,
        "action_validity_rate": valid_action_count / action_count if action_count else 0.0,
        "parser_crashes": sum(1 for result in results if result.parser_crash),
        "fixed_mode_fallbacks": sum(1 for result in results if result.fixed_mode_fallback_used),
        "multi_step_sessions": sum(1 for result in results if result.action_count > 1),
        "terminal_sessions": sum(1 for result in results if result.terminal_action is not None),
        "total_latency_s": round(total_latency, 6),
        "median_action_latency_s": _median([step.latency_s for result in results for step in result.steps]),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 6)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 6)
