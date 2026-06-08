"""Canonical AMALGAM analytic action contract.

The contract below is the single source for JSON Schema, GBNF, and the local
runtime validator. The GBNF intentionally fixes object key order because
llama.cpp grammars are simpler and more reliable when the action surface is
canonical rather than accepting every JSON spelling.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Literal


ScalarKind = Literal["string", "integer", "number", "boolean", "null"]
FieldKind = Literal["string", "integer", "number", "boolean", "nullable_string", "string_array", "json"]


class ActionValidationError(ValueError):
    """Raised when generated text is not a valid action object."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: FieldKind
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args: tuple[FieldSpec, ...]


@dataclass(frozen=True)
class ActionContract:
    version: str
    tools: tuple[ToolSpec, ...]
    sql_prefixes: tuple[str, ...]
    final_fields: tuple[FieldSpec, ...]


ACTION_CONTRACT = ActionContract(
    version="amalgam_action_contract_v1_primitive_engine",
    tools=(
        ToolSpec(
            "inspect_schema",
            (
                FieldSpec("tables", "string_array"),
                FieldSpec("include_columns", "boolean"),
                FieldSpec("include_relationships", "boolean"),
                FieldSpec("max_tables", "integer", minimum=1, maximum=500),
                FieldSpec("max_columns_per_table", "integer", minimum=1, maximum=500),
            ),
        ),
        ToolSpec(
            "profile_columns",
            (
                FieldSpec("table", "string", required=True),
                FieldSpec("columns", "string_array"),
                FieldSpec("max_exact_distinct", "integer", minimum=0, maximum=1000000),
                FieldSpec("top_values_limit", "integer", minimum=0, maximum=100),
                FieldSpec("allow_value_examples", "boolean"),
            ),
        ),
        ToolSpec(
            "sample",
            (
                FieldSpec("table", "string", required=True),
                FieldSpec("columns", "string_array"),
                FieldSpec("limit", "integer", minimum=0, maximum=100),
                FieldSpec("where_sql", "nullable_string"),
                FieldSpec("method", "string"),
            ),
        ),
        ToolSpec(
            "retrieve_exemplars",
            (
                FieldSpec("question", "string", required=True),
                FieldSpec("k", "integer", minimum=0, maximum=10),
                FieldSpec("similarity_cap", "number", minimum=0.0, maximum=1.0),
                FieldSpec("source", "string"),
                FieldSpec("schema_scope", "string"),
                FieldSpec("db_id", "string"),
                FieldSpec("current_gold_sql", "nullable_string"),
            ),
        ),
        ToolSpec(
            "exec_sql",
            (
                FieldSpec("sql", "string", required=True),
                FieldSpec("row_limit", "integer", minimum=0, maximum=1000),
                FieldSpec("timeout_ms", "integer", minimum=1, maximum=60000),
                FieldSpec("purpose", "string"),
            ),
        ),
        ToolSpec(
            "file_store",
            (
                FieldSpec("key", "string", required=True),
                FieldSpec("value", "json", required=True),
                FieldSpec("scope", "string"),
                FieldSpec("ttl_turns", "integer", minimum=0, maximum=1000000),
                FieldSpec("visibility", "string"),
                FieldSpec("overwrite", "boolean"),
                FieldSpec("catalog_hash", "nullable_string"),
                FieldSpec("profile_hash", "nullable_string"),
                FieldSpec("allow_long_lived_raw", "boolean"),
            ),
        ),
        ToolSpec(
            "file_read",
            (
                FieldSpec("key", "string", required=True),
                FieldSpec("scope", "string"),
                FieldSpec("max_bytes", "integer", minimum=1, maximum=262144),
                FieldSpec("version", "nullable_string"),
                FieldSpec("catalog_hash", "nullable_string"),
                FieldSpec("profile_hash", "nullable_string"),
            ),
        ),
        ToolSpec(
            "file_list",
            (
                FieldSpec("prefix", "string"),
                FieldSpec("scope", "string"),
                FieldSpec("include_values", "boolean"),
                FieldSpec("limit", "integer", minimum=1, maximum=500),
                FieldSpec("catalog_hash", "nullable_string"),
            ),
        ),
        ToolSpec(
            "ask_clarification",
            (
                FieldSpec("question", "string", required=True),
                FieldSpec("missing_slots", "string_array"),
            ),
        ),
        ToolSpec(
            "answerability_probe",
            (
                FieldSpec("question", "string", required=True),
                FieldSpec("candidate_sql", "nullable_string"),
                FieldSpec("required_tables", "string_array"),
                FieldSpec("required_columns", "string_array"),
            ),
        ),
        ToolSpec("schema_catalog", (FieldSpec("include_columns", "boolean"),)),
        ToolSpec("table_profile", (FieldSpec("table", "string", required=True),)),
        ToolSpec(
            "sample_rows",
            (
                FieldSpec("table", "string", required=True),
                FieldSpec("limit", "integer", minimum=0, maximum=100),
                FieldSpec("columns", "string_array"),
            ),
        ),
        ToolSpec(
            "find_join_paths",
            (
                FieldSpec("start_table", "string"),
                FieldSpec("end_table", "string"),
                FieldSpec("max_hops", "integer", minimum=1, maximum=8),
                FieldSpec("max_paths", "integer", minimum=1, maximum=50),
            ),
        ),
        ToolSpec(
            "aggregate_window_template",
            (
                FieldSpec("pattern", "string", required=True),
                FieldSpec("table", "string", required=True),
                FieldSpec("params", "json"),
            ),
        ),
        ToolSpec("explain_plan", (FieldSpec("sql", "string", required=True),)),
        ToolSpec(
            "result_to_chart_spec",
            (
                FieldSpec("result_schema", "json", required=True),
                FieldSpec("rows", "json", required=True),
                FieldSpec("intent", "string"),
                FieldSpec("max_rows", "integer", minimum=0, maximum=500),
            ),
        ),
    ),
    sql_prefixes=("SELECT", "WITH"),
    final_fields=(
        FieldSpec("answer", "string", required=True),
        FieldSpec("sql_id", "nullable_string"),
        FieldSpec("chart_id", "nullable_string"),
        FieldSpec("confidence", "number", minimum=0.0, maximum=1.0),
    ),
)


def _field_schema(field: FieldSpec) -> dict[str, Any]:
    if field.kind == "nullable_string":
        schema: dict[str, Any] = {"type": ["string", "null"]}
    elif field.kind == "string_array":
        schema = {"type": "array", "items": {"type": "string"}}
    elif field.kind == "json":
        schema = True
    else:
        schema = {"type": field.kind}
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum
    return schema


def _object_schema(properties: dict[str, Any], required: Iterable[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def action_contract_json_schema(contract: ActionContract = ACTION_CONTRACT) -> dict[str, Any]:
    """Return the JSON Schema generated from the canonical action contract."""

    tool_branches: list[dict[str, Any]] = []
    for tool in contract.tools:
        arg_properties = {field.name: _field_schema(field) for field in tool.args}
        required_args = [field.name for field in tool.args if field.required]
        tool_branches.append(
            _object_schema(
                {
                    "type": {"const": "tool_call"},
                    "tool": {"const": tool.name},
                    "args": _object_schema(arg_properties, required_args),
                },
                ("type", "tool", "args"),
            )
        )

    sql_pattern = r"^\s*(?:" + "|".join(re.escape(prefix) for prefix in contract.sql_prefixes) + r")\b"
    sql_branch = _object_schema(
        {
            "type": {"const": "sql"},
            "sql": {"type": "string", "pattern": sql_pattern},
        },
        ("type", "sql"),
    )
    clarification_branch = _object_schema(
        {
            "type": {"const": "clarification"},
            "question": {"type": "string", "minLength": 1},
        },
        ("type", "question"),
    )
    final_branch = _object_schema(
        {
            "type": {"const": "final_answer"},
            **{field.name: _field_schema(field) for field in contract.final_fields},
        },
        ("type", *[field.name for field in contract.final_fields if field.required]),
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://cothogonal.local/schemas/{contract.version}.schema.json",
        "title": "AMALGAM Analytic Action",
        "description": "Canonical bounded action object for the in-process GBNF action runtime.",
        "oneOf": [*tool_branches, sql_branch, clarification_branch, final_branch],
    }


def _gbnf_literal(value: str) -> str:
    return json.dumps(value)


def _gbnf_json_string(value: str) -> str:
    return json.dumps(json.dumps(value))


def _gbnf_value_rule(field: FieldSpec) -> str:
    if field.kind == "string":
        return "json-string"
    if field.kind == "integer":
        if field.minimum is not None and field.minimum >= 0:
            return "uint"
        return "integer"
    if field.kind == "number":
        if field.minimum == 0.0 and field.maximum == 1.0:
            return "unit-number"
        return "number"
    if field.kind == "boolean":
        return "boolean"
    if field.kind == "nullable_string":
        return "nullable-string"
    if field.kind == "string_array":
        return "string-array"
    if field.kind == "json":
        return "json-value"
    raise AssertionError(field.kind)


def _field_fragment(field: FieldSpec) -> str:
    return f"{_gbnf_json_string(field.name)} ws \":\" ws {_gbnf_value_rule(field)}"


def _object_variant(fields: tuple[FieldSpec, ...]) -> str:
    if not fields:
        return "\"{\" ws \"}\""
    joined = " ws \",\" ws ".join(_field_fragment(field) for field in fields)
    return f"\"{{\" ws {joined} ws \"}}\""


def _optional_field_variants(fields: tuple[FieldSpec, ...]) -> list[tuple[FieldSpec, ...]]:
    required = tuple(field for field in fields if field.required)
    optional = tuple(field for field in fields if not field.required)
    variants: list[tuple[FieldSpec, ...]] = []
    for count in range(len(optional) + 1):
        for selected in combinations(optional, count):
            selected_names = {field.name for field in selected}
            variant = tuple(field for field in fields if field.required or field.name in selected_names)
            variants.append(variant)
    variants.sort(key=lambda item: (len(item), tuple(field.name for field in item)))
    return variants


def _args_rule_name(tool_name: str) -> str:
    return f"{tool_name.replace('_', '-')}-args"


def _tool_rule_name(tool_name: str) -> str:
    return f"tool-{tool_name.replace('_', '-')}"


def _tool_rule(tool: ToolSpec) -> str:
    rule = _tool_rule_name(tool.name)
    args_rule = _args_rule_name(tool.name)
    return (
        f"{rule} ::= "
        "\"{\" ws "
        "\"\\\"type\\\"\" ws \":\" ws \"\\\"tool_call\\\"\" ws \",\" ws "
        "\"\\\"tool\\\"\" ws \":\" ws "
        f"{_gbnf_json_string(tool.name)} ws \",\" ws "
        "\"\\\"args\\\"\" ws \":\" ws "
        f"{args_rule} ws "
        "\"}\""
    )


def _tool_args_rule(tool: ToolSpec) -> str:
    variants = [_object_variant(variant) for variant in _optional_field_variants(tool.args)]
    return f"{_args_rule_name(tool.name)} ::= " + " | ".join(variants)


def _final_answer_rule(contract: ActionContract) -> str:
    variants = [_object_variant(variant) for variant in _optional_field_variants(contract.final_fields)]
    object_variants = []
    for variant in variants:
        body = (
            "\"{\" ws "
            "\"\\\"type\\\"\" ws \":\" ws \"\\\"final_answer\\\"\" ws \",\" ws "
            + variant.removeprefix("\"{\" ws ").removesuffix(" ws \"}\"")
            + " ws \"}\""
        )
        object_variants.append(body)
    return "final-answer ::= " + " | ".join(object_variants)


def action_contract_gbnf(contract: ActionContract = ACTION_CONTRACT) -> str:
    """Return llama.cpp GBNF generated from the canonical action contract."""

    branches = [
        *[_tool_rule_name(tool.name) for tool in contract.tools],
        "sql-action",
        "clarification-action",
        "final-answer",
    ]
    sql_prefixes = " | ".join(f"{_gbnf_literal(prefix + ' ')} sql-char*" for prefix in contract.sql_prefixes)
    lines = [
        f"# Generated from actionrt.contract ACTION_CONTRACT version {contract.version}.",
        "root ::= " + " | ".join(branches),
        "",
        *[_tool_rule(tool) for tool in contract.tools],
        "",
        *[_tool_args_rule(tool) for tool in contract.tools],
        "",
        (
            "sql-action ::= "
            "\"{\" ws "
            "\"\\\"type\\\"\" ws \":\" ws \"\\\"sql\\\"\" ws \",\" ws "
            "\"\\\"sql\\\"\" ws \":\" ws sql-string ws "
            "\"}\""
        ),
        (
            "clarification-action ::= "
            "\"{\" ws "
            "\"\\\"type\\\"\" ws \":\" ws \"\\\"clarification\\\"\" ws \",\" ws "
            "\"\\\"question\\\"\" ws \":\" ws json-string ws "
            "\"}\""
        ),
        _final_answer_rule(contract),
        "",
        f"sql-string ::= \"\\\"\" ({sql_prefixes}) \"\\\"\"",
        "json-string ::= \"\\\"\" json-char* \"\\\"\"",
        "nullable-string ::= json-string | \"null\"",
        "string-array ::= \"[\" ws (json-string (ws \",\" ws json-string)*)? ws \"]\"",
        "json-value ::= json-object | json-array | json-string | number | boolean | \"null\"",
        "json-object ::= \"{\" ws (json-member (ws \",\" ws json-member)*)? ws \"}\"",
        "json-member ::= json-string ws \":\" ws json-value",
        "json-array ::= \"[\" ws (json-value (ws \",\" ws json-value)*)? ws \"]\"",
        "integer ::= \"-\"? ([0-9] | [1-9] [0-9]*)",
        "uint ::= [0-9] | [1-9] [0-9]*",
        "number ::= \"-\"? ([0-9] | [1-9] [0-9]*) (\".\" [0-9]+)?",
        "unit-number ::= \"0\" (\".\" [0-9]+)? | \"1\" (\".\" \"0\"+)?",
        "boolean ::= \"true\" | \"false\"",
        "",
        "sql-char ::= [^\"\\\\\\x00-\\x1f] | escape",
        "json-char ::= [^\"\\\\\\x00-\\x1f] | escape",
        "escape ::= \"\\\\\" ([\"\\\\/bfnrt] | \"u\" hex hex hex hex)",
        "hex ::= [0-9a-fA-F]",
        "ws ::= [ \\t\\n\\r]*",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _require_keys(obj: dict[str, Any], required: set[str], allowed: set[str], where: str) -> None:
    keys = set(obj)
    missing = required - keys
    extra = keys - allowed
    if missing:
        raise ActionValidationError(f"{where} missing required keys: {sorted(missing)}")
    if extra:
        raise ActionValidationError(f"{where} has unexpected keys: {sorted(extra)}")


def _validate_field(value: Any, field: FieldSpec, where: str) -> None:
    if field.kind == "string":
        ok = isinstance(value, str)
    elif field.kind == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif field.kind == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif field.kind == "boolean":
        ok = isinstance(value, bool)
    elif field.kind == "nullable_string":
        ok = value is None or isinstance(value, str)
    elif field.kind == "string_array":
        ok = isinstance(value, list) and all(isinstance(item, str) for item in value)
    elif field.kind == "json":
        ok = isinstance(value, (dict, list, str, int, float, bool)) or value is None
    else:
        ok = False
    if not ok:
        raise ActionValidationError(f"{where}.{field.name} must be {field.kind}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if field.minimum is not None and value < field.minimum:
            raise ActionValidationError(f"{where}.{field.name} must be >= {field.minimum}")
        if field.maximum is not None and value > field.maximum:
            raise ActionValidationError(f"{where}.{field.name} must be <= {field.maximum}")


def _validate_args(args: Any, tool: ToolSpec) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ActionValidationError("tool_call.args must be an object")
    required = {field.name for field in tool.args if field.required}
    allowed = {field.name for field in tool.args}
    _require_keys(args, required, allowed, f"args[{tool.name}]")
    fields = {field.name: field for field in tool.args}
    for key, value in args.items():
        _validate_field(value, fields[key], f"args[{tool.name}]")
    return dict(args)


def _validate_sql(sql: Any, contract: ActionContract) -> str:
    if not isinstance(sql, str):
        raise ActionValidationError("sql.sql must be a string")
    stripped = sql.lstrip()
    if not any(stripped.upper().startswith(prefix + " ") or stripped.upper() == prefix for prefix in contract.sql_prefixes):
        raise ActionValidationError("sql.sql must start with SELECT or WITH")
    return sql


def validate_action(obj: Any, contract: ActionContract = ACTION_CONTRACT) -> dict[str, Any]:
    """Validate and normalize an action object using the canonical contract."""

    if not isinstance(obj, dict):
        raise ActionValidationError("action must be a JSON object")
    action_type = obj.get("type")
    if action_type == "tool_call":
        _require_keys(obj, {"type", "tool", "args"}, {"type", "tool", "args"}, "tool_call")
        tool_name = obj.get("tool")
        tool = next((candidate for candidate in contract.tools if candidate.name == tool_name), None)
        if tool is None:
            raise ActionValidationError(f"unknown tool: {tool_name}")
        return {"type": "tool_call", "tool": tool.name, "args": _validate_args(obj.get("args"), tool)}
    if action_type == "sql":
        _require_keys(obj, {"type", "sql"}, {"type", "sql"}, "sql")
        return {"type": "sql", "sql": _validate_sql(obj.get("sql"), contract)}
    if action_type == "clarification":
        _require_keys(obj, {"type", "question"}, {"type", "question"}, "clarification")
        question = obj.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ActionValidationError("clarification.question must be a nonempty string")
        return {"type": "clarification", "question": question}
    if action_type == "final_answer":
        fields = {field.name: field for field in contract.final_fields}
        required = {"type", *[field.name for field in contract.final_fields if field.required]}
        allowed = {"type", *fields}
        _require_keys(obj, required, allowed, "final_answer")
        normalized = {"type": "final_answer"}
        for key, value in obj.items():
            if key == "type":
                continue
            _validate_field(value, fields[key], "final_answer")
            normalized[key] = value
        return normalized
    raise ActionValidationError(f"unknown action type: {action_type}")


def parse_and_validate_action(text: str, contract: ActionContract = ACTION_CONTRACT) -> dict[str, Any]:
    """Parse a generated JSON action and validate it without best-effort repair."""

    try:
        parsed = json.loads(text.strip())
    except Exception as exc:
        raise ActionValidationError(f"invalid JSON action: {exc}") from exc
    return validate_action(parsed, contract)


def write_contract_artifacts(
    *,
    grammar_path: str | Path,
    schema_path: str | Path,
    contract: ActionContract = ACTION_CONTRACT,
) -> None:
    """Write generated GBNF and JSON Schema artifacts."""

    grammar_target = Path(grammar_path)
    schema_target = Path(schema_path)
    grammar_target.parent.mkdir(parents=True, exist_ok=True)
    schema_target.parent.mkdir(parents=True, exist_ok=True)
    grammar_target.write_text(action_contract_gbnf(contract), encoding="utf-8")
    schema_target.write_text(
        json.dumps(action_contract_json_schema(contract), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
