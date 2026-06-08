"""Fixed-mode static grounding fallback for weak primitive tenants."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from .exemplars import retrieve_structural_exemplars

ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = ROOT / "tasks" / "micro_sql_agent_harness"
if HARNESS_ROOT.exists() and str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from micro_sql_agent_harness.analytic_tools import sample_rows, schema_catalog, table_profile  # noqa: E402


def _catalog_map(catalog: dict[str, Any]) -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = {}
    for table in catalog.get("tables") or []:
        name = table.get("name")
        if not isinstance(name, str):
            continue
        mapped[name] = [str(column.get("name")) for column in table.get("columns") or [] if column.get("name")]
    return mapped


def build_fixed_mode_fallback(
    *,
    db_path: str | Path,
    question: str,
    db_id: str,
    exemplar_bank: dict[str, list[dict[str, Any]]] | None = None,
    max_chars: int = 3600,
) -> dict[str, Any]:
    """Build the proven static richer-grounding digest as a labeled fallback."""

    catalog = schema_catalog(db_path, include_columns=True)
    if not catalog.get("ok"):
        return {
            "ok": False,
            "type": "fixed_mode_fallback",
            "fallback_policy": "fixed_static_grounding_digest",
            "error_class": catalog.get("error_class"),
            "errors": catalog.get("errors"),
        }

    lines = [
        "FIXED-MODE FALLBACK: static richer-grounding digest.",
        "Policy label: fixed_static_grounding_digest. This is a fallback path, not primitive-composition evidence.",
    ]
    for table in catalog.get("tables") or []:
        table_name = str(table.get("name") or "")
        if not table_name:
            continue
        profile = table_profile(db_path, table_name)
        lines.append(f"TABLE {table_name} rows={profile.get('row_count', table.get('row_count'))}")
        for column in (profile.get("columns") or [])[:8]:
            detail = (
                f"  {column.get('name')} {column.get('data_type')} "
                f"null_pct={column.get('null_pct')} distinct={column.get('distinct_count')}"
            )
            top_values = column.get("top_values") or []
            if top_values:
                rendered = [f"{item.get('value')}:{item.get('count')}" for item in top_values[:4]]
                detail += " top=[" + ", ".join(rendered) + "]"
            lines.append(detail)
        sample = sample_rows(db_path, table_name, limit=2)
        if sample.get("ok") and sample.get("rows"):
            lines.append(f"  sample columns={sample.get('columns')}")
            lines.append(f"  sample rows={(sample.get('rows') or [])[:2]}")
        if len("\n".join(lines)) >= max_chars:
            break

    exemplar_obs = {"exemplars": []}
    if exemplar_bank:
        exemplar_obs = retrieve_structural_exemplars(
            db_id=db_id,
            question=question,
            catalog=_catalog_map(catalog),
            exemplar_bank=exemplar_bank,
            k=2,
            similarity_cap=0.75,
        )
        exemplars = exemplar_obs.get("exemplars") or []
        if exemplars:
            lines.append("Leak-safe structural same-DB exemplars:")
            for row in exemplars[:2]:
                q = str(row.get("question") or "").replace("\n", " ")[:160]
                sql = str(row.get("gold_sql") or "").replace("\n", " ")[:320]
                lines.append(f"- Q: {q}")
                lines.append(f"  SQL: {sql}")

    digest = "\n".join(lines)
    truncated = False
    if len(digest) > max_chars:
        digest = digest[: max_chars - 32].rstrip() + "\n[truncated digest]"
        truncated = True
    return {
        "ok": True,
        "type": "fixed_mode_fallback",
        "fallback_policy": "fixed_static_grounding_digest",
        "digest": digest,
        "digest_chars": len(digest),
        "truncated": truncated,
        "catalog_hash": catalog.get("catalog_hash"),
        "exemplars": exemplar_obs.get("exemplars") or [],
    }
