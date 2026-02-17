#!/usr/bin/env python3
"""Flask app for exploring Reykjavik arsuppgjor data."""
from __future__ import annotations

import os
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
from flask import Flask, Response, redirect, render_template, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

DEFAULT_PARQUET = os.getenv("PARQUET_PATH", "data/processed/arsuppgjor_combined.parquet")
DEFAULT_ANOMALIES_PARQUET = os.getenv("ANOMALIES_PARQUET_PATH", "data/processed/anomalies_flagged.parquet")
DEFAULT_ANOMALIES_ALL_PARQUET = os.getenv("ANOMALIES_ALL_PARQUET_PATH", "data/processed/anomalies_yoy_all.parquet")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")

REYKJAVIK_PREFIX = os.getenv("REYKJAVIK_PREFIX", "").rstrip("/")
if REYKJAVIK_PREFIX:
    app.config["APPLICATION_ROOT"] = REYKJAVIK_PREFIX
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

    class PrefixMiddleware:
        def __init__(self, app, prefix: str):
            self.app = app
            self.prefix = prefix

        def __call__(self, environ, start_response):
            script_name = self.prefix
            path_info = environ.get("PATH_INFO", "")
            if path_info.startswith(script_name):
                environ["SCRIPT_NAME"] = script_name
                environ["PATH_INFO"] = path_info[len(script_name):] or "/"
            return self.app(environ, start_response)

    app.wsgi_app = PrefixMiddleware(app.wsgi_app, REYKJAVIK_PREFIX)

DISPLAY_NAMES = {
    "year": "Year",
    "ingested_at": "Ingested",
    "samtala0": "Institution",
    "samtala1": "Division",
    "samtala2": "Department",
    "samtala2_raw": "Department (raw)",
    "samtala2_canonical": "Department",
    "samtala2_family": "Department",
    "samtala3": "Unit",
    "tegund0": "Group",
    "tegund1": "Type",
    "tegund2": "Category",
    "tegund3": "Item",
    "fyrirtaeki": "Entity",
    "vm_numer": "VSK number",
    "vm_nafn": "VSK name",
    "raun": "Actual",
}
DEPARTMENT_CANONICAL_MAP = {
    "Frístund": "Frístundastarf",
    "Búsetuúrræði": "Búsetuþjónusta",
    "Heimaþjónusta - Heimahjúkrun": "Heimaþjónusta og Heimahjúkrun",
    "Heimastuðningur": "Heimaþjónusta og Heimahjúkrun",
    "Heimahjúkrun": "Heimaþjónusta og Heimahjúkrun",
}
DEPARTMENT_FAMILY_MAP = {
    "Frístund": "Frístund / Frístundastarf",
    "Frístundastarf": "Frístund / Frístundastarf",
    "Búsetuúrræði": "Búsetuúrræði / Búsetuþjónusta",
    "Búsetuþjónusta": "Búsetuúrræði / Búsetuþjónusta",
    "Heimaþjónusta - Heimahjúkrun": "Heimaþjónusta / Heimastuðningur",
    "Heimaþjónusta og Heimahjúkrun": "Heimaþjónusta / Heimastuðningur",
    "Heimastuðningur": "Heimaþjónusta / Heimastuðningur",
    "Heimahjúkrun": "Heimaþjónusta / Heimastuðningur",
}
DEPARTMENT_MODE = os.getenv("REYKJAVIK_DEPARTMENT_MODE", "canonical").strip().lower()
DEPARTMENT_COLUMN_BY_MODE = {
    "raw": "samtala2_raw",
    "canonical": "samtala2_canonical",
    "family": "samtala2_family",
}
if DEPARTMENT_MODE not in DEPARTMENT_COLUMN_BY_MODE:
    DEPARTMENT_MODE = "canonical"
DEPARTMENT_COLUMN = DEPARTMENT_COLUMN_BY_MODE[DEPARTMENT_MODE]
CLICKABLE_COLUMNS = {
    "fyrirtaeki",
    "samtala0",
    "samtala1",
    DEPARTMENT_COLUMN,
    "samtala3",
    "tegund0",
    "tegund1",
    "tegund2",
    "tegund3",
    "vm_numer",
    "vm_nafn",
}
FILTER_COLUMNS = [
    "fyrirtaeki",
    "samtala0",
    "samtala1",
    DEPARTMENT_COLUMN,
    "samtala3",
    "tegund0",
    "tegund1",
    "tegund2",
    "tegund3",
    "vm_numer",
    "vm_nafn",
]
ANALYSIS_PARENT_COLUMNS = {
    "group": "tegund0",
    "vsk_name": "vm_nafn",
}
ANALYSIS_CHILD_COLUMNS = [
    "samtala0",
    "samtala1",
    DEPARTMENT_COLUMN,
    "samtala3",
    "tegund1",
    "tegund2",
    "tegund3",
]
ANOMALY_PARENT_COLUMNS = [
    "samtala0",
    "samtala1",
    DEPARTMENT_COLUMN,
    "samtala3",
    "tegund0",
    "tegund1",
    "tegund2",
    "tegund3",
    "vm_nafn",
    "vm_numer",
]


def _sql_string_literal(value: str) -> str:
    return value.replace("'", "''")


def _build_case_expr(column_expr: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return f"TRIM({column_expr})"
    parts = ["CASE"]
    for src, dst in mapping.items():
        parts.append(
            f"WHEN TRIM({column_expr}) = '{_sql_string_literal(src)}' THEN '{_sql_string_literal(dst)}'"
        )
    parts.append(f"ELSE TRIM({column_expr}) END")
    return " ".join(parts)


def _mapped_source_select_sql(read_sql: str, alias: str = "t") -> str:
    canonical_expr = _build_case_expr(f"{alias}.samtala2", DEPARTMENT_CANONICAL_MAP)
    family_expr = _build_case_expr(f"{alias}.samtala2", DEPARTMENT_FAMILY_MAP)
    return f"""
        SELECT
            {alias}.*,
            TRIM({alias}.samtala2) AS samtala2_raw,
            {canonical_expr} AS samtala2_canonical,
            {family_expr} AS samtala2_family
        FROM {read_sql} {alias}
    """.strip()


def _display_name(col: str) -> str:
    return DISPLAY_NAMES.get(col, col)


def _is_excluded_column(name: str) -> bool:
    if name in {
        "source_file",
        "source_url",
        "tgr1",
        "tgr2",
        "tgr3",
        "ar_fjordungur",
        "arsfjordungur",
        "ar",
        "tsundl",
        "fjarfesting",
        "eining1",
        "eining2",
        "eining3",
        "eining4",
        "eining5",
        "jon1",
        "jon2",
        "jon3",
        "uttak",
        "samtala4",
        "tegund4",
        "period_month",
        "month",
        "ingested_at",
        "year",
    }:
        return True
    if name in {"samtala2_raw", "samtala2_canonical", "samtala2_family"} and name != DEPARTMENT_COLUMN:
        return True
    return name.startswith("x")


def _get_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    try:
        rows = con.execute("PRAGMA table_info('arsuppgjor')").fetchall()
        return [row[1] for row in rows]
    except Exception:
        return []


def _numeric_expr(column: str) -> str:
    return f"TRY_CAST(REPLACE(REPLACE({column}, '.', ''), ',', '.') AS DOUBLE)"


def _numeric_candidates(con: duckdb.DuckDBPyConnection) -> list[str]:
    candidates = []
    try:
        rows = con.execute("DESCRIBE arsuppgjor").fetchall()
    except Exception:
        return candidates
    for name, dtype, *_ in rows:
        if _is_excluded_column(name):
            continue
        if name in {"year"}:
            continue
        if any(token in dtype.lower() for token in ("int", "double", "decimal", "float")):
            candidates.append(name)
    return candidates


def _category_candidates(con: duckdb.DuckDBPyConnection) -> list[str]:
    allowed = {
        "fyrirtaeki",
        "samtala0",
        "samtala1",
        DEPARTMENT_COLUMN,
        "samtala3",
        "tegund0",
        "tegund1",
        "tegund2",
        "tegund3",
    }
    categories = []
    for col in allowed:
        if _is_excluded_column(col):
            continue
        try:
            distinct = con.execute(
                f"SELECT COUNT(DISTINCT {col}) FROM arsuppgjor WHERE {col} IS NOT NULL"
            ).fetchone()[0]
        except Exception:
            continue
        if 1 < distinct <= 200:
            categories.append(col)
    return sorted(categories)


@lru_cache(maxsize=1)
def get_connection(parquet_path: str) -> duckdb.DuckDBPyConnection:
    path = Path(parquet_path)
    con = duckdb.connect(database=":memory:")
    if not path.exists():
        return con
    con.execute("PRAGMA threads=4")
    safe_path = str(path).replace("'", "''")
    select_sql = _mapped_source_select_sql(f"read_parquet('{safe_path}')", alias="t")
    con.execute(f"CREATE OR REPLACE VIEW arsuppgjor AS {select_sql}")
    return con


def _table_exists(con: duckdb.DuckDBPyConnection) -> bool:
    try:
        con.execute("SELECT 1 FROM arsuppgjor LIMIT 1")
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_anomaly_connection(parquet_path: str) -> duckdb.DuckDBPyConnection:
    path = Path(parquet_path)
    con = duckdb.connect(database=":memory:")
    if not path.exists():
        return con
    con.execute("PRAGMA threads=4")
    safe_path = str(path).replace("'", "''")
    select_sql = _mapped_source_select_sql(f"read_parquet('{safe_path}')", alias="t")
    con.execute(f"CREATE OR REPLACE VIEW anomalies AS {select_sql}")
    return con


def _anomaly_table_exists(con: duckdb.DuckDBPyConnection) -> bool:
    try:
        con.execute("SELECT 1 FROM anomalies LIMIT 1")
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_anomaly_all_connection(parquet_path: str) -> duckdb.DuckDBPyConnection:
    path = Path(parquet_path)
    con = duckdb.connect(database=":memory:")
    if not path.exists():
        return con
    con.execute("PRAGMA threads=4")
    safe_path = str(path).replace("'", "''")
    select_sql = _mapped_source_select_sql(f"read_parquet('{safe_path}')", alias="t")
    con.execute(f"CREATE OR REPLACE VIEW anomalies_all AS {select_sql}")
    return con


def _anomaly_all_table_exists(con: duckdb.DuckDBPyConnection) -> bool:
    try:
        con.execute("SELECT 1 FROM anomalies_all LIMIT 1")
        return True
    except Exception:
        return False


def _available_years(con: duckdb.DuckDBPyConnection) -> list[int]:
    try:
        rows = con.execute(
            "SELECT DISTINCT year FROM arsuppgjor WHERE year IS NOT NULL ORDER BY year"
        ).fetchall()
        return [int(row[0]) for row in rows]
    except Exception:
        return []


def _build_where(year: str, filters: dict[str, str], exclude_col: str | None = None) -> tuple[str, list]:
    clauses = []
    params: list = []
    if year and year != "all":
        try:
            year_value = int(year)
            clauses.append("year = ?")
            params.append(year_value)
        except ValueError:
            pass
    for col, value in filters.items():
        if exclude_col and col == exclude_col:
            continue
        clauses.append(f"{col} = ?")
        params.append(value)
    where_sql = " AND ".join(clauses)
    if where_sql:
        where_sql = "WHERE " + where_sql
    return where_sql, params


def _value_options(con: duckdb.DuckDBPyConnection) -> list[str]:
    candidates = _numeric_candidates(con)
    return sorted(candidates)


def _category_options(con: duckdb.DuckDBPyConnection) -> list[str]:
    candidates = _category_candidates(con)
    return sorted(candidates)


def _distinct_values(
    con: duckdb.DuckDBPyConnection,
    column: str,
    where_sql: str,
    params: list,
    limit: int = 200,
) -> list[str]:
    if where_sql:
        query = f"SELECT DISTINCT {column} FROM arsuppgjor {where_sql} AND {column} IS NOT NULL ORDER BY {column} LIMIT {limit}"
    else:
        query = f"SELECT DISTINCT {column} FROM arsuppgjor WHERE {column} IS NOT NULL ORDER BY {column} LIMIT {limit}"
    return [row[0] for row in con.execute(query, params).fetchall()]


def _distinct_non_null(con: duckdb.DuckDBPyConnection, column: str, limit: int = 1000) -> list[str]:
    query = f"SELECT DISTINCT {column} FROM arsuppgjor WHERE {column} IS NOT NULL ORDER BY {column} LIMIT {limit}"
    return [row[0] for row in con.execute(query).fetchall()]


def _select_preview_columns(
    columns: Iterable[str],
    category: str | None,
    value: str | None,
    numeric_cols: Iterable[str],
    limit: int = 12,
) -> list[str]:
    excluded = {"source_file", "source_url", "tgr1", "tgr2", "tgr3"}
    preferred: list[str] = []
    for col in ("vm_numer", "fyrirtaeki", category, value):
        if col and col in columns and col not in preferred and col not in excluded and not _is_excluded_column(col):
            preferred.append(col)
    for col in numeric_cols:
        if col in columns and col not in preferred and col not in excluded and not _is_excluded_column(col):
            preferred.append(col)
    for col in columns:
        if col in preferred or col in excluded or _is_excluded_column(col):
            continue
        preferred.append(col)
        if len(preferred) >= limit:
            break
    # Ensure samtala0-3 appear after ingested_at and before tegund1 if present.
    samtala_order = ["samtala0", "samtala1", DEPARTMENT_COLUMN, "samtala3"]
    if any(col in preferred for col in samtala_order):
        preferred = [col for col in preferred if col not in samtala_order]
        insert_after = "ingested_at" if "ingested_at" in preferred else None
        insert_before = "tegund1" if "tegund1" in preferred else None
        ordered = [col for col in samtala_order if col in columns and col not in preferred and not _is_excluded_column(col)]
        if ordered:
            if insert_after and insert_after in preferred:
                idx = preferred.index(insert_after) + 1
                for col in reversed(ordered):
                    preferred.insert(idx, col)
            elif insert_before and insert_before in preferred:
                idx = preferred.index(insert_before)
                for col in reversed(ordered):
                    preferred.insert(idx, col)
            else:
                preferred.extend(ordered)
    if "vm_numer" in preferred:
        preferred = [col for col in preferred if col != "vm_numer"]
    if "raun" in preferred:
        preferred = [col for col in preferred if col != "raun"]
    if "vm_numer" in columns and not _is_excluded_column("vm_numer"):
        preferred.append("vm_numer")
    if "raun" in columns and not _is_excluded_column("raun"):
        preferred.append("raun")
    return preferred[:limit]


def _format_preview_rows(rows: list[dict], columns: list[str]) -> list[dict]:
    formatted = []
    for row in rows:
        out = dict(row)
        if "raun" in columns and out.get("raun") is not None:
            out["raun"] = _format_number(out.get("raun"))
        formatted.append(out)
    return formatted


def _format_number(value) -> str:
    if value is None:
        return ""
    number = _parse_number(value)
    if number is None:
        return str(value)
    if number.is_integer():
        return f"{int(number):,}".replace(",", ".")
    formatted = f"{number:,.2f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def _parse_number(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    # Interpret Icelandic-style numbers: '.' thousands, ',' decimal.
    text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _build_links(
    items: Iterable,
    current: str,
    key: str,
    base_params: dict,
    label_fn=None,
    overrides: dict | None = None,
) -> list[dict]:
    links = []
    for item in items:
        value = str(item)
        params = dict(base_params)
        if overrides:
            params.update(overrides)
        params[key] = value
        label = label_fn(item) if label_fn else value
        links.append({
            "label": label,
            "value": value,
            "url": url_for("index", **params),
            "active": value == current,
        })
    return links


def _build_filter_url(col: str, value: str, base_params: dict) -> str:
    params = dict(base_params)
    params[f"f_{col}"] = value
    return url_for("index", **params)


def _analysis_url(base: dict, **updates) -> str:
    params = dict(base)
    params.update(updates)
    return url_for("analysis", **params)


def _anomalies_url(base: dict, **updates) -> str:
    params = dict(base)
    params.update(updates)
    return url_for("anomalies", **params)


def _reports_url(base: dict, **updates) -> str:
    params = dict(base)
    params.update(updates)
    return url_for("reports", **params)


def _analysis_scope_from_request(
    con: duckdb.DuckDBPyConnection,
) -> tuple[str, str, str, str, str, list[str], list[str], str]:
    parent_type = request.args.get("parent_type", "group")
    if parent_type not in ANALYSIS_PARENT_COLUMNS:
        parent_type = "group"
    parent_col = ANALYSIS_PARENT_COLUMNS[parent_type]

    columns = _get_columns(con)
    child_keys = [c for c in ANALYSIS_CHILD_COLUMNS if c in columns]
    child_key = request.args.get("child_key") or (child_keys[0] if child_keys else "")
    if child_key not in child_keys:
        child_key = child_keys[0] if child_keys else ""

    parent_options = _distinct_non_null(con, parent_col, limit=1500) if parent_col in columns else []
    parent_value = request.args.get("parent_value") or (parent_options[0] if parent_options else "")
    if parent_value and parent_value not in parent_options:
        parent_value = parent_options[0] if parent_options else ""

    child_value = request.args.get("child_value", "")
    analysis_year = request.args.get("analysis_year", "all")
    return parent_type, parent_col, child_key, parent_value, child_value, child_keys, parent_options, analysis_year


@app.route("/")
def index():
    parquet_path = DEFAULT_PARQUET
    con = get_connection(parquet_path)
    if not _table_exists(con):
        return render_template(
            "index.html",
            parquet_path=parquet_path,
            data_loaded=False,
            error=f"No data found at {parquet_path}. Run the pipeline first.",
        )

    columns = _get_columns(con)
    years = _available_years(con)
    categories = _category_options(con)
    values = _value_options(con)

    # Defaults
    year = request.args.get("year") or (str(years[-1]) if years else "all")
    category = request.args.get("category") or (categories[0] if categories else "none")
    value = request.args.get("value") or (values[0] if values else "none")
    filter_value = request.args.get("filter_value") or "all"
    limit = request.args.get("limit") or "50"

    try:
        limit_int = max(1, min(500, int(limit)))
    except ValueError:
        limit_int = 50
        limit = "50"

    value_locked = False
    if "raun" in columns:
        value = "raun"
        value_locked = True
    elif value not in values and value != "none":
        value = values[0] if values else "none"

    filters: dict[str, str] = {}
    for col in FILTER_COLUMNS:
        val = request.args.get(f"f_{col}") or "all"
        if val != "all":
            filters[col] = val
    if category and category != "none" and filter_value != "all" and category in FILTER_COLUMNS and category not in filters:
        filters[category] = filter_value

    where_sql, where_params = _build_where(year, filters)

    # Summary tables
    summary_rows = []
    metric_key = None
    metric_label = None
    metric_display_key = None
    if category and category != "none" and category in columns:
        metric_key = "raun"
        metric_label = "raun"
        numeric_expr = _numeric_expr("raun")
        summary_query = f"""
            SELECT {category}, SUM({numeric_expr}) AS raun
            FROM arsuppgjor
            {where_sql}
            GROUP BY {category}
            ORDER BY raun DESC NULLS LAST
            LIMIT 50
        """.strip()
        summary_df = con.execute(summary_query, where_params).fetchdf()
        summary_rows = summary_df.to_dict(orient="records")
        for row in summary_rows:
            row["_metric_display"] = _format_number(row.get("raun"))
        metric_display_key = "_metric_display"

    totals = {"rows": 0}
    totals_display_sum = None
    totals_display_pos = None
    totals_display_neg = None
    if value and value != "none" and value in columns and value != category:
        numeric_expr = _numeric_expr(value)
        totals_query = f"""
            SELECT
                COUNT(*) AS rows,
                SUM({numeric_expr}) AS sum,
                SUM(CASE WHEN {numeric_expr} > 0 THEN {numeric_expr} END) AS sum_pos,
                SUM(CASE WHEN {numeric_expr} < 0 THEN {numeric_expr} END) AS sum_neg
            FROM arsuppgjor
            {where_sql}
        """.strip()
        totals_row = con.execute(totals_query, where_params).fetchone()
        totals["rows"] = int(totals_row[0] or 0)
        totals["sum"] = float(totals_row[1] or 0)
        totals["sum_pos"] = float(totals_row[2] or 0)
        totals["sum_neg"] = float(totals_row[3] or 0)
        metric_label = metric_label or value
        if value == "raun":
            totals_display_sum = _format_number(totals["sum"])
            totals_display_pos = _format_number(totals["sum_pos"])
            totals_display_neg = _format_number(totals["sum_neg"])

    filter_values = []
    if category and category != "none" and category in columns:
        filter_query = f"""
            SELECT {category}, COUNT(*) AS count
            FROM arsuppgjor
            {where_sql}
            GROUP BY {category}
            ORDER BY count DESC
            LIMIT 50
        """.strip()
        filter_values = con.execute(filter_query, where_params).fetchall()

    filter_options: dict[str, list[str]] = {}
    for col in FILTER_COLUMNS:
        if col not in columns:
            continue
        col_where, col_params = _build_where(year, filters, exclude_col=col)
        filter_options[col] = _distinct_values(con, col, col_where, col_params, limit=200)

    preview_columns = _select_preview_columns(
        columns,
        category,
        value,
        values,
        limit=12,
    )
    if preview_columns:
        cols_sql = ", ".join(preview_columns)
        preview_query = f"SELECT {cols_sql} FROM arsuppgjor {where_sql} LIMIT ?"
        preview_rows = con.execute(preview_query, [*where_params, limit_int]).fetchdf().to_dict(orient="records")
    else:
        preview_rows = []
    preview_rows = _format_preview_rows(preview_rows, preview_columns)

    base_params = {
        "year": year,
        "category": category,
        "value": value,
        "filter_value": filter_value,
        "limit": limit,
    }
    for col in FILTER_COLUMNS:
        base_params[f"f_{col}"] = filters.get(col, "all")
    year_links = _build_links(["all"] + [str(y) for y in years], year, "year", base_params, label_fn=lambda x: "All" if x == "all" else x)
    category_links = _build_links(
        ["none"] + categories,
        category,
        "category",
        base_params,
        label_fn=lambda x: "None" if x == "none" else _display_name(x),
        overrides={"filter_value": "all"},
    )
    value_links = []
    if not value_locked:
        value_links = _build_links(["none"] + values, value, "value", base_params, label_fn=lambda x: "None" if x == "none" else x)
    filter_value_links = []
    filter_value_links.append({
        "label": "All",
        "value": "all",
        "url": url_for("index", **{**base_params, "filter_value": "all"}),
        "active": filter_value == "all",
    })
    for v, count in filter_values:
        label = f"{v} ({count})"
        params = dict(base_params)
        params["filter_value"] = v
        if category in FILTER_COLUMNS:
            params[f"f_{category}"] = v
        filter_value_links.append({
            "label": label,
            "value": v,
            "url": url_for("index", **params),
            "active": v == filter_value,
        })

    return render_template(
        "index.html",
        parquet_path=parquet_path,
        data_loaded=True,
        year=year,
        category=category,
        value=value,
        filter_value=filter_value,
        limit=limit,
        totals=totals,
        totals_display_sum=totals_display_sum,
        totals_display_pos=totals_display_pos,
        totals_display_neg=totals_display_neg,
        summary_rows=summary_rows,
        metric_key=metric_key,
        metric_label=metric_label,
        metric_display_key=metric_display_key,
        year_links=year_links,
        category_links=category_links,
        value_links=value_links,
        filter_value_links=filter_value_links,
        preview_columns=preview_columns,
        preview_rows=preview_rows,
        value_locked=value_locked,
        display_name=_display_name,
        clickable_columns=CLICKABLE_COLUMNS,
        filter_columns=FILTER_COLUMNS,
        filter_options=filter_options,
        filters=filters,
        base_params=base_params,
        build_filter_url=_build_filter_url,
    )


@app.route("/analysis")
def analysis():
    parquet_path = DEFAULT_PARQUET
    con = get_connection(parquet_path)
    if not _table_exists(con):
        return render_template(
            "analysis.html",
            data_loaded=False,
            error=f"No data found at {parquet_path}. Run the pipeline first.",
        )

    (
        parent_type,
        parent_col,
        child_key,
        parent_value,
        child_value,
        child_keys,
        parent_options,
        analysis_year,
    ) = _analysis_scope_from_request(con)
    columns = _get_columns(con)

    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1
    try:
        page_size = int(request.args.get("page_size", "100"))
    except ValueError:
        page_size = 100
    page_size = max(25, min(500, page_size))

    base_params = {
        "parent_type": parent_type,
        "parent_value": parent_value,
        "child_key": child_key,
        "child_value": child_value,
        "analysis_year": analysis_year,
        "page": page,
        "page_size": page_size,
    }

    if not parent_value or not child_key:
        return render_template(
            "analysis.html",
            data_loaded=True,
            parent_type=parent_type,
            parent_col=parent_col,
            child_key=child_key,
            parent_options=parent_options,
            child_keys=child_keys,
            parent_value=parent_value,
            child_value=child_value,
            analysis_year=analysis_year,
            parent_label=_display_name(parent_col),
            child_label=_display_name(child_key) if child_key else "",
            breakdown_rows=[],
            record_rows=[],
            record_columns=[],
            yearly_labels=[],
            yearly_values=[],
            year_links=[],
            chart_title="",
            page=page,
            page_size=page_size,
            total_records=0,
            total_pages=1,
            build_analysis_url=_analysis_url,
            base_params=base_params,
            display_name=_display_name,
        )

    numeric_expr = _numeric_expr("raun")

    scope_params = [parent_value]
    scope_where = f"{parent_col} = ?"
    if child_value:
        scope_where += f" AND {child_key} = ?"
        scope_params.append(child_value)

    graph_query = f"""
        SELECT year, SUM({numeric_expr}) AS actual_sum
        FROM arsuppgjor
        WHERE {scope_where} AND year IS NOT NULL
        GROUP BY year
        ORDER BY year
    """.strip()
    graph_rows = con.execute(graph_query, scope_params).fetchall()
    yearly_labels = [str(int(row[0])) for row in graph_rows]
    yearly_values = [float(row[1] or 0) for row in graph_rows]
    if analysis_year != "all" and analysis_year not in yearly_labels:
        analysis_year = "all"
    base_params["analysis_year"] = analysis_year

    table_where = scope_where
    table_params = list(scope_params)
    if analysis_year != "all":
        table_where += " AND year = ?"
        table_params.append(int(analysis_year))

    record_columns = [
        c
        for c in ["year", parent_col, child_key, DEPARTMENT_COLUMN, "samtala3", "tegund1", "tegund2", "tegund3", "vm_nafn", "vm_numer", "raun"]
        if c in columns
    ]
    breakdown_query = f"""
        SELECT {child_key} AS child_value, SUM({numeric_expr}) AS actual_sum, COUNT(*) AS row_count
        FROM arsuppgjor
        WHERE {table_where} AND {child_key} IS NOT NULL
        GROUP BY {child_key}
        ORDER BY ABS(actual_sum) DESC NULLS LAST
        LIMIT 500
    """.strip()
    breakdown_df = con.execute(breakdown_query, table_params).fetchdf()
    breakdown_rows = breakdown_df.to_dict(orient="records")
    for row in breakdown_rows:
        row["actual_sum_fmt"] = _format_number(row.get("actual_sum"))

    if record_columns:
        count_query = f"SELECT COUNT(*) FROM arsuppgjor WHERE {table_where}"
        total_records = int(con.execute(count_query, table_params).fetchone()[0] or 0)
        total_pages = max(1, int(math.ceil(total_records / page_size)))
        if page > total_pages:
            page = total_pages
            base_params["page"] = page
        offset = (page - 1) * page_size

        records_query = f"""
            SELECT {", ".join(record_columns)}
            FROM arsuppgjor
            WHERE {table_where}
            ORDER BY year DESC
            LIMIT ?
            OFFSET ?
        """.strip()
        record_rows = con.execute(records_query, [*table_params, page_size, offset]).fetchdf().to_dict(orient="records")
    else:
        total_records = 0
        total_pages = 1
        record_rows = []
    record_rows = _format_preview_rows(record_rows, record_columns)

    chart_title = f"{_display_name(parent_col)}: {parent_value}"
    if child_value:
        chart_title += f" | {_display_name(child_key)}: {child_value}"

    return render_template(
        "analysis.html",
        data_loaded=True,
        parent_type=parent_type,
        parent_col=parent_col,
        child_key=child_key,
        parent_options=parent_options,
        child_keys=child_keys,
        parent_value=parent_value,
        child_value=child_value,
        analysis_year=analysis_year,
        parent_label=_display_name(parent_col),
        child_label=_display_name(child_key),
        breakdown_rows=breakdown_rows,
        record_rows=record_rows,
        record_columns=record_columns,
        yearly_labels=yearly_labels,
        yearly_values=yearly_values,
        yearly_labels_json=json.dumps(yearly_labels),
        yearly_values_json=json.dumps(yearly_values),
        year_links=["all"] + yearly_labels,
        chart_title=chart_title,
        page=page,
        page_size=page_size,
        total_records=total_records,
        total_pages=total_pages,
        build_analysis_url=_analysis_url,
        base_params=base_params,
        display_name=_display_name,
    )


@app.route("/analysis/export")
def analysis_export():
    parquet_path = DEFAULT_PARQUET
    con = get_connection(parquet_path)
    if not _table_exists(con):
        return Response("No data available", status=404)

    (
        _parent_type,
        parent_col,
        child_key,
        parent_value,
        child_value,
        _child_keys,
        _parent_options,
        analysis_year,
    ) = _analysis_scope_from_request(con)
    columns = _get_columns(con)
    if not parent_value or not child_key:
        return Response("Missing selection", status=400)

    scope_params = [parent_value]
    scope_where = f"{parent_col} = ?"
    if child_value:
        scope_where += f" AND {child_key} = ?"
        scope_params.append(child_value)
    if analysis_year != "all":
        scope_where += " AND year = ?"
        scope_params.append(int(analysis_year))

    export_columns = [
        c
        for c in [
            "year",
            parent_col,
            child_key,
            "samtala0",
            "samtala1",
            DEPARTMENT_COLUMN,
            "samtala3",
            "tegund0",
            "tegund1",
            "tegund2",
            "tegund3",
            "vm_nafn",
            "vm_numer",
            "raun",
        ]
        if c in columns
    ]
    query = f"SELECT {', '.join(export_columns)} FROM arsuppgjor WHERE {scope_where} ORDER BY year DESC"
    csv_text = con.execute(query, scope_params).fetchdf().to_csv(index=False)
    filename = f"analysis_{parent_col}.csv"
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/anomalies")
def anomalies():
    con = get_anomaly_connection(DEFAULT_ANOMALIES_PARQUET)
    if not _anomaly_table_exists(con):
        return render_template(
            "anomalies.html",
            data_loaded=False,
            error=f"No anomalies file found at {DEFAULT_ANOMALIES_PARQUET}. Run `make anomalies` first.",
        )

    year = request.args.get("year", "all")
    direction = request.args.get("direction", "all")
    parent_col = request.args.get("parent_col", "tegund0")
    if parent_col not in ANOMALY_PARENT_COLUMNS:
        parent_col = "tegund0"
    parent_value = request.args.get("parent_value", "all")

    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1
    try:
        page_size = int(request.args.get("page_size", "100"))
    except ValueError:
        page_size = 100
    page_size = max(25, min(500, page_size))

    where = []
    params: list = []
    if year != "all":
        where.append("year = ?")
        params.append(int(year))
    if direction != "all":
        where.append("direction = ?")
        params.append(direction)
    if parent_value != "all":
        where.append(f"{parent_col} = ?")
        params.append(parent_value)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    count_row = con.execute(f"SELECT COUNT(*) FROM anomalies {where_sql}", params).fetchone()
    total_records = int(count_row[0] or 0)
    total_pages = max(1, int(math.ceil(total_records / page_size)))
    if page > total_pages:
        page = total_pages
    offset = (page - 1) * page_size

    row_dims = ["samtala0", "samtala1", DEPARTMENT_COLUMN, "samtala3", "tegund0", "tegund1", "tegund2", "tegund3", "vm_nafn", "vm_numer"]
    row_dims = list(dict.fromkeys(row_dims))
    rows_query = f"""
        SELECT year, direction, anomaly_score, yoy_real_pct, yoy_real_change, actual_real, prior_real,
               {", ".join(row_dims)}
        FROM anomalies
        {where_sql}
        ORDER BY anomaly_score DESC, abs_change_real DESC
        LIMIT ?
        OFFSET ?
    """.strip()
    rows_df = con.execute(rows_query, [*params, page_size, offset]).fetchdf()
    rows = rows_df.to_dict(orient="records")
    for row in rows:
        row["anomaly_score"] = f"{float(row.get('anomaly_score') or 0):.2f}"
        row["yoy_real_pct_fmt"] = f"{float(row.get('yoy_real_pct') or 0) * 100:.1f}%"
        row["yoy_real_change_fmt"] = _format_number(row.get("yoy_real_change"))
        row["actual_real_fmt"] = _format_number(row.get("actual_real"))
        row["prior_real_fmt"] = _format_number(row.get("prior_real"))

    def _norm_key_value(v):
        if pd.isna(v):
            return None
        return str(v)

    series_key_cols = [c for c in ANOMALY_PARENT_COLUMNS if c in rows_df.columns]
    totals_map: dict[tuple, list[dict]] = {}
    if series_key_cols and not rows_df.empty:
        all_con = get_anomaly_all_connection(DEFAULT_ANOMALIES_ALL_PARQUET)
        source_view = "anomalies_all" if _anomaly_all_table_exists(all_con) else "anomalies"
        if source_view == "anomalies":
            all_con = con

        key_rows = rows_df[series_key_cols].drop_duplicates().values.tolist()
        if key_rows:
            row_placeholder = "(" + ", ".join(["?"] * len(series_key_cols)) + ")"
            values_sql = ", ".join([row_placeholder] * len(key_rows))
            key_cols_sql = ", ".join(series_key_cols)
            join_sql = " AND ".join([f"a.{c} = k.{c}" for c in series_key_cols])
            select_cols_sql = ", ".join([f"a.{c} AS {c}" for c in series_key_cols])
            group_cols_sql = ", ".join([f"a.{c}" for c in series_key_cols])
            year_totals_query = f"""
                WITH k({key_cols_sql}) AS (
                    VALUES {values_sql}
                )
                SELECT a.year, {select_cols_sql}, SUM(a.actual_real) AS actual_real_sum
                FROM {source_view} a
                JOIN k ON {join_sql}
                GROUP BY a.year, {group_cols_sql}
                ORDER BY a.year
            """.strip()
            flat_params = [item for key in key_rows for item in key]
            year_totals_df = all_con.execute(year_totals_query, flat_params).fetchdf()
            for _, r in year_totals_df.iterrows():
                key = tuple(_norm_key_value(r[c]) for c in series_key_cols)
                totals_map.setdefault(key, []).append(
                    {
                        "year": int(r["year"]),
                        "actual_real": float(r["actual_real_sum"] or 0),
                        "actual_real_fmt": _format_number(r["actual_real_sum"]),
                    }
                )
    for row in rows:
        key = tuple(_norm_key_value(row.get(c)) for c in series_key_cols)
        row["year_totals"] = totals_map.get(key, [])

    years = [str(int(r[0])) for r in con.execute("SELECT DISTINCT year FROM anomalies ORDER BY year").fetchall()]
    parent_values = [r[0] for r in con.execute(f"SELECT DISTINCT {parent_col} FROM anomalies WHERE {parent_col} IS NOT NULL ORDER BY {parent_col} LIMIT 1500").fetchall()]

    chart_where = []
    chart_params: list = []
    if parent_value != "all":
        chart_where.append(f"{parent_col} = ?")
        chart_params.append(parent_value)
    chart_where_sql = f"WHERE {' AND '.join(chart_where)}" if chart_where else ""
    chart_rows = con.execute(
        f"""
        SELECT
            year,
            COUNT(*) AS anomalies_count,
            SUM(abs_change_real) AS abs_change_sum,
            SUM(CASE WHEN yoy_real_change > 0 THEN ABS(yoy_real_change) ELSE 0 END) AS abs_change_increase,
            SUM(CASE WHEN yoy_real_change < 0 THEN ABS(yoy_real_change) ELSE 0 END) AS abs_change_decrease
        FROM anomalies
        {chart_where_sql}
        GROUP BY year
        ORDER BY year
        """.strip(),
        chart_params,
    ).fetchall()
    chart_labels = [str(int(r[0])) for r in chart_rows]
    chart_counts = [int(r[1] or 0) for r in chart_rows]
    chart_abs_change = [float(r[2] or 0) for r in chart_rows]
    chart_abs_increase = [float(r[3] or 0) for r in chart_rows]
    chart_abs_decrease = [float(r[4] or 0) for r in chart_rows]

    base_params = {
        "year": year,
        "direction": direction,
        "parent_col": parent_col,
        "parent_value": parent_value,
        "page": page,
        "page_size": page_size,
    }
    return render_template(
        "anomalies.html",
        data_loaded=True,
        rows=rows,
        years=years,
        directions=["all", "increase", "decrease"],
        year=year,
        direction=direction,
        parent_col=parent_col,
        parent_columns=ANOMALY_PARENT_COLUMNS,
        parent_value=parent_value,
        parent_values=parent_values,
        page=page,
        page_size=page_size,
        total_records=total_records,
        total_pages=total_pages,
        base_params=base_params,
        build_anomalies_url=_anomalies_url,
        display_name=_display_name,
        chart_labels_json=json.dumps(chart_labels),
        chart_counts_json=json.dumps(chart_counts),
        chart_abs_change_json=json.dumps(chart_abs_change),
        chart_abs_increase_json=json.dumps(chart_abs_increase),
        chart_abs_decrease_json=json.dumps(chart_abs_decrease),
    )


@app.route("/anomalies/export")
def anomalies_export():
    con = get_anomaly_connection(DEFAULT_ANOMALIES_PARQUET)
    if not _anomaly_table_exists(con):
        return Response("No anomalies data available", status=404)

    year = request.args.get("year", "all")
    direction = request.args.get("direction", "all")
    parent_col = request.args.get("parent_col", "tegund0")
    if parent_col not in ANOMALY_PARENT_COLUMNS:
        parent_col = "tegund0"
    parent_value = request.args.get("parent_value", "all")

    where = []
    params: list = []
    if year != "all":
        where.append("year = ?")
        params.append(int(year))
    if direction != "all":
        where.append("direction = ?")
        params.append(direction)
    if parent_value != "all":
        where.append(f"{parent_col} = ?")
        params.append(parent_value)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    csv_text = con.execute(
        f"""
        SELECT *
        FROM anomalies
        {where_sql}
        ORDER BY anomaly_score DESC, abs_change_real DESC
        """.strip(),
        params,
    ).fetchdf().to_csv(index=False)
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=anomalies_filtered.csv"},
    )


@app.route("/reports")
def reports():
    parquet_path = DEFAULT_PARQUET
    con = get_connection(parquet_path)
    if not _table_exists(con):
        return render_template(
            "reports.html",
            data_loaded=False,
            error=f"No data found at {parquet_path}. Run the pipeline first.",
        )

    columns = _get_columns(con)
    required = {"tegund0", DEPARTMENT_COLUMN, "samtala0", "year", "raun"}
    if not required.issubset(set(columns)):
        return render_template(
            "reports.html",
            data_loaded=False,
            error="Dataset is missing required columns for the Laun inspection.",
        )

    numeric_expr = _numeric_expr("raun")
    excluded_departments = [
        "Fasteignatekjur",
        "Fasteignir í umsjón Eignaskrifstofu",
        "Jöfnunarsjóðstekjur",
        "Fasteignir",
        "Útsvarstekjur",
        "Áhöld, tæki og stofnbúnaður",
        "Lóðir og lönd",
        "Götur, göngul. og opin svæði",
    ]
    excluded_placeholders = ", ".join(["?"] * len(excluded_departments))

    full_year_filter_sql = ""
    if "period_month" in columns:
        month_expr = "TRY_CAST(TRY_CAST(period_month AS DOUBLE) AS INTEGER)"
        full_year_filter_sql = f"""
            year IN (
                SELECT year
                FROM arsuppgjor
                WHERE year IS NOT NULL
                GROUP BY year
                HAVING MAX({month_expr}) >= 12
            )
        """.strip()

    def _avg_yoy_pct_for_scope(scope_clause: str, scope_params: list) -> float | None:
        yearly_where = f"{scope_clause} AND year IS NOT NULL"
        if full_year_filter_sql:
            yearly_where += f" AND {full_year_filter_sql}"
        avg_row = con.execute(
            f"""
            WITH yearly AS (
                SELECT year, SUM(CASE WHEN tegund0 = 'Laun' THEN {numeric_expr} ELSE 0 END) AS laun_year_sum
                FROM arsuppgjor
                WHERE {yearly_where}
                GROUP BY year
            ),
            deltas AS (
                SELECT
                    CASE
                        WHEN LAG(laun_year_sum) OVER (ORDER BY year) IS NULL THEN NULL
                        WHEN LAG(laun_year_sum) OVER (ORDER BY year) = 0 THEN NULL
                        ELSE ((laun_year_sum / LAG(laun_year_sum) OVER (ORDER BY year)) - 1) * 100.0
                    END AS yoy_pct
                FROM yearly
            )
            SELECT AVG(yoy_pct) AS avg_yoy_pct
            FROM deltas
            WHERE yoy_pct IS NOT NULL
            """,
            scope_params,
        ).fetchone()
        if not avg_row:
            return None
        value = avg_row[0]
        return None if value is None else float(value)

    department = request.args.get("department", "").strip()
    institution = request.args.get("institution", "").strip()
    expanded_department = request.args.get("expanded_department", "").strip()
    expanded_institution = request.args.get("expanded_institution", "").strip()
    report_year = request.args.get("report_year", "all").strip()
    try:
        report_year_int = int(report_year) if report_year != "all" else None
    except ValueError:
        report_year = "all"
        report_year_int = None
    year_filter_sql = " AND CAST(a.year AS INTEGER) = ?" if report_year_int is not None else ""
    year_filter_params = [report_year_int] if report_year_int is not None else []
    share_level = request.args.get("share_level", "").strip().lower()
    share_value = request.args.get("share_value", "").strip()
    share_department = request.args.get("share_department", "").strip()
    if share_level not in {"department", "institution"}:
        share_level = ""
        share_value = ""
        share_department = ""

    dept_df = con.execute(
        f"""
        WITH yearly AS (
            SELECT
                {DEPARTMENT_COLUMN} AS department,
                year,
                SUM(CASE WHEN tegund0 = 'Laun' THEN {numeric_expr} ELSE 0 END) AS laun_year_sum
            FROM arsuppgjor
            WHERE
                {DEPARTMENT_COLUMN} IS NOT NULL
                AND {DEPARTMENT_COLUMN} NOT IN ({excluded_placeholders})
                AND year IS NOT NULL
                {f"AND {full_year_filter_sql}" if full_year_filter_sql else ""}
            GROUP BY {DEPARTMENT_COLUMN}, year
        ),
        yearly_deltas AS (
            SELECT
                department,
                CASE
                    WHEN LAG(laun_year_sum) OVER (PARTITION BY department ORDER BY year) IS NULL THEN NULL
                    WHEN LAG(laun_year_sum) OVER (PARTITION BY department ORDER BY year) = 0 THEN NULL
                    ELSE (
                        (laun_year_sum / LAG(laun_year_sum) OVER (PARTITION BY department ORDER BY year)) - 1
                    ) * 100.0
                END AS delta_pct
            FROM yearly
        ),
        delta_avg AS (
            SELECT department, AVG(delta_pct) AS avg_increase_pct
            FROM yearly_deltas
            WHERE delta_pct IS NOT NULL
            GROUP BY department
        ),
        department_totals AS (
            SELECT
                a.{DEPARTMENT_COLUMN} AS department,
                SUM(CASE WHEN a.tegund0 = 'Laun' THEN {numeric_expr} ELSE 0 END) AS actual_sum,
                COUNT(*) AS row_count
            FROM arsuppgjor a
            WHERE a.{DEPARTMENT_COLUMN} IS NOT NULL AND a.{DEPARTMENT_COLUMN} NOT IN ({excluded_placeholders}){year_filter_sql}
            GROUP BY a.{DEPARTMENT_COLUMN}
        ),
        cost_basis AS (
            SELECT
                a.{DEPARTMENT_COLUMN} AS department,
                CAST(a.year AS INTEGER) AS year,
                COALESCE(a.samtala0, '') AS k_samtala0,
                COALESCE(a.samtala1, '') AS k_samtala1,
                COALESCE(a.samtala3, '') AS k_samtala3,
                COALESCE(a.tegund1, '') AS k_tegund1,
                COALESCE(a.tegund2, '') AS k_tegund2,
                COALESCE(a.tegund3, '') AS k_tegund3,
                COALESCE(a.vm_numer, '') AS k_vm_numer,
                SUM(CASE WHEN a.tegund0 = 'Laun' THEN {numeric_expr} ELSE 0 END) AS laun_net_detail,
                SUM(CASE WHEN a.tegund0 <> 'Laun' OR a.tegund0 IS NULL THEN {numeric_expr} ELSE 0 END) AS other_net_detail
            FROM arsuppgjor a
            WHERE a.{DEPARTMENT_COLUMN} IS NOT NULL AND a.{DEPARTMENT_COLUMN} NOT IN ({excluded_placeholders}) AND a.year IS NOT NULL{year_filter_sql}
            GROUP BY
                a.{DEPARTMENT_COLUMN},
                CAST(a.year AS INTEGER),
                COALESCE(a.samtala0, ''),
                COALESCE(a.samtala1, ''),
                COALESCE(a.samtala3, ''),
                COALESCE(a.tegund1, ''),
                COALESCE(a.tegund2, ''),
                COALESCE(a.tegund3, ''),
                COALESCE(a.vm_numer, '')
        ),
        cost_totals AS (
            SELECT
                department,
                SUM(GREATEST(laun_net_detail, 0)) AS laun_cost_sum,
                SUM(GREATEST(other_net_detail, 0)) AS other_sum
            FROM cost_basis
            GROUP BY department
        )
        SELECT
            t.department AS department,
            t.actual_sum AS actual_sum,
            c.other_sum AS other_sum,
            c.laun_cost_sum AS laun_cost_sum,
            t.row_count AS row_count,
            d.avg_increase_pct AS avg_increase_pct
        FROM department_totals t
        LEFT JOIN delta_avg d ON d.department = t.department
        LEFT JOIN cost_totals c ON c.department = t.department
        ORDER BY actual_sum DESC NULLS LAST
        """,
        excluded_departments
        + excluded_departments
        + year_filter_params
        + excluded_departments
        + year_filter_params,
    ).fetchdf()
    department_rows = dept_df.to_dict(orient="records")
    for row in department_rows:
        row["actual_sum_fmt"] = _format_number(row.get("actual_sum"))
        avg_increase_pct = row.get("avg_increase_pct")
        row["avg_increase_fmt"] = "" if avg_increase_pct is None else f"{float(avg_increase_pct):.1f}%"
        row["other_sum_fmt"] = _format_number(row.get("other_sum"))
        total_sum = (row.get("laun_cost_sum") or 0) + (row.get("other_sum") or 0)
        row["laun_share_pct"] = (float(row.get("laun_cost_sum") or 0) / total_sum * 100.0) if total_sum else 0.0
        row["laun_share_pct_fmt"] = f"{row['laun_share_pct']:.1f}%"
    dept_total_actual = sum(float(r.get("actual_sum") or 0) for r in department_rows)
    dept_total_other = sum(float(r.get("other_sum") or 0) for r in department_rows)
    dept_total_laun_cost = sum(float(r.get("laun_cost_sum") or 0) for r in department_rows)
    dept_total_rows = int(sum(int(r.get("row_count") or 0) for r in department_rows))
    dept_total_den = dept_total_laun_cost + dept_total_other
    dept_total_share = (dept_total_laun_cost / dept_total_den * 100.0) if dept_total_den else 0.0
    dept_total_avg = _avg_yoy_pct_for_scope(
        f"{DEPARTMENT_COLUMN} IS NOT NULL AND {DEPARTMENT_COLUMN} NOT IN ({excluded_placeholders})",
        excluded_departments,
    )
    department_total_row = {
        "label": "Total",
        "actual_sum_fmt": _format_number(dept_total_actual),
        "avg_increase_fmt": "" if dept_total_avg is None else f"{dept_total_avg:.1f}%",
        "other_sum_fmt": _format_number(dept_total_other),
        "laun_share_pct_fmt": f"{dept_total_share:.1f}%",
        "row_count": dept_total_rows,
    }

    if department and (department in set(excluded_departments) or department not in set(dept_df["department"].astype(str))):
        department = ""
    if expanded_department and (
        expanded_department in set(excluded_departments)
        or expanded_department not in set(dept_df["department"].astype(str))
    ):
        expanded_department = ""
    if expanded_department and not department:
        department = expanded_department

    valid_departments = set(dept_df["department"].astype(str))
    if share_level == "department":
        if not share_value or share_value not in valid_departments:
            share_level = ""
            share_value = ""
            share_department = ""
        else:
            share_department = share_value

    institution_rows = []
    institution_total_row = None
    if department:
        inst_df = con.execute(
            f"""
            WITH yearly AS (
                SELECT
                    samtala0 AS institution,
                    year,
                    SUM(CASE WHEN tegund0 = 'Laun' THEN {numeric_expr} ELSE 0 END) AS laun_year_sum
                FROM arsuppgjor
                WHERE {DEPARTMENT_COLUMN} = ? AND samtala0 IS NOT NULL AND year IS NOT NULL {f"AND {full_year_filter_sql}" if full_year_filter_sql else ""}
                GROUP BY samtala0, year
            ),
            yearly_deltas AS (
                SELECT
                    institution,
                    CASE
                        WHEN LAG(laun_year_sum) OVER (PARTITION BY institution ORDER BY year) IS NULL THEN NULL
                        WHEN LAG(laun_year_sum) OVER (PARTITION BY institution ORDER BY year) = 0 THEN NULL
                        ELSE (
                            (laun_year_sum / LAG(laun_year_sum) OVER (PARTITION BY institution ORDER BY year)) - 1
                        ) * 100.0
                    END AS delta_pct
                FROM yearly
            ),
            delta_avg AS (
                SELECT institution, AVG(delta_pct) AS avg_increase_pct
                FROM yearly_deltas
                WHERE delta_pct IS NOT NULL
                GROUP BY institution
            ),
            institution_totals AS (
                SELECT
                    a.samtala0 AS institution,
                    SUM(CASE WHEN a.tegund0 = 'Laun' THEN {numeric_expr} ELSE 0 END) AS actual_sum,
                    COUNT(*) AS row_count
                FROM arsuppgjor a
                WHERE a.{DEPARTMENT_COLUMN} = ? AND a.samtala0 IS NOT NULL{year_filter_sql}
                GROUP BY a.samtala0
            ),
            cost_basis AS (
                SELECT
                    a.samtala0 AS institution,
                    CAST(a.year AS INTEGER) AS year,
                    COALESCE(a.samtala1, '') AS k_samtala1,
                    COALESCE(a.samtala3, '') AS k_samtala3,
                    COALESCE(a.tegund1, '') AS k_tegund1,
                    COALESCE(a.tegund2, '') AS k_tegund2,
                    COALESCE(a.tegund3, '') AS k_tegund3,
                    COALESCE(a.vm_numer, '') AS k_vm_numer,
                    SUM(CASE WHEN a.tegund0 = 'Laun' THEN {numeric_expr} ELSE 0 END) AS laun_net_detail,
                    SUM(CASE WHEN a.tegund0 <> 'Laun' OR a.tegund0 IS NULL THEN {numeric_expr} ELSE 0 END) AS other_net_detail
                FROM arsuppgjor a
                WHERE a.{DEPARTMENT_COLUMN} = ? AND a.samtala0 IS NOT NULL AND a.year IS NOT NULL{year_filter_sql}
                GROUP BY
                    a.samtala0,
                    CAST(a.year AS INTEGER),
                    COALESCE(a.samtala1, ''),
                    COALESCE(a.samtala3, ''),
                    COALESCE(a.tegund1, ''),
                    COALESCE(a.tegund2, ''),
                    COALESCE(a.tegund3, ''),
                    COALESCE(a.vm_numer, '')
            ),
            cost_totals AS (
                SELECT
                    institution,
                    SUM(GREATEST(laun_net_detail, 0)) AS laun_cost_sum,
                    SUM(GREATEST(other_net_detail, 0)) AS other_sum
                FROM cost_basis
                GROUP BY institution
            )
            SELECT
                t.institution AS institution,
                t.actual_sum AS actual_sum,
                c.other_sum AS other_sum,
                c.laun_cost_sum AS laun_cost_sum,
                t.row_count AS row_count,
                d.avg_increase_pct AS avg_increase_pct
            FROM institution_totals t
            LEFT JOIN delta_avg d ON d.institution = t.institution
            LEFT JOIN cost_totals c ON c.institution = t.institution
            ORDER BY actual_sum DESC NULLS LAST
            """,
            [department, department, *year_filter_params, department, *year_filter_params],
        ).fetchdf()
        institution_rows = inst_df.to_dict(orient="records")
        for row in institution_rows:
            row["actual_sum_fmt"] = _format_number(row.get("actual_sum"))
            avg_increase_pct = row.get("avg_increase_pct")
            row["avg_increase_fmt"] = "" if avg_increase_pct is None else f"{float(avg_increase_pct):.1f}%"
            row["other_sum_fmt"] = _format_number(row.get("other_sum"))
            total_sum = (row.get("laun_cost_sum") or 0) + (row.get("other_sum") or 0)
            row["laun_share_pct"] = (float(row.get("laun_cost_sum") or 0) / total_sum * 100.0) if total_sum else 0.0
            row["laun_share_pct_fmt"] = f"{row['laun_share_pct']:.1f}%"
        inst_total_actual = sum(float(r.get("actual_sum") or 0) for r in institution_rows)
        inst_total_other = sum(float(r.get("other_sum") or 0) for r in institution_rows)
        inst_total_laun_cost = sum(float(r.get("laun_cost_sum") or 0) for r in institution_rows)
        inst_total_rows = int(sum(int(r.get("row_count") or 0) for r in institution_rows))
        inst_total_den = inst_total_laun_cost + inst_total_other
        inst_total_share = (inst_total_laun_cost / inst_total_den * 100.0) if inst_total_den else 0.0
        inst_total_avg = _avg_yoy_pct_for_scope(f"{DEPARTMENT_COLUMN} = ? AND samtala0 IS NOT NULL", [department])
        institution_total_row = {
            "label": "Total",
            "actual_sum_fmt": _format_number(inst_total_actual),
            "avg_increase_fmt": "" if inst_total_avg is None else f"{inst_total_avg:.1f}%",
            "other_sum_fmt": _format_number(inst_total_other),
            "laun_share_pct_fmt": f"{inst_total_share:.1f}%",
            "row_count": inst_total_rows,
        }
        if institution and institution not in set(inst_df["institution"].astype(str)):
            institution = ""
    else:
        institution = ""
    if expanded_institution and not expanded_department:
        expanded_institution = ""
    if expanded_institution and institution == "":
        institution = expanded_institution

    if share_level == "institution":
        if not share_department or share_department not in valid_departments or not share_value:
            share_level = ""
            share_value = ""
            share_department = ""
        else:
            check = con.execute(
                f"SELECT 1 FROM arsuppgjor WHERE {DEPARTMENT_COLUMN} = ? AND samtala0 = ? LIMIT 1",
                [share_department, share_value],
            ).fetchone()
            if not check:
                share_level = ""
                share_value = ""
                share_department = ""

    expanded_institution_rows = []
    expanded_actual_rows = []
    expanded_actual_columns = []
    expanded_actual_total = 0
    if expanded_department:
        expanded_inst_df = con.execute(
            f"""
            WITH institution_totals AS (
                SELECT
                    a.samtala0 AS institution,
                    SUM(CASE WHEN a.tegund0 = 'Laun' THEN {numeric_expr} ELSE 0 END) AS actual_sum,
                    COUNT(*) AS row_count
                FROM arsuppgjor a
                WHERE a.{DEPARTMENT_COLUMN} = ? AND a.samtala0 IS NOT NULL{year_filter_sql}
                GROUP BY a.samtala0
            ),
            cost_basis AS (
                SELECT
                    a.samtala0 AS institution,
                    CAST(a.year AS INTEGER) AS year,
                    COALESCE(a.samtala1, '') AS k_samtala1,
                    COALESCE(a.samtala3, '') AS k_samtala3,
                    COALESCE(a.tegund1, '') AS k_tegund1,
                    COALESCE(a.tegund2, '') AS k_tegund2,
                    COALESCE(a.tegund3, '') AS k_tegund3,
                    COALESCE(a.vm_numer, '') AS k_vm_numer,
                    SUM(CASE WHEN a.tegund0 = 'Laun' THEN {numeric_expr} ELSE 0 END) AS laun_net_detail,
                    SUM(CASE WHEN a.tegund0 <> 'Laun' OR a.tegund0 IS NULL THEN {numeric_expr} ELSE 0 END) AS other_net_detail
                FROM arsuppgjor a
                WHERE a.{DEPARTMENT_COLUMN} = ? AND a.samtala0 IS NOT NULL AND a.year IS NOT NULL{year_filter_sql}
                GROUP BY
                    a.samtala0,
                    CAST(a.year AS INTEGER),
                    COALESCE(a.samtala1, ''),
                    COALESCE(a.samtala3, ''),
                    COALESCE(a.tegund1, ''),
                    COALESCE(a.tegund2, ''),
                    COALESCE(a.tegund3, ''),
                    COALESCE(a.vm_numer, '')
            ),
            cost_totals AS (
                SELECT
                    institution,
                    SUM(GREATEST(laun_net_detail, 0)) AS laun_cost_sum,
                    SUM(GREATEST(other_net_detail, 0)) AS other_sum
                FROM cost_basis
                GROUP BY institution
            )
            SELECT
                t.institution AS institution,
                t.actual_sum AS actual_sum,
                t.row_count AS row_count,
                c.laun_cost_sum AS laun_cost_sum,
                c.other_sum AS other_sum
            FROM institution_totals t
            LEFT JOIN cost_totals c ON c.institution = t.institution
            ORDER BY actual_sum DESC NULLS LAST
            """,
            [expanded_department, *year_filter_params, expanded_department, *year_filter_params],
        ).fetchdf()
        expanded_institution_rows = expanded_inst_df.to_dict(orient="records")
        for row in expanded_institution_rows:
            row["actual_sum_fmt"] = _format_number(row.get("actual_sum"))
            share_den = (row.get("laun_cost_sum") or 0) + (row.get("other_sum") or 0)
            row["laun_share_pct"] = (float(row.get("laun_cost_sum") or 0) / share_den * 100.0) if share_den else 0.0
            row["laun_share_pct_fmt"] = f"{row['laun_share_pct']:.1f}%"

        if expanded_institution and expanded_institution not in set(expanded_inst_df["institution"].astype(str)):
            expanded_institution = ""
        elif expanded_institution and not institution:
            institution = expanded_institution

        row_scope_where = [f"{DEPARTMENT_COLUMN} = ?"]
        row_scope_params: list = [expanded_department]
        if expanded_institution:
            row_scope_where.append("samtala0 = ?")
            row_scope_params.append(expanded_institution)
        if report_year_int is not None:
            row_scope_where.append("CAST(year AS INTEGER) = ?")
            row_scope_params.append(report_year_int)
        where_sql = " AND ".join(row_scope_where)
        expanded_actual_columns = [
            c
            for c in [
                "year",
                "samtala0",
                "samtala1",
                DEPARTMENT_COLUMN,
                "samtala3",
                "tegund0",
                "tegund1",
                "tegund2",
                "tegund3",
                "vm_nafn",
                "vm_numer",
                "raun",
            ]
            if c in columns
        ]
        expanded_actual_total = int(
            con.execute(
                f"SELECT COUNT(*) FROM arsuppgjor WHERE {where_sql}",
                row_scope_params,
            ).fetchone()[0]
            or 0
        )
        if expanded_actual_columns:
            expanded_actual_rows = con.execute(
                f"""
                SELECT {", ".join(expanded_actual_columns)}
                FROM arsuppgjor
                WHERE {where_sql}
                ORDER BY year DESC
                LIMIT 200
                """,
                row_scope_params,
            ).fetchdf().to_dict(orient="records")
            expanded_actual_rows = _format_preview_rows(expanded_actual_rows, expanded_actual_columns)

    base_filter = f"tegund0 = 'Laun' AND {DEPARTMENT_COLUMN} NOT IN ({excluded_placeholders})"
    scope_where = f"{base_filter}"
    params: list = list(excluded_departments)
    scope_label = "Laun"
    if department:
        scope_where += f" AND {DEPARTMENT_COLUMN} = ?"
        params.append(department)
        scope_label += f" | {_display_name(DEPARTMENT_COLUMN)}: {department}"
    if institution:
        scope_where += " AND samtala0 = ?"
        params.append(institution)
        scope_label += f" | {_display_name('samtala0')}: {institution}"

    yearly_df = con.execute(
        f"""
        SELECT year, SUM({numeric_expr}) AS actual_sum
        FROM arsuppgjor
        WHERE {scope_where} AND year IS NOT NULL
        GROUP BY year
        ORDER BY year
        """,
        params,
    ).fetchdf()
    yearly_rows = yearly_df.to_dict(orient="records")
    for row in yearly_rows:
        row["actual_sum_fmt"] = _format_number(row.get("actual_sum"))
    yearly_labels = [str(int(y)) for y in yearly_df["year"].tolist()] if not yearly_df.empty else []
    yearly_values = [float(v or 0) for v in yearly_df["actual_sum"].tolist()] if not yearly_df.empty else []

    share_scope_label = ""
    share_labels: list[str] = []
    share_values: list[float] = []
    if share_level:
        share_where = ["a.year IS NOT NULL"]
        share_params: list = []
        if share_level == "department":
            share_where.append(f"a.{DEPARTMENT_COLUMN} = ?")
            share_params.append(share_value)
            share_scope_label = f"{_display_name(DEPARTMENT_COLUMN)}: {share_value}"
        elif share_level == "institution":
            share_where.append(f"a.{DEPARTMENT_COLUMN} = ?")
            share_where.append("a.samtala0 = ?")
            share_params.extend([share_department, share_value])
            share_scope_label = f"{_display_name('samtala0')}: {share_value} ({_display_name(DEPARTMENT_COLUMN)}: {share_department})"
        share_df = con.execute(
            f"""
            WITH cost_basis AS (
                SELECT
                    CAST(a.year AS INTEGER) AS year,
                    COALESCE(a.samtala1, '') AS k_samtala1,
                    COALESCE(a.samtala3, '') AS k_samtala3,
                    COALESCE(a.tegund1, '') AS k_tegund1,
                    COALESCE(a.tegund2, '') AS k_tegund2,
                    COALESCE(a.tegund3, '') AS k_tegund3,
                    COALESCE(a.vm_numer, '') AS k_vm_numer,
                    SUM(CASE WHEN a.tegund0 = 'Laun' THEN {numeric_expr} ELSE 0 END) AS laun_net_detail,
                    SUM(CASE WHEN a.tegund0 <> 'Laun' OR a.tegund0 IS NULL THEN {numeric_expr} ELSE 0 END) AS other_net_detail
                FROM arsuppgjor a
                WHERE {" AND ".join(share_where)}
                GROUP BY
                    CAST(a.year AS INTEGER),
                    COALESCE(a.samtala1, ''),
                    COALESCE(a.samtala3, ''),
                    COALESCE(a.tegund1, ''),
                    COALESCE(a.tegund2, ''),
                    COALESCE(a.tegund3, ''),
                    COALESCE(a.vm_numer, '')
            ),
            yearly AS (
                SELECT
                    year,
                    SUM(GREATEST(laun_net_detail, 0)) AS laun_cost_sum,
                    SUM(GREATEST(other_net_detail, 0)) AS other_sum
                FROM cost_basis
                GROUP BY year
            )
            SELECT
                year,
                CASE
                    WHEN (laun_cost_sum + other_sum) > 0 THEN (laun_cost_sum / (laun_cost_sum + other_sum)) * 100.0
                    ELSE NULL
                END AS laun_share_pct
            FROM yearly
            ORDER BY year
            """,
            share_params,
        ).fetchdf()
        if not share_df.empty:
            share_labels = [str(int(y)) for y in share_df["year"].tolist()]
            share_values = [float(v) if v is not None else 0.0 for v in share_df["laun_share_pct"].tolist()]

    base_params = {
        "department": department,
        "institution": institution,
        "expanded_department": expanded_department,
        "expanded_institution": expanded_institution,
        "report_year": report_year,
        "share_level": share_level,
        "share_value": share_value,
        "share_department": share_department,
    }
    return render_template(
        "reports.html",
        data_loaded=True,
        inspection_name="Laun by Department/Institution",
        department=department,
        institution=institution,
        department_rows=department_rows,
        department_total_row=department_total_row,
        institution_rows=institution_rows,
        institution_total_row=institution_total_row,
        expanded_department=expanded_department,
        expanded_institution=expanded_institution,
        expanded_institution_rows=expanded_institution_rows,
        expanded_actual_rows=expanded_actual_rows,
        expanded_actual_columns=expanded_actual_columns,
        expanded_actual_total=expanded_actual_total,
        yearly_rows=yearly_rows,
        report_year=report_year,
        report_year_links=["all"] + yearly_labels,
        share_level=share_level,
        share_value=share_value,
        share_department=share_department,
        share_scope_label=share_scope_label,
        share_labels_json=json.dumps(share_labels),
        share_values_json=json.dumps(share_values),
        yearly_labels_json=json.dumps(yearly_labels),
        yearly_values_json=json.dumps(yearly_values),
        scope_label=scope_label,
        department_label=_display_name(DEPARTMENT_COLUMN),
        institution_label=_display_name("samtala0"),
        build_reports_url=_reports_url,
        base_params=base_params,
        display_name=_display_name,
    )


@app.route("/reload")
def reload_data():
    get_connection.cache_clear()
    get_anomaly_connection.cache_clear()
    get_anomaly_all_connection.cache_clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
