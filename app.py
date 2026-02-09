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
CLICKABLE_COLUMNS = {
    "fyrirtaeki",
    "samtala0",
    "samtala1",
    "samtala2",
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
    "samtala2",
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
    "samtala2",
    "samtala3",
    "tegund1",
    "tegund2",
    "tegund3",
]


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
        "samtala2",
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
    con.execute(f"CREATE OR REPLACE VIEW arsuppgjor AS SELECT * FROM read_parquet('{safe_path}')")
    return con


def _table_exists(con: duckdb.DuckDBPyConnection) -> bool:
    try:
        con.execute("SELECT 1 FROM arsuppgjor LIMIT 1")
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
    samtala_order = ["samtala0", "samtala1", "samtala2", "samtala3"]
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

    record_columns = [c for c in ["year", parent_col, child_key, "samtala2", "samtala3", "tegund1", "tegund2", "tegund3", "vm_nafn", "vm_numer", "raun"] if c in columns]
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

    export_columns = [c for c in ["year", parent_col, child_key, "samtala0", "samtala1", "samtala2", "samtala3", "tegund0", "tegund1", "tegund2", "tegund3", "vm_nafn", "vm_numer", "raun"] if c in columns]
    query = f"SELECT {', '.join(export_columns)} FROM arsuppgjor WHERE {scope_where} ORDER BY year DESC"
    csv_text = con.execute(query, scope_params).fetchdf().to_csv(index=False)
    filename = f"analysis_{parent_col}.csv"
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/reload")
def reload_data():
    get_connection.cache_clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
