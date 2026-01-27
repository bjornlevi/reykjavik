#!/usr/bin/env python3
"""Flask app for exploring Reykjavik arsuppgjor data."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
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
    con.execute(f"CREATE OR REPLACE VIEW arsuppgjor AS SELECT * FROM read_parquet('{str(path)}')")
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


def _build_where(year: str, category: str, filter_value: str) -> tuple[str, list]:
    clauses = []
    params: list = []
    if year and year != "all":
        try:
            year_value = int(year)
            clauses.append("year = ?")
            params.append(year_value)
        except ValueError:
            pass
    if category and category != "none" and filter_value and filter_value != "all":
        clauses.append(f"{category} = ?")
        params.append(filter_value)
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

    where_sql, where_params = _build_where(year, category, filter_value)

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
    )


@app.route("/reload")
def reload_data():
    get_connection.cache_clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
