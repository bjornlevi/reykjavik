#!/usr/bin/env python3
"""Flask app for exploring Reykjavik arsuppgjor data."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

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
    "tegund1": "Type",
    "tegund2": "Category",
    "tegund3": "Item",
    "fyrirtaeki": "Entity",
    "vm_numer": "VSK number",
    "raun": "Actual",
}
CLICKABLE_COLUMNS = {
    "fyrirtaeki",
    "samtala0",
    "samtala1",
    "samtala2",
    "samtala3",
    "tegund1",
    "tegund2",
    "tegund3",
}


def _display_name(col: str) -> str:
    return DISPLAY_NAMES.get(col, col)


def _is_numeric_series(series: pd.Series) -> bool:
    if series.empty:
        return False
    numeric = _coerce_numeric(series)
    return numeric.notna().mean() >= 0.6


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
        "period_month",
        "month",
        "ingested_at",
        "year",
    }:
        return True
    return name.startswith("x")


def _numeric_candidates(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if _is_excluded_column(col):
            continue
        if col in {"year"}:
            continue
        if _is_numeric_series(df[col].dropna().head(500)):
            candidates.append(col)
    return candidates


def _category_candidates(df: pd.DataFrame) -> list[str]:
    allowed = {
        "fyrirtaeki",
        "samtala0",
        "samtala1",
        "samtala2",
        "samtala3",
        "tegund1",
        "tegund2",
        "tegund3",
    }
    categories = []
    for col in df.columns:
        if _is_excluded_column(col):
            continue
        if col in {"source_file", "source_url", "ingested_at"}:
            continue
        if col in {"year"}:
            continue
        if col not in allowed:
            continue
        # Prefer low-cardinality columns for categories
        distinct = df[col].nunique(dropna=True)
        if 1 < distinct <= 200:
            categories.append(col)
    return categories


@lru_cache(maxsize=1)
def load_data(parquet_path: str) -> pd.DataFrame:
    path = Path(parquet_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _available_years(df: pd.DataFrame) -> list[int]:
    if "year" not in df.columns:
        return []
    years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    return sorted(years.unique().tolist())


def _apply_filters(
    df: pd.DataFrame,
    year: str | None,
    category: str | None,
    value: str | None,
    filter_value: str | None,
) -> pd.DataFrame:
    filtered = df
    if year and year != "all" and "year" in filtered.columns:
        filtered = filtered[filtered["year"].astype(str) == year]
    if category and category != "none" and filter_value and filter_value != "all":
        if category in filtered.columns:
            filtered = filtered[filtered[category].astype(str) == filter_value]
    if value and value in filtered.columns:
        # Coerce to numeric for aggregation
        filtered = filtered.copy()
        filtered[value] = _coerce_numeric(filtered[value])
    return filtered


def _value_options(df: pd.DataFrame) -> list[str]:
    candidates = _numeric_candidates(df)
    # Ensure deterministic ordering
    return sorted(candidates)


def _category_options(df: pd.DataFrame) -> list[str]:
    candidates = _category_candidates(df)
    return sorted(candidates)


def _filter_values(df: pd.DataFrame, category: str | None) -> list[str]:
    if not category or category == "none" or category not in df.columns:
        return []
    values = df[category].dropna().astype(str).unique().tolist()
    values.sort()
    return values


def _top_filter_values(df: pd.DataFrame, category: str, limit: int = 50) -> list[tuple[str, int]]:
    if category not in df.columns:
        return []
    counts = (
        df[category]
        .dropna()
        .astype(str)
        .value_counts()
        .head(limit)
    )
    return list(zip(counts.index.tolist(), counts.values.tolist()))


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


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.apply(_parse_number), errors="coerce")


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
    df = load_data(parquet_path)

    if df.empty:
        return render_template(
            "index.html",
            parquet_path=parquet_path,
            data_loaded=False,
            error=f"No data found at {parquet_path}. Run the pipeline first.",
        )

    years = _available_years(df)
    categories = _category_options(df)
    values = _value_options(df)

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
    if "raun" in df.columns:
        value = "raun"
        value_locked = True
    elif value not in values and value != "none":
        value = values[0] if values else "none"

    filtered = _apply_filters(df, year, category, value, filter_value)

    # Summary tables
    summary_rows = []
    metric_key = None
    metric_label = None
    metric_display_key = None
    if category and category != "none" and category in filtered.columns:
        if not value or value == "none" or value not in filtered.columns or value == category:
            summary = (
                filtered.groupby(category, dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            metric_key = "count"
            metric_label = "count"
        else:
            summary = (
                filtered.groupby(category, dropna=False)[value]
                .sum(min_count=1)
                .reset_index()
                .sort_values(value, ascending=False)
            )
            metric_key = value
            metric_label = value
        summary_rows = summary.head(50).to_dict(orient="records")
        if metric_key == "raun":
            for row in summary_rows:
                row["_metric_display"] = _format_number(row.get(metric_key))
            metric_display_key = "_metric_display"

    totals = {"rows": int(filtered.shape[0])}
    totals_display_sum = None
    totals_display_pos = None
    totals_display_neg = None
    if value and value != "none" and value in filtered.columns and value != category:
        numeric_series = pd.to_numeric(filtered[value], errors="coerce")
        totals["sum"] = float(numeric_series.sum(skipna=True))
        totals["sum_pos"] = float(numeric_series[numeric_series > 0].sum(skipna=True))
        totals["sum_neg"] = float(numeric_series[numeric_series < 0].sum(skipna=True))
        metric_label = metric_label or value
        if value == "raun":
            totals_display_sum = _format_number(totals["sum"])
            totals_display_pos = _format_number(totals["sum_pos"])
            totals_display_neg = _format_number(totals["sum_neg"])

    if year != "all" and "year" in df.columns:
        try:
            mask = df["year"].astype(str).to_numpy() == year
            year_filtered = df.loc[mask]
        except Exception:
            year_filtered = df
    else:
        year_filtered = df
    filter_values = _top_filter_values(year_filtered, category, limit=50)

    preview_columns = _select_preview_columns(
        filtered.columns,
        category,
        value,
        values,
        limit=12,
    )
    preview_rows = filtered.head(limit_int)[preview_columns].to_dict(orient="records")
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
    load_data.cache_clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
    if category not in categories and category != "none":
        category = categories[0] if categories else "none"
    if "raun" in df.columns:
        value = "raun"
    elif value not in values and value != "none":
        value = values[0] if values else "none"
