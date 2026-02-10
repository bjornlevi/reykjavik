#!/usr/bin/env python3
"""Detect unusual yearly expenditure changes using CPI-adjusted actual amounts."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

CPI_API_URL = "https://px.hagstofa.is:443/pxis/api/v1/is/Efnahagur/visitolur/1_vnv/1_vnv/VIS01000.px"

DEFAULT_KEY_COLUMNS = [
    "samtala0",
    "samtala1",
    "samtala2",
    "samtala3",
    "tegund0",
    "tegund1",
    "tegund2",
    "tegund3",
    "vm_nafn",
    "vm_numer",
]


@dataclass
class Config:
    parquet: Path
    out_dir: Path
    cpi_api_url: str
    z_threshold: float
    pct_threshold: float
    min_abs_real_change: float
    min_prior_real_abs: float
    timeout: int


def _log(message: str) -> None:
    print(f"[anomalies] {message}", flush=True)


def _parse_icelandic_number(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _month_values(start_year: int, end_year: int) -> list[str]:
    vals = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            vals.append(f"{year}M{month:02d}")
    return vals


def _fetch_available_month_values(api_url: str, timeout: int) -> list[str]:
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()
    meta = response.json()
    for variable in meta.get("variables", []):
        if variable.get("code") == "Mánuður":
            return list(variable.get("values", []))
    raise RuntimeError("Could not find 'Mánuður' values in CPI metadata.")


def _cpi_payload(month_values: list[str]) -> dict:
    return {
        "query": [
            {
                "code": "Mánuður",
                "selection": {
                    "filter": "item",
                    "values": month_values,
                },
            },
            {
                "code": "Vísitala",
                "selection": {
                    "filter": "item",
                    "values": ["CPI"],
                },
            },
            {
                "code": "Liður",
                "selection": {
                    "filter": "item",
                    "values": ["index"],
                },
            },
        ],
        "response": {
            "format": "json",
        },
    }


def _fetch_cpi_monthly(start_year: int, end_year: int, api_url: str, timeout: int) -> pd.DataFrame:
    _log(f"Loading CPI metadata from {api_url}")
    requested = set(_month_values(start_year, end_year))
    available = _fetch_available_month_values(api_url, timeout)
    selected = [m for m in available if m in requested]
    if not selected:
        raise RuntimeError("No overlapping CPI months found for requested years.")
    _log(f"Requesting CPI for {len(selected)} months ({selected[0]} to {selected[-1]})")
    payload = _cpi_payload(selected)
    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    rows = []
    for item in data.get("data", []):
        keys = item.get("key", [])
        vals = item.get("values", [])
        if not keys or not vals:
            continue
        month_key = keys[0]
        cpi_value = _parse_icelandic_number(vals[0])
        if cpi_value is None:
            continue
        year = int(month_key[:4])
        month = int(month_key[-2:])
        rows.append({"year": year, "month": month, "month_key": month_key, "cpi": cpi_value})

    cpi_df = pd.DataFrame(rows)
    if cpi_df.empty:
        raise RuntimeError("CPI API returned no usable rows.")
    _log(f"CPI monthly rows loaded: {len(cpi_df)}")
    return cpi_df.sort_values(["year", "month"]).reset_index(drop=True)


def _load_actuals(parquet_path: Path, keys: Iterable[str]) -> pd.DataFrame:
    _log(f"Reading source parquet: {parquet_path}")
    base_cols = ["year", "raun"]
    df = pd.read_parquet(parquet_path)

    available_keys = [c for c in keys if c in df.columns]
    needed = [c for c in base_cols + available_keys if c in df.columns]
    df = df[needed].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["actual_nominal"] = df["raun"].apply(_parse_icelandic_number)
    df = df.drop(columns=["raun"]) 
    df = df.dropna(subset=["year", "actual_nominal"]) 
    df["year"] = df["year"].astype(int)

    for col in available_keys:
        df[col] = df[col].fillna("(missing)").astype(str)

    agg = df.groupby(["year", *available_keys], dropna=False, as_index=False)["actual_nominal"].sum()
    _log(f"Aggregated annual nominal rows: {len(agg)}")
    return agg


def _annual_cpi(cpi_monthly: pd.DataFrame) -> pd.DataFrame:
    annual = cpi_monthly.groupby("year", as_index=False)["cpi"].mean()
    annual = annual.rename(columns={"cpi": "cpi_avg"})
    if annual.empty:
        raise RuntimeError("No annual CPI values after aggregation.")
    latest = annual.sort_values("year").iloc[-1]
    annual["cpi_latest_base"] = float(latest["cpi_avg"])
    annual["deflator_to_latest"] = annual["cpi_latest_base"] / annual["cpi_avg"]
    return annual


def _modified_zscore(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    median = clean.median(skipna=True)
    mad = (clean - median).abs().median(skipna=True)
    if pd.isna(mad) or mad == 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    return 0.6745 * (clean - median) / mad


def _compute_anomalies(annual_actuals: pd.DataFrame, key_cols: list[str], cfg: Config) -> pd.DataFrame:
    sort_cols = key_cols + ["year"]
    df = annual_actuals.sort_values(sort_cols).copy()

    grp = df.groupby(key_cols, dropna=False)
    df["prior_real"] = grp["actual_real"].shift(1)
    df["yoy_real_change"] = df["actual_real"] - df["prior_real"]
    df["yoy_real_pct"] = np.where(
        df["prior_real"].abs() > 1e-9,
        df["yoy_real_change"] / df["prior_real"].abs(),
        np.nan,
    )

    df["series_year_count"] = grp["year"].transform("count")
    df["mz_yoy_pct"] = grp["yoy_real_pct"].transform(_modified_zscore)

    df["direction"] = np.where(df["yoy_real_change"] >= 0, "increase", "decrease")
    df["abs_pct"] = df["yoy_real_pct"].abs()
    df["abs_mz"] = df["mz_yoy_pct"].abs()
    df["abs_change_real"] = df["yoy_real_change"].abs()

    enough_history = df["series_year_count"] >= 4
    flagged = (
        enough_history
        & (df["abs_pct"] >= cfg.pct_threshold)
        & (df["abs_mz"] >= cfg.z_threshold)
        & (df["prior_real"].abs() >= cfg.min_prior_real_abs)
        & (df["abs_change_real"] >= cfg.min_abs_real_change)
    )

    df["is_anomaly"] = flagged
    df["anomaly_score"] = (df["abs_mz"].fillna(0) * 0.7) + (df["abs_pct"].fillna(0) * 100 * 0.3)
    return df


def run(cfg: Config) -> int:
    if not cfg.parquet.exists():
        print(f"Missing parquet: {cfg.parquet}")
        return 1

    _log("Starting anomaly detection")
    df0 = pd.read_parquet(cfg.parquet, columns=["year"])
    year_min = int(pd.to_numeric(df0["year"], errors="coerce").dropna().min())
    year_max = int(pd.to_numeric(df0["year"], errors="coerce").dropna().max())
    cpi_end_year = max(year_max, date.today().year)
    _log(f"Data year range: {year_min}-{year_max}")

    cpi_monthly = _fetch_cpi_monthly(year_min, cpi_end_year, cfg.cpi_api_url, cfg.timeout)
    cpi_annual = _annual_cpi(cpi_monthly)
    _log(f"CPI annual rows loaded: {len(cpi_annual)}")

    annual_nominal = _load_actuals(cfg.parquet, DEFAULT_KEY_COLUMNS)
    key_cols = [c for c in DEFAULT_KEY_COLUMNS if c in annual_nominal.columns]
    _log(f"Series key columns in use: {', '.join(key_cols)}")

    _log("Applying CPI deflator and computing real amounts")
    merged = annual_nominal.merge(cpi_annual[["year", "deflator_to_latest"]], on="year", how="left")
    merged = merged.dropna(subset=["deflator_to_latest"]).copy()
    merged["actual_real"] = merged["actual_nominal"] * merged["deflator_to_latest"]
    _log(f"Rows after CPI merge: {len(merged)}")

    _log("Scoring anomalies")
    anomalies = _compute_anomalies(merged, key_cols, cfg)
    flagged = anomalies[anomalies["is_anomaly"]].copy()
    flagged = flagged.sort_values(["anomaly_score", "abs_change_real"], ascending=False)
    _log(f"Flagged anomalies: {len(flagged)} of {len(anomalies)} rows")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Writing outputs to {cfg.out_dir}")

    cpi_monthly.to_csv(cfg.out_dir / "cpi_monthly.csv", index=False)
    cpi_annual.to_csv(cfg.out_dir / "cpi_annual.csv", index=False)

    anomalies.to_parquet(cfg.out_dir / "anomalies_yoy_all.parquet", index=False)
    anomalies.to_csv(cfg.out_dir / "anomalies_yoy_all.csv", index=False)
    flagged.to_parquet(cfg.out_dir / "anomalies_flagged.parquet", index=False)
    flagged.to_csv(cfg.out_dir / "anomalies_flagged.csv", index=False)

    summary = {
        "rows_total": int(len(anomalies)),
        "rows_flagged": int(len(flagged)),
        "years_covered": [int(cpi_annual["year"].min()), int(cpi_annual["year"].max())],
        "thresholds": {
            "z_threshold": cfg.z_threshold,
            "pct_threshold": cfg.pct_threshold,
            "min_abs_real_change": cfg.min_abs_real_change,
            "min_prior_real_abs": cfg.min_prior_real_abs,
        },
    }
    (cfg.out_dir / "anomalies_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True))

    _log(f"Wrote anomaly outputs to {cfg.out_dir}")
    _log(f"Done. Flagged rows: {len(flagged)} / {len(anomalies)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect CPI-adjusted expenditure anomalies")
    parser.add_argument("--parquet", default="data/processed/arsuppgjor_combined.parquet")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--cpi-api-url", default=CPI_API_URL)
    parser.add_argument("--z-threshold", type=float, default=3.5)
    parser.add_argument("--pct-threshold", type=float, default=0.25)
    parser.add_argument("--min-abs-real-change", type=float, default=5_000_000)
    parser.add_argument("--min-prior-real-abs", type=float, default=2_000_000)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args(argv)

    cfg = Config(
        parquet=Path(args.parquet),
        out_dir=Path(args.out_dir),
        cpi_api_url=args.cpi_api_url,
        z_threshold=args.z_threshold,
        pct_threshold=args.pct_threshold,
        min_abs_real_change=args.min_abs_real_change,
        min_prior_real_abs=args.min_prior_real_abs,
        timeout=args.timeout,
    )
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
