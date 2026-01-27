#!/usr/bin/env python3
"""Combine arsuppgjor CSVs into a single exploration-ready dataset."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")
PERIOD_RE = re.compile(r"(20\d{2})(\d{2})")


def _normalize_column(name: str) -> str:
    # Strip accents and normalize to ASCII for consistent column names
    normalized = unicodedata.normalize("NFKD", name)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")
    normalized = re.sub(r"_+", "_", normalized).lower()
    return normalized or "col"


def _make_unique(columns: list[str]) -> list[str]:
    seen: Dict[str, int] = {}
    result: list[str] = []
    for col in columns:
        count = seen.get(col, 0)
        if count == 0:
            result.append(col)
        else:
            result.append(f"{col}_{count + 1}")
        seen[col] = count + 1
    return result


def _read_manifest(raw_dir: Path) -> dict:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}


def _url_lookup(manifest: dict) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for entry in manifest.get("files", []):
        filename = entry.get("filename")
        url = entry.get("url")
        if filename and url:
            lookup[filename] = url
    return lookup


def _parse_year_period(filename: str) -> Tuple[int | None, int | None]:
    year = None
    period_month = None

    year_match = YEAR_RE.search(filename)
    if year_match:
        year = int(year_match.group(1))

    period_match = PERIOD_RE.search(filename)
    if period_match:
        year = int(period_match.group(1))
        month = int(period_match.group(2))
        if 1 <= month <= 12:
            period_month = month

    return year, period_month


def _read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "iso-8859-1"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                dtype=str,
                sep=None,
                engine="python",
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(path, dtype=str, sep=None, engine="python")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare arsuppgjor dataset")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted([p for p in raw_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        print("No CSV files found in raw directory.", file=sys.stderr)
        return 1

    manifest = _read_manifest(raw_dir)
    url_lookup = _url_lookup(manifest)

    frames = []
    column_maps = []
    ingested_at = datetime.now(timezone.utc).isoformat()

    for path in csv_files:
        try:
            df = _read_csv(path)
        except Exception as exc:
            print(f"Failed to read {path.name}: {exc}", file=sys.stderr)
            return 2

        year, period_month = _parse_year_period(path.name)

        df.insert(0, "source_file", path.name)
        df.insert(1, "source_url", url_lookup.get(path.name))
        df.insert(2, "year", year)
        # Skip month column entirely per UI requirements.
        df.insert(3, "ingested_at", ingested_at)

        original_columns = list(df.columns)
        normalized = _make_unique([_normalize_column(col) for col in original_columns])
        df.columns = normalized
        column_maps.append({"file": path.name, "columns": dict(zip(original_columns, normalized))})

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    parquet_path = processed_dir / "arsuppgjor_combined.parquet"
    combined.to_parquet(parquet_path, index=False)

    metadata = {
        "generated_at": ingested_at,
        "rows": len(combined),
        "files": [p.name for p in csv_files],
        "column_maps": column_maps,
    }
    metadata_path = processed_dir / "arsuppgjor_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True))

    print(f"Wrote {parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
